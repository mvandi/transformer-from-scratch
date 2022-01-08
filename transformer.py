import math
from typing import Optional

import torch
from torch import nn, BoolTensor, Tensor

import functional as F


class MultiheadAttention(nn.Module):
    """
    Masked Multi-Head Self-Attention
    """

    def __init__(self, d_model: int, h: int, bias: bool = True) -> None:
        super(self.__class__, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d = d_model // h

        if self.d * self.h != d_model:
            raise ValueError('d_model must be divisible by h')

        self.values_projection = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.keys_projection = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.queries_projection = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor, keys: Tensor, queries: Tensor, mask: Optional[BoolTensor]) -> Tensor:
        """
        :param values:  Size([N, d_k, d_model])
        :param keys:    Size([N, d_k, d_model])
        :param queries: Size([N, d_q, d_model])
        :param mask:    Size([N, 1, M, d_k])
        :return:        Size([N, d_q, d_model])
        """
        batch_size, d_q = queries.size(0), queries.size(1)

        values = self.project(self.values_projection, values)     # (N, d_k, h, d_head)
        keys = self.project(self.keys_projection, keys)           # (N, d_k, h, d_head)
        queries = self.project(self.queries_projection, queries)  # (N, d_q, h, d_head)

        out, _ = F.scaled_dot_product_attention(values, keys, queries, mask)
        # then we combine the last two dimensions
        out = out.reshape(batch_size, d_q, self.d_model)

        out = self.fc(out)
        # Linear block doesn't modify the shape, final shape will be (N, d_q, d_model)
        return out

    def project(self, projection: nn.Module, x: Tensor) -> Tensor:
        batch_size, seq_length = x.size(0), x.size(1)
        out = projection(x)
        # Split the embedding into h different pieces (heads)
        return out.reshape(batch_size, seq_length, self.h, self.d)


class ResidualLayerNorm(nn.Module):

    def __init__(self, d_model, dropout_p):
        super(self.__class__, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, residual):
        """
        :param x:        Size([N, d_q, d_model])
        :param residual: Size([N, d_q, d_model])
        :return:         Size([N, d_q, d_model])
        """
        out = self.norm(x + residual)
        out = self.dropout(out)
        return out


class PWFFN(nn.Module):
    """
    Position-wise Feed-Forward Network
    """
    activations = dict(
        relu=nn.ReLU,
        gelu=nn.GELU
    )

    def __init__(
            self, d_model: int, d_ff: Optional[int], activation: Optional[str]
    ) -> None:
        super(self.__class__, self).__init__()

        if d_ff is None:
            self.fc = nn.Linear(d_model, d_model, bias=False)
        else:
            if activation is None:
                activation = 'relu'
            activation = self.activations.get(activation)
            if activation is None:
                raise ValueError(
                    f'Activation function `{activation}` is not supported. '
                    f'Supported activations are {self.activations.keys()}'
                )

            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                activation(),
                nn.Linear(d_ff, d_model, bias=False)
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Size([N, d_q, d_model])
        :return:  Size([N, d_q, d_model])
        """
        return self.fc(x)


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int]) -> None:
        super(self.__class__, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Size([N, d_q])
        :return:  Size([N, d_q, d_model])
        """
        out = self.embedding(x)
        out *= math.sqrt(out.size(-1))
        out = F.positional_encoding(out)
        return out


class EncoderBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            dropout_p: float
    ) -> None:
        super(self.__class__, self).__init__()

        self.source_attn = MultiheadAttention(d_model, h)
        self.source_norm = ResidualLayerNorm(d_model, dropout_p)
        self.pwffn = PWFFN(d_model, d_ff, activation)
        self.pwffn_norm = ResidualLayerNorm(d_model, dropout_p)

    def forward(self, x: Tensor, mask: Optional[BoolTensor]) -> Tensor:
        """
        :param x:       Size([N, d_k, d_model])
        :param mask:    Size([N, 1, 1, d_k])
        :return:        Size([N, d_k, d_model])
        """
        residual = x
        x = self.source_attn(x, x, x, mask)
        x = self.source_norm(x, residual)
        residual = x
        x = self.pwffn(x)
        x = self.pwffn_norm(x, residual)
        return x


class Encoder(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            padding_idx: Optional[int],
            d_model: int,
            h: int,
            d_ff: Optional[int],
            dropout_p: float,
            activation: Optional[str],
            num_layers: int
    ) -> None:
        super(self.__class__, self).__init__()

        self.embedding = Embedding(vocab_size, d_model, padding_idx)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.stack = nn.ModuleList([
            EncoderBlock(d_model, h, d_ff, activation, dropout_p)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[BoolTensor]) -> Tensor:
        """
        :param x:    Size([N, d_k])
        :param mask: Size([N, 1, 1, d_k])
        :return:     Size([N, d_k, d_model])
        """
        out = self.embedding(x)
        out = self.embedding_dropout(out)
        for layer in self.stack:
            out = layer(out, mask)
        return out


class DecoderBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            dropout_p: float
    ) -> None:
        super(self.__class__, self).__init__()

        self.target_attn = MultiheadAttention(d_model, h)
        self.target_norm = ResidualLayerNorm(d_model, dropout_p)
        self.encoder_attn = MultiheadAttention(d_model, h)
        self.encoder_norm = ResidualLayerNorm(d_model, dropout_p)
        self.pwffn = PWFFN(d_model, d_ff, activation)
        self.pwffn_norm = ResidualLayerNorm(d_model, dropout_p)

    def forward(
            self, x: Tensor, z: Tensor, padding_mask: Optional[BoolTensor], lookahead_mask: Optional[BoolTensor]
    ) -> Tensor:
        """
        :param x:               Size([N, d_q, d_model])
        :param z:               Size([N, d_k, d_model])
        :param padding_mask:    Size([N, 1, 1, d_k])
        :param lookahead_mask:  Size([N, 1, d_q, d_q])
        :return:                Size([N, d_q, d_model])
        """
        residual = x
        x = self.target_attn(x, x, x, lookahead_mask)
        x = self.target_norm(x, residual)
        residual = x
        x = self.encoder_attn(z, z, x, padding_mask)
        x = self.encoder_norm(x, residual)
        residual = x
        x = self.pwffn(x)
        x = self.pwffn_norm(x, residual)
        return x


class Decoder(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            padding_idx: Optional[int],
            d_model: int,
            h: int,
            d_ff: Optional[int],
            dropout_p: float,
            activation: Optional[str],
            num_layers: int
    ) -> None:
        super(self.__class__, self).__init__()

        self.embedding = Embedding(vocab_size, d_model, padding_idx)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.stack = nn.ModuleList([
            DecoderBlock(d_model, h, d_ff, activation, dropout_p)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
            self, x: Tensor, z: Tensor, padding_mask: Optional[BoolTensor], lookahead_mask: Optional[BoolTensor]
    ) -> Tensor:
        """
        :param x:               Size([N, d_q])
        :param z:               Size([N, d_k, d_model])
        :param padding_mask:    Size([N, 1, 1, d_k])
        :param lookahead_mask:  Size([N, 1, d_q, d_q])
        :return:                Size([N, d_q, d_t])
        """
        out = self.embedding(x)
        out = self.embedding_dropout(out)
        for layer in self.stack:
            out = layer(out, z, padding_mask, lookahead_mask)
        out = self.fc(out)
        return out


class Transformer(nn.Module):

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            src_padding_idx: Optional[int],
            tgt_padding_idx: Optional[int],
            d_model: int = 512,
            h: int = 8,
            d_ff: Optional[int] = 2048,
            dropout_p: float = 0.1,
            activation: Optional[str] = None,
            num_encoder_layers: int = 6,
            num_decoder_layers: Optional[int] = None
    ) -> None:
        super(self.__class__, self).__init__()

        if num_decoder_layers is None:
            num_decoder_layers = num_encoder_layers

        self.encoder = Encoder(
            src_vocab_size,
            src_padding_idx,
            d_model,
            h,
            d_ff,
            dropout_p,
            activation,
            num_encoder_layers
        )
        self.decoder = Decoder(
            tgt_vocab_size,
            tgt_padding_idx,
            d_model,
            h,
            d_ff,
            dropout_p,
            activation,
            num_decoder_layers
        )

        self.src_padding_idx = src_padding_idx

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """
        :param source: Size([N, d_k])
        :param target: Size([N, d_q])
        :return:    Size([N, d_q, d_t])
        """
        padding_mask = self.make_padding_mask(source)
        lookahead_mask = self.make_lookahead_mask(target)
        z = self.encoder(source, padding_mask)
        out = self.decoder(target, z, padding_mask, lookahead_mask)
        return out

    def make_padding_mask(self, source: Tensor) -> BoolTensor:
        batch_size, seq_length = source.size()
        # Helps to ignore paddings while computing the attention scores
        # padding_mask: Size([N, 1, 1, seq_length])
        padding_mask = torch.ne(source, self.src_padding_idx) == 0
        return padding_mask.reshape(batch_size, 1, 1, seq_length).to(source.device)

    def make_lookahead_mask(self, target: Tensor) -> BoolTensor:
        batch_size, seq_length = target.size()
        # Helps to ignore future values while computing the attention scores
        # lookahead_mask: (N, 1, seq_length, seq_length)
        lookahead_mask = torch.ones(seq_length, seq_length).tril() == 0

        return lookahead_mask.expand(batch_size, 1, seq_length, seq_length).to(target.device)
