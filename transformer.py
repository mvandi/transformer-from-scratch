import math
from typing import Optional

import torch.nn as nn
from torch import BoolTensor, Tensor

import functional as F


class MultiheadAttention(nn.Module):
    """
    Attention Is All You Need §3.2.2 - Multi-Head Attention
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
        # Split the projected tensor into heads
        return out.reshape(batch_size, seq_length, self.h, self.d)


class ResidualLayerNorm(nn.Module):
    """
    Attention Is All You Need §5.4 - Regularization
    """

    def __init__(self, d_model: int, p_drop: float) -> None:
        super(self.__class__, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        """
        :param x:        Size([N, d_q, d_model])
        :param residual: Size([N, d_q, d_model])
        :return:         Size([N, d_q, d_model])
        """
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class PWFFN(nn.Module):
    """
    Attention Is All You Need §3.3 - Position-wise Feed-Forward Networks
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


class TransformerEmbedding(nn.Module):
    """
    Attention Is All You Need §3.4 - Embeddings and Softmax
    """

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


class TransformerEncoderLayer(nn.Module):
    """
    Attention Is All You Need §3.1 - Encoder and Decoder Stacks
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            p_drop: float
    ) -> None:
        super(self.__class__, self).__init__()

        self.source_attn = MultiheadAttention(d_model, h)
        self.source_norm = ResidualLayerNorm(d_model, p_drop)
        self.pwffn = PWFFN(d_model, d_ff, activation)
        self.pwffn_norm = ResidualLayerNorm(d_model, p_drop)

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


class TransformerEncoder(nn.Module):
    """
    Attention Is All You Need §3.1 - Encoder and Decoder Stacks
    """

    def __init__(
            self,
            vocab_size: int,
            padding_idx: Optional[int],
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            p_drop: float,
            num_layers: int
    ) -> None:
        super(self.__class__, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, padding_idx)
        self.embedding_dropout = nn.Dropout(p_drop)
        self.stack = nn.ModuleList([
            TransformerEncoderLayer(d_model, h, d_ff, activation, p_drop)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[BoolTensor]) -> Tensor:
        """
        :param x:    Size([N, d_k])
        :param mask: Size([N, 1, 1, d_k])
        :return:     Size([N, d_k, d_model])
        """
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        for layer in self.stack:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Attention Is All You Need §3.1 - Encoder and Decoder Stacks
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            p_drop: float
    ) -> None:
        super(self.__class__, self).__init__()

        self.target_attn = MultiheadAttention(d_model, h)
        self.target_norm = ResidualLayerNorm(d_model, p_drop)
        self.encoder_attn = MultiheadAttention(d_model, h)
        self.encoder_norm = ResidualLayerNorm(d_model, p_drop)
        self.pwffn = PWFFN(d_model, d_ff, activation)
        self.pwffn_norm = ResidualLayerNorm(d_model, p_drop)

    def forward(
            self, x: Tensor, z: Tensor, source_mask: Optional[BoolTensor], target_mask: Optional[BoolTensor]
    ) -> Tensor:
        """
        :param x:           Size([N, d_q, d_model])
        :param z:           Size([N, d_k, d_model])
        :param source_mask: Size([N, 1, 1, d_k])
        :param target_mask: Size([N, 1, d_q, d_q])
        :return:            Size([N, d_q, d_model])
        """
        residual = x
        x = self.target_attn(x, x, x, target_mask)
        x = self.target_norm(x, residual)
        residual = x
        x = self.encoder_attn(z, z, x, source_mask)
        x = self.encoder_norm(x, residual)
        residual = x
        x = self.pwffn(x)
        x = self.pwffn_norm(x, residual)
        return x


class TransformerDecoder(nn.Module):
    """
    Attention Is All You Need §3.1 - Encoder and Decoder Stacks
    """

    def __init__(
            self,
            vocab_size: int,
            padding_idx: Optional[int],
            d_model: int,
            h: int,
            d_ff: Optional[int],
            activation: Optional[str],
            p_drop: float,
            num_layers: int
    ) -> None:
        super(self.__class__, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, d_model, padding_idx)
        self.embedding_dropout = nn.Dropout(p_drop)
        self.stack = nn.ModuleList([
            TransformerDecoderLayer(d_model, h, d_ff, activation, p_drop)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
            self, x: Tensor, z: Tensor, source_mask: Optional[BoolTensor], target_mask: Optional[BoolTensor]
    ) -> Tensor:
        """
        :param x:           Size([N, d_q])
        :param z:           Size([N, d_k, d_model])
        :param source_mask: Size([N, 1, 1, d_k])
        :param target_mask: Size([N, 1, d_q, d_q])
        :return:            Size([N, d_q, d_t])
        """
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        for layer in self.stack:
            x = layer(x, z, source_mask, target_mask)
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    """
    Attention Is All You Need §3.1 - Encoder and Decoder Stacks
    """

    def __init__(
            self,
            source_vocab_size: int,
            target_vocab_size: int,
            source_padding_idx: Optional[int],
            target_padding_idx: Optional[int],
            d_model: int = 512,
            h: int = 8,
            d_ff: Optional[int] = 2048,
            activation: Optional[str] = None,
            p_drop: float = 0.1,
            num_encoder_layers: int = 6,
            num_decoder_layers: Optional[int] = None
    ) -> None:
        super(self.__class__, self).__init__()

        if num_decoder_layers is None:
            num_decoder_layers = num_encoder_layers

        self.encoder = TransformerEncoder(
            source_vocab_size,
            source_padding_idx,
            d_model,
            h,
            d_ff,
            activation,
            p_drop,
            num_encoder_layers
        )
        self.decoder = TransformerDecoder(
            target_vocab_size,
            target_padding_idx,
            d_model,
            h,
            d_ff,
            activation,
            p_drop,
            num_decoder_layers
        )

        self.source_padding_idx = source_padding_idx
        self.target_padding_idx = target_padding_idx

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """
        :param source: Size([N, d_k])
        :param target: Size([N, d_q])
        :return:    Size([N, d_q, d_t])
        """
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)
        z = self.encoder(source, source_mask)
        y = self.decoder(target, z, source_mask, target_mask)
        return y

    def make_source_mask(self, source: Tensor) -> BoolTensor:
        """
        Attention Is All You Need §3.2.3 - Applications of Attention in our Model
        """
        batch_size, seq_length = source.size()
        # Helps to ignore paddings while computing the attention scores
        # source_mask: Size([N, 1, 1, seq_length])
        mask = F.make_padding_mask(source, self.source_padding_idx)

        return mask.reshape(batch_size, 1, 1, seq_length)

    def make_target_mask(self, target: Tensor) -> BoolTensor:
        """
        Attention Is All You Need §3.2.3 - Applications of Attention in our Model
        """
        batch_size, seq_length = target.size()
        # Helps to ignore future values while computing the attention scores
        # target_mask: (N, 1, seq_length, seq_length)
        lookahead_mask = F.make_lookahead_mask(seq_length).to(target.device)
        lookahead_mask = lookahead_mask.expand(batch_size, 1, seq_length, seq_length)

        padding_mask = F.make_padding_mask(target, self.target_padding_idx)
        padding_mask = padding_mask.reshape(batch_size, 1, 1, seq_length)
        padding_mask = padding_mask.repeat(1, 1, seq_length, 1)

        return lookahead_mask & padding_mask
