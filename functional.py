import math
from typing import Optional, Tuple

import torch
from torch import BoolTensor, Tensor


def scaled_dot_product_attention(
        values: Tensor, keys: Tensor, queries: Tensor, mask: Optional[BoolTensor] = None, ninf: float = -1e4
) -> Tuple[Tensor, Tensor]:
    h, d = queries.size(2), queries.size(3)
    # Multiply queries and keys for each training example with every other training example
    # queries: (N, d_q, h, d_head),
    # keys:    (N, d_k, h, d_head)
    # energy:  (N, h, d_q, d_k)
    energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

    # Masked indices so their weights become 0 after applying softmax
    if mask is not None:
        energy.masked_fill_(mask, ninf)

    # Normalize energy values similarly to seq2seq + attention so that they sum to 1.
    # Also divide by scaling factor for better stability
    attn = torch.softmax(energy / math.sqrt(h * d), dim=3)
    # attn:   (N, h, d_q, d_k)
    # values: (N, d_k, h, d_head)

    out = torch.einsum('nhqk,nkhd->nqhd', [attn, values])

    return out, attn


def positional_encoding(x: Tensor) -> Tensor:
    batch_size, seq_length, d_model = x.size()

    pe = positional_encoding_matrix(
        seq_length, d_model, x.dtype, x.device
    ).repeat(batch_size, 1, 1)

    return x + pe


def positional_encoding_matrix(
        seq_length: int,
        d_model: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
) -> Tensor:
    pos = torch.arange(seq_length, dtype=dtype, device=device).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=dtype, device=device)
    div_term = torch.pow(10_000, (two_i / d_model))
    radians = pos / div_term

    pe = torch.empty(seq_length, d_model, dtype=dtype, device=device)
    pe[:, 0::2] = torch.sin(radians)
    pe[:, 1::2] = torch.cos(radians)

    return pe
