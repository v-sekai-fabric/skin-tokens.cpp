"""Reference-container fallback for upstream's optional FlashAttention import.

The released code imports ``flash_attn_interface`` unconditionally in two
modules even though its other attention paths already fall back to PyTorch
scaled-dot-product attention. Keeping this shim outside the vendored upstream
tree lets the pinned reference run on CUDA systems without a matching
FlashAttention wheel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    *args, **kwargs):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if query.shape[1] != key.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    output = F.scaled_dot_product_attention(query, key, value)
    return output.transpose(1, 2), None
