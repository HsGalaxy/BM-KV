"""INT8 symmetric per-channel quantization for KV cache simulation.

For experiment purposes we round-trip values through int8 representation and back
to the original floating dtype. This introduces the same numerical error a real
INT8-stored cache would suffer, while keeping the rest of the pipeline running on
standard floating tensors.
"""
from __future__ import annotations

import torch


def symmetric_quantize_dequantize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-channel symmetric INT8 quantize+dequantize round-trip.

    Args:
        x: tensor to quantize, any floating dtype.
        dim: channel dimension along which to compute the per-channel scale.
            For KV tensors of shape ``[batch, heads, seq, head_dim]`` we use the
            last dim so each (batch, head, token) gets its own scale.

    Returns:
        Tensor of the same shape and dtype as ``x`` whose values have been
        rounded to the nearest INT8 quantization grid.
    """
    orig_dtype = x.dtype
    x_f = x.float()
    abs_max = x_f.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-8)
    q = torch.round(x_f / scale).clamp(-127.0, 127.0)
    return (q * scale).to(orig_dtype)
