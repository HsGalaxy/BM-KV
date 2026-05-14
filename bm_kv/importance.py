"""Token importance scoring used by BM-KV.

Score_i = alpha * Norm(A_i) + beta * Norm(R_i) + gamma * P_i

A_i: accumulated attention received by key-position i across heads/queries/layers.
R_i: recency weight, exp(-lambda * (t - i)).
P_i: prefix bonus, 1 for the first ``prefix_len`` tokens, 0 otherwise.
"""
from __future__ import annotations

from typing import Sequence

import torch


def compute_attention_received(
    attentions: Sequence[torch.Tensor],
    last_n_layers: int | None = None,
) -> torch.Tensor:
    """Sum attention received by each key position.

    Args:
        attentions: tuple of L tensors, each ``[batch, heads, q_len, k_len]``
            as returned by HuggingFace with ``output_attentions=True``.
        last_n_layers: if given, only aggregate the last N layers (cheaper).

    Returns:
        Tensor of shape ``[batch, k_len]`` with the per-key attention mass.
    """
    layers = list(attentions)
    if last_n_layers is not None and last_n_layers > 0:
        layers = layers[-last_n_layers:]
    if not layers:
        raise ValueError("Empty attentions provided")

    total = None
    for attn in layers:
        # Attention from query q to key k. Sum over heads (dim=1) and queries
        # (dim=2) gives the total attention each key position received.
        recv = attn.sum(dim=(1, 2))
        total = recv if total is None else total + recv
    assert total is not None
    return total / float(len(layers))


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each row to [0, 1]. ``x`` shape ``[batch, seq]``."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    rng = (x_max - x_min).clamp(min=eps)
    out = (x - x_min) / rng
    return out.squeeze(0) if squeeze else out


def compute_recency_weight(
    seq_len: int,
    current_t: int,
    lambd: float,
    device: torch.device,
) -> torch.Tensor:
    """R_i = exp(-lambda * (t - i)) for i in [0, seq_len)."""
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    distance = (current_t - positions).clamp(min=0.0)
    return torch.exp(-float(lambd) * distance)


def compute_prefix_weight(
    seq_len: int, prefix_len: int, device: torch.device
) -> torch.Tensor:
    """1.0 for the first ``prefix_len`` positions, 0 elsewhere."""
    w = torch.zeros(seq_len, device=device, dtype=torch.float32)
    if prefix_len > 0:
        w[: min(prefix_len, seq_len)] = 1.0
    return w


def compute_importance(
    attentions: Sequence[torch.Tensor],
    current_t: int,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    lambd: float = 0.01,
    prefix_len: int = 4,
    last_n_layers: int | None = None,
) -> torch.Tensor:
    """Compute the BM-KV importance score for each key position.

    Returns a tensor of shape ``[batch, seq_len]``.
    """
    a = compute_attention_received(attentions, last_n_layers=last_n_layers)
    seq_len = a.shape[-1]
    device = a.device
    r = compute_recency_weight(seq_len, current_t, lambd, device)
    p = compute_prefix_weight(seq_len, prefix_len, device)

    a_norm = _normalize(a)
    r_norm = _normalize(r).unsqueeze(0).expand_as(a_norm)
    p_expanded = p.unsqueeze(0).expand_as(a_norm)

    return alpha * a_norm + beta * r_norm + gamma * p_expanded
