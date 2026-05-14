"""Apply per-token compression actions to a HuggingFace past_key_values cache."""
from __future__ import annotations

from typing import Sequence

import torch

from .quantize import symmetric_quantize_dequantize

Action = str
PastKV = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def count_memory_cost(actions: Sequence[Action]) -> int:
    """Return the total cost in units (FP16=2, INT8=1, DROP=0)."""
    cost = 0
    for a in actions:
        if a == "FP16":
            cost += 2
        elif a == "INT8":
            cost += 1
    return cost


def memory_ratio(actions: Sequence[Action]) -> float:
    """Compressed memory size relative to a full FP16 cache of the same length."""
    n = len(actions)
    if n == 0:
        return 0.0
    return count_memory_cost(actions) / (2.0 * n)


def apply_compression(
    past_key_values: PastKV,
    actions: Sequence[Action],
) -> tuple[PastKV, list[int]]:
    """Compress past_key_values according to per-token actions.

    DROP tokens are removed from the cache (the K/V tensors get shorter).
    INT8 tokens have their K and V values run through a quantize+dequantize
    round-trip to introduce the same numerical error a real INT8 store would
    have. FP16 tokens are kept exactly.

    Args:
        past_key_values: tuple of L (key, value) pairs; each tensor is shaped
            ``[batch, num_heads, seq_len, head_dim]``.
        actions: list of length ``seq_len`` with values "FP16"/"INT8"/"DROP".

    Returns:
        ``(new_past_key_values, kept_indices)`` where ``kept_indices`` is the
        list of original sequence positions that survived (the caller needs
        these positions to feed correct ``position_ids`` to the model).
    """
    seq_len = past_key_values[0][0].shape[2]
    if len(actions) != seq_len:
        raise ValueError(
            f"actions length {len(actions)} != cache seq_len {seq_len}"
        )

    keep_indices = [i for i, a in enumerate(actions) if a != "DROP"]
    kept_actions = [actions[i] for i in keep_indices]
    if not keep_indices:
        # Empty cache. Use index_select with empty index to preserve dtype/device.
        empty_idx = torch.empty(0, dtype=torch.long, device=past_key_values[0][0].device)
        new_past = tuple(
            (k.index_select(2, empty_idx), v.index_select(2, empty_idx))
            for k, v in past_key_values
        )
        return new_past, []

    device = past_key_values[0][0].device
    keep_idx_t = torch.tensor(keep_indices, dtype=torch.long, device=device)
    int8_local_mask = torch.tensor(
        [a == "INT8" for a in kept_actions], dtype=torch.bool, device=device
    )

    new_past_list = []
    for k, v in past_key_values:
        k_kept = k.index_select(2, keep_idx_t).clone()
        v_kept = v.index_select(2, keep_idx_t).clone()

        if int8_local_mask.any():
            k_int8 = k_kept[:, :, int8_local_mask, :]
            v_int8 = v_kept[:, :, int8_local_mask, :]
            k_kept[:, :, int8_local_mask, :] = symmetric_quantize_dequantize(k_int8)
            v_kept[:, :, int8_local_mask, :] = symmetric_quantize_dequantize(v_int8)

        new_past_list.append((k_kept, v_kept))
    return tuple(new_past_list), keep_indices
