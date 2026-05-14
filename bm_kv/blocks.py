"""Block-level helpers for BM-KV v2.

Tokens are grouped into fixed-size blocks of ``block_size`` consecutive
positions. Each block then receives a single FP16/INT8/DROP action. Working at
the block level mirrors PagedAttention-style memory management: a real
implementation can put FP16 blocks and INT8 blocks in separate contiguous
pools, while dropped blocks return their pages to a free pool.
"""
from __future__ import annotations

from typing import Sequence

import torch


def num_blocks(seq_len: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (seq_len + block_size - 1) // block_size


def token_to_block(token_idx: int, block_size: int) -> int:
    return token_idx // block_size


def block_to_token_range(block_idx: int, block_size: int, seq_len: int) -> tuple[int, int]:
    start = block_idx * block_size
    end = min(start + block_size, seq_len)
    return start, end


def aggregate_token_scores_to_blocks(
    token_scores: torch.Tensor,
    block_size: int,
    mean_weight: float = 0.6,
    max_weight: float = 0.4,
) -> torch.Tensor:
    """Aggregate per-token scores into per-block scores.

    BlockScore_j = mean_weight * mean(scores in block) + max_weight * max(scores in block).
    Using both mean and max ensures a block with a single very important token
    is not averaged down to oblivion, while blocks of uniformly mid-range
    tokens still get a fair shake.

    Args:
        token_scores: ``[seq_len]`` tensor of per-token importance.
        block_size: block size in tokens.
        mean_weight, max_weight: combination weights, should sum to ~1.

    Returns:
        ``[num_blocks]`` tensor of block scores.
    """
    n = token_scores.shape[-1]
    nb = num_blocks(n, block_size)
    # Pad token_scores to a multiple of block_size with -inf so the partial
    # last block is well-defined.
    pad_len = nb * block_size - n
    if pad_len > 0:
        pad = torch.full(
            (pad_len,), float("-inf"),
            dtype=token_scores.dtype, device=token_scores.device,
        )
        padded = torch.cat([token_scores, pad])
    else:
        padded = token_scores
    grid = padded.view(nb, block_size)
    # For mean, mask -inf entries.
    valid = grid != float("-inf")
    safe = torch.where(valid, grid, torch.zeros_like(grid))
    counts = valid.sum(dim=1).clamp(min=1).to(safe.dtype)
    means = safe.sum(dim=1) / counts
    # Max: treat -inf as missing and fall back to mean for empty blocks.
    masked_max = torch.where(valid, grid, torch.full_like(grid, float("-inf")))
    maxes = masked_max.max(dim=1).values
    maxes = torch.where(torch.isfinite(maxes), maxes, means)
    return mean_weight * means + max_weight * maxes


def expand_block_actions_to_tokens(
    block_actions: Sequence[str],
    seq_len: int,
    block_size: int,
) -> list[str]:
    """Replicate a block-level action list to a per-token action list."""
    out: list[str] = []
    for ba in block_actions:
        out.extend([ba] * block_size)
    return out[:seq_len]
