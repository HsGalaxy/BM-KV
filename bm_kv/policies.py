"""Five cache compression policies, all returning a per-token action list.

Each policy takes the importance score for every existing key position, plus a
``budget`` measured in cost units (FP16 = 2, INT8 = 1, DROP = 0). The "memory
ratio" of the resulting cache equals ``sum(cost) / (2 * seq_len)``.

Policies do NOT modify the cache themselves; that is the job of
``apply_compression`` in ``compress.py``.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch

Action = str  # "FP16" | "INT8" | "DROP"


def _to_list(scores: torch.Tensor | Sequence[float]) -> list[float]:
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().tolist()
    return list(scores)


def policy_full_cache(
    scores: torch.Tensor | Sequence[float],
    budget: int,
) -> list[Action]:
    """Keep every token at FP16 (ignores ``budget``)."""
    seq_len = len(_to_list(scores))
    return ["FP16"] * seq_len


def policy_recent_only(
    scores: torch.Tensor | Sequence[float],
    budget: int,
) -> list[Action]:
    """Keep the most recent ``budget // 2`` tokens at FP16, drop the rest."""
    seq_len = len(_to_list(scores))
    n_keep = max(0, budget // 2)
    actions: list[Action] = ["DROP"] * seq_len
    start = max(0, seq_len - n_keep)
    for i in range(start, seq_len):
        actions[i] = "FP16"
    return actions


def policy_attention_only(
    scores: torch.Tensor | Sequence[float],
    budget: int,
) -> list[Action]:
    """Keep the top ``budget // 2`` tokens by score at FP16, drop the rest.

    NOTE: this policy expects ``scores`` to already reflect *only* attention
    importance (the caller controls that by zeroing beta and gamma when
    computing the score).
    """
    s = _to_list(scores)
    seq_len = len(s)
    n_keep = max(0, min(budget // 2, seq_len))
    if n_keep == 0:
        return ["DROP"] * seq_len
    order = sorted(range(seq_len), key=lambda i: s[i], reverse=True)
    keep = set(order[:n_keep])
    return ["FP16" if i in keep else "DROP" for i in range(seq_len)]


def policy_full_int8(
    scores: torch.Tensor | Sequence[float],
    budget: int,
) -> list[Action]:
    """Quantize as many recent tokens as possible to INT8 within the budget.

    With INT8 cost = 1, a budget of B fits B tokens. When ``budget >= seq_len``
    every token is kept at INT8 (memory ratio 0.5 of full FP16).
    """
    seq_len = len(_to_list(scores))
    n_keep = max(0, min(budget, seq_len))
    actions: list[Action] = ["DROP"] * seq_len
    start = max(0, seq_len - n_keep)
    for i in range(start, seq_len):
        actions[i] = "INT8"
    return actions


def policy_bm_kv(
    scores: torch.Tensor | Sequence[float],
    budget: int,
    fp16_ratio: float = 0.35,
) -> list[Action]:
    """Greedy mixed-precision allocation matching algorithm 3.1 in the paper.

    Tokens are sorted by descending score. The top tokens get FP16 until either
    the FP16 sub-budget is exhausted or the global budget is reached, after
    which the remaining high-score tokens are stored at INT8 until the budget
    is fully consumed; the rest are dropped.
    """
    s = _to_list(scores)
    seq_len = len(s)
    if budget <= 0 or seq_len == 0:
        return ["DROP"] * seq_len

    order = sorted(range(seq_len), key=lambda i: s[i], reverse=True)
    actions: list[Action] = ["DROP"] * seq_len
    used = 0
    fp16_limit = int(budget * fp16_ratio) * 2  # cost units reserved for FP16

    for i in order:
        if used + 2 <= budget and used < fp16_limit:
            actions[i] = "FP16"
            used += 2
        elif used + 1 <= budget:
            actions[i] = "INT8"
            used += 1
        else:
            actions[i] = "DROP"
    return actions


def policy_bm_kv_no_int8(
    scores: torch.Tensor | Sequence[float],
    budget: int,
) -> list[Action]:
    """Ablation: same BM-KV importance ranking but FP16/DROP only (no INT8).

    Used to measure how much of BM-KV's gain comes from the mixed-precision
    action set versus from the importance score itself.
    """
    s = _to_list(scores)
    seq_len = len(s)
    if budget <= 0 or seq_len == 0:
        return ["DROP"] * seq_len
    order = sorted(range(seq_len), key=lambda i: s[i], reverse=True)
    actions: list[Action] = ["DROP"] * seq_len
    n_fp16 = min(seq_len, budget // 2)
    for i in order[:n_fp16]:
        actions[i] = "FP16"
    return actions


def policy_bm_kv_v2(
    scores: torch.Tensor | Sequence[float],
    budget: int,
    block_size: int = 16,
    theta_fp16_quantile: float = 0.70,
    theta_drop_quantile: float = 0.20,
    mean_weight: float = 0.6,
    max_weight: float = 0.4,
    fp16_ratio: float = 0.35,
) -> list[Action]:
    """Block-level BM-KV with absolute thresholds (revision v2).

    Workflow:
      1. Aggregate per-token scores into per-block scores using a
         weighted mean + max combination.
      2. Pick θ_drop and θ_fp16 as the requested quantiles of the block
         score distribution. Blocks below θ_drop are dropped outright,
         even if budget remains — this prevents preserving truly
         low-value history just because the cache is roomy.
      3. Sort the surviving blocks by score (descending) and greedily
         allocate FP16 (cost 2 per token in the block) up to the
         FP16 sub-budget, then INT8 (cost 1 per token) until the global
         budget is exhausted. Anything left over is dropped.

    The budget is expressed in *token-equivalent* cost units, matching
    v1 (FP16=2, INT8=1, DROP=0 per token). A whole FP16 block of size
    ``g`` therefore costs ``2g``.
    """
    import torch as _torch
    from .blocks import (
        aggregate_token_scores_to_blocks,
        expand_block_actions_to_tokens,
        num_blocks,
    )

    if isinstance(scores, _torch.Tensor):
        scores_t = scores.detach().float()
    else:
        scores_t = _torch.tensor(list(scores), dtype=_torch.float32)

    seq_len = scores_t.shape[-1]
    if budget <= 0 or seq_len == 0:
        return ["DROP"] * seq_len

    nb = num_blocks(seq_len, block_size)
    block_scores = aggregate_token_scores_to_blocks(
        scores_t, block_size, mean_weight=mean_weight, max_weight=max_weight,
    )

    # Absolute thresholds from quantiles of the current block score distribution.
    sorted_bs, _ = _torch.sort(block_scores)
    theta_drop = sorted_bs[int(theta_drop_quantile * (nb - 1))].item() if nb > 1 else float("-inf")
    theta_fp16 = sorted_bs[int(theta_fp16_quantile * (nb - 1))].item() if nb > 1 else float("inf")

    # Build sort order on score; we want descending.
    order = _torch.argsort(block_scores, descending=True).tolist()

    block_actions: list[Action] = ["DROP"] * nb
    used = 0  # cost in token-units
    fp16_budget = int(budget * fp16_ratio) * 2

    for j in order:
        score = block_scores[j].item()
        if score < theta_drop:
            # Hard drop regardless of budget — paper's contribution vs v1.
            continue
        # Determine cost of putting block j at FP16 or INT8.
        start, end = j * block_size, min((j + 1) * block_size, seq_len)
        block_token_count = end - start
        cost_fp16 = 2 * block_token_count
        cost_int8 = 1 * block_token_count

        if (
            score >= theta_fp16
            and used + cost_fp16 <= budget
            and used + cost_fp16 <= fp16_budget + cost_fp16  # gentle cap
        ):
            block_actions[j] = "FP16"
            used += cost_fp16
        elif used + cost_int8 <= budget:
            block_actions[j] = "INT8"
            used += cost_int8
        else:
            # Out of budget; the rest stay DROP.
            break

    return expand_block_actions_to_tokens(block_actions, seq_len, block_size)


PolicyFn = Callable[[torch.Tensor, int], list[Action]]

POLICIES: dict[str, PolicyFn] = {
    "Full": policy_full_cache,
    "Recent": policy_recent_only,
    "AttnOnly": policy_attention_only,
    "FullINT8": policy_full_int8,
    "BM-KV": policy_bm_kv,
}

# v2 policies (block-level, absolute thresholds).
POLICIES_V2: dict[str, PolicyFn] = {
    "Full": policy_full_cache,
    "Recent": policy_recent_only,
    "AttnOnly": policy_attention_only,
    "FullINT8": policy_full_int8,
    "BM-KV-v2": policy_bm_kv_v2,
}
