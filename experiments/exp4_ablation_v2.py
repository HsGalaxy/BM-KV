"""Experiment 4 v2: ablation for the revised BM-KV-v2 on Qwen2.5.

Variants:
- A. Full BM-KV-v2 (block-level + θ_drop/θ_fp16 + INT8 action).
- B. v2 without the absolute θ_drop threshold (only budget gates DROP).
- C. v2 reverted to token-level (block_size = 1).
- D. v2 without INT8 (FP16/DROP only — same idea as v1's D ablation).
- E. v1-style token-level BM-KV (no threshold, no block).

Plus a separate lazy-update comparison:
- L0. Static BM-KV-v2 (no rebalance).
- L1. Lazy BM-KV-v2 with delta=16.
- L2. Lazy BM-KV-v2 with delta=8.
The lazy run reports both quality (PPL on a held-out continuation) and TPOT.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from statistics import mean

import torch

import common  # noqa: F401
from common import compute_budget, get_wikitext_chunks, load_qwen, RESULTS_DIR

from bm_kv import (
    CompressedRunner, POLICIES_V2,
    policy_bm_kv, policy_bm_kv_v2, policy_bm_kv_no_int8,
)
from bm_kv.runner import CompressionConfig

PROMPT_LEN = 512
TARGET_LEN = 128
ABLATION_RATIOS = [0.50, 0.25]
N_CHUNKS = 8


def policy_no_threshold(scores, budget, block_size=16, fp16_ratio=0.35):
    """v2 minus the absolute θ_drop: a block is kept iff the budget permits."""
    return policy_bm_kv_v2(
        scores, budget,
        block_size=block_size,
        theta_drop_quantile=0.0,   # never drop on threshold
        theta_fp16_quantile=0.7,
        fp16_ratio=fp16_ratio,
    )


def policy_token_level(scores, budget):
    """v2 with block_size=1 (token-level granularity)."""
    return policy_bm_kv_v2(scores, budget, block_size=1)


def policy_no_int8_v2(scores, budget, block_size=16):
    """Block-level FP16/DROP only — disables the INT8 action set."""
    from bm_kv.blocks import (
        aggregate_token_scores_to_blocks,
        expand_block_actions_to_tokens,
        num_blocks,
    )
    if isinstance(scores, torch.Tensor):
        s = scores.detach().float()
    else:
        s = torch.tensor(list(scores), dtype=torch.float32)
    seq_len = s.shape[-1]
    nb = num_blocks(seq_len, block_size)
    block_scores = aggregate_token_scores_to_blocks(s, block_size)
    # Keep top blocks at FP16 until budget exhausted (FP16 cost = 2 * block_size).
    fp16_cost_per_block = 2 * block_size
    max_fp16_blocks = max(0, min(nb, budget // fp16_cost_per_block))
    order = torch.argsort(block_scores, descending=True).tolist()
    block_actions = ["DROP"] * nb
    for j in order[:max_fp16_blocks]:
        block_actions[j] = "FP16"
    return expand_block_actions_to_tokens(block_actions, seq_len, block_size)


VARIANTS = [
    ("A_full_v2",       lambda: CompressionConfig(),                         policy_bm_kv_v2),
    ("B_no_threshold",  lambda: CompressionConfig(),                         policy_no_threshold),
    ("C_token_level",   lambda: CompressionConfig(),                         policy_token_level),
    ("D_no_int8",       lambda: CompressionConfig(),                         policy_no_int8_v2),
    ("E_v1_token_kv",   lambda: CompressionConfig(),                         policy_bm_kv),
]


def run_ppl_ablation(model, tok):
    print(f"PPL ablation: {N_CHUNKS} chunks")
    chunks = get_wikitext_chunks(tok, PROMPT_LEN + TARGET_LEN, N_CHUNKS)
    records = defaultdict(lambda: defaultdict(list))

    for ci, chunk in enumerate(chunks):
        prompt_ids = chunk[:, :PROMPT_LEN].to(model.device)
        target_ids = chunk[:, PROMPT_LEN: PROMPT_LEN + TARGET_LEN].to(model.device)
        prefill_out = None
        for ratio in ABLATION_RATIOS:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, cfg_factory, policy_fn in VARIANTS:
                cfg = cfg_factory()
                runner = CompressedRunner(model, tok, cfg)
                if prefill_out is None:
                    prefill_out = runner.prefill(prompt_ids)
                nll, stats = runner.teacher_force_logprob_with_prefill(
                    prefill_out, prompt_ids, target_ids,
                    policy_fn=policy_fn, budget=budget,
                )
                records[name][ratio].append({
                    "nll": nll.mean().item(),
                    "mem": stats.memory_ratio,
                })
        prefill_out = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = []
    for name, _, _ in VARIANTS:
        for ratio in ABLATION_RATIOS:
            recs = records[name][ratio]
            ppls = [math.exp(r["nll"]) for r in recs]
            summary.append({
                "variant": name, "ratio": ratio,
                "mean_ppl": mean(ppls), "n": len(ppls),
                "mem_ratio": mean(r["mem"] for r in recs),
            })
    return summary


def run_lazy_comparison(model, tok):
    """Compare static vs lazy(delta=16) vs lazy(delta=8) on a long-decode task."""
    print("Lazy update comparison (decode 64 tokens)")
    chunks = get_wikitext_chunks(tok, PROMPT_LEN, 5)
    LAZY_RATIO = 0.35
    budget = compute_budget(PROMPT_LEN, LAZY_RATIO)
    DECODE = 64

    rows = []
    for ci, chunk in enumerate(chunks):
        prompt = chunk[:, :PROMPT_LEN].to(model.device)
        cfg = CompressionConfig()
        runner = CompressedRunner(model, tok, cfg)

        # Static (no rebalance): use generate_lazy with delta huge.
        out_ids_s, info_s = runner.generate_lazy(
            prompt, policy_bm_kv_v2, budget,
            max_new_tokens=DECODE, delta=DECODE + 1, drift_threshold=None,
        )
        rows.append({
            "trial": ci, "mode": "static",
            "rebalances": len(info_s["rebalances"]),
            "mean_step_ms": mean(info_s["per_step_ms"]),
            "first_text": tok.decode(out_ids_s[0][:32], skip_special_tokens=True),
        })

        out_ids_l16, info_l16 = runner.generate_lazy(
            prompt, policy_bm_kv_v2, budget,
            max_new_tokens=DECODE, delta=16, drift_threshold=0.5,
        )
        rows.append({
            "trial": ci, "mode": "lazy_delta16",
            "rebalances": len(info_l16["rebalances"]),
            "mean_step_ms": mean(info_l16["per_step_ms"]),
            "first_text": tok.decode(out_ids_l16[0][:32], skip_special_tokens=True),
        })

        out_ids_l8, info_l8 = runner.generate_lazy(
            prompt, policy_bm_kv_v2, budget,
            max_new_tokens=DECODE, delta=8, drift_threshold=0.5,
        )
        rows.append({
            "trial": ci, "mode": "lazy_delta8",
            "rebalances": len(info_l8["rebalances"]),
            "mean_step_ms": mean(info_l8["per_step_ms"]),
            "first_text": tok.decode(out_ids_l8[0][:32], skip_special_tokens=True),
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    summary = []
    for mode, recs in by_mode.items():
        summary.append({
            "mode": mode,
            "n_trials": len(recs),
            "mean_step_ms": mean(r["mean_step_ms"] for r in recs),
            "mean_rebalances": mean(r["rebalances"] for r in recs),
        })
    return summary, rows


def run_experiment():
    print("Loading Qwen2.5-0.5B-Instruct")
    model, tok = load_qwen()

    ppl = run_ppl_ablation(model, tok)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lazy_summary, lazy_rows = run_lazy_comparison(model, tok)

    out = {
        "config": {
            "model": "Qwen2.5-0.5B-Instruct",
            "ratios": ABLATION_RATIOS,
            "prompt_len": PROMPT_LEN,
            "target_len": TARGET_LEN,
            "n_chunks": N_CHUNKS,
        },
        "ppl_ablation": ppl,
        "lazy_summary": lazy_summary,
        "lazy_samples": lazy_rows,
    }
    out_path = RESULTS_DIR / "exp4_ablation_v2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")

    print("\nPPL ablation (lower better):")
    print(f"{'Variant':<18} {'@0.50':>10} {'@0.25':>10}")
    by = defaultdict(dict)
    for r in ppl:
        by[r["variant"]][r["ratio"]] = r["mean_ppl"]
    for v, _, _ in VARIANTS:
        print(f"{v:<18} {by[v].get(0.50, float('nan')):>10.2f} "
              f"{by[v].get(0.25, float('nan')):>10.2f}")

    print("\nLazy update comparison:")
    for r in lazy_summary:
        print(f"  {r['mode']:<14} steps_ms={r['mean_step_ms']:.2f} "
              f"rebalances={r['mean_rebalances']:.1f}")
    return out


if __name__ == "__main__":
    run_experiment()
