"""Experiment 1: language-modeling perplexity on WikiText-2.

For each chunk we feed the first ``PROMPT_LEN`` tokens to the model, run a
compression policy, and then teacher-force the next ``TARGET_LEN`` tokens to
measure their NLL using the (now compressed) prompt cache. We compare every
policy at several memory ratios.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

import common  # noqa: F401 - sets sys.path
from common import compute_budget, get_wikitext_chunks, load_gpt2, RESULTS_DIR

from bm_kv import POLICIES, CompressedRunner
from bm_kv.runner import CompressionConfig

PROMPT_LEN = 384
TARGET_LEN = 96
DEFAULT_RATIOS = [1.0, 0.75, 0.50, 0.35, 0.25]
DEFAULT_N_CHUNKS = 30


def policy_kwargs_for(name: str) -> dict:
    """Some baselines need a different importance score (e.g. attention-only)."""
    cfg = CompressionConfig()
    if name == "AttnOnly":
        cfg.use_recency = False
        cfg.use_prefix = False
    elif name == "Recent":
        cfg.use_attention = False
        cfg.use_prefix = False
    elif name == "FullINT8":
        # Score is irrelevant for FullINT8 (keeps recent), but cheap to compute.
        pass
    elif name == "Full":
        pass
    return {"config": cfg}


def run_experiment(
    n_chunks: int = DEFAULT_N_CHUNKS,
    ratios: list[float] = None,
    model_name: str = "gpt2",
):
    ratios = ratios or DEFAULT_RATIOS
    print(f"Loading model {model_name}")
    model, tok = load_gpt2(model_name)

    print(f"Tokenizing WikiText-2 and slicing {n_chunks} chunks of "
          f"{PROMPT_LEN + TARGET_LEN} tokens")
    chunks = get_wikitext_chunks(tok, PROMPT_LEN + TARGET_LEN, n_chunks)
    print(f"Got {len(chunks)} chunks")

    # nll_records[policy][ratio] -> list of mean NLL per chunk
    nll_records: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    mem_records: dict[str, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prompt_ids = chunk[:, :PROMPT_LEN].to(model.device)
        target_ids = chunk[:, PROMPT_LEN : PROMPT_LEN + TARGET_LEN].to(model.device)

        # Run a single prefill PER POLICY-config since some policies use a
        # different importance score. We share the underlying prefill_out
        # within each unique config to avoid redundant work.
        prefill_cache: dict[tuple, tuple] = {}

        def get_prefill(cfg_key):
            if cfg_key not in prefill_cache:
                prefill_cache[cfg_key] = None  # placeholder
            return prefill_cache[cfg_key]

        # Build per-policy runners (each is just a thin wrapper).
        for ratio in ratios:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, policy_fn in POLICIES.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue

                kw = policy_kwargs_for(name)
                cfg = kw["config"]
                key = (cfg.use_attention, cfg.use_recency, cfg.use_prefix,
                       cfg.last_n_layers)
                runner = CompressedRunner(model, tok, cfg)

                if prefill_cache.get(key) is None:
                    prefill_cache[key] = runner.prefill(prompt_ids)
                prefill_out = prefill_cache[key]

                nll, stats = runner.teacher_force_logprob_with_prefill(
                    prefill_out, prompt_ids, target_ids,
                    policy_fn=policy_fn, budget=budget,
                )
                nll_records[name][ratio].append(nll.mean().item())
                mem_records[name][ratio].append(stats.memory_ratio)

        if (ci + 1) % max(1, len(chunks) // 5) == 0:
            elapsed = time.time() - t0
            print(f"  chunk {ci+1}/{len(chunks)}  elapsed {elapsed:.1f}s")

    # Build summary table.
    summary: list[dict] = []
    for name in ["Full"] + [n for n in POLICIES if n != "Full"]:
        for ratio in ratios:
            if ratio == 1.0 and name != "Full":
                continue
            if ratio != 1.0 and name == "Full":
                continue
            nlls = nll_records[name][ratio]
            mems = mem_records[name][ratio]
            if not nlls:
                continue
            ppl_per_chunk = [math.exp(n) for n in nlls]
            summary.append({
                "policy": name,
                "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(mems),
                "mean_nll": mean(nlls),
                "mean_ppl": mean(ppl_per_chunk),
                "std_ppl": stdev(ppl_per_chunk) if len(ppl_per_chunk) > 1 else 0.0,
                "n_chunks": len(nlls),
            })

    # Add PPL increment vs Full Cache.
    full_row = next((r for r in summary if r["policy"] == "Full"), None)
    if full_row is not None:
        full_ppl = full_row["mean_ppl"]
        for row in summary:
            row["ppl_increment"] = row["mean_ppl"] - full_ppl
            row["ppl_increment_pct"] = (row["mean_ppl"] - full_ppl) / full_ppl * 100

    out = {
        "config": {
            "model": model_name,
            "prompt_len": PROMPT_LEN,
            "target_len": TARGET_LEN,
            "n_chunks": n_chunks,
            "ratios": ratios,
        },
        "results": summary,
    }
    out_path = RESULTS_DIR / "exp1_ppl.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")

    # Pretty-print.
    print()
    print(f"{'Policy':<10} {'Ratio':>6} {'MemUsed':>8} {'PPL':>10} {'ΔPPL':>10} {'Δ%':>8}")
    for row in summary:
        print(
            f"{row['policy']:<10} "
            f"{row['memory_ratio_target']:>6.2f} "
            f"{row['memory_ratio_actual']:>8.3f} "
            f"{row['mean_ppl']:>10.3f} "
            f"{row.get('ppl_increment', 0):>+10.3f} "
            f"{row.get('ppl_increment_pct', 0):>+7.2f}%"
        )

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    p.add_argument("--model", default="gpt2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n_chunks=args.n_chunks, model_name=args.model)
