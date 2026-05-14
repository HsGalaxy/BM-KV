"""Experiment 1 v2: WikiText-2 perplexity on Qwen2.5 with block-level BM-KV.

Mirrors exp1_ppl.py but uses Qwen2.5-0.5B-Instruct, longer prompts (1k tokens)
and the revised v2 policies (BM-KV-v2 = block-level + θ_drop/θ_fp16 thresholds).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from statistics import mean, stdev

import torch

import common  # noqa: F401 sets sys.path
from common import compute_budget, get_wikitext_chunks, load_qwen, RESULTS_DIR

from bm_kv import POLICIES_V2, CompressedRunner
from bm_kv.runner import CompressionConfig

PROMPT_LEN = 512
TARGET_LEN = 128
DEFAULT_RATIOS = [1.0, 0.75, 0.50, 0.35, 0.25]
DEFAULT_N_CHUNKS = 15


def policy_kwargs_for(name: str) -> dict:
    cfg = CompressionConfig()
    if name == "AttnOnly":
        cfg.use_recency = False
        cfg.use_prefix = False
    elif name == "Recent":
        cfg.use_attention = False
        cfg.use_prefix = False
    return {"config": cfg}


def run_experiment(n_chunks=DEFAULT_N_CHUNKS, ratios=None):
    ratios = ratios or DEFAULT_RATIOS
    print("Loading Qwen2.5-0.5B-Instruct")
    model, tok = load_qwen()
    print(f"Tokenizing WikiText into {n_chunks} chunks of {PROMPT_LEN + TARGET_LEN}")
    chunks = get_wikitext_chunks(tok, PROMPT_LEN + TARGET_LEN, n_chunks)
    print(f"Got {len(chunks)} chunks")

    nll_records: dict = defaultdict(lambda: defaultdict(list))
    mem_records: dict = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prompt_ids = chunk[:, :PROMPT_LEN].to(model.device)
        target_ids = chunk[:, PROMPT_LEN: PROMPT_LEN + TARGET_LEN].to(model.device)

        # Cache prefill output per importance config to avoid redoing it,
        # but clear the cache at the end of each chunk to keep VRAM bounded.
        prefill_cache = {}

        for ratio in ratios:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, policy_fn in POLICIES_V2.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue
                cfg = policy_kwargs_for(name)["config"]
                key = (cfg.use_attention, cfg.use_recency,
                       cfg.use_prefix, cfg.last_n_layers)
                runner = CompressedRunner(model, tok, cfg)
                if key not in prefill_cache:
                    prefill_cache[key] = runner.prefill(prompt_ids)
                prefill_out = prefill_cache[key]
                nll, stats = runner.teacher_force_logprob_with_prefill(
                    prefill_out, prompt_ids, target_ids,
                    policy_fn=policy_fn, budget=budget,
                )
                nll_records[name][ratio].append(nll.mean().item())
                mem_records[name][ratio].append(stats.memory_ratio)

        # Release the prefill cache (which holds the heavy attentions tensors).
        prefill_cache.clear()
        del prompt_ids, target_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (ci + 1) % max(1, len(chunks) // 5) == 0:
            print(f"  chunk {ci+1}/{len(chunks)}  elapsed {time.time() - t0:.1f}s")

    summary = []
    for name in ["Full"] + [n for n in POLICIES_V2 if n != "Full"]:
        for ratio in ratios:
            if ratio == 1.0 and name != "Full":
                continue
            if ratio != 1.0 and name == "Full":
                continue
            nlls = nll_records[name][ratio]
            mems = mem_records[name][ratio]
            if not nlls:
                continue
            ppls = [math.exp(n) for n in nlls]
            summary.append({
                "policy": name,
                "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(mems),
                "mean_nll": mean(nlls),
                "mean_ppl": mean(ppls),
                "std_ppl": stdev(ppls) if len(ppls) > 1 else 0.0,
                "n_chunks": len(nlls),
            })

    full_row = next((r for r in summary if r["policy"] == "Full"), None)
    if full_row:
        for r in summary:
            r["ppl_increment"] = r["mean_ppl"] - full_row["mean_ppl"]
            r["ppl_increment_pct"] = (
                (r["mean_ppl"] - full_row["mean_ppl"]) / full_row["mean_ppl"] * 100
            )

    out = {
        "config": {
            "model": "Qwen2.5-0.5B-Instruct",
            "prompt_len": PROMPT_LEN,
            "target_len": TARGET_LEN,
            "n_chunks": n_chunks,
            "ratios": ratios,
        },
        "results": summary,
    }
    out_path = RESULTS_DIR / "exp1_ppl_v2.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")

    print()
    print(f"{'Policy':<12} {'Ratio':>6} {'MemUsed':>8} {'PPL':>10} {'ΔPPL':>10} {'Δ%':>8}")
    for r in summary:
        print(
            f"{r['policy']:<12} "
            f"{r['memory_ratio_target']:>6.2f} "
            f"{r['memory_ratio_actual']:>8.3f} "
            f"{r['mean_ppl']:>10.3f} "
            f"{r.get('ppl_increment', 0):>+10.3f} "
            f"{r.get('ppl_increment_pct', 0):>+7.2f}%"
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n_chunks=args.n_chunks)
