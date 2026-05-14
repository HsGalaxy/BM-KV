"""Experiment 4: ablation. We start from full BM-KV and remove one component
at a time to see how much each contributes to the result. Variants:

- A. BM-KV (full)             — attention + recency + prefix + INT8 action
- B. BM-KV w/o prefix         — set gamma = 0 (no prefix bonus)
- C. BM-KV w/o recency        — set beta = 0 (no recency bonus)
- D. BM-KV w/o INT8           — same score, but FP16/DROP only
- E. BM-KV w/o attention      — set alpha = 0 (importance from recency+prefix only)

We evaluate each variant on PPL (WikiText) and Needle-in-Haystack.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

import common  # noqa: F401 sets sys.path
from common import compute_budget, get_wikitext_chunks, load_gpt2, RESULTS_DIR

from bm_kv import (
    POLICIES,
    CompressedRunner,
    policy_bm_kv,
    policy_bm_kv_no_int8,
)
from bm_kv.runner import CompressionConfig

# Each variant: (name, config_factory, policy_fn)
VARIANTS = [
    ("A_full",       lambda: CompressionConfig(),                                       policy_bm_kv),
    ("B_no_prefix",  lambda: CompressionConfig(use_prefix=False),                       policy_bm_kv),
    ("C_no_recency", lambda: CompressionConfig(use_recency=False),                      policy_bm_kv),
    ("D_no_int8",    lambda: CompressionConfig(),                                       policy_bm_kv_no_int8),
    ("E_no_attn",    lambda: CompressionConfig(use_attention=False),                    policy_bm_kv),
]

PROMPT_LEN = 384
TARGET_LEN = 96
PPL_CHUNKS = 30
NEEDLE_FILLERS = 3
NEEDLE_CTX = 768

# We sweep just two ratios for the ablation: a tight budget and a loose one.
ABLATION_RATIOS = [0.50, 0.25]


def run_ppl(model, tok):
    print(f"PPL ablation: {PPL_CHUNKS} chunks of {PROMPT_LEN}+{TARGET_LEN}")
    chunks = get_wikitext_chunks(tok, PROMPT_LEN + TARGET_LEN, PPL_CHUNKS)
    records = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prompt_ids = chunk[:, :PROMPT_LEN].to(model.device)
        target_ids = chunk[:, PROMPT_LEN : PROMPT_LEN + TARGET_LEN].to(model.device)

        # Cache prefill per (use_attention,use_recency,use_prefix).
        prefill_cache = {}

        for ratio in ABLATION_RATIOS:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, cfg_factory, policy_fn in VARIANTS:
                cfg = cfg_factory()
                runner = CompressedRunner(model, tok, cfg)
                key = (cfg.use_attention, cfg.use_recency,
                       cfg.use_prefix, cfg.last_n_layers)
                if key not in prefill_cache:
                    prefill_cache[key] = runner.prefill(prompt_ids)
                prefill_out = prefill_cache[key]

                nll, stats = runner.teacher_force_logprob_with_prefill(
                    prefill_out, prompt_ids, target_ids,
                    policy_fn=policy_fn, budget=budget,
                )
                records[name][ratio].append(nll.mean().item())
        if (ci + 1) % max(1, len(chunks) // 5) == 0:
            print(f"  PPL chunk {ci+1}/{len(chunks)} elapsed {time.time()-t0:.1f}s")

    summary = []
    for name, _, _ in VARIANTS:
        for ratio in ABLATION_RATIOS:
            nlls = records[name][ratio]
            ppls = [math.exp(n) for n in nlls]
            summary.append({
                "variant": name,
                "ratio": ratio,
                "mean_ppl": mean(ppls),
                "n": len(ppls),
            })
    return summary


def run_needle(model, tok):
    print(f"Needle ablation: {NEEDLE_FILLERS} fillers x 5 needles x 5 depths")
    from exp2_needle import NEEDLES, NEEDLE_DEPTHS, build_prompt, MAX_NEW_TOKENS, check_answer
    from bm_kv.runner import _to_cache  # noqa: F401

    filler_chunks = get_wikitext_chunks(tok, NEEDLE_CTX, NEEDLE_FILLERS)
    records = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    n_trials = 0
    for ci, filler in enumerate(filler_chunks):
        filler_ids = filler[0]
        for ni, (needle, suffix, expected) in enumerate(NEEDLES):
            for depth in NEEDLE_DEPTHS:
                prompt_ids = build_prompt(
                    tok, filler_ids, needle, suffix, depth,
                    target_len=NEEDLE_CTX,
                ).to(model.device)
                prompt_len = prompt_ids.shape[1]
                prefill_cache = {}
                for ratio in ABLATION_RATIOS:
                    budget = compute_budget(prompt_len, ratio)
                    for name, cfg_factory, policy_fn in VARIANTS:
                        cfg = cfg_factory()
                        runner = CompressedRunner(model, tok, cfg)
                        key = (cfg.use_attention, cfg.use_recency,
                               cfg.use_prefix, cfg.last_n_layers)
                        if key not in prefill_cache:
                            prefill_cache[key] = runner.prefill(prompt_ids)
                        prefill_out = prefill_cache[key]

                        bridge_logits, bridge_past, _, stats = (
                            runner._compress_from_prefill(
                                prefill_out, prompt_ids, policy_fn, budget
                            )
                        )
                        next_tok = bridge_logits[:, -1, :].argmax(
                            dim=-1, keepdim=True
                        )
                        gen = [next_tok]
                        cur_past = bridge_past
                        with torch.no_grad():
                            for step in range(1, MAX_NEW_TOKENS):
                                pos = prompt_len + step - 1
                                pos_ids = torch.tensor(
                                    [[pos]], device=model.device
                                )
                                out = model(
                                    input_ids=next_tok,
                                    past_key_values=cur_past,
                                    position_ids=pos_ids,
                                    use_cache=True,
                                    return_dict=True,
                                )
                                cur_past = out.past_key_values
                                next_tok = out.logits[:, -1, :].argmax(
                                    dim=-1, keepdim=True
                                )
                                gen.append(next_tok)
                        gen_text = tok.decode(
                            torch.cat(gen, dim=1)[0],
                            skip_special_tokens=True,
                        )
                        correct = check_answer(gen_text, expected)
                        records[name][ratio].append(int(correct))
                        n_trials += 1
        elapsed = time.time() - t0
        print(f"  Needle filler {ci+1}/{len(filler_chunks)} trials={n_trials} "
              f"elapsed={elapsed:.1f}s")

    summary = []
    for name, _, _ in VARIANTS:
        for ratio in ABLATION_RATIOS:
            xs = records[name][ratio]
            summary.append({
                "variant": name,
                "ratio": ratio,
                "accuracy": mean(xs) if xs else 0.0,
                "n": len(xs),
                "n_correct": sum(xs),
            })
    return summary


def run_experiment():
    print("Loading GPT-2 (FP16)")
    model, tok = load_gpt2()

    ppl = run_ppl(model, tok)
    needle = run_needle(model, tok)

    out = {
        "config": {
            "variants": [v[0] for v in VARIANTS],
            "ratios": ABLATION_RATIOS,
            "ppl_chunks": PPL_CHUNKS,
            "needle_fillers": NEEDLE_FILLERS,
            "needle_ctx": NEEDLE_CTX,
            "prompt_len": PROMPT_LEN,
            "target_len": TARGET_LEN,
        },
        "ppl": ppl,
        "needle": needle,
    }
    out_path = RESULTS_DIR / "exp4_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")

    print("\nPPL ablation (lower is better):")
    print(f"{'Variant':<14} {'@0.50':>10} {'@0.25':>10}")
    by = defaultdict(dict)
    for r in ppl:
        by[r["variant"]][r["ratio"]] = r["mean_ppl"]
    for name, _, _ in VARIANTS:
        print(f"{name:<14} {by[name].get(0.50, float('nan')):>10.2f} "
              f"{by[name].get(0.25, float('nan')):>10.2f}")

    print("\nNeedle ablation (higher is better):")
    print(f"{'Variant':<14} {'@0.50':>10} {'@0.25':>10}")
    by = defaultdict(dict)
    for r in needle:
        by[r["variant"]][r["ratio"]] = r["accuracy"]
    for name, _, _ in VARIANTS:
        print(f"{name:<14} {by[name].get(0.50, 0):>10.3f} "
              f"{by[name].get(0.25, 0):>10.3f}")
    return out


if __name__ == "__main__":
    run_experiment()
