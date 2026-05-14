"""Experiment 2 v2: Needle-in-a-Haystack on Qwen2.5 with longer contexts.

Qwen2.5-0.5B-Instruct supports 32k context, so we can place the needle at
realistic long-context depths. We compare every v2 policy (BM-KV-v2 vs
baselines) at multiple memory budgets.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean

import torch

import common  # noqa: F401
from common import compute_budget, get_wikitext_chunks, load_qwen, RESULTS_DIR

from bm_kv import POLICIES_V2, CompressedRunner
from bm_kv.runner import CompressionConfig

# Same kind of LM-friendly needles as v1: short, unpredictable single-token
# answers so substring matching is fair when the model degenerates into
# repetition under aggressive compression.
NEEDLES = [
    (
        " The launch code we agreed on is 743 according to my notes.",
        " The launch code we agreed on is",
        "743",
    ),
    (
        " The signal we picked up was on channel 218 yesterday.",
        " The signal we picked up was on channel",
        "218",
    ),
    (
        " The team finished the round with a score of 526 points.",
        " The team finished the round with a score of",
        "526",
    ),
    (
        " The hidden treasure is in box number 309 of the storage room.",
        " The hidden treasure is in box number",
        "309",
    ),
    (
        " The combination to the safe is 891 my friend please remember.",
        " The combination to the safe is",
        "891",
    ),
]
DEFAULT_RATIOS = [1.0, 0.75, 0.50, 0.35, 0.25]
NEEDLE_DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTEXT_TOKENS = 1536
MAX_NEW_TOKENS = 12


def policy_kwargs_for(name: str) -> dict:
    cfg = CompressionConfig()
    if name == "AttnOnly":
        cfg.use_recency = False
        cfg.use_prefix = False
    elif name == "Recent":
        cfg.use_attention = False
        cfg.use_prefix = False
    return {"config": cfg}


def build_prompt(tokenizer, filler_ids, needle, suffix, depth, target_len):
    needle_ids = tokenizer(needle, return_tensors="pt").input_ids[0]
    suffix_ids = tokenizer(suffix, return_tensors="pt").input_ids[0]
    budget_for_filler = target_len - needle_ids.shape[0] - suffix_ids.shape[0]
    filler = filler_ids[:budget_for_filler]
    insert_at = int(filler.shape[0] * depth)
    full = torch.cat([
        filler[:insert_at], needle_ids, filler[insert_at:], suffix_ids,
    ])
    return full.unsqueeze(0)


def check_answer(text: str, expected: str) -> bool:
    return expected.lower() in text.lower()


def run_experiment(
    n_fillers: int = 3,
    context_tokens: int = CONTEXT_TOKENS,
    depths: list[float] = None,
    ratios: list[float] = None,
):
    depths = depths or NEEDLE_DEPTHS
    ratios = ratios or DEFAULT_RATIOS
    print("Loading Qwen2.5-0.5B-Instruct")
    model, tok = load_qwen()
    filler_chunks = get_wikitext_chunks(tok, context_tokens, n_fillers)
    print(f"{len(filler_chunks)} fillers x {len(NEEDLES)} needles x "
          f"{len(depths)} depths = "
          f"{len(filler_chunks) * len(NEEDLES) * len(depths)} prompts")

    from bm_kv.runner import _to_cache  # noqa: F401
    records = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    n_trials = 0
    for ci, filler in enumerate(filler_chunks):
        filler_ids = filler[0]
        for ni, (needle, suffix, expected) in enumerate(NEEDLES):
            for depth in depths:
                prompt_ids = build_prompt(
                    tok, filler_ids, needle, suffix, depth,
                    target_len=context_tokens,
                ).to(model.device)
                prompt_len = prompt_ids.shape[1]

                prefill_cache = {}
                for ratio in ratios:
                    budget = compute_budget(prompt_len, ratio)
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

                        bridge_logits, bridge_past, _, stats = (
                            runner._compress_from_prefill(
                                prefill_out, prompt_ids, policy_fn, budget,
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
                        records[name][ratio].append({
                            "correct": correct,
                            "gen": gen_text,
                            "needle_idx": ni,
                            "depth": depth,
                            "filler_idx": ci,
                            "expected": expected,
                            "mem_ratio": stats.memory_ratio,
                        })
                        n_trials += 1

                prefill_cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"  filler {ci+1}/{len(filler_chunks)} trials={n_trials} "
              f"elapsed={elapsed:.1f}s")

    summary = []
    for name in ["Full"] + [n for n in POLICIES_V2 if n != "Full"]:
        for ratio in ratios:
            if ratio == 1.0 and name != "Full":
                continue
            if ratio != 1.0 and name == "Full":
                continue
            recs = records[name][ratio]
            if not recs:
                continue
            n_correct = sum(1 for r in recs if r["correct"])
            summary.append({
                "policy": name,
                "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(r["mem_ratio"] for r in recs),
                "n_trials": len(recs),
                "accuracy": n_correct / len(recs),
                "n_correct": n_correct,
            })

    out = {
        "config": {
            "model": "Qwen2.5-0.5B-Instruct",
            "context_tokens": context_tokens,
            "n_fillers": n_fillers,
            "depths": depths,
            "n_needles": len(NEEDLES),
            "max_new_tokens": MAX_NEW_TOKENS,
            "ratios": ratios,
        },
        "results": summary,
        "raw": records,
    }
    out_path = RESULTS_DIR / "exp2_needle_v2.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")

    print()
    print(f"{'Policy':<12} {'Ratio':>6} {'MemUsed':>8} {'Acc':>8} {'Correct':>10}")
    for row in summary:
        print(
            f"{row['policy']:<12} {row['memory_ratio_target']:>6.2f} "
            f"{row['memory_ratio_actual']:>8.3f} "
            f"{row['accuracy']:>7.3f} "
            f"{row['n_correct']:>4}/{row['n_trials']:<5}"
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-fillers", type=int, default=3)
    p.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n_fillers=args.n_fillers, context_tokens=args.context_tokens)
