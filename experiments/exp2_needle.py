"""Experiment 2: needle-in-a-haystack retrieval.

We embed a short factual statement (the "needle") at a configurable position
inside a long context of filler text, then end with a prompt that asks the
model to repeat the fact. We measure whether each compression policy retains
enough information for the model to recover the needle.

Because GPT-2 (124M) is a relatively weak base, we use language-model-friendly
needles (sentence completions) rather than instruction-following questions.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

import common  # noqa: F401 sets sys.path
from common import compute_budget, get_wikitext_chunks, load_gpt2, RESULTS_DIR

from bm_kv import POLICIES, CompressedRunner
from bm_kv.runner import CompressionConfig

# (needle_statement, prompt_suffix, expected_answer_substring)
# Answers are kept short (typically 1-2 GPT-2 tokens) and unpredictable from
# the suffix alone, so substring matching gives a fair signal even when the
# model degenerates into repetition.
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
NEEDLE_DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]  # fraction of context where needle is placed
DEFAULT_CONTEXT_TOKENS = 768
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


def build_prompt(
    tokenizer,
    filler_ids: torch.Tensor,
    needle_text: str,
    prompt_suffix: str,
    depth: float,
    target_len: int,
    include_needle: bool = True,
) -> torch.Tensor:
    """Insert needle at relative depth, then append the question suffix.

    The filler is sliced to fit the requested ``target_len`` after the needle
    and suffix are added. If ``include_needle`` is False the needle is omitted
    entirely (used to measure the no-needle baseline accuracy).
    """
    needle_ids = tokenizer(needle_text, return_tensors="pt").input_ids[0]
    suffix_ids = tokenizer(prompt_suffix, return_tensors="pt").input_ids[0]
    if include_needle:
        budget_for_filler = target_len - needle_ids.shape[0] - suffix_ids.shape[0]
    else:
        budget_for_filler = target_len - suffix_ids.shape[0]
    if budget_for_filler < 50:
        raise ValueError("target_len too small for filler")

    filler = filler_ids[:budget_for_filler]
    if include_needle:
        insert_at = int(filler.shape[0] * depth)
        full = torch.cat([
            filler[:insert_at],
            needle_ids,
            filler[insert_at:],
            suffix_ids,
        ])
    else:
        full = torch.cat([filler, suffix_ids])
    return full.unsqueeze(0)


def check_answer(text: str, expected: str) -> bool:
    """Case-insensitive substring match (with light token-friendly cleanup)."""
    return expected.lower() in text.lower()


def run_experiment(
    n_filler_sources: int = 5,
    ratios: list[float] = None,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    depths: list[float] = None,
):
    ratios = ratios or DEFAULT_RATIOS
    depths = depths or NEEDLE_DEPTHS
    print(f"Loading GPT-2")
    model, tok = load_gpt2()

    # Get filler chunks (raw token sequences from WikiText). We'll use each
    # chunk as filler for all needle templates and depths.
    filler_chunks = get_wikitext_chunks(tok, context_tokens, n_filler_sources)
    print(f"Got {len(filler_chunks)} filler chunks; needles={len(NEEDLES)}; "
          f"depths={depths}")

    # results[policy][ratio] -> list of (correct, generated_text, needle_idx, depth)
    records: dict[str, dict[float, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    t0 = time.time()
    n_trials = 0
    total_trials = (
        len(filler_chunks) * len(NEEDLES) * len(depths)
        * (1 + (len(POLICIES) - 1) * len(ratios) - len(ratios))
    )

    for ci, filler in enumerate(filler_chunks):
        filler_ids = filler[0]  # 1D tensor
        for ni, (needle_text, suffix, expected) in enumerate(NEEDLES):
            for depth in depths:
                prompt_ids = build_prompt(
                    tok, filler_ids, needle_text, suffix, depth,
                    target_len=context_tokens,
                )
                prompt_len = prompt_ids.shape[1]
                # Cache prefill per importance-config to avoid redundant work.
                prefill_cache = {}
                for ratio in ratios:
                    budget = compute_budget(prompt_len, ratio)
                    for name, policy_fn in POLICIES.items():
                        if ratio == 1.0 and name != "Full":
                            continue
                        if ratio != 1.0 and name == "Full":
                            continue
                        kw = policy_kwargs_for(name)
                        cfg = kw["config"]
                        key = (cfg.use_attention, cfg.use_recency,
                               cfg.use_prefix, cfg.last_n_layers)
                        runner = CompressedRunner(model, tok, cfg)

                        if key not in prefill_cache:
                            prefill_cache[key] = runner.prefill(
                                prompt_ids.to(model.device)
                            )
                        prefill_out = prefill_cache[key]

                        # Use the bridge-based compression so the first
                        # generated token reflects the compressed cache.
                        bridge_logits, bridge_past, _, stats = (
                            runner._compress_from_prefill(
                                prefill_out, prompt_ids.to(model.device),
                                policy_fn, budget,
                            )
                        )
                        next_logits = bridge_logits[:, -1, :]
                        next_tok = next_logits.argmax(dim=-1, keepdim=True)
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

                        gen_ids = torch.cat(gen, dim=1)[0]
                        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
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

        elapsed = time.time() - t0
        print(f"  filler {ci+1}/{len(filler_chunks)}  trials={n_trials}  "
              f"elapsed={elapsed:.1f}s")

    # Summarize accuracy.
    summary = []
    for name in ["Full"] + [n for n in POLICIES if n != "Full"]:
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
            "context_tokens": context_tokens,
            "n_filler_sources": n_filler_sources,
            "depths": depths,
            "n_needles": len(NEEDLES),
            "max_new_tokens": MAX_NEW_TOKENS,
            "ratios": ratios,
        },
        "results": summary,
        "raw": records,
    }
    out_path = RESULTS_DIR / "exp2_needle.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")

    print()
    print(f"{'Policy':<10} {'Ratio':>6} {'MemUsed':>8} {'Acc':>8} {'Correct':>10}")
    for row in summary:
        print(
            f"{row['policy']:<10} "
            f"{row['memory_ratio_target']:>6.2f} "
            f"{row['memory_ratio_actual']:>8.3f} "
            f"{row['accuracy']:>7.3f} "
            f"{row['n_correct']:>4}/{row['n_trials']:<5}"
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-fillers", type=int, default=5)
    p.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        n_filler_sources=args.n_fillers,
        context_tokens=args.context_tokens,
    )
