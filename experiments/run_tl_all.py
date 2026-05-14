"""Run the full v2 experiment suite on TinyLlama-1.1B-Chat.

Replays exp1 (PPL), exp2 (Needle), exp3 (Runtime), exp4 (Ablation) using
TinyLlama instead of Qwen2.5, and writes results to ``exp*_tl.json``.

This script consolidates the four experiments to share a single model load
(loading takes a few seconds for the 1.1B model)."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from statistics import mean, stdev

import torch

import common  # noqa: F401
from common import (
    compute_budget, get_wikitext_chunks, load_tinyllama, RESULTS_DIR,
)

from bm_kv import (
    POLICIES_V2, CompressedRunner,
    policy_bm_kv, policy_bm_kv_v2,
)
from bm_kv.runner import CompressionConfig, _to_legacy


# ----- shared config ------------------------------------------------------
PPL_PROMPT = 512
PPL_TARGET = 128
PPL_CHUNKS = 15
NEEDLE_CTX = 1024
NEEDLE_FILLERS = 3
NEEDLE_MAX_NEW = 12
RUNTIME_PROMPT = 1024
RUNTIME_DECODE_STEPS = 64
RUNTIME_N_TRIALS = 5
ABLATION_PROMPT = 512
ABLATION_TARGET = 128
ABLATION_CHUNKS = 8
RATIOS = [1.0, 0.75, 0.50, 0.35, 0.25]
ABLATION_RATIOS = [0.50, 0.25]
RUNTIME_RATIOS = [1.0, 0.50, 0.25]

NEEDLE_DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
NEEDLES = [
    (" The launch code we agreed on is 743 according to my notes.",
     " The launch code we agreed on is", "743"),
    (" The signal we picked up was on channel 218 yesterday.",
     " The signal we picked up was on channel", "218"),
    (" The team finished the round with a score of 526 points.",
     " The team finished the round with a score of", "526"),
    (" The hidden treasure is in box number 309 of the storage room.",
     " The hidden treasure is in box number", "309"),
    (" The combination to the safe is 891 my friend please remember.",
     " The combination to the safe is", "891"),
]


def cfg_for(name: str) -> CompressionConfig:
    cfg = CompressionConfig()
    if name == "AttnOnly":
        cfg.use_recency = False
        cfg.use_prefix = False
    elif name == "Recent":
        cfg.use_attention = False
        cfg.use_prefix = False
    return cfg


# ----- exp1: PPL ---------------------------------------------------------
def run_ppl(model, tok):
    print(f"[exp1] PPL: {PPL_CHUNKS} chunks of {PPL_PROMPT}+{PPL_TARGET}")
    chunks = get_wikitext_chunks(tok, PPL_PROMPT + PPL_TARGET, PPL_CHUNKS)
    nll_rec = defaultdict(lambda: defaultdict(list))
    mem_rec = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prompt = chunk[:, :PPL_PROMPT].to(model.device)
        target = chunk[:, PPL_PROMPT : PPL_PROMPT + PPL_TARGET].to(model.device)
        prefill_cache = {}
        for ratio in RATIOS:
            budget = compute_budget(PPL_PROMPT, ratio)
            for name, pol in POLICIES_V2.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue
                cfg = cfg_for(name)
                key = (cfg.use_attention, cfg.use_recency, cfg.use_prefix, cfg.last_n_layers)
                runner = CompressedRunner(model, tok, cfg)
                if key not in prefill_cache:
                    prefill_cache[key] = runner.prefill(prompt)
                pf = prefill_cache[key]
                nll, stats = runner.teacher_force_logprob_with_prefill(
                    pf, prompt, target, policy_fn=pol, budget=budget,
                )
                nll_rec[name][ratio].append(nll.mean().item())
                mem_rec[name][ratio].append(stats.memory_ratio)
        prefill_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (ci + 1) % max(1, len(chunks) // 5) == 0:
            print(f"  chunk {ci+1}/{len(chunks)} elapsed {time.time()-t0:.1f}s")

    summary = []
    for name in ["Full"] + [n for n in POLICIES_V2 if n != "Full"]:
        for ratio in RATIOS:
            if ratio == 1.0 and name != "Full":
                continue
            if ratio != 1.0 and name == "Full":
                continue
            nlls = nll_rec[name][ratio]
            mems = mem_rec[name][ratio]
            if not nlls:
                continue
            ppls = [math.exp(n) for n in nlls]
            summary.append({
                "policy": name, "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(mems),
                "mean_nll": mean(nlls), "mean_ppl": mean(ppls),
                "std_ppl": stdev(ppls) if len(ppls) > 1 else 0.0,
                "n_chunks": len(nlls),
            })
    fr = next((r for r in summary if r["policy"] == "Full"), None)
    if fr:
        for r in summary:
            r["ppl_increment"] = r["mean_ppl"] - fr["mean_ppl"]
            r["ppl_increment_pct"] = (r["mean_ppl"] - fr["mean_ppl"]) / fr["mean_ppl"] * 100
    out = {
        "config": {"model": "TinyLlama-1.1B-Chat-v1.0",
                   "prompt_len": PPL_PROMPT, "target_len": PPL_TARGET,
                   "n_chunks": PPL_CHUNKS, "ratios": RATIOS},
        "results": summary,
    }
    (RESULTS_DIR / "exp1_ppl_tl.json").write_text(json.dumps(out, indent=2))
    return out


# ----- exp2: Needle -------------------------------------------------------
def run_needle(model, tok):
    print(f"[exp2] Needle: ctx={NEEDLE_CTX}, {NEEDLE_FILLERS} fillers")
    fillers = get_wikitext_chunks(tok, NEEDLE_CTX, NEEDLE_FILLERS)
    records = defaultdict(lambda: defaultdict(list))

    def build_prompt(filler_ids, needle, suffix, depth):
        n_ids = tok(needle, return_tensors="pt").input_ids[0]
        s_ids = tok(suffix, return_tensors="pt").input_ids[0]
        budget_for_filler = NEEDLE_CTX - n_ids.shape[0] - s_ids.shape[0]
        f = filler_ids[:budget_for_filler]
        at = int(f.shape[0] * depth)
        return torch.cat([f[:at], n_ids, f[at:], s_ids]).unsqueeze(0)

    t0 = time.time()
    n_trials = 0
    for ci, filler in enumerate(fillers):
        fid = filler[0]
        for ni, (needle, suf, expected) in enumerate(NEEDLES):
            for depth in NEEDLE_DEPTHS:
                prompt = build_prompt(fid, needle, suf, depth).to(model.device)
                pl = prompt.shape[1]
                pcache = {}
                for ratio in RATIOS:
                    budget = compute_budget(pl, ratio)
                    for name, pol in POLICIES_V2.items():
                        if ratio == 1.0 and name != "Full":
                            continue
                        if ratio != 1.0 and name == "Full":
                            continue
                        cfg = cfg_for(name)
                        key = (cfg.use_attention, cfg.use_recency, cfg.use_prefix, cfg.last_n_layers)
                        runner = CompressedRunner(model, tok, cfg)
                        if key not in pcache:
                            pcache[key] = runner.prefill(prompt)
                        pf = pcache[key]
                        bl, bp, _, stats = runner._compress_from_prefill(
                            pf, prompt, pol, budget,
                        )
                        nt = bl[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = [nt]
                        cur = bp
                        with torch.no_grad():
                            for step in range(1, NEEDLE_MAX_NEW):
                                pos = pl + step - 1
                                pos_ids = torch.tensor([[pos]], device=model.device)
                                out = model(input_ids=nt, past_key_values=cur,
                                            position_ids=pos_ids,
                                            use_cache=True, return_dict=True)
                                cur = out.past_key_values
                                nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                                gen.append(nt)
                        gen_text = tok.decode(torch.cat(gen, dim=1)[0],
                                              skip_special_tokens=True)
                        ok = expected.lower() in gen_text.lower()
                        records[name][ratio].append({
                            "correct": ok, "gen": gen_text,
                            "needle_idx": ni, "depth": depth,
                            "filler_idx": ci, "expected": expected,
                            "mem_ratio": stats.memory_ratio,
                        })
                        n_trials += 1
                pcache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        print(f"  filler {ci+1}/{len(fillers)} trials={n_trials} "
              f"elapsed={time.time()-t0:.1f}s")

    summary = []
    for name in ["Full"] + [n for n in POLICIES_V2 if n != "Full"]:
        for ratio in RATIOS:
            if ratio == 1.0 and name != "Full":
                continue
            if ratio != 1.0 and name == "Full":
                continue
            recs = records[name][ratio]
            if not recs:
                continue
            nok = sum(1 for r in recs if r["correct"])
            summary.append({
                "policy": name, "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(r["mem_ratio"] for r in recs),
                "n_trials": len(recs), "accuracy": nok / len(recs),
                "n_correct": nok,
            })
    out = {
        "config": {"model": "TinyLlama-1.1B-Chat-v1.0",
                   "context_tokens": NEEDLE_CTX,
                   "n_fillers": NEEDLE_FILLERS,
                   "depths": NEEDLE_DEPTHS,
                   "n_needles": len(NEEDLES),
                   "max_new_tokens": NEEDLE_MAX_NEW,
                   "ratios": RATIOS},
        "results": summary, "raw": records,
    }
    (RESULTS_DIR / "exp2_needle_tl.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# ----- exp3: Runtime ------------------------------------------------------
def run_runtime(model, tok):
    print(f"[exp3] Runtime: prompt={RUNTIME_PROMPT} decode={RUNTIME_DECODE_STEPS}")
    chunks = get_wikitext_chunks(tok, RUNTIME_PROMPT, RUNTIME_N_TRIALS)
    has_cuda = torch.cuda.is_available()
    def sync():
        if has_cuda:
            torch.cuda.synchronize()

    print("  warmup")
    warm = CompressedRunner(model, tok, CompressionConfig())
    for _ in range(2):
        _ = warm.generate(
            chunks[0].to(model.device), POLICIES_V2["Full"],
            compute_budget(RUNTIME_PROMPT, 1.0), max_new_tokens=RUNTIME_DECODE_STEPS,
        )
    sync()

    records = defaultdict(lambda: defaultdict(list))
    for ti, chunk in enumerate(chunks):
        prompt = chunk[:, :RUNTIME_PROMPT].to(model.device)
        for ratio in RUNTIME_RATIOS:
            budget = compute_budget(RUNTIME_PROMPT, ratio)
            for name, pol in POLICIES_V2.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue
                cfg = cfg_for(name)
                runner = CompressedRunner(model, tok, cfg)
                sync(); t0 = time.perf_counter()
                bl, bp, _, stats = runner.compress(prompt, pol, budget)
                sync(); ttft = time.perf_counter() - t0
                next_tok = bl[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = bp
                per_step = []
                with torch.no_grad():
                    for step in range(1, RUNTIME_DECODE_STEPS):
                        sync(); ts = time.perf_counter()
                        pos = RUNTIME_PROMPT + step - 1
                        pos_ids = torch.tensor([[pos]], device=model.device)
                        out = model(input_ids=next_tok, past_key_values=cur,
                                    position_ids=pos_ids, use_cache=True,
                                    return_dict=True)
                        cur = out.past_key_values
                        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        sync(); per_step.append((time.perf_counter() - ts) * 1000)
                records[f"{name}-static"][ratio].append({
                    "ttft_s": ttft, "per_step_ms": per_step,
                    "tpot_ms": mean(per_step),
                    "tokens_per_s": 1000.0 * (RUNTIME_DECODE_STEPS - 1) / sum(per_step),
                    "memory_ratio_target": ratio,
                    "memory_ratio_actual": stats.memory_ratio,
                })
        # Lazy variant for BM-KV-v2.
        for ratio in RUNTIME_RATIOS:
            if ratio == 1.0:
                continue
            budget = compute_budget(RUNTIME_PROMPT, ratio)
            cfg = cfg_for("BM-KV-v2")
            runner = CompressedRunner(model, tok, cfg)
            sync(); t0 = time.perf_counter()
            _, info = runner.generate_lazy(
                prompt, POLICIES_V2["BM-KV-v2"], budget,
                max_new_tokens=RUNTIME_DECODE_STEPS, delta=16,
                drift_threshold=0.5,
            )
            sync(); total_s = time.perf_counter() - t0
            ttft = max(0.0, total_s - sum(info["per_step_ms"]) / 1000.0)
            records["BM-KV-v2-lazy16"][ratio].append({
                "ttft_s": ttft,
                "per_step_ms": info["per_step_ms"],
                "tpot_ms": mean(info["per_step_ms"]),
                "tokens_per_s": 1000.0 * (RUNTIME_DECODE_STEPS - 1) / sum(info["per_step_ms"]),
                "memory_ratio_target": ratio,
                "memory_ratio_actual": info["initial_stats"].memory_ratio,
                "n_rebalances": len(info["rebalances"]),
            })
        if (ti + 1) % max(1, len(chunks) // 4) == 0:
            print(f"  trial {ti+1}/{len(chunks)}")

    summary = []
    for key in records:
        for ratio in records[key]:
            recs = records[key][ratio]
            tpots = [r["tpot_ms"] for r in recs]
            tps = [r["tokens_per_s"] for r in recs]
            summary.append({
                "policy": key, "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(r["memory_ratio_actual"] for r in recs),
                "ttft_ms": mean(r["ttft_s"] * 1000 for r in recs),
                "tpot_ms": mean(tpots),
                "tpot_ms_std": stdev(tpots) if len(tpots) > 1 else 0.0,
                "tokens_per_s": mean(tps),
                "n_rebalances": mean([r.get("n_rebalances", 0) for r in recs]),
                "n_trials": len(recs),
            })
    out = {
        "config": {"model": "TinyLlama-1.1B-Chat-v1.0",
                   "prompt_len": RUNTIME_PROMPT,
                   "decode_steps": RUNTIME_DECODE_STEPS,
                   "n_trials": RUNTIME_N_TRIALS, "ratios": RUNTIME_RATIOS,
                   "device": "cuda" if has_cuda else "cpu"},
        "results": summary,
    }
    (RESULTS_DIR / "exp3_runtime_tl.json").write_text(json.dumps(out, indent=2))
    return out


# ----- exp4: Ablation -----------------------------------------------------
def run_ablation(model, tok):
    print(f"[exp4] Ablation: {ABLATION_CHUNKS} chunks")
    chunks = get_wikitext_chunks(tok, ABLATION_PROMPT + ABLATION_TARGET, ABLATION_CHUNKS)
    from bm_kv import policy_bm_kv_no_int8
    from bm_kv.blocks import (
        aggregate_token_scores_to_blocks,
        expand_block_actions_to_tokens, num_blocks,
    )

    def policy_no_threshold(scores, budget):
        return policy_bm_kv_v2(scores, budget,
                               theta_drop_quantile=0.0, theta_fp16_quantile=0.7)

    def policy_token_level(scores, budget):
        return policy_bm_kv_v2(scores, budget, block_size=1)

    def policy_no_int8_v2(scores, budget, block_size=16):
        if isinstance(scores, torch.Tensor):
            s = scores.detach().float()
        else:
            s = torch.tensor(list(scores), dtype=torch.float32)
        n = s.shape[-1]
        nb = num_blocks(n, block_size)
        bs = aggregate_token_scores_to_blocks(s, block_size)
        fp16_cost = 2 * block_size
        nfp = max(0, min(nb, budget // fp16_cost))
        order = torch.argsort(bs, descending=True).tolist()
        ba = ["DROP"] * nb
        for j in order[:nfp]:
            ba[j] = "FP16"
        return expand_block_actions_to_tokens(ba, n, block_size)

    variants = [
        ("A_full_v2", policy_bm_kv_v2),
        ("B_no_threshold", policy_no_threshold),
        ("C_token_level", policy_token_level),
        ("D_no_int8", policy_no_int8_v2),
        ("E_v1_token_kv", policy_bm_kv),
    ]

    records = defaultdict(lambda: defaultdict(list))
    for ci, chunk in enumerate(chunks):
        prompt = chunk[:, :ABLATION_PROMPT].to(model.device)
        target = chunk[:, ABLATION_PROMPT : ABLATION_PROMPT + ABLATION_TARGET].to(model.device)
        pf = None
        for ratio in ABLATION_RATIOS:
            budget = compute_budget(ABLATION_PROMPT, ratio)
            for name, pol in variants:
                runner = CompressedRunner(model, tok, CompressionConfig())
                if pf is None:
                    pf = runner.prefill(prompt)
                nll, stats = runner.teacher_force_logprob_with_prefill(
                    pf, prompt, target, policy_fn=pol, budget=budget,
                )
                records[name][ratio].append({
                    "nll": nll.mean().item(), "mem": stats.memory_ratio,
                })
        pf = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ppl_summary = []
    for name, _ in variants:
        for ratio in ABLATION_RATIOS:
            recs = records[name][ratio]
            ppls = [math.exp(r["nll"]) for r in recs]
            ppl_summary.append({
                "variant": name, "ratio": ratio,
                "mean_ppl": mean(ppls), "n": len(ppls),
                "mem_ratio": mean(r["mem"] for r in recs),
            })

    # Lazy comparison (short version)
    print("  lazy comparison")
    lazy_chunks = get_wikitext_chunks(tok, RUNTIME_PROMPT, 3)
    LAZY_RATIO = 0.35
    budget = compute_budget(RUNTIME_PROMPT, LAZY_RATIO)
    DECODE = 64
    lazy_rows = []
    for ci, c in enumerate(lazy_chunks):
        prompt = c[:, :RUNTIME_PROMPT].to(model.device)
        runner = CompressedRunner(model, tok, CompressionConfig())
        _, info_s = runner.generate_lazy(prompt, policy_bm_kv_v2, budget,
                                         max_new_tokens=DECODE,
                                         delta=DECODE + 1, drift_threshold=None)
        lazy_rows.append({"mode": "static",
                          "rebalances": len(info_s["rebalances"]),
                          "mean_step_ms": mean(info_s["per_step_ms"])})
        _, info_16 = runner.generate_lazy(prompt, policy_bm_kv_v2, budget,
                                          max_new_tokens=DECODE,
                                          delta=16, drift_threshold=0.5)
        lazy_rows.append({"mode": "lazy_delta16",
                          "rebalances": len(info_16["rebalances"]),
                          "mean_step_ms": mean(info_16["per_step_ms"])})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lazy_by = defaultdict(list)
    for r in lazy_rows:
        lazy_by[r["mode"]].append(r)
    lazy_summary = [{
        "mode": m, "n_trials": len(recs),
        "mean_step_ms": mean(r["mean_step_ms"] for r in recs),
        "mean_rebalances": mean(r["rebalances"] for r in recs),
    } for m, recs in lazy_by.items()]

    out = {
        "config": {"model": "TinyLlama-1.1B-Chat-v1.0",
                   "ratios": ABLATION_RATIOS,
                   "prompt_len": ABLATION_PROMPT,
                   "target_len": ABLATION_TARGET,
                   "n_chunks": ABLATION_CHUNKS},
        "ppl_ablation": ppl_summary,
        "lazy_summary": lazy_summary,
    }
    (RESULTS_DIR / "exp4_ablation_tl.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main():
    print("Loading TinyLlama-1.1B-Chat-v1.0")
    model, tok = load_tinyllama()

    ppl = run_ppl(model, tok)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    needle = run_needle(model, tok)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    runtime = run_runtime(model, tok)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ablation = run_ablation(model, tok)

    print("\n=== Summary ===")
    print("\nPPL:")
    for r in ppl["results"]:
        print(f"  {r['policy']:<12} ratio={r['memory_ratio_target']:.2f} "
              f"PPL={r['mean_ppl']:8.2f} Δ%={r.get('ppl_increment_pct', 0):+.1f}%")
    print("\nNeedle:")
    for r in needle["results"]:
        print(f"  {r['policy']:<12} ratio={r['memory_ratio_target']:.2f} "
              f"acc={r['accuracy']:.3f} ({r['n_correct']}/{r['n_trials']})")
    print("\nRuntime:")
    for r in runtime["results"]:
        print(f"  {r['policy']:<22} ratio={r['memory_ratio_target']:.2f} "
              f"TPOT={r['tpot_ms']:.2f}ms tps={r['tokens_per_s']:.2f}")
    print("\nAblation PPL:")
    for r in ablation["ppl_ablation"]:
        print(f"  {r['variant']:<18} ratio={r['ratio']:.2f} PPL={r['mean_ppl']:.2f}")
    print("Saved exp1_ppl_tl.json, exp2_needle_tl.json, exp3_runtime_tl.json, exp4_ablation_tl.json")


if __name__ == "__main__":
    main()
