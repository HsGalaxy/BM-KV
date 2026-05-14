"""Experiment 3 v2: TTFT, TPOT, tokens/s and KV cache memory on Qwen2.5.

We measure:
- TTFT (Time To First Token): prefill + compress + bridge time.
- TPOT (Time Per Output Token): mean per-step decode latency after the first.
- tokens/s: (max_new_tokens - 1) / sum(per_step_times[1:]).
- Theoretical KV cache bytes vs Full.

For BM-KV-v2 we report two variants:
- "BM-KV-v2 (static)": compress once at prefill and decode straight through.
- "BM-KV-v2 (lazy delta=16)": rebalance every 16 steps + drift-triggered.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from statistics import mean, stdev

import torch

import common  # noqa: F401
from common import compute_budget, get_wikitext_chunks, load_qwen, RESULTS_DIR

from bm_kv import POLICIES_V2, CompressedRunner, policy_bm_kv_v2
from bm_kv.runner import CompressionConfig, _to_legacy

PROMPT_LEN = 1024
DECODE_STEPS = 64
DEFAULT_RATIOS = [1.0, 0.50, 0.25]
N_TRIALS = 5
LAZY_DELTAS = [16]


def policy_kwargs_for(name: str) -> dict:
    cfg = CompressionConfig()
    if name == "AttnOnly":
        cfg.use_recency = False
        cfg.use_prefix = False
    elif name == "Recent":
        cfg.use_attention = False
        cfg.use_prefix = False
    return {"config": cfg}


def kv_bytes(past_key_values) -> int:
    total = 0
    for k, v in past_key_values:
        total += k.element_size() * k.numel()
        total += v.element_size() * v.numel()
    return total


def theoretical_kv_bytes(actions, past_key_values) -> int:
    if not past_key_values:
        return 0
    n_layers = len(past_key_values)
    k0, _ = past_key_values[0]
    per_token = k0.shape[1] * k0.shape[3] * 2  # heads * head_dim * 2 (K+V)
    total = 0
    for a in actions:
        if a == "FP16":
            total += per_token * 2
        elif a == "INT8":
            total += per_token * 1
    return total * n_layers


def time_static(runner, prompt, policy_fn, budget, decode_steps, sync):
    """Return (ttft_s, per_step_ms, final_cache_bytes_theoretical, stats)."""
    from bm_kv.runner import _to_cache

    sync(); t0 = time.perf_counter()
    bridge_logits, bridge_past, _, stats = runner.compress(
        prompt, policy_fn, budget
    )
    sync(); ttft = time.perf_counter() - t0  # prefill + compress + bridge

    next_tok = bridge_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cur_past = bridge_past
    per_step_ms = []
    prompt_len = prompt.shape[1]
    with torch.no_grad():
        for step in range(1, decode_steps):
            sync(); t = time.perf_counter()
            pos = prompt_len + step - 1
            pos_ids = torch.tensor([[pos]], device=runner.device)
            out = runner.model(
                input_ids=next_tok,
                past_key_values=cur_past,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )
            cur_past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sync(); per_step_ms.append((time.perf_counter() - t) * 1000)

    final_legacy = _to_legacy(cur_past)
    actions_final = stats.actions + ["FP16"] * (decode_steps - 1)
    return ttft, per_step_ms, theoretical_kv_bytes(actions_final, final_legacy), stats


def time_lazy(runner, prompt, policy_fn, budget, decode_steps, delta, sync):
    """Same metrics but using generate_lazy."""
    sync(); t0 = time.perf_counter()
    out_ids, info = runner.generate_lazy(
        prompt, policy_fn, budget,
        max_new_tokens=decode_steps,
        delta=delta, drift_threshold=0.5,
    )
    sync(); total_s = time.perf_counter() - t0

    # info["per_step_ms"] starts at step 1 (skipping the bridge/first token).
    per_step_ms = info["per_step_ms"]
    # Approximate TTFT as total - decode time.
    ttft = max(0.0, total_s - sum(per_step_ms) / 1000.0)
    # Final cache size estimate: take the last rebalance's snapshot.
    if info["rebalances"]:
        last = info["rebalances"][-1]
        # Approximate: assume final cache mix mirrors the last rebalance.
        # cost = FP16 * 2 + INT8 * 1 per token of *prefill*; the appended
        # decode tokens are FP16. We compute theoretical bytes from the
        # current cache shape.
        pass
    # Simpler: use heuristic from initial stats since exact final structure
    # depends on rebalance history.
    final_theo = None
    return ttft, per_step_ms, final_theo, info


def run_experiment(n_trials=N_TRIALS, ratios=None, decode_steps=DECODE_STEPS):
    ratios = ratios or DEFAULT_RATIOS
    print("Loading Qwen2.5-0.5B-Instruct")
    model, tok = load_qwen()
    chunks = get_wikitext_chunks(tok, PROMPT_LEN, n_trials)
    print(f"Got {len(chunks)} chunks. Decode steps = {decode_steps}")

    records = defaultdict(lambda: defaultdict(list))
    has_cuda = torch.cuda.is_available()
    def sync():
        if has_cuda:
            torch.cuda.synchronize()

    # Warmup.
    print("Warming up...")
    warm_runner = CompressedRunner(model, tok, CompressionConfig())
    for _ in range(2):
        time_static(warm_runner, chunks[0].to(model.device),
                    POLICIES_V2["Full"],
                    compute_budget(PROMPT_LEN, 1.0), decode_steps, sync)
    sync()

    for ti, chunk in enumerate(chunks):
        prompt = chunk[:, :PROMPT_LEN].to(model.device)

        # Static runs for each policy / ratio.
        for ratio in ratios:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, policy_fn in POLICIES_V2.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue
                cfg = policy_kwargs_for(name)["config"]
                runner = CompressedRunner(model, tok, cfg)
                ttft, per_step_ms, theo, stats = time_static(
                    runner, prompt, policy_fn, budget, decode_steps, sync
                )
                records[f"{name}-static"][ratio].append({
                    "ttft_s": ttft,
                    "per_step_ms": per_step_ms,
                    "tpot_ms": mean(per_step_ms),
                    "tokens_per_s": 1000.0 * (decode_steps - 1) / sum(per_step_ms),
                    "memory_ratio_target": ratio,
                    "memory_ratio_actual": stats.memory_ratio,
                    "theoretical_kv_bytes": theo,
                })

        # Lazy variant for BM-KV-v2 only.
        for ratio in ratios:
            if ratio == 1.0:
                continue
            budget = compute_budget(PROMPT_LEN, ratio)
            for delta in LAZY_DELTAS:
                cfg = policy_kwargs_for("BM-KV-v2")["config"]
                runner = CompressedRunner(model, tok, cfg)
                ttft, per_step_ms, _, info = time_lazy(
                    runner, prompt, POLICIES_V2["BM-KV-v2"], budget,
                    decode_steps, delta, sync,
                )
                records[f"BM-KV-v2-lazy{delta}"][ratio].append({
                    "ttft_s": ttft,
                    "per_step_ms": per_step_ms,
                    "tpot_ms": mean(per_step_ms),
                    "tokens_per_s": 1000.0 * (decode_steps - 1) / sum(per_step_ms),
                    "memory_ratio_target": ratio,
                    "memory_ratio_actual": info["initial_stats"].memory_ratio,
                    "n_rebalances": len(info["rebalances"]),
                })

        if (ti + 1) % max(1, len(chunks) // 4) == 0:
            print(f"  trial {ti+1}/{len(chunks)}")

    # Aggregate.
    summary = []
    for key in records:
        for ratio in records[key]:
            recs = records[key][ratio]
            if not recs:
                continue
            tpots = [r["tpot_ms"] for r in recs]
            tps = [r["tokens_per_s"] for r in recs]
            ttfts = [r["ttft_s"] * 1000 for r in recs]
            summary.append({
                "policy": key,
                "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(r["memory_ratio_actual"] for r in recs),
                "ttft_ms": mean(ttfts),
                "tpot_ms": mean(tpots),
                "tpot_ms_std": stdev(tpots) if len(tpots) > 1 else 0.0,
                "tokens_per_s": mean(tps),
                "n_rebalances": mean([r.get("n_rebalances", 0) for r in recs]),
                "n_trials": len(recs),
            })

    out = {
        "config": {
            "model": "Qwen2.5-0.5B-Instruct",
            "prompt_len": PROMPT_LEN,
            "decode_steps": decode_steps,
            "n_trials": n_trials,
            "ratios": ratios,
            "lazy_deltas": LAZY_DELTAS,
            "device": "cuda" if has_cuda else "cpu",
        },
        "results": summary,
    }
    out_path = RESULTS_DIR / "exp3_runtime_v2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")

    print()
    print(f"{'Policy':<22} {'Ratio':>5} {'TTFT(ms)':>9} {'TPOT(ms)':>9} "
          f"{'tok/s':>7} {'reb':>4}")
    for r in summary:
        print(
            f"{r['policy']:<22} {r['memory_ratio_target']:>5.2f} "
            f"{r['ttft_ms']:>9.1f} {r['tpot_ms']:>9.2f} "
            f"{r['tokens_per_s']:>7.2f} {r['n_rebalances']:>4.1f}"
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n_trials=args.n_trials, decode_steps=args.decode_steps)
