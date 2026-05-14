"""Experiment 3: prefill / compression / decoding latency and observed cache
size. We do NOT claim BM-KV is faster than full cache (the score+sort step
adds work). We do show that the per-decoding-step latency is comparable to
the baselines, so the algorithm is practical for a small extra prefill cost.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch

import common  # noqa: F401 sets sys.path
from common import compute_budget, get_wikitext_chunks, load_gpt2, RESULTS_DIR

from bm_kv import POLICIES, CompressedRunner
from bm_kv.runner import CompressionConfig

PROMPT_LEN = 512
DECODE_STEPS = 32
DEFAULT_RATIOS = [1.0, 0.50, 0.25]
N_TRIALS = 8


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
    """Sum element_size * numel across every K/V tensor (actual storage)."""
    total = 0
    for k, v in past_key_values:
        total += k.element_size() * k.numel()
        total += v.element_size() * v.numel()
    return total


def theoretical_kv_bytes(actions: list[str], past_key_values) -> int:
    """Bytes a real implementation would use for the cache, given that
    INT8-marked tokens cost half as much as FP16 ones. Counts every layer
    once, with FP16=2 bytes per scalar and INT8=1 byte per scalar."""
    if not past_key_values:
        return 0
    n_layers = len(past_key_values)
    k0, v0 = past_key_values[0]
    # k0 shape: [batch, heads, seq, head_dim]
    per_token_floats = k0.shape[1] * k0.shape[3] * 2  # K + V
    bytes_per_layer = 0
    for a in actions:
        if a == "FP16":
            bytes_per_layer += per_token_floats * 2
        elif a == "INT8":
            bytes_per_layer += per_token_floats * 1
        # DROP -> 0
    return bytes_per_layer * n_layers


def run_experiment(n_trials: int = N_TRIALS, ratios: list[float] = None):
    ratios = ratios or DEFAULT_RATIOS
    print("Loading GPT-2")
    model, tok = load_gpt2()
    chunks = get_wikitext_chunks(tok, PROMPT_LEN, n_trials)
    print(f"Got {len(chunks)} chunks")

    records: dict[str, dict[float, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    has_cuda = torch.cuda.is_available()

    def sync():
        if has_cuda:
            torch.cuda.synchronize()

    # Warmup: run two full prefill+decode cycles before timing.
    print("Warming up...")
    warm_runner = CompressedRunner(model, tok, CompressionConfig())
    for _ in range(2):
        _ = warm_runner.generate(
            chunks[0], policy_fn=POLICIES["Full"],
            budget=compute_budget(PROMPT_LEN, 1.0),
            max_new_tokens=DECODE_STEPS,
        )
    sync()

    for ti, chunk in enumerate(chunks):
        prompt = chunk[:, :PROMPT_LEN].to(model.device)

        # Cache prefill per importance config.
        prefill_cache = {}
        prefill_time_cache = {}
        for ratio in ratios:
            budget = compute_budget(PROMPT_LEN, ratio)
            for name, policy_fn in POLICIES.items():
                if ratio == 1.0 and name != "Full":
                    continue
                if ratio != 1.0 and name == "Full":
                    continue

                kw = policy_kwargs_for(name)
                runner = CompressedRunner(model, tok, kw["config"])
                cfg = kw["config"]
                key = (cfg.use_attention, cfg.use_recency,
                       cfg.use_prefix, cfg.last_n_layers)

                # Prefill (timed once per cfg key, cached for reuse below).
                if key not in prefill_cache:
                    sync()
                    t0 = time.perf_counter()
                    prefill_out = runner.prefill(prompt)
                    sync()
                    t1 = time.perf_counter()
                    prefill_cache[key] = prefill_out
                    prefill_time_cache[key] = t1 - t0
                prefill_out = prefill_cache[key]
                prefill_time = prefill_time_cache[key]

                # Compression + bridge step.
                sync()
                t0 = time.perf_counter()
                bridge_logits, bridge_past, _, stats = (
                    runner._compress_from_prefill(
                        prefill_out, prompt, policy_fn, budget
                    )
                )
                sync()
                t_compress = time.perf_counter() - t0

                # Measure cache memory after compression. Two metrics:
                #   * cache_bytes: actual storage in our FP16-simulated cache
                #     (INT8 round-trip does not save bytes here).
                #   * theoretical_bytes: what a real INT8 backend would use.
                from bm_kv.runner import _to_legacy
                bridge_legacy = _to_legacy(bridge_past)
                cache_bytes = kv_bytes(bridge_legacy)
                theo_bytes = theoretical_kv_bytes(stats.actions, bridge_legacy)

                # Pure decoding loop (token by token, like generate).
                sync()
                t0 = time.perf_counter()
                next_tok = bridge_logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                cur_past = bridge_past
                with torch.no_grad():
                    for step in range(1, DECODE_STEPS):
                        pos = PROMPT_LEN + step - 1
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
                sync()
                t_decode = time.perf_counter() - t0

                records[name][ratio].append({
                    "prefill_s": prefill_time,
                    "compress_s": t_compress,
                    "decode_s": t_decode,
                    "ms_per_token": t_decode / max(1, DECODE_STEPS - 1) * 1000,
                    "cache_bytes": cache_bytes,
                    "theoretical_bytes": theo_bytes,
                    "memory_ratio": stats.memory_ratio,
                })

        if (ti + 1) % max(1, len(chunks) // 4) == 0:
            print(f"  trial {ti+1}/{len(chunks)}")

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
            ms = [r["ms_per_token"] for r in recs]
            cbytes = [r["cache_bytes"] for r in recs]
            theo = [r["theoretical_bytes"] for r in recs]
            summary.append({
                "policy": name,
                "memory_ratio_target": ratio,
                "memory_ratio_actual": mean(r["memory_ratio"] for r in recs),
                "prefill_s": mean(r["prefill_s"] for r in recs),
                "compress_s": mean(r["compress_s"] for r in recs),
                "decode_s": mean(r["decode_s"] for r in recs),
                "ms_per_token": mean(ms),
                "ms_per_token_std": stdev(ms) if len(ms) > 1 else 0.0,
                "cache_bytes_simulated": mean(cbytes),
                "cache_kb_simulated": mean(cbytes) / 1024.0,
                "cache_bytes_theoretical": mean(theo),
                "cache_kb_theoretical": mean(theo) / 1024.0,
                "n_trials": len(recs),
            })

    full_row = next((r for r in summary if r["policy"] == "Full"), None)
    if full_row:
        full_bytes = full_row["cache_bytes_simulated"]
        for row in summary:
            row["cache_bytes_ratio_simulated"] = (
                row["cache_bytes_simulated"] / full_bytes
            )
            row["cache_bytes_ratio_theoretical"] = (
                row["cache_bytes_theoretical"] / full_bytes
            )

    out = {
        "config": {
            "prompt_len": PROMPT_LEN,
            "decode_steps": DECODE_STEPS,
            "n_trials": n_trials,
            "ratios": ratios,
            "device": "cuda" if has_cuda else "cpu",
            "decoding_latency_includes": "per-step decode loop excluding prefill+compression",
        },
        "results": summary,
    }
    out_path = RESULTS_DIR / "exp3_runtime.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")

    print()
    print(
        f"{'Policy':<10} {'Ratio':>5} "
        f"{'Prefill':>9} {'Compr':>9} {'ms/tok':>8} "
        f"{'KV-real':>9} {'KV-theo':>9} {'theo%':>7}"
    )
    for r in summary:
        print(
            f"{r['policy']:<10} "
            f"{r['memory_ratio_target']:>5.2f} "
            f"{r['prefill_s']*1000:>7.1f}ms "
            f"{r['compress_s']*1000:>7.1f}ms "
            f"{r['ms_per_token']:>7.2f} "
            f"{r['cache_kb_simulated']:>8.0f}K "
            f"{r['cache_kb_theoretical']:>8.0f}K "
            f"{r.get('cache_bytes_ratio_theoretical', 0)*100:>6.1f}%"
        )

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(n_trials=args.n_trials)
