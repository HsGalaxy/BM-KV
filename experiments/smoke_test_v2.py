"""Smoke test for v2 features: Qwen2.5 + block-level BM-KV + lazy update."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_qwen():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()
    return m, tok


def main():
    from bm_kv import POLICIES_V2, CompressedRunner
    from bm_kv.runner import CompressionConfig

    print("Loading Qwen2.5-0.5B-Instruct...")
    model, tok = load_qwen()

    prompt = (
        "Once upon a time, in a faraway kingdom, there lived a wise old wizard "
        "named Eldrin who guarded a single secret: the seven crystals of "
        "Vesperhall were hidden beneath the ancient oak that stood by the well. "
        "Years passed and many adventurers came searching for them, but none "
        "found the right tree. One day, a young apprentice arrived at the "
        "wizard's cottage and asked where the crystals were. Eldrin smiled, "
        "took a sip of tea, and after a long pause, he answered:"
    )
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    print(f"Prompt length: {ids.shape[1]} tokens")

    seq_len = ids.shape[1]
    full_budget = 2 * seq_len
    half_budget = full_budget // 2

    runner = CompressedRunner(model, tok, CompressionConfig())

    print("\n=== Static compression (v2 block-level BM-KV) ===")
    for name, pol in POLICIES_V2.items():
        budget = full_budget if name == "Full" else half_budget
        out_ids, stats = runner.generate(ids, pol, budget=budget, max_new_tokens=24)
        text = tok.decode(out_ids[0], skip_special_tokens=True)
        print(f"[{name:<10}] mem={stats.memory_ratio:.2f} "
              f"FP16/INT8/DROP={stats.n_fp16}/{stats.n_int8}/{stats.n_drop}")
        print(f"  -> {text!r}")

    print("\n=== Lazy compression with BM-KV-v2 (delta=8) ===")
    out_ids, info = runner.generate_lazy(
        ids, POLICIES_V2["BM-KV-v2"], budget=half_budget,
        max_new_tokens=32, delta=8, drift_threshold=0.5,
    )
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    print(f"Initial: FP16/INT8/DROP="
          f"{info['initial_stats'].n_fp16}/"
          f"{info['initial_stats'].n_int8}/"
          f"{info['initial_stats'].n_drop} "
          f"mem={info['initial_stats'].memory_ratio:.2f}")
    print(f"Rebalances triggered: {len(info['rebalances'])}")
    for r in info["rebalances"]:
        print(f"  @step {r['step']} trigger={r['trigger']} "
              f"{r['old_cache_len']} -> {r['new_cache_len']} "
              f"FP16/INT8/DROP="
              f"{r['actions_summary']['FP16']}/"
              f"{r['actions_summary']['INT8']}/"
              f"{r['actions_summary']['DROP']}")
    avg_ms = sum(info['per_step_ms']) / len(info['per_step_ms'])
    print(f"Decode {len(info['per_step_ms'])} steps, mean {avg_ms:.2f} ms/step")
    print(f"  -> {text!r}")


if __name__ == "__main__":
    main()
