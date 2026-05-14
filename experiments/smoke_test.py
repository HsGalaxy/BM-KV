"""Quick sanity check that compression + decoding produces sensible output."""
from __future__ import annotations

import os
import sys
import warnings

import torch

# Quiet some noisy warnings/logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore")

import logging  # noqa: E402
logging.getLogger("transformers").setLevel(logging.ERROR)

# Make ``import bm_kv`` work when run from the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402

from bm_kv import POLICIES, CompressedRunner  # noqa: E402
from bm_kv.runner import CompressionConfig  # noqa: E402


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading GPT-2 on {device}")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", attn_implementation="eager"
    ).to(device).eval()

    runner = CompressedRunner(model, tokenizer, CompressionConfig())

    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the English alphabet, "
        "which makes it a useful pangram for typography testing. "
        "However, today we are not interested in typography. "
        "Instead, we want to study how a language model behaves when "
        "we compress its key-value cache aggressively. "
        "Continue the story:"
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Prompt length: {ids.shape[1]} tokens")

    seq_len = ids.shape[1]
    full_budget = 2 * seq_len
    half_budget = full_budget // 2

    print(f"Full FP16 budget: {full_budget}, half budget: {half_budget}")
    print("-" * 60)

    for name, policy_fn in POLICIES.items():
        budget = full_budget if name == "Full" else half_budget
        out_ids, stats = runner.generate(
            ids, policy_fn=policy_fn, budget=budget, max_new_tokens=20
        )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(
            f"[{name:<10}] mem_ratio={stats.memory_ratio:.3f}  "
            f"FP16/INT8/DROP={stats.n_fp16}/{stats.n_int8}/{stats.n_drop}"
        )
        print(f"  -> {text!r}")


if __name__ == "__main__":
    main()
