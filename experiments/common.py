"""Shared experiment utilities."""
from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import torch

# Quiet logs/warnings before importing transformers.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure ``import bm_kv`` works regardless of CWD.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gpt2(model_name: str = "gpt2", dtype: torch.dtype = torch.float16):
    """Load GPT-2 with eager attention (we need real attention weights).

    By default the model is loaded in FP16 so the FP16-vs-INT8 cache memory
    accounting in our experiments is consistent with the paper's analysis.
    Pass ``dtype=torch.float32`` to compare against an FP32 baseline.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=dtype,
    )
    model.to(get_device()).eval()
    return model, tok


def load_qwen(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dtype: torch.dtype = torch.float16,
):
    """Load a Qwen2.5 chat model with eager attention.

    Qwen2.5 uses GQA (2 KV heads, 14 query heads in 0.5B) and RoPE, supports
    32k context and is the v2 experimental target."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=dtype,
    )
    model.to(get_device()).eval()
    return model, tok


def load_tinyllama(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype: torch.dtype = torch.float16,
):
    """Load TinyLlama-1.1B-Chat with eager attention.

    Llama-2 architecture clone: 22 layers, 32 query heads, 4 KV heads
    (GQA 8:1), RoPE, 2048 max position. Bigger than Qwen2.5-0.5B but with
    shorter context.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=dtype,
    )
    model.to(get_device()).eval()
    return model, tok


def load_causal_lm(name: str):
    """Dispatch a friendly model alias to its loader."""
    name = name.lower()
    if name in ("gpt2", "gpt-2"):
        return load_gpt2()
    if name in ("qwen", "qwen2.5", "qwen2.5-0.5b"):
        return load_qwen()
    if name in ("tinyllama", "tl", "tinyllama1.1b"):
        return load_tinyllama()
    raise ValueError(f"Unknown model alias: {name}")


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_wikitext_chunks(
    tokenizer,
    chunk_token_count: int,
    n_chunks: int,
    split: str = "test",
) -> list[torch.Tensor]:
    """Return ``n_chunks`` non-overlapping token chunks from WikiText-2.

    The text is cached to ``data/wikitext2_<split>.txt`` on first load so we do
    not have to re-enter the HuggingFace datasets pipeline (and its hub probe)
    on every run.
    """
    cache_path = DATA_DIR / f"wikitext2_{split}.txt"
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
        cache_path.write_text(text, encoding="utf-8")

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    chunks: list[torch.Tensor] = []
    stride = chunk_token_count
    for start in range(0, ids.shape[0] - chunk_token_count, stride):
        if len(chunks) >= n_chunks:
            break
        chunks.append(ids[start : start + chunk_token_count].unsqueeze(0))
    return chunks


def compute_budget(seq_len: int, memory_ratio: float) -> int:
    """Memory ratio 1.0 = full FP16 cache."""
    return int(memory_ratio * 2 * seq_len)
