"""Cross-model comparison: GPT-2 (v1) vs Qwen2.5-0.5B (v2) vs TinyLlama-1.1B (tl).

Reads all six result files and produces:
- A unified markdown table per metric.
- Side-by-side plots so the model architecture effect is visible.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: F401
from common import RESULTS_DIR

MODELS = [
    ("GPT-2",    "exp1_ppl.json",     "exp2_needle.json",     "exp3_runtime.json",     "exp4_ablation.json",     "BM-KV"),
    ("Qwen2.5",  "exp1_ppl_v2.json",  "exp2_needle_v2.json",  "exp3_runtime_v2.json",  "exp4_ablation_v2.json",  "BM-KV-v2"),
    ("TinyLlama","exp1_ppl_tl.json",  "exp2_needle_tl.json",  "exp3_runtime_tl.json",  "exp4_ablation_tl.json",  "BM-KV-v2"),
]
BASELINES = ["Full", "Recent", "AttnOnly", "FullINT8"]


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_ppl_table():
    """Per-model, per-ratio PPL table for every policy."""
    cols = []
    for model_name, ppl_f, _, _, _, bm_name in MODELS:
        d = load(ppl_f)
        if d is None:
            continue
        by = defaultdict(dict)
        for r in d["results"]:
            policy = r["policy"]
            by[policy][r["memory_ratio_target"]] = r["mean_ppl"]
        cols.append((model_name, bm_name, by))
    return cols


def fmt_or_dash(x, prec=2):
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):  # NaN check
        return "—"
    return f"{x:.{prec}f}"


def write_cross_report():
    ppl_cols = build_ppl_table()
    lines = ["# Cross-Model Comparison\n",
             "Three models, identical algorithm (with the v2 refinements where indicated).\n",
             "All numbers come from `results/exp*.json` produced by the matching run scripts.\n"]

    # ----- PPL table -----
    lines.append("\n## WikiText Perplexity\n")
    ratios = [1.0, 0.75, 0.50, 0.35, 0.25]
    for policy in ["Full", "Recent", "AttnOnly", "FullINT8", "BM-KV/BM-KV-v2"]:
        lines.append(f"\n### Policy: {policy}\n")
        header = "| Memory ratio | " + " | ".join(name for name, _, _ in ppl_cols) + " |"
        sep = "|--------------|" + "|".join(["----"] * len(ppl_cols)) + "|"
        lines.append(header)
        lines.append(sep)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for _, bm_name, by in ppl_cols:
                lookup = bm_name if policy.startswith("BM-KV") else policy
                v = by.get(lookup, {}).get(ratio)
                row.append(fmt_or_dash(v, 2))
            lines.append("| " + " | ".join(row) + " |")

    # ----- Needle table -----
    needle_cols = []
    for model_name, _, needle_f, _, _, bm_name in MODELS:
        d = load(needle_f)
        if d is None:
            continue
        by = defaultdict(dict)
        for r in d["results"]:
            by[r["policy"]][r["memory_ratio_target"]] = r["accuracy"]
        needle_cols.append((model_name, bm_name, by))

    lines.append("\n## Needle Retrieval Accuracy\n")
    for policy in ["Full", "Recent", "AttnOnly", "FullINT8", "BM-KV/BM-KV-v2"]:
        lines.append(f"\n### Policy: {policy}\n")
        header = "| Memory ratio | " + " | ".join(name for name, _, _ in needle_cols) + " |"
        sep = "|--------------|" + "|".join(["----"] * len(needle_cols)) + "|"
        lines.append(header)
        lines.append(sep)
        for ratio in ratios:
            row = [f"{ratio:.2f}"]
            for _, bm_name, by in needle_cols:
                lookup = bm_name if policy.startswith("BM-KV") else policy
                v = by.get(lookup, {}).get(ratio)
                row.append(fmt_or_dash(v, 3) if v is not None else "—")
            lines.append("| " + " | ".join(row) + " |")

    # ----- Runtime table -----
    lines.append("\n## Decode Speed (TPOT and tokens/s)\n")
    lines.append("\n### TPOT (ms / token) at memory ratio 0.50\n")
    rows = []
    for model_name, _, _, runtime_f, _, bm_name in MODELS:
        d = load(runtime_f)
        if d is None:
            continue
        by = defaultdict(dict)
        for r in d["results"]:
            policy = r["policy"]
            by[policy][r["memory_ratio_target"]] = r.get("tpot_ms")
        rows.append((model_name, bm_name, by))
    header = "| Policy | " + " | ".join(name for name, _, _ in rows) + " |"
    lines.append(header)
    lines.append("|--------|" + "|".join(["----"] * len(rows)) + "|")
    for policy_name in ["Full", "Recent", "AttnOnly", "FullINT8"]:
        row = [policy_name]
        for _, _, by in rows:
            # GPT-2 column has no -static suffix.
            v = by.get(policy_name, {}).get(0.50)
            if v is None:
                v = by.get(f"{policy_name}-static", {}).get(0.50)
            row.append(fmt_or_dash(v, 2))
        lines.append("| " + " | ".join(row) + " |")
    # BM-KV row.
    row = ["BM-KV(-v2)"]
    for _, bm_name, by in rows:
        v = by.get(bm_name, {}).get(0.50) or by.get(f"{bm_name}-static", {}).get(0.50)
        row.append(fmt_or_dash(v, 2))
    lines.append("| " + " | ".join(row) + " |")
    # Lazy row (only v2 and tl have it).
    row = ["BM-KV-v2 lazy16"]
    for _, _, by in rows:
        v = by.get("BM-KV-v2-lazy16", {}).get(0.50)
        row.append(fmt_or_dash(v, 2))
    lines.append("| " + " | ".join(row) + " |")

    # ----- Ablation table -----
    lines.append("\n## Ablation (PPL @ ratio 0.50)\n")
    rows = []
    for model_name, _, _, _, ab_f, _ in MODELS:
        d = load(ab_f)
        if d is None:
            continue
        ppl = d.get("ppl_ablation", d.get("ppl"))
        if not ppl:
            continue
        by = {}
        for r in ppl:
            by[r["variant"]] = r["mean_ppl"]
        rows.append((model_name, by))
    if rows:
        all_variants = set()
        for _, by in rows:
            all_variants.update(by.keys())
        header = "| Variant | " + " | ".join(name for name, _ in rows) + " |"
        lines.append(header)
        lines.append("|---------|" + "|".join(["----"] * len(rows)) + "|")
        for v in sorted(all_variants):
            row = [v]
            for _, by in rows:
                row.append(fmt_or_dash(by.get(v), 2))
            lines.append("| " + " | ".join(row) + " |")

    out_path = RESULTS_DIR / "REPORT_cross_model.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def make_cross_plot():
    """Three-panel: PPL, Needle, TPOT across all models. BM-KV always blue."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    model_styles = {
        "GPT-2":    {"marker": "o", "linestyle": "-"},
        "Qwen2.5":  {"marker": "s", "linestyle": "-"},
        "TinyLlama":{"marker": "D", "linestyle": "-"},
    }

    # Panel 1: PPL (BM-KV variant only, log scale)
    for model_name, ppl_f, _, _, _, bm_name in MODELS:
        d = load(ppl_f)
        if d is None:
            continue
        ys = []
        xs = []
        for r in d["results"]:
            if r["policy"] in (bm_name,):
                xs.append(r["memory_ratio_target"])
                ys.append(r["mean_ppl"])
        rows = sorted(zip(xs, ys))
        if rows:
            xs2, ys2 = zip(*rows)
            axes[0].plot(xs2, ys2, label=f"{model_name} ({bm_name})",
                         linewidth=2, **model_styles.get(model_name, {}))
        # Full Cache reference (dashed).
        full = next((r for r in d["results"] if r["policy"] == "Full"), None)
        if full:
            axes[0].axhline(full["mean_ppl"], color="#888888", linestyle="--",
                            alpha=0.4)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Memory Ratio")
    axes[0].set_ylabel("PPL (log scale)")
    axes[0].set_title("BM-KV PPL across models")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    # Panel 2: Needle BM-KV vs Recent.
    for model_name, _, needle_f, _, _, bm_name in MODELS:
        d = load(needle_f)
        if d is None:
            continue
        xs = []; ys = []
        for r in d["results"]:
            if r["policy"] == bm_name:
                xs.append(r["memory_ratio_target"]); ys.append(r["accuracy"])
        rows = sorted(zip(xs, ys))
        if rows:
            xs2, ys2 = zip(*rows)
            axes[1].plot(xs2, ys2, label=f"{model_name} BM-KV",
                         linewidth=2, **model_styles.get(model_name, {}))
    axes[1].set_xlabel("Memory Ratio")
    axes[1].set_ylabel("Needle Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("BM-KV Needle accuracy across models")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel 3: TPOT.
    for model_name, _, _, runtime_f, _, bm_name in MODELS:
        d = load(runtime_f)
        if d is None:
            continue
        xs, ys = [], []
        for r in d["results"]:
            policy = r["policy"]
            if policy in (bm_name, f"{bm_name}-static"):
                xs.append(r["memory_ratio_target"])
                ys.append(r["tpot_ms"])
        rows = sorted(zip(xs, ys))
        if rows:
            xs2, ys2 = zip(*rows)
            axes[2].plot(xs2, ys2, label=f"{model_name}", linewidth=2,
                         **model_styles.get(model_name, {}))
    axes[2].set_xlabel("Memory Ratio")
    axes[2].set_ylabel("TPOT (ms / token)")
    axes[2].set_title("BM-KV TPOT across models")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cross_model.png", dpi=150)
    plt.close(fig)
    print("Saved fig_cross_model.png")


def main():
    write_cross_report()
    make_cross_plot()


if __name__ == "__main__":
    main()
