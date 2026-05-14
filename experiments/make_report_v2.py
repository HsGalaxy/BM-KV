"""Aggregate v2 experimental results into tables and plots."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: F401 sets sys.path
from common import RESULTS_DIR

POLICY_ORDER = ["Full", "Recent", "AttnOnly", "FullINT8", "BM-KV-v2"]
POLICY_COLORS = {
    "Full": "#444444",
    "Recent": "#d62728",
    "AttnOnly": "#ff7f0e",
    "FullINT8": "#2ca02c",
    "BM-KV-v2": "#1f77b4",
}
POLICY_MARKERS = {
    "Full": "*",
    "Recent": "o",
    "AttnOnly": "s",
    "FullINT8": "^",
    "BM-KV-v2": "D",
}


def load(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"WARNING: missing {path}")
        return None
    return json.loads(path.read_text())


def plot_ppl(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by = defaultdict(list)
    for r in data["results"]:
        by[r["policy"]].append((r["memory_ratio_target"], r["mean_ppl"]))
    for p in POLICY_ORDER:
        rows = sorted(by.get(p, []))
        if not rows:
            continue
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        if p == "Full":
            ax.axhline(ys[0], color=POLICY_COLORS[p], linestyle="--",
                       alpha=0.6, label=f"Full Cache ({ys[0]:.1f})")
        else:
            ax.plot(xs, ys, marker=POLICY_MARKERS[p],
                    color=POLICY_COLORS[p], label=p, linewidth=2)
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"WikiText PPL on {data['config']['model']}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_needle(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by = defaultdict(list)
    for r in data["results"]:
        by[r["policy"]].append((r["memory_ratio_target"], r["accuracy"]))
    for p in POLICY_ORDER:
        rows = sorted(by.get(p, []))
        if not rows:
            continue
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        if p == "Full":
            ax.axhline(ys[0], color=POLICY_COLORS[p], linestyle="--",
                       alpha=0.6, label=f"Full Cache ({ys[0]:.2f})")
        else:
            ax.plot(xs, ys, marker=POLICY_MARKERS[p],
                    color=POLICY_COLORS[p], label=p, linewidth=2)
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Needle Retrieval Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Needle-in-a-Haystack on {data['config']['model']}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tpot = defaultdict(list)
    tps = defaultdict(list)
    for r in data["results"]:
        tpot[r["policy"]].append((r["memory_ratio_target"], r["tpot_ms"]))
        tps[r["policy"]].append((r["memory_ratio_target"], r["tokens_per_s"]))

    palette = {
        "Full-static": "#444444",
        "Recent-static": "#d62728",
        "AttnOnly-static": "#ff7f0e",
        "FullINT8-static": "#2ca02c",
        "BM-KV-v2-static": "#1f77b4",
        "BM-KV-v2-lazy16": "#9467bd",
    }
    markers = {
        "Full-static": "*",
        "Recent-static": "o",
        "AttnOnly-static": "s",
        "FullINT8-static": "^",
        "BM-KV-v2-static": "D",
        "BM-KV-v2-lazy16": "P",
    }
    for p in tpot:
        rows = sorted(tpot[p])
        xs = [x for x, _ in rows]; ys = [y for _, y in rows]
        axes[0].plot(xs, ys, marker=markers.get(p, "o"),
                     color=palette.get(p, "#888"), label=p, linewidth=2)
    for p in tps:
        rows = sorted(tps[p])
        xs = [x for x, _ in rows]; ys = [y for _, y in rows]
        axes[1].plot(xs, ys, marker=markers.get(p, "o"),
                     color=palette.get(p, "#888"), label=p, linewidth=2)
    axes[0].set_xlabel("Memory Ratio")
    axes[0].set_ylabel("TPOT (ms / token)")
    axes[0].set_title("Time Per Output Token")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Memory Ratio")
    axes[1].set_ylabel("Tokens / sec")
    axes[1].set_title("Decode Throughput")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation(data, out_path):
    variants = [r["variant"] for r in data["ppl_ablation"] if r["ratio"] == 0.50]
    ppl_by = defaultdict(dict)
    for r in data["ppl_ablation"]:
        ppl_by[r["variant"]][r["ratio"]] = r["mean_ppl"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(variants)))
    w = 0.35
    p50 = [ppl_by[v].get(0.50, 0) for v in variants]
    p25 = [ppl_by[v].get(0.25, 0) for v in variants]
    ax.bar([i - w/2 for i in x], p50, w, label="ratio=0.50", color="#1f77b4")
    ax.bar([i + w/2 for i in x], p25, w, label="ratio=0.25", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20)
    ax.set_ylabel("Perplexity (lower better)")
    ax.set_title("BM-KV-v2 Ablation (PPL)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(out_path):
    ppl = load("exp1_ppl_v2.json")
    needle = load("exp2_needle_v2.json")
    runtime = load("exp3_runtime_v2.json")
    ablation = load("exp4_ablation_v2.json")

    lines = ["# BM-KV-v2 Experimental Results\n",
             "Model: Qwen2.5-0.5B-Instruct (FP16), Hardware: CUDA\n",
             "Algorithm: block-level (g=16) + dual thresholds (θ_drop/θ_fp16 from quantiles) + lazy update.\n"]

    if ppl:
        lines.append("\n## 1. WikiText-2 Perplexity\n")
        lines.append(
            f"Configuration: prompt={ppl['config']['prompt_len']}, "
            f"target={ppl['config']['target_len']}, "
            f"chunks={ppl['config']['n_chunks']}.\n"
        )
        lines.append("| Policy | Mem ratio | Actual | PPL | Δ vs Full | Δ% |")
        lines.append("|--------|-----------|--------|-----|-----------|----|")
        for r in ppl["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['memory_ratio_actual']:.3f} | {r['mean_ppl']:.2f} | "
                f"{r.get('ppl_increment', 0):+.2f} | "
                f"{r.get('ppl_increment_pct', 0):+.1f}% |"
            )
    if needle:
        lines.append("\n## 2. Needle-in-a-Haystack\n")
        lines.append(
            f"Configuration: context={needle['config']['context_tokens']}, "
            f"{needle['config']['n_fillers']} fillers x {needle['config']['n_needles']} "
            f"needles x {len(needle['config']['depths'])} depths = "
            f"{needle['config']['n_fillers'] * needle['config']['n_needles'] * len(needle['config']['depths'])} "
            f"trials/cell.\n"
        )
        lines.append("| Policy | Mem ratio | Accuracy | Correct |")
        lines.append("|--------|-----------|----------|---------|")
        for r in needle["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['accuracy']:.3f} | {r['n_correct']}/{r['n_trials']} |"
            )

    if runtime:
        lines.append("\n## 3. Runtime: TTFT, TPOT, tokens/s\n")
        lines.append(
            f"Configuration: prompt={runtime['config']['prompt_len']}, "
            f"decode_steps={runtime['config']['decode_steps']}, "
            f"trials={runtime['config']['n_trials']}.\n"
        )
        lines.append("| Policy | Mem ratio | TTFT (ms) | TPOT (ms) | tokens/s | Rebalances |")
        lines.append("|--------|-----------|-----------|-----------|----------|-----------|")
        for r in runtime["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['ttft_ms']:.1f} | {r['tpot_ms']:.2f} ± {r.get('tpot_ms_std', 0):.2f} | "
                f"{r['tokens_per_s']:.2f} | {r.get('n_rebalances', 0):.1f} |"
            )

    if ablation:
        lines.append("\n## 4. Ablation Study\n")
        lines.append("Variants: A=full BM-KV-v2, B=no θ_drop, C=token-level (block=1), "
                     "D=no INT8, E=v1 BM-KV.\n")
        lines.append("\n### PPL\n")
        lines.append("| Variant | ratio=0.50 | ratio=0.25 |")
        lines.append("|---------|-----------|------------|")
        by = defaultdict(dict)
        for r in ablation["ppl_ablation"]:
            by[r["variant"]][r["ratio"]] = r["mean_ppl"]
        for v in by:
            lines.append(
                f"| {v} | {by[v].get(0.50, float('nan')):.2f} | "
                f"{by[v].get(0.25, float('nan')):.2f} |"
            )
        lines.append("\n### Lazy Update Comparison\n")
        lines.append("| Mode | mean ms/step | mean rebalances |")
        lines.append("|------|--------------|-----------------|")
        for r in ablation["lazy_summary"]:
            lines.append(
                f"| {r['mode']} | {r['mean_step_ms']:.2f} | "
                f"{r['mean_rebalances']:.1f} |"
            )

    lines.append("\n## 5. Plots\n")
    for f in ["fig_ppl_v2.png", "fig_needle_v2.png", "fig_runtime_v2.png",
              "fig_ablation_v2.png"]:
        lines.append(f"- `{f}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ppl = load("exp1_ppl_v2.json")
    needle = load("exp2_needle_v2.json")
    runtime = load("exp3_runtime_v2.json")
    ablation = load("exp4_ablation_v2.json")
    if ppl:
        plot_ppl(ppl, RESULTS_DIR / "fig_ppl_v2.png")
    if needle:
        plot_needle(needle, RESULTS_DIR / "fig_needle_v2.png")
    if runtime:
        plot_runtime(runtime, RESULTS_DIR / "fig_runtime_v2.png")
    if ablation:
        plot_ablation(ablation, RESULTS_DIR / "fig_ablation_v2.png")
    write_markdown(RESULTS_DIR / "REPORT_v2.md")
    print("Saved REPORT_v2.md and v2 figures")


if __name__ == "__main__":
    main()
