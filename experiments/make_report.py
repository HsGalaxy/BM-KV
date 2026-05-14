"""Aggregate results from all experiments into tables and plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: F401 - sets sys.path
from common import RESULTS_DIR

POLICY_ORDER = ["Full", "Recent", "AttnOnly", "FullINT8", "BM-KV"]
POLICY_COLORS = {
    "Full": "#444444",
    "Recent": "#d62728",
    "AttnOnly": "#ff7f0e",
    "FullINT8": "#2ca02c",
    "BM-KV": "#1f77b4",
}
POLICY_MARKERS = {
    "Full": "*",
    "Recent": "o",
    "AttnOnly": "s",
    "FullINT8": "^",
    "BM-KV": "D",
}


def fmt_num(x, prec=2):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def load(name):
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"WARNING: missing {path}")
        return None
    return json.loads(path.read_text())


def make_ppl_plot(data, out_path):
    """PPL vs memory ratio plot. Recent's catastrophic values get clipped on a
    log y-axis so the rest stays readable."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by = {}
    for r in data["results"]:
        by.setdefault(r["policy"], []).append(
            (r["memory_ratio_target"], r["mean_ppl"])
        )
    for policy in POLICY_ORDER:
        if policy not in by:
            continue
        rows = sorted(by[policy])
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        if policy == "Full":
            # Single point — also draw as horizontal reference line.
            ax.axhline(ys[0], color=POLICY_COLORS[policy], linestyle="--",
                       alpha=0.6, label=f"Full Cache ({ys[0]:.1f})")
        else:
            ax.plot(xs, ys, marker=POLICY_MARKERS[policy],
                    color=POLICY_COLORS[policy], label=policy, linewidth=2)
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_yscale("log")
    ax.set_title("WikiText-2 PPL vs KV Cache Memory Ratio")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_needle_plot(data, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by = {}
    for r in data["results"]:
        by.setdefault(r["policy"], []).append(
            (r["memory_ratio_target"], r["accuracy"])
        )
    for policy in POLICY_ORDER:
        if policy not in by:
            continue
        rows = sorted(by[policy])
        xs = [x for x, _ in rows]
        ys = [y for _, y in rows]
        if policy == "Full":
            ax.axhline(ys[0], color=POLICY_COLORS[policy], linestyle="--",
                       alpha=0.6, label=f"Full Cache ({ys[0]:.2f})")
        else:
            ax.plot(xs, ys, marker=POLICY_MARKERS[policy],
                    color=POLICY_COLORS[policy], label=policy, linewidth=2)
    ax.set_xlabel("Memory Ratio")
    ax.set_ylabel("Needle Retrieval Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Needle-in-a-Haystack Accuracy vs Memory Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_runtime_plot(data, out_path):
    """Two-panel: ms/token (left) and theoretical KV bytes (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ms_by = {}
    bytes_by = {}
    for r in data["results"]:
        ms_by.setdefault(r["policy"], []).append(
            (r["memory_ratio_target"], r["ms_per_token"])
        )
        bytes_by.setdefault(r["policy"], []).append(
            (r["memory_ratio_target"], r["cache_kb_theoretical"])
        )
    for policy in POLICY_ORDER:
        if policy in ms_by:
            rows = sorted(ms_by[policy])
            xs = [x for x, _ in rows]; ys = [y for _, y in rows]
            axes[0].plot(xs, ys, marker=POLICY_MARKERS[policy],
                         color=POLICY_COLORS[policy], label=policy, linewidth=2)
        if policy in bytes_by:
            rows = sorted(bytes_by[policy])
            xs = [x for x, _ in rows]; ys = [y for _, y in rows]
            axes[1].plot(xs, ys, marker=POLICY_MARKERS[policy],
                         color=POLICY_COLORS[policy], label=policy, linewidth=2)
    axes[0].set_xlabel("Memory Ratio")
    axes[0].set_ylabel("Decode latency (ms / token)")
    axes[0].set_title("Per-Token Decoding Latency")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Memory Ratio")
    axes[1].set_ylabel("KV cache size (KB, theoretical)")
    axes[1].set_title("Theoretical KV Cache Size")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_ablation_plot(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    variants = [v for v in data["config"]["variants"]]

    def by_variant(records, key):
        return {v: {0.50: None, 0.25: None} for v in variants} | {
            r["variant"]: {**({k: v for k, v in by_variant(records, key).get(r["variant"], {}).items()}),
                           r["ratio"]: r[key]}
            for r in records
        }

    # Simpler: build with defaultdict.
    from collections import defaultdict
    ppl_by = defaultdict(dict)
    for r in data["ppl"]:
        ppl_by[r["variant"]][r["ratio"]] = r["mean_ppl"]
    needle_by = defaultdict(dict)
    for r in data["needle"]:
        needle_by[r["variant"]][r["ratio"]] = r["accuracy"]

    x = list(range(len(variants)))
    bar_w = 0.35

    ppl_50 = [ppl_by[v].get(0.50, 0) for v in variants]
    ppl_25 = [ppl_by[v].get(0.25, 0) for v in variants]
    axes[0].bar([i - bar_w/2 for i in x], ppl_50, bar_w, label="ratio=0.50",
                color="#1f77b4")
    axes[0].bar([i + bar_w/2 for i in x], ppl_25, bar_w, label="ratio=0.25",
                color="#ff7f0e")
    axes[0].set_xticks(x); axes[0].set_xticklabels(variants, rotation=20)
    axes[0].set_ylabel("Perplexity (lower better)")
    axes[0].set_title("Ablation: PPL")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend()

    n_50 = [needle_by[v].get(0.50, 0) for v in variants]
    n_25 = [needle_by[v].get(0.25, 0) for v in variants]
    axes[1].bar([i - bar_w/2 for i in x], n_50, bar_w, label="ratio=0.50",
                color="#1f77b4")
    axes[1].bar([i + bar_w/2 for i in x], n_25, bar_w, label="ratio=0.25",
                color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(variants, rotation=20)
    axes[1].set_ylabel("Needle Accuracy (higher better)")
    axes[1].set_title("Ablation: Needle Retrieval")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_report(out_path):
    """Aggregated markdown report with all tables."""
    ppl = load("exp1_ppl.json")
    needle = load("exp2_needle.json")
    runtime = load("exp3_runtime.json")
    ablation = load("exp4_ablation.json")

    lines: list[str] = []
    lines.append("# BM-KV Experimental Results\n")
    lines.append(f"Model: GPT-2 (124M, FP16)  |  Hardware: {runtime['config']['device'].upper() if runtime else 'unknown'}\n")

    if ppl:
        lines.append("\n## 1. WikiText-2 Perplexity\n")
        lines.append(
            f"Configuration: prompt={ppl['config']['prompt_len']} tokens, "
            f"target={ppl['config']['target_len']} tokens, "
            f"chunks={ppl['config']['n_chunks']}.\n"
        )
        lines.append("| Policy | Memory ratio | Actual mem | PPL | Δ vs Full | Δ% |")
        lines.append("|--------|--------------|------------|-----|-----------|----|")
        for r in ppl["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['memory_ratio_actual']:.3f} | {r['mean_ppl']:.2f} | "
                f"{r.get('ppl_increment', 0):+.2f} | "
                f"{r.get('ppl_increment_pct', 0):+.1f}% |"
            )

    if needle:
        lines.append("\n## 2. Needle-in-a-Haystack Retrieval\n")
        lines.append(
            f"Configuration: context={needle['config']['context_tokens']} tokens, "
            f"{needle['config']['n_filler_sources']} fillers x "
            f"{needle['config']['n_needles']} needles x "
            f"{len(needle['config']['depths'])} depths = "
            f"{needle['config']['n_filler_sources'] * needle['config']['n_needles'] * len(needle['config']['depths'])} "
            f"trials per (policy, ratio).\n"
        )
        lines.append("| Policy | Memory ratio | Accuracy | Correct |")
        lines.append("|--------|--------------|----------|---------|")
        for r in needle["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['accuracy']:.3f} | "
                f"{r['n_correct']}/{r['n_trials']} |"
            )

    if runtime:
        lines.append("\n## 3. Runtime and Memory\n")
        lines.append(
            f"Configuration: prompt={runtime['config']['prompt_len']} tokens, "
            f"{runtime['config']['decode_steps']} decoding steps, "
            f"{runtime['config']['n_trials']} trials per row, "
            f"device={runtime['config']['device']}.\n"
        )
        lines.append(
            "Theoretical KV bytes assume an FP16 (=2B) baseline with INT8 (=1B) "
            "tokens; actual bytes are what our FP16-simulated implementation uses, "
            "where the INT8 quantization is round-tripped to FP16.\n"
        )
        lines.append(
            "| Policy | Memory ratio | Prefill (ms) | Compress (ms) | "
            "Decode (ms/tok) | KV theoretical | KV ratio |"
        )
        lines.append("|--------|--------------|--------------|---------------|"
                     "------------------|-----------------|----------|")
        for r in runtime["results"]:
            lines.append(
                f"| {r['policy']} | {r['memory_ratio_target']:.2f} | "
                f"{r['prefill_s']*1000:.1f} | {r['compress_s']*1000:.1f} | "
                f"{r['ms_per_token']:.2f} ± {r['ms_per_token_std']:.2f} | "
                f"{r['cache_kb_theoretical']:.0f} KB | "
                f"{r.get('cache_bytes_ratio_theoretical', 0)*100:.1f}% |"
            )

    if ablation:
        lines.append("\n## 4. Ablation Study\n")
        lines.append(
            "Variants: A=full BM-KV, B=no prefix, C=no recency, D=no INT8 "
            "(FP16/DROP only), E=no attention.\n"
        )
        lines.append("### PPL\n")
        lines.append("| Variant | ratio=0.50 | ratio=0.25 |")
        lines.append("|---------|-----------|------------|")
        from collections import defaultdict
        ppl_by = defaultdict(dict)
        for r in ablation["ppl"]:
            ppl_by[r["variant"]][r["ratio"]] = r["mean_ppl"]
        for v in ablation["config"]["variants"]:
            lines.append(
                f"| {v} | {ppl_by[v].get(0.50, float('nan')):.2f} | "
                f"{ppl_by[v].get(0.25, float('nan')):.2f} |"
            )
        lines.append("\n### Needle Accuracy\n")
        lines.append("| Variant | ratio=0.50 | ratio=0.25 |")
        lines.append("|---------|-----------|------------|")
        n_by = defaultdict(dict)
        for r in ablation["needle"]:
            n_by[r["variant"]][r["ratio"]] = r["accuracy"]
        for v in ablation["config"]["variants"]:
            lines.append(
                f"| {v} | {n_by[v].get(0.50, 0):.3f} | "
                f"{n_by[v].get(0.25, 0):.3f} |"
            )

    lines.append("\n## 5. Plots\n")
    lines.append("- `fig_ppl.png`: WikiText PPL vs memory ratio (log scale)")
    lines.append("- `fig_needle.png`: Needle retrieval accuracy vs memory ratio")
    lines.append("- `fig_runtime.png`: Per-token decode latency and theoretical KV size")
    lines.append("- `fig_ablation.png`: Ablation bar chart for PPL and Needle\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ppl = load("exp1_ppl.json")
    needle = load("exp2_needle.json")
    runtime = load("exp3_runtime.json")
    ablation = load("exp4_ablation.json")

    if ppl:
        make_ppl_plot(ppl, RESULTS_DIR / "fig_ppl.png")
        print("Saved fig_ppl.png")
    if needle:
        make_needle_plot(needle, RESULTS_DIR / "fig_needle.png")
        print("Saved fig_needle.png")
    if runtime:
        make_runtime_plot(runtime, RESULTS_DIR / "fig_runtime.png")
        print("Saved fig_runtime.png")
    if ablation:
        make_ablation_plot(ablation, RESULTS_DIR / "fig_ablation.png")
        print("Saved fig_ablation.png")

    write_markdown_report(RESULTS_DIR / "REPORT.md")
    print("Saved REPORT.md")


if __name__ == "__main__":
    main()
