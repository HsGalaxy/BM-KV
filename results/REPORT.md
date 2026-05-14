# BM-KV Experimental Results

Model: GPT-2 (124M, FP16)  |  Hardware: CUDA


## 1. WikiText-2 Perplexity

Configuration: prompt=384 tokens, target=96 tokens, chunks=50.

| Policy | Memory ratio | Actual mem | PPL | Δ vs Full | Δ% |
|--------|--------------|------------|-----|-----------|----|
| Full | 1.00 | 1.000 | 30.48 | +0.00 | +0.0% |
| Recent | 0.75 | 0.753 | 162.56 | +132.08 | +433.3% |
| Recent | 0.50 | 0.503 | 1212.93 | +1182.45 | +3879.3% |
| Recent | 0.35 | 0.352 | 2481.07 | +2450.59 | +8039.8% |
| Recent | 0.25 | 0.253 | 4218.82 | +4188.34 | +13740.9% |
| AttnOnly | 0.75 | 0.753 | 39.53 | +9.05 | +29.7% |
| AttnOnly | 0.50 | 0.503 | 47.46 | +16.98 | +55.7% |
| AttnOnly | 0.35 | 0.352 | 55.46 | +24.98 | +82.0% |
| AttnOnly | 0.25 | 0.253 | 64.15 | +33.67 | +110.5% |
| FullINT8 | 0.75 | 0.501 | 30.50 | +0.02 | +0.1% |
| FullINT8 | 0.50 | 0.501 | 30.50 | +0.02 | +0.1% |
| FullINT8 | 0.35 | 0.352 | 217.31 | +186.83 | +612.9% |
| FullINT8 | 0.25 | 0.253 | 1206.85 | +1176.36 | +3859.4% |
| BM-KV | 0.75 | 0.753 | 30.51 | +0.03 | +0.1% |
| BM-KV | 0.50 | 0.503 | 38.08 | +7.60 | +24.9% |
| BM-KV | 0.35 | 0.352 | 51.20 | +20.72 | +68.0% |
| BM-KV | 0.25 | 0.253 | 60.34 | +29.86 | +98.0% |

## 2. Needle-in-a-Haystack Retrieval

Configuration: context=768 tokens, 5 fillers x 5 needles x 5 depths = 125 trials per (policy, ratio).

| Policy | Memory ratio | Accuracy | Correct |
|--------|--------------|----------|---------|
| Full | 1.00 | 0.928 | 116/125 |
| Recent | 0.75 | 0.000 | 0/125 |
| Recent | 0.50 | 0.000 | 0/125 |
| Recent | 0.35 | 0.000 | 0/125 |
| Recent | 0.25 | 0.000 | 0/125 |
| AttnOnly | 0.75 | 0.272 | 34/125 |
| AttnOnly | 0.50 | 0.080 | 10/125 |
| AttnOnly | 0.35 | 0.080 | 10/125 |
| AttnOnly | 0.25 | 0.064 | 8/125 |
| FullINT8 | 0.75 | 0.928 | 116/125 |
| FullINT8 | 0.50 | 0.928 | 116/125 |
| FullINT8 | 0.35 | 0.000 | 0/125 |
| FullINT8 | 0.25 | 0.000 | 0/125 |
| BM-KV | 0.75 | 0.872 | 109/125 |
| BM-KV | 0.50 | 0.568 | 71/125 |
| BM-KV | 0.35 | 0.360 | 45/125 |
| BM-KV | 0.25 | 0.248 | 31/125 |

## 3. Runtime and Memory

Configuration: prompt=512 tokens, 32 decoding steps, 8 trials per row, device=cuda.

Theoretical KV bytes assume an FP16 (=2B) baseline with INT8 (=1B) tokens; actual bytes are what our FP16-simulated implementation uses, where the INT8 quantization is round-tripped to FP16.

| Policy | Memory ratio | Prefill (ms) | Compress (ms) | Decode (ms/tok) | KV theoretical | KV ratio |
|--------|--------------|--------------|---------------|------------------|-----------------|----------|
| Full | 1.00 | 15.2 | 15.3 | 10.50 ± 0.42 | 18432 KB | 100.0% |
| Recent | 0.50 | 14.3 | 13.9 | 10.34 ± 0.32 | 9252 KB | 50.2% |
| Recent | 0.25 | 14.3 | 16.0 | 10.90 ± 1.08 | 4644 KB | 25.2% |
| AttnOnly | 0.50 | 15.2 | 16.1 | 11.05 ± 1.33 | 9252 KB | 50.2% |
| AttnOnly | 0.25 | 15.2 | 15.3 | 10.78 ± 0.79 | 4644 KB | 25.2% |
| FullINT8 | 0.50 | 15.2 | 28.2 | 10.44 ± 0.35 | 9234 KB | 50.1% |
| FullINT8 | 0.25 | 15.2 | 24.9 | 10.59 ± 0.64 | 4644 KB | 25.2% |
| BM-KV | 0.50 | 15.2 | 25.7 | 10.23 ± 0.60 | 9252 KB | 50.2% |
| BM-KV | 0.25 | 15.2 | 25.3 | 11.24 ± 1.51 | 4644 KB | 25.2% |

## 4. Ablation Study

Variants: A=full BM-KV, B=no prefix, C=no recency, D=no INT8 (FP16/DROP only), E=no attention.

### PPL

| Variant | ratio=0.50 | ratio=0.25 |
|---------|-----------|------------|
| A_full | 37.97 | 61.37 |
| B_no_prefix | 37.55 | 65.38 |
| C_no_recency | 40.49 | 56.49 |
| D_no_int8 | 49.23 | 68.92 |
| E_no_attn | 37.94 | 61.31 |

### Needle Accuracy

| Variant | ratio=0.50 | ratio=0.25 |
|---------|-----------|------------|
| A_full | 0.547 | 0.253 |
| B_no_prefix | 0.547 | 0.213 |
| C_no_recency | 0.200 | 0.080 |
| D_no_int8 | 0.373 | 0.053 |
| E_no_attn | 0.533 | 0.253 |

## 5. Plots

- `fig_ppl.png`: WikiText PPL vs memory ratio (log scale)
- `fig_needle.png`: Needle retrieval accuracy vs memory ratio
- `fig_runtime.png`: Per-token decode latency and theoretical KV size
- `fig_ablation.png`: Ablation bar chart for PPL and Needle
