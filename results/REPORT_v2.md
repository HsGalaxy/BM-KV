# BM-KV-v2 Experimental Results

Model: Qwen2.5-0.5B-Instruct (FP16), Hardware: CUDA

Algorithm: block-level (g=16) + dual thresholds (θ_drop/θ_fp16 from quantiles) + lazy update.


## 1. WikiText-2 Perplexity

Configuration: prompt=512, target=128, chunks=15.

| Policy | Mem ratio | Actual | PPL | Δ vs Full | Δ% |
|--------|-----------|--------|-----|-----------|----|
| Full | 1.00 | 1.000 | 14.68 | +0.00 | +0.0% |
| Recent | 0.75 | 0.752 | 1058.59 | +1043.91 | +7110.9% |
| Recent | 0.50 | 0.502 | 973.18 | +958.50 | +6529.1% |
| Recent | 0.35 | 0.352 | 993.44 | +978.75 | +6667.1% |
| Recent | 0.25 | 0.252 | 882.85 | +868.17 | +5913.8% |
| AttnOnly | 0.75 | 0.752 | 16.34 | +1.66 | +11.3% |
| AttnOnly | 0.50 | 0.502 | 18.14 | +3.46 | +23.6% |
| AttnOnly | 0.35 | 0.352 | 19.05 | +4.37 | +29.8% |
| AttnOnly | 0.25 | 0.252 | 19.61 | +4.93 | +33.6% |
| FullINT8 | 0.75 | 0.501 | 14.73 | +0.05 | +0.3% |
| FullINT8 | 0.50 | 0.501 | 14.73 | +0.05 | +0.3% |
| FullINT8 | 0.35 | 0.352 | 1042.21 | +1027.53 | +6999.4% |
| FullINT8 | 0.25 | 0.252 | 962.00 | +947.32 | +6453.0% |
| BM-KV-v2 | 0.75 | 0.578 | 15.50 | +0.82 | +5.6% |
| BM-KV-v2 | 0.50 | 0.500 | 15.87 | +1.19 | +8.1% |
| BM-KV-v2 | 0.35 | 0.344 | 16.97 | +2.29 | +15.6% |
| BM-KV-v2 | 0.25 | 0.250 | 17.62 | +2.94 | +20.0% |

## 2. Needle-in-a-Haystack

Configuration: context=1024, 3 fillers x 5 needles x 5 depths = 75 trials/cell.

| Policy | Mem ratio | Accuracy | Correct |
|--------|-----------|----------|---------|
| Full | 1.00 | 0.987 | 74/75 |
| Recent | 0.75 | 0.787 | 59/75 |
| Recent | 0.50 | 0.400 | 30/75 |
| Recent | 0.35 | 0.400 | 30/75 |
| Recent | 0.25 | 0.200 | 15/75 |
| AttnOnly | 0.75 | 0.307 | 23/75 |
| AttnOnly | 0.50 | 0.120 | 9/75 |
| AttnOnly | 0.35 | 0.093 | 7/75 |
| AttnOnly | 0.25 | 0.027 | 2/75 |
| FullINT8 | 0.75 | 0.987 | 74/75 |
| FullINT8 | 0.50 | 0.987 | 74/75 |
| FullINT8 | 0.35 | 0.760 | 57/75 |
| FullINT8 | 0.25 | 0.400 | 30/75 |
| BM-KV-v2 | 0.75 | 0.787 | 59/75 |
| BM-KV-v2 | 0.50 | 0.600 | 45/75 |
| BM-KV-v2 | 0.35 | 0.387 | 29/75 |
| BM-KV-v2 | 0.25 | 0.387 | 29/75 |

## 3. Runtime: TTFT, TPOT, tokens/s

Configuration: prompt=1024, decode_steps=64, trials=5.

| Policy | Mem ratio | TTFT (ms) | TPOT (ms) | tokens/s | Rebalances |
|--------|-----------|-----------|-----------|----------|-----------|
| Full-static | 1.00 | 162.2 | 46.90 ± 5.91 | 21.58 | 0.0 |
| Recent-static | 0.50 | 146.9 | 48.92 ± 7.44 | 20.81 | 0.0 |
| Recent-static | 0.25 | 163.0 | 48.12 ± 4.45 | 20.93 | 0.0 |
| AttnOnly-static | 0.50 | 147.9 | 47.87 ± 5.58 | 21.11 | 0.0 |
| AttnOnly-static | 0.25 | 153.6 | 50.11 ± 7.37 | 20.29 | 0.0 |
| FullINT8-static | 0.50 | 179.7 | 50.89 ± 5.11 | 19.81 | 0.0 |
| FullINT8-static | 0.25 | 175.8 | 49.71 ± 2.55 | 20.16 | 0.0 |
| BM-KV-v2-static | 0.50 | 191.4 | 47.52 ± 4.90 | 21.22 | 0.0 |
| BM-KV-v2-static | 0.25 | 197.9 | 49.55 ± 5.71 | 20.40 | 0.0 |
| BM-KV-v2-lazy16 | 0.50 | 223.8 | 58.28 ± 7.38 | 17.36 | 9.4 |
| BM-KV-v2-lazy16 | 0.25 | 224.5 | 58.81 ± 3.16 | 17.04 | 7.0 |

## 4. Ablation Study

Variants: A=full BM-KV-v2, B=no θ_drop, C=token-level (block=1), D=no INT8, E=v1 BM-KV.


### PPL

| Variant | ratio=0.50 | ratio=0.25 |
|---------|-----------|------------|
| A_full_v2 | 16.51 | 18.76 |
| B_no_threshold | 16.51 | 18.76 |
| C_token_level | 16.40 | 18.61 |
| D_no_int8 | 16.73 | 18.83 |
| E_v1_token_kv | 16.54 | 18.44 |

### Lazy Update Comparison

| Mode | mean ms/step | mean rebalances |
|------|--------------|-----------------|
| static | 46.97 | 0.0 |
| lazy_delta16 | 50.85 | 6.6 |
| lazy_delta8 | 53.15 | 12.6 |

## 5. Plots

- `fig_ppl_v2.png`
- `fig_needle_v2.png`
- `fig_runtime_v2.png`
- `fig_ablation_v2.png`