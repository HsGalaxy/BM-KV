# BM-KV: Budget-Aware Mixed-Precision KV Cache Compression

BM-KV 是一个面向长上下文推理的 KV Cache 压缩原型。它把缓存管理从简单的“保留或删除”扩展为一个带预算约束的多动作分配问题：重要内容用 FP16 保留，中等重要内容用 INT8 量化保留，低价值内容直接淘汰。

这个仓库包含 BM-KV 的核心实现、对比策略、实验脚本和已生成的实验结果。实验主要在 GPT-2 与 Qwen2.5-0.5B-Instruct 上完成，评估指标包括困惑度、长距离关键词检索准确率、TTFT、TPOT、tokens/s 和消融结果。

## Motivation

自回归语言模型在解码阶段会缓存每个历史 token 的 Key 和 Value，避免重复计算完整上下文。KV Cache 能提升生成效率，但它的显存占用会随序列长度、层数、注意力头数、数据精度和 batch size 线性增长。

在长文档问答、代码补全、多轮对话和长篇生成中，KV Cache 往往比模型参数本身更容易成为瓶颈。直接保留完整缓存成本高；只保留最近窗口又容易丢失开头指令、函数签名、全局约束和长距离事实。

BM-KV 的思路是：历史 token 的重要性不同，缓存动作也不应该只有二元选择。高价值位置完整保留，中等价值位置低精度保留，低价值位置删除，可以在相同预算下覆盖更多有用上下文。

## Core Idea

BM-KV 为每个历史 token 或缓存块计算重要性分数：

```text
Score = alpha * attention + beta * recency + gamma * prefix
```

其中：

- `attention` 表示历史位置在过去注意力中被关注的程度。
- `recency` 表示距离当前生成位置的远近，用于保护局部连贯性。
- `prefix` 表示开头任务约束、特殊标记或固定前缀区域的保护权重。

根据分数和缓存预算，BM-KV 为缓存位置分配三类动作：

- `FP16`：完整保留，适合最高价值的 block。
- `INT8`：量化保留，适合中等价值但仍可能有用的 block。
- `DROP`：淘汰，适合低价值历史状态。

v2 版本把 token 级分配改为 block 级分配。连续 token 被聚合成固定大小 block，每个 block 使用统一动作。这样更接近真实推理系统中的分页式 KV Cache 管理，也减少逐 token 混合精度带来的碎片化问题。

v2 还加入了两个约束：

- `theta_drop` / `theta_fp16`：用阈值决定哪些 block 必须丢弃、哪些 block 优先完整保留。
- lazy update：不在每个 decode step 重新排序缓存，而是在固定间隔或查询漂移较大时更新动作映射。

## What Is Implemented

代码里实现了两套 BM-KV：

- `BM-KV v1`：token-level 策略，在 GPT-2 上测试。
- `BM-KV v2`：block-level 策略，在 Qwen2.5-0.5B-Instruct 上测试，包含阈值和 lazy update。

同时实现了这些 baseline：

- `Full`：完整 FP16 KV Cache。
- `Recent`：只保留最近窗口。
- `AttnOnly`：只按注意力贡献保留高分 token。
- `FullINT8`：尽可能把缓存转为 INT8。

## Results

### GPT-2 / BM-KV v1

GPT-2 实验中，Recent-only 在低预算下明显退化，长距离关键词检索几乎失效。BM-KV 在 50% 和 25% 预算下比 Recent、AttnOnly 和低预算 FullINT8 更稳定。

| Method | 50% PPL | 50% Needle | 25% PPL | 25% Needle |
|---|---:|---:|---:|---:|
| Full | 30.48 | 0.928 | - | - |
| Recent | 1212.93 | 0.000 | 4218.82 | 0.000 |
| AttnOnly | 47.46 | 0.080 | 64.15 | 0.064 |
| FullINT8 | 30.50 | 0.928 | 1206.85 | 0.000 |
| BM-KV | 38.08 | 0.568 | 60.34 | 0.248 |

### Qwen2.5-0.5B / BM-KV v2

Qwen2.5-0.5B-Instruct 实验中，BM-KV-v2 在 50% 预算下把 PPL 从 14.68 提高到 15.87，在 25% 预算下提高到 17.62；相比 Recent-only，质量退化小得多。

| Method | 50% PPL | 50% Needle | 25% PPL | 25% Needle |
|---|---:|---:|---:|---:|
| Full | 14.68 | 0.987 | - | - |
| Recent | 973.18 | 0.400 | 882.85 | 0.200 |
| AttnOnly | 18.14 | 0.120 | 19.61 | 0.027 |
| FullINT8 | 14.73 | 0.987 | 962.00 | 0.400 |
| BM-KV-v2 | 15.87 | 0.600 | 17.62 | 0.387 |

运行时结果表明，当前实现主要用于验证策略行为。静态 BM-KV-v2 的 TPOT 与 Full 接近；lazy update 版本由于依赖 HuggingFace 的 `output_attentions=True` 重新统计注意力，会引入额外开销。要获得真实系统加速，还需要对接底层 KV layout 或自定义 attention kernel。

## Takeaways

- KV Cache 压缩不只是“保留多少”的问题，也涉及“用什么精度保留”。
- Recent-only 容易丢失 attention sink、开头指令和长距离事实。
- INT8 在中高预算下很有效，但低预算时单纯量化仍可能失败。
- BM-KV 在低于 50% 的预算区间更有价值，因为它能用 INT8 保存中等重要信息，而不是直接删除。
- block-level 管理和 lazy update 是让算法接近真实推理系统时必须考虑的工程约束。

## Repository Layout

```text
BM-KV/
├── bm_kv/
│   ├── quantize.py        # INT8 symmetric quantization
│   ├── importance.py      # attention / recency / prefix scoring
│   ├── policies.py        # cache allocation policies
│   ├── blocks.py          # block-level aggregation for v2
│   ├── compress.py        # apply actions to past_key_values
│   └── runner.py          # prefill -> compress -> bridge -> decode
├── experiments/
│   ├── exp1_ppl*.py       # WikiText-2 perplexity
│   ├── exp2_needle*.py    # Needle-in-a-Haystack
│   ├── exp3_runtime*.py   # TTFT / TPOT / tokens/s
│   ├── exp4_ablation*.py  # ablation experiments
│   └── make_report*.py    # report and plot generation
├── data/
│   └── wikitext2_test.txt
├── results/
│   ├── REPORT.md
│   ├── REPORT_v2.md
│   └── fig_*.png
└── requirements.txt
```

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run smoke tests:

```bash
python experiments/smoke_test.py
python experiments/smoke_test_v2.py
```

Run GPT-2 experiments:

```bash
python experiments/exp1_ppl.py --n-chunks 50
python experiments/exp2_needle.py --n-fillers 5 --context-tokens 768
python experiments/exp3_runtime.py --n-trials 8
python experiments/exp4_ablation.py
python experiments/make_report.py
```

Run Qwen2.5 experiments:

```bash
python experiments/exp1_ppl_v2.py --n-chunks 15
python experiments/exp2_needle_v2.py --n-fillers 3 --context-tokens 1024
python experiments/exp3_runtime_v2.py --n-trials 5
python experiments/exp4_ablation_v2.py
python experiments/make_report_v2.py
```

The scripts may download models and datasets from HuggingFace. CUDA is recommended for the runtime experiments.

## Limitations

- INT8 KV is simulated through quantize-dequantize in the prototype; it is not a production INT8 attention kernel.
- Lazy update currently needs attention outputs from HuggingFace, so its measured latency is higher than the ideal system design.
- Block-level allocation improves memory-layout realism but loses some token-level precision.
- The scoring weights, block size and thresholds are manually configured rather than learned automatically.
