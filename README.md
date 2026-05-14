# BM-KV 实验代码与结果

实现并验证论文《基于预算分配的混合精度 KV 缓存压缩算法研究》提出的
BM-KV 算法，包括 **v1**（token 级 + 单一预算决策）和 **v2 修订版**（块级 +
绝对阈值 + 延迟更新）。

## 目录结构

```
algorithm_paper/
├── bm_kv/
│   ├── quantize.py        # INT8 对称量化
│   ├── importance.py      # 重要性评分 α·A + β·R + γ·P
│   ├── policies.py        # v1 + v2 全部 policy
│   ├── blocks.py          # v2 块级聚合 (mean + max 加权)
│   ├── compress.py        # 把 actions 应用到 past_key_values
│   └── runner.py          # prefill→bridge→decode + generate_lazy (Δ + drift)
├── experiments/
│   ├── exp1_ppl[_v2].py   # WikiText 困惑度
│   ├── exp2_needle[_v2].py# Needle-in-a-Haystack
│   ├── exp3_runtime[_v2].py # TTFT / TPOT / tokens-per-sec
│   ├── exp4_ablation[_v2].py
│   ├── make_report[_v2].py
│   └── smoke_test[_v2].py
└── results/
    ├── exp1-4_*.json       # v1 (GPT-2)
    ├── exp1-4_*_v2.json    # v2 (Qwen2.5-0.5B-Instruct)
    ├── fig_*.png / fig_*_v2.png
    └── REPORT.md / REPORT_v2.md
```

## 两个版本

| 维度          | v1 (GPT-2)              | v2 (Qwen2.5-0.5B)         |
|---------------|-------------------------|---------------------------|
| 模型          | GPT-2 124M, 1024 ctx    | Qwen2.5-0.5B-Instruct, 32k ctx |
| 注意力        | MHA (12 heads)          | GQA (14 Q / 2 KV) + RoPE  |
| 缓存粒度      | per-token               | per-block (g=16)          |
| 动作决策      | 排序后按预算贪心        | + 绝对阈值 θ_drop/θ_fp16  |
| 在线更新      | prefill 后一次性压缩    | + 每 Δ 步 / 查询漂移触发  |
| 指标          | PPL/Acc/ms-per-token    | + TTFT/TPOT/tokens-per-s |

## v1 主要结论 (GPT-2)

| 方法       | 25% PPL  | 25% Needle | 50% PPL | 50% Needle |
|------------|---------:|-----------:|--------:|-----------:|
| Full       |    30.5  |      0.93  |       — |         — |
| Recent     |   4218.8 |      0.00  | 1212.9 |      0.00 |
| AttnOnly   |    64.1  |      0.06  |   47.5 |      0.08 |
| FullINT8   |  1206.8  |      0.00  |   30.5 |      0.93 |
| **BM-KV**  |   60.3  |      0.25  |   38.1 |      0.57 |

故事：GPT-2 的学习位置编码使 Recent 在丢失 attention sink 后崩盘，FullINT8 出现 50% "内存悬崖"，BM-KV 是 <50% 区间的唯一稳定方法。

## v2 主要结论 (Qwen2.5-0.5B-Instruct)

| 方法         | 25% PPL  | 25% Needle | 50% PPL | 50% Needle | 50% TPOT |
|--------------|---------:|-----------:|--------:|-----------:|---------:|
| Full         |    14.7  |      0.99  |      — |         — |  46.9 ms |
| Recent       |   882.9  |      0.20  |  973.2 |      0.40 |  48.9 ms |
| AttnOnly     |    19.6  |      0.03  |   18.1 |      0.12 |  47.9 ms |
| FullINT8     |   962.0  |      0.40  |   14.7 |      0.99 |  50.9 ms |
| **BM-KV-v2** |    17.6  |      0.39  |   15.9 |      0.60 |  47.5 ms |
| **+ lazy16** |       —  |         —  |      — |         — |  58.3 ms |

故事：换到 Qwen + RoPE 后：
- BM-KV-v2 PPL 几乎不掉（+8% @50%, +20% @25%），远好于 v1 在 GPT-2 上的 +24/+97%
- 但 FullINT8 在 Qwen 上没有 50% 悬崖，所以 BM-KV-v2 的"低预算优势"不像 GPT-2 上那么戏剧化
- TPOT 几乎所有静态方法都跟 Full 持平，lazy update 额外 +24% 时延
- Recent 在 Qwen 上仍然崩盘（RoPE 也保留不住 sink）

## v2 消融

| 变体            | @0.50 PPL | @0.25 PPL | 解读 |
|-----------------|----------:|----------:|------|
| A 完整 BM-KV-v2 |     16.51 |     18.76 | 基线 |
| B 去掉 θ_drop    |     16.51 |     18.76 | 阈值在静态场景里不咬合 |
| C 用 token 粒度  |     16.40 |     18.61 | 略好，块级换布局换质量 |
| D **去掉 INT8** |    *16.73*|    *18.83*| INT8 仍是关键 |
| E v1 token-BM-KV |     16.54 |     18.44 | 与 v2 相当 |

| 模式            | ms/step | rebalances/64-step |
|-----------------|--------:|-------------------:|
| static          |    46.97|                0   |
| lazy Δ=16       |    50.85|                6.6 |
| lazy Δ=8        |    53.15|               12.6 |

## 关键 takeaway 写论文

1. **Recent 失败原因是 attention sink**（GPT-2 学习位置 + Qwen RoPE 都中招）
2. **FullINT8 不是免费午餐**：内存预算 ≥50% 几乎无损，<50% 看具体模型——
   GPT-2 上崩盘，Qwen 上仍工作
3. **BM-KV 主要价值在 <50% 区间**：在 GPT-2 上显著领先；在 Qwen 上和 FullINT8 互有胜负
4. **块级 + 阈值的修订对质量影响很小**（差 0.1 PPL），价值主要在 **系统层面**
   （连续内存布局、与 PagedAttention 对接）
5. **lazy update 当前实现在算法层面有 ~24% TPOT 开销**——主要因为
   `output_attentions=True` 必须重新算 attention。要真的省时，需要绕开 HF
   eager attention 或自定义算子

## 复现

```bash
pip install datasets matplotlib

# v1 (GPT-2)
python experiments/smoke_test.py
python experiments/exp1_ppl.py --n-chunks 50
python experiments/exp2_needle.py --n-fillers 5 --context-tokens 768
python experiments/exp3_runtime.py --n-trials 8
python experiments/exp4_ablation.py
python experiments/make_report.py

# v2 (Qwen2.5-0.5B-Instruct)
python experiments/smoke_test_v2.py
python experiments/exp1_ppl_v2.py --n-chunks 15
python experiments/exp2_needle_v2.py --n-fillers 3 --context-tokens 1024
python experiments/exp3_runtime_v2.py --n-trials 5
python experiments/exp4_ablation_v2.py
python experiments/make_report_v2.py
```
