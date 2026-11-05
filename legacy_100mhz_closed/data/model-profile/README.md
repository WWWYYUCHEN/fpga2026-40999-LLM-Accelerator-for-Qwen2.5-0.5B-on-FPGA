# 模型静态分析

Qwen2.5-0.5B-Instruct 的关键模型参数与硬件成本汇总见：

```text
../model_geometry_and_cost.csv
```

模型配置包括 hidden size 896、24 层 Transformer、14 个 Q 头、2 个 KV 头、head_dim 64、MLP 宽度 4864、词表大小 151936，以及共享 Embedding/LM Head。

模型总参数为 494,032,768。INT8 矩阵权重为 493,961,216 B，输入列 scale 为 246,656 个、493,312 B。BF16 KV cache 每 token 为 12,288 B，16K 上下文为 192 MiB。

