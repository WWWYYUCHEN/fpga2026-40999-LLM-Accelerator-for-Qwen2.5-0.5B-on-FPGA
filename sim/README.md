# 仿真与验证

系统默认入口为加速器 RTL 逐周期仿真。PS 软件负责聊天模板、分词、embedding 和 AXI-Lite 调度，PL 执行 Transformer 推理。

| 目录 | 内容 |
| --- | --- |
| `simulator/` | RTL 逐周期仿真器和 PS 动态库 |
| `reports/system_regression/final_*` | 6 类最终场景的输入、结果、UART 日志和 token 数据 |
| `reports/parameter_fixed/` | 参数 RAM 验证摘要 |
| `reports/rtl_current_acceptance/` | 当前 RTL 数值验收摘要 |
| `reports/regression_status.json` | 总体验证状态 |
| `reports/source_lineage.json` | 源码与构建产物溯源 |

| 指标 | 结果 |
| --- | ---: |
| 单 token prefill 逐层最大相对 L2 | 0.00044046511 |
| 最终归一化 896 元素相对 L2 | 0 |
| LM Head 151936 logits 相对 L2 | 3.3777615e-05 |
| 已测试 token ID | 全部一致 |

完整场景汇总见 [测试结果汇总.csv](../data/测试结果汇总.csv)。
