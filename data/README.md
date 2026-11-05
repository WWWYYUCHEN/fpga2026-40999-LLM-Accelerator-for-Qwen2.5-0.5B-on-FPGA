# 测试数据

本分支**没有上板测量数据**，因此没有可汇总的延迟/时钟实测表格。可用的数值证据全部来自仿真，
放在 `sim/candidate9_chat_result/`：

| 数据 | 值 |
| --- | --- |
| 6 token chat 输出 | `9707, 0, 2585, 646, 358, 7789` |
| 首轮 151936 logits SHA-256 | `BBA14D322C6F33940DBCEC531E7F0B3A97C38E650A0212880581A664AA39DE95` |
| 各 token 周期数 | 见 `hardware_tokens.csv`（首轮 91,281,969 cycles，逐 token 约 1,360 万） |
| GEMM 局部综合时序 | WNS −5.238 ns @300 MHz（`build/reports/timing_summary.rpt`） |

模型几何（来自 Qwen2.5-0.5B-Instruct 公开配置，与本分支实现一致）：

| 项 | 值 |
| --- | --- |
| hidden_size / layers | 896 / 24 |
| attention heads / KV heads | 14 / 2 |
| head_dim | 64 |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| 总参数量 | 494,032,768 |

如需 100 MHz 闭合版本的上板实测汇总，请参见 `report/` 中早期交付的设计报告——那是另一套设计。
