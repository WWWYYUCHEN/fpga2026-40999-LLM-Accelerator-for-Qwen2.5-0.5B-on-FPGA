# 原生测试向量实板复测

结论：在 Image A / FSBL 2020.2 / U-Boot 2020.01; clocks captured in this boot 的 READY 状态下，在真实 KV260 上复跑基线交付包提供的 6 类 final 场景，共 7 次请求、22 个输出 token，全部与参考一致。40-token 分词自检通过。比对包含 prompt token 数、max_new、token ID/字节、输出数量、结束原因和 status；所有请求 status=0 并返回 READY。

本轮执行 PS 裸机 + PL 推理，主机只控制串口和整理数据。复用基线交付包测试输入、模板和输出上限；交付包的原生入口默认是 PC 上的 RTL 逐周期仿真，本轮没有重跑仿真。`final_decode_trace` 仅验证功能输出，未导出逐层 trace；不将端到端 token 一致写成该场景全项 PASS。

## 功能结果

| 场景 | 模板 | 输入 | prompt tokens | max_new | 实测 token IDs（不同请求用 / 分隔） | 对照 |
| --- | --- | --- | ---: | ---: | --- | --- |
| final_all_layers | raw | 你好 | 1 | 2 | 3837,35946 | token一致 |
| final_repeat_context_reset | raw | 你好 | 1 | 1 | 3837 / 3837 | token一致 |
| final_rows9_prefill | raw | 123456789 | 9 | 1 | 16 | token一致 |
| final_decode_trace | raw | 你好 | 1 | 2 | 3837,35946 | token一致 |
| final_chat_nihao | chat | 你好 | 30 | 8 | 108386,6313,112169,102804,47874,1773,104139,109944 | token一致 |
| final_chat_eos | chat | 1+1等于几？ | 35 | 20 | 16,10,16,107106,17,1773,151645 | token一致 |

raw「你好」前两 token 为「，我」，符合 raw 参考。chat「你好」输出「你好！很高兴为您服务。有什么我可以」，8 token 正常截断。算术回答「1+1等于2。」，第7个 token 151645 为 EOS，reason=eos；其余请求 reason=limit。重复 raw 请求两次均输出3837。

## 板内计时记录

| 场景 | 首输出窗口 / ms | decode 样本数 | decode 平均 / ms |
| --- | ---: | ---: | ---: |
| final_all_layers | 134.571 | 1 | 136.986 |
| final_repeat_context_reset | 133.861 / 133.865 | 0 | — |
| final_rows9_prefill | 522.898 | 0 | — |
| final_decode_trace | 133.761 | 1 | 136.996 |
| final_chat_nihao | 1675.433 | 7 | 240.591 |
| final_chat_eos | 2074.512 | 6 | 255.404 |

逐 token 原始 ticks、换算 ms 与阶段见 tokens.csv。频率来源为交付 BSP `XPAR_CPU_TIMESTAMP_CLK_FREQ=99999001` Hz；公式 `ms=ticks×1000/99999001`，未独立校准。generation.c 通过 PS XTime 包围 START 至观察 DONE，包含 MMIO、屏障和100μs轮询；排除分词、embedding、UART和SD日志。iteration=0为prefill/首输出，后续才是decode。此数值不是纯PL周期或用户端到端耗时，不与参考仿真 cycles 比较，也不作为新优化效果。

## 版本、证据与边界

- 基线回归数据已集成于 `sim/reports/system_regression/final_*`，来源与 SHA-256 见 `test-results.json`。
- 沿用 QWUART.ELF；本轮未重编应用或 BSP，未替换模型或 PL。
- 启动标识：Image A / FSBL 2020.2 / U-Boot 2020.01; clocks captured in this boot；证据见 `../boot/capture-state.json` 与 `../boot/uart-capture.log`。功能输出一致不等于从物理配置反解了完整网表。当前KV容量256。
- 串口单拥有者，COM11、115200 8N1，无流控；持续日志与冻结记录分别位于 `uart-live.log`、`startup-uart-transcript.log` 和 `uart-transcript.log`。
- 未发送闪存写入或启动配置变更命令，未使用GPU。启动选择器可自行更新Last Booted；固件按原逻辑向SD的CHAT.CSV追加请求记录。
- 当前固件不能导出逐层tensor/完成事件、全部logits或PL cycles/stall。本轮未独立复验基线 RTL 仿真的逐层数值误差、argmax与低层计数；也未做长上下文、功耗或长稳测试。
- 本目录为独立原生向量实板复测，未覆盖既有功能报告、长度扫描或其他启动版本的数据。
- 结束已恢复chat / text / max20并确认READY，命令槽清空，可见串口监听保留。

复现工具：`../../tools/run_kv260_native_regression.py`，通过 `../../tools/kv260_uart_native.py` 的显式命令槽顺序发送，一条命令收到 READY 才发下一条；错误、超时或参考不一致即停止。既有结果不覆盖、不自动重放。
