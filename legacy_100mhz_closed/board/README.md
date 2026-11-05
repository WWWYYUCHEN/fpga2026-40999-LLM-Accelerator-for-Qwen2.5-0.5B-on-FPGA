# 上板工程与实测

KV260 正式运行链为 FSBL 2020.2、U-Boot 2020.01，PL0 配置为 100 MHz。SD 卡自动启动后完成 PL 配置、模型资源加载、tokenizer 自检，并进入 `READY / prompt>`。

## 目录

| 目录 | 内容 |
| --- | --- |
| `deployment/` | 可直接写入 SD 卡的启动脚本、PL 镜像、PS 固件和模型资产 |
| `measurements/boot/` | 启动版本、时钟、持久元数据与串口记录 |
| `measurements/functional/` | A 链 6 场景、7 请求、22 token 原始记录 |
| `measurements/autoboot/` | 上电自动启动到 READY 的记录 |
| `measurements/functional_repeat/` | 自动启动后的重复功能验证 |
| `measurements/length_scan/` | decode 长度扫描原始日志与结果 |
| `tools/` | 串口、SD 校验、回归和启动复现工具 |
| `模型对话实际演示.mp4` | KV260 实机模型对话演示录像 |

## 部署文件

| 文件或项目 | 值 |
| --- | --- |
| PL 镜像 | `QWEN25.BIN`，7,797,692 字节 |
| PL CRC32 | `137125e6` |
| PS 固件 | `QWUART.ELF`，999,656 字节 |
| PS 固件 SHA-256 | `0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c` |
| PS 固件 CRC32 | `f195fdca` |
| CSR ID | `0x515732354b563236` |
| CSR VERSION | `0x0000000000010000` |
| UART | COM11，115200 8N1，无流控 |
| KV 容量 | 256 token |

## 启动流程

```text
上电
→ QSPI 引导
→ FSBL 2020.2
→ U-Boot 2020.01
→ 自动执行 SD 卡 boot.scr
→ 配置 QWEN25.BIN
→ 加载 QWUART.ELF
→ 加载模型资源
→ TOKENIZER_SELFTEST_OK
→ READY / prompt>
```

## 实板功能结果

上板测试覆盖 6 类场景、7 次请求和 22 个生成 token，token ID、字节、数量、结束原因和应用状态全部与交付参考一致。

| 场景 | 模板 | 输入 | 输出 token |
| --- | --- | --- | --- |
| final_all_layers | raw | 你好 | `3837,35946` |
| final_repeat_context_reset | raw | 你好 | `3837 / 3837` |
| final_rows9_prefill | raw | 123456789 | `16` |
| final_decode_trace | raw | 你好 | `3837,35946` |
| final_chat_nihao | chat | 你好 | `108386,6313,112169,102804,47874,1773,104139,109944` |
| final_chat_eos | chat | 1+1等于几？ | `16,10,16,107106,17,1773,151645` |

板内计时使用 PS `XTime`，从启动前读取到观察到 DONE，包含 MMIO、屏障和 100 μs 轮询。首输出与 decode 阶段分别记录，原始 ticks 见 `measurements/functional/tokens.csv` 和 `measurements/length_scan/tokens.csv`。

