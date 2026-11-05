# 设计源码

本目录收录完整工程设计源文件。

| 目录 | 内容 |
| --- | --- |
| `rtl/` | AFU、attention、RoPE、storage 等生产 RTL |
| `software/interactive/` | PS 分词、生成、UART 和 AXI-Lite 调度 |
| `verification/` | RTL 与系统验证源码 |
| `BSP/` | A53 standalone 头文件和预编译库 |
| `reference/` | tokenizer 等模型参考资产 |
| `PC/` | 串口、RTL 仿真和交互界面 |

部署 PS 固件位于 `../board/deployment/QWUART.ELF`，SHA-256：

```text
0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c
```

