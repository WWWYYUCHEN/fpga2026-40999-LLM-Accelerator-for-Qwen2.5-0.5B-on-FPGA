# Qwen2.5-0.5B KV260 Transformer Accelerator

本项目在 AMD Kria KV260 上实现 Qwen2.5-0.5B-Instruct 的 PS 裸机控制与 PL 加速推理。

## 核心指标

| 项目 | 结果 |
| --- | --- |
| 模型 | Qwen2.5-0.5B-Instruct，24 层 |
| 数据格式 | INT8 权重、BF16 激活、FP32 累加、BF16 输出 |
| PL 频率 | 100 MHz |
| 上下文容量 | 256 token |
| RTL 系统回归 | 6 类场景 PASS |
| 上板功能测试 | 7 次请求、22/22 token 与参考一致 |
| Tokenizer 自检 | 40 tokens OK |
| 100 MHz 时序 | Setup WNS +1.700 ns，Hold WHS +0.010 ns |
| Decode 延迟模型 | `T_ms = 133.680537 + 3.244886 × n_history` |

## 目录

```text
qwen25_kv260_2026_competition/
├── README.md
├── src/       # RTL、PS 软件、BSP、验证和上位机源码
├── sim/       # RTL 仿真器与最终验证结果
├── build/     # Vivado 工程、bitstream、XSA 和时序报告
├── board/     # SD 部署、上板记录、工具和演示视频
├── data/      # token、时延、模型和版本汇总
├── skill/     # 技能包说明
└── report/    # 设计报告、验证矩阵和复现说明
```

## 快速运行

1. 将 `board/deployment/` 的全部内容复制到 FAT32 SD 卡根目录。
2. 将 SD 卡插入 KV260，连接 USB UART 并上电。
3. 串口设置为 `115200 8N1`、无流控。
4. 等待 `TOKENIZER_SELFTEST_OK 40 tokens` 和 `READY / prompt>`。
5. 输入中文或英文开始推理。

常用命令：

```text
/selftest
/template chat
/mode text
/max 20
/reset
```

## 仿真

在项目根目录运行：

```powershell
python .\src\PC\run_cli.py
python .\src\PC\run_cli.py --prompt "你好" --tokens 8
```

最终验证结果位于 `sim/reports/system_regression/final_*`。
完整设计见 [report/设计报告.md](report/设计报告.md)，协作记录见 [report/大模型协作记录.md](report/大模型协作记录.md)。
