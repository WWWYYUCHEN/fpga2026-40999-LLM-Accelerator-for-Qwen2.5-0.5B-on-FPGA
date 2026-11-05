# 构建与实现

工程采用 Vivado 100 MHz 约束完成综合、布局布线和 bitstream 导出。

| 项目 | 路径 | 结果 |
| --- | --- | --- |
| 综合输入 | `synth/` | Vivado 工程基线 |
| bitstream | `hardware/qwen25_system_100mhz.bit` | 7,797,912 字节 |
| PL 启动镜像 | `../board/deployment/QWEN25.BIN` | 7,797,692 字节 |
| XSA | `hardware/qwen25_system_100mhz.xsa` | 6,274,719 字节 |
| 时序报告 | `timing/` | 100 MHz 实现 |

## 关键哈希

- bitstream：`e1c3110a27d7ab07b4d4e88af222c42c4dc2c9bde4050eeaff5cd2f92e4de12a`
- PL BIN：`73a06211577e199bb4ff3735581899a50c48c2e4f94f81189d3469e8d9874d26`

XSA 内的 bitstream 与独立 bitstream 已逐字节核对一致。

## 时序

| 项目 | 结果 |
| --- | ---: |
| 时钟约束 | 10.000 ns / 100 MHz |
| Setup uncertainty | 0.100 ns |
| Setup WNS | +1.700 ns |
| Hold WHS | +0.010 ns |

