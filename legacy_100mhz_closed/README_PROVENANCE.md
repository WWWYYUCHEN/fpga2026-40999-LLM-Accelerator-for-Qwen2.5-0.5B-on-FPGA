# 历史交付：100 MHz 闭合版本（legacy）

这个目录是**上一版已完整闭合并上板的交付**，原样保留，未做任何修改。

## 它包含什么

| 内容 | 说明 |
| --- | --- |
| `build/hardware/qwen25_system_100mhz.bit` | bitstream，7,797,912 B，SHA-256 `e1c3110a…e4de12a` |
| `build/hardware/qwen25_system_100mhz.xsa` | 硬件平台，6,274,719 B |
| `build/timing/` | 时序、利用率、DRC、route 报告与 Vivado 回归日志 |
| `board/deployment/` | 可直接写 SD 卡的启动包：`boot.scr`、`QWEN25.BIN`、`QWUART.ELF`、模型资产 |
| `board/measurements/` | 上板实测原始记录：boot、functional、functional_repeat、autoboot、length_scan |
| `board/模型对话实际演示.mp4` | 实机演示录像 |
| `data/测试结果汇总.csv` | 全项结果汇总 |
| `src/`、`sim/` | 该版本的设计源码与仿真结果 |

## 关键指标（该版本）

- 100 MHz：Setup WNS **+1.700 ns**，Hold WHS **+0.010 ns**
- 上板：6 类场景、7 次请求、**22/22 token** 与参考一致
- RTL 系统回归：6 类场景 PASS
- 逐层最大相对 L2：0.00044046511；LM Head 相对 L2：3.3777615e-05

## 与仓库根目录（candidate9）的关系

**两者是不同的设计，不可混用。**

| | 根目录 candidate9 | 本目录 legacy |
| --- | --- | --- |
| 浮点实现 | 厂商 IP（floating_point / cordic / fp_exp…）+ shim | 自研 `bf16_mul` / `fp32_add` 路径 |
| KV cache | BF16 + **KV8 量化**（`kv8_*` 模块族） | BF16 |
| 时序 | 300 MHz 局部 WNS **−5.238 ns**，未闭合 | 100 MHz 闭合 |
| bitstream | 无 | 有 |
| 上板数据 | 无 | 有 |

根目录 `README.md` 描述的是 candidate9；本目录 `README.md` 是 legacy 版本自己的说明。
仓库根 `report/` 下的设计报告对应的是**本目录这一版**。

## 模型资产

`board/deployment/QWEN25/` 下三个大文件同样由 Git LFS 托管，其内容与根目录
`board/sd/QWEN25/` 完全一致（SHA-256 相同），LFS 存储是同一份对象。
