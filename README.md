# Qwen2.5-0.5B KV260 Transformer Accelerator — Candidate 9

2026 全国大学生嵌入式芯片与系统设计竞赛（AMD 赛题）参赛项目。

在 AMD Kria KV260 上加速 Qwen2.5-0.5B-Instruct：PS（Cortex-A53 裸机）负责聊天模板、分词、
Embedding 查表与 AXI-Lite 调度，PL 执行 24 层 Transformer 的 GEMM、RoPE、Attention、FFN 与 LM Head。

本仓库根目录收录 **candidate9** 分支的完整交付：在 INT8 权重 / BF16 激活的基础上引入 **KV8 KV-cache 量化**
（`kv8_*` 模块族），并用 Xilinx 浮点 IP（`floating_point`、`cordic_sincos`、`fp_exp/mul/recip/rsqrt`、
`fixed_to_float`）替代部分自研浮点路径。

上一版已闭合上板的交付完整保留在 [`legacy_100mhz_closed/`](legacy_100mhz_closed/README_PROVENANCE.md)。

## 状态与限制

| 项 | 结果 |
| --- | --- |
| 端到端仿真（6 token） | PASS：`9707, 0, 2585, 646, 358, 7789` |
| 首轮 151936 logits SHA-256 | `BBA14D322C6F33940DBCEC531E7F0B3A97C38E650A0212880581A664AA39DE95` |
| GEMM 局部综合 @300 MHz | WNS **−5.238 ns**（未收敛，见 `build/reports/timing_summary.rpt`） |
| 完整 BD 布局布线 | **未完成** |
| bitstream / 上板实测 | **本分支没有** |
| 顶层 IRQ | 已移除，上板软件与系统接口需与此变更匹配 |

一句话：**这个分支仿真能跑通，但时序没有闭合，也没有上板数据。** 已闭合上板的版本见 `legacy_100mhz_closed/`。

## 目录

```text
.
├── README.md          # 项目简介 + 复现步骤
├── src/               # 设计源码：RTL、厂商 IP 配置、PS 软件、BSP、验证模型
├── sim/               # 单元 TB、端到端验证结果、仿真器
├── build/             # Vivado 工程与脚本、时序/利用率报告、构建日志
├── board/             # SD 卡模型资源与 PS 固件
├── data/              # 本分支可用数据说明
├── skill/             # 技能包
├── report/            # 设计报告（对应 legacy 版本）
└── legacy_100mhz_closed/  # 上一版已闭合上板的完整交付
```

各子目录的详细说明见其内部 `README.md`。

## 复现步骤

### 0. 环境

| 用途 | 依赖 |
| --- | --- |
| RTL 综合 / 实现 | Vivado 2026.1，器件 `xck26-sfvc784-2LV-c` |
| 端到端仿真 | oss-cad-suite（iverilog / verilator）+ zig（充当 C++ 编译器） |
| PS 固件编译 | Vitis 2020.2 及以上（aarch64-none-elf 工具链） |
| 模型权重 | Git LFS（克隆后必须 `git lfs pull`） |

构建脚本里默认引用了本机的 `C:\Users\ZKHC\tools\oss-cad-suite` 与
`C:\Users\ZKHC\tools\ziglang_verilator\ziglang\zig.exe`，**换机器必须改这两个路径**。

### 1. 端到端仿真（C→RTL→C）

在仓库根目录运行，首次构建要用一个新的 Tag，对象目录已存在时脚本会拒绝覆盖：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build\scripts\build_candidate9.ps1 -Tag candidate9_new -TreeSel stage4
powershell -NoProfile -ExecutionPolicy Bypass -File .\build\scripts\run_candidate.ps1 -Exe .\qwen_candidate9_new.exe -Dir run_chat_new -Template chat -Max 6 -Prompt hi -Stop 6
```

也可以直接把 `-Exe` 指向随包的 `sim\simulator\qwen_candidate9.exe`（已构建好的仿真程序）。
运行目录必须是新目录。注意：这里的验证用的是**行为级浮点模型**，不是完整厂商 IP 的门级仿真。

期望输出（已随包的结果见 `sim/candidate9_chat_result/`）：

```text
RTL_TOKEN_VERIFIED iteration=5 id=7789 cycles=13805955 logits=151936
RTL_STOP_AFTER reached (6 token(s)); exiting 0
```

### 2. 单元验证

`sim/tb/` 下 18 个自检测试平台（GEMM 列、KV8 量化各阶段、EXP/ADD 对照 VHDL 等），
由 `sim/scripts/` 的 `build.ps1` / `run.ps1` / `run_custom.ps1` / `preflight.ps1` 驱动，
`verify_ab.py` 用于 A/B 结果比对。

### 3. 重建 Vivado 工程

```text
build/vivado/create_rtl_project.tcl   → 生成 build/vivado/rtl_project/candidate9.xpr
build/vivado/ip_package.tcl           → 打包加速器 IP（输出到 package_candidate9/）
build/vivado/bd_build_ps.tcl          → 重建 KV260 PS Block Design
```

`clk.xdc` 是 100 MHz 顶层 OOC 时钟约束，**不是已闭合的性能保证**。`bd_build_ps.tcl` 依赖本机
KV260 board preset 路径，且本次交付的 BD 未生成、未验证。

## 数据与报告

- `sim/candidate9_chat_result/`：6 个 token 的完整证据——每个 iteration 的 151936 维 logits、
  token 序列、周期数、`run.out` 与 `timing.txt`。
- `build/reports/`：`timing_summary.rpt`（含 −5.238 ns 违规路径）、`utilization.rpt`。
- `report/`：设计报告，对应 `legacy_100mhz_closed/` 那一版（100 MHz 闭合、有上板数据）。

## 历史交付

[`legacy_100mhz_closed/`](legacy_100mhz_closed/README_PROVENANCE.md) 存放上一版**已完整闭合、已上板**的交付：
100 MHz 时序收敛（Setup WNS +1.700 ns、Hold WHS +0.010 ns）、bitstream 与 XSA、KV260 上板
6 场景 22 token 实测记录、演示视频与数据汇总。它用自研浮点路径（无 `kv8_*`），与根目录分支不是同一套设计。

`report/` 下的设计报告描述的就是这一版，两份材料配套阅读；根目录内容不作为该报告的成果来源。

## 许可

MIT License。
