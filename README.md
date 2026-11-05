# Qwen2.5-0.5B KV260 Transformer Accelerator

2026 全国大学生嵌入式芯片与系统设计竞赛（AMD 赛题）参赛项目。

本项目在 AMD Kria KV260 上实现 Qwen2.5-0.5B-Instruct 的 PS 裸机控制与 PL 加速推理：
PS（Cortex-A53，裸机）负责聊天模板、分词、Embedding 查表与 AXI-Lite 调度，PL（100 MHz）执行
Transformer 24 层的 GEMM、RoPE、Attention、FFN 与 LM Head。

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
.
├── README.md      # 项目简介 + 复现步骤
├── src/           # 设计源码：RTL、PS 侧 C 软件、BSP、验证平台、上位机
├── sim/           # 仿真器与最终验证结果
├── build/         # 综合与实现报告、bitstream、XSA
├── board/         # 上板工程、SD 部署包、实测输出、演示视频
├── data/          # 测试结果汇总、模型档案
├── skill/         # 技能包（可复用方法）
└── report/        # 设计报告 + 大模型协作记录
```

各子目录的详细说明见其内部 `README.md`。

## 复现步骤

### 0. 环境要求

| 用途 | 依赖 |
| --- | --- |
| 仿真（RTL 后端） | Windows x64，无需额外运行时（`sim/simulator/` 内为预编译 exe/dll） |
| 仿真（数学后端）/ 数据核对 | Python 3.10+，`pip install -r src/PC/requirements_math.txt` |
| 数值验证脚本 | `pip install -r src/PC/requirements_verify.txt` |
| 上板 | KV260 Starter Kit、FSBL 2020.2 + U-Boot 2020.01、FAT32 SD 卡、USB 串口线 |
| 重新综合/实现 | Vivado 2026.1 |

### 1. 获取代码（含 LFS 大文件）

```bash
git clone https://github.com/WWWYYUCHEN/fpga2026-40999-LLM-Accelerator-for-Qwen2.5-0.5B-on-FPGA.git
cd fpga2026-40999-LLM-Accelerator-for-Qwen2.5-0.5B-on-FPGA
git lfs pull      # 必做
```

`board/deployment/QWEN25/` 下三个模型资产超过 GitHub 100 MB 单文件限制，由 Git LFS 托管：

| 文件 | 大小 |
| --- | --- |
| `WEIGHTS.BIN` | 341 MB |
| `EMBED.BIN` | 260 MB |
| `LMW.BIN` | 130 MB |

未安装 git-lfs 时这三个文件只是指针文件，仿真会缺权重、上板会加载失败。

### 2. 仿真复现（无需硬件）

在仓库根目录运行：

```powershell
python .\src\PC\run_cli.py
python .\src\PC\run_cli.py --prompt "你好" --tokens 8
```

- 默认 `--backend rtl`：调用 `sim/simulator/qwen_rtl_sim.exe`（Verilator 逐周期 RTL 模型），
  PS 侧跑的是 `src/software/interactive/` 里的真实固件逻辑（分词、模板、AXI-Lite 调度）。
- `--backend math`：用 PyTorch 数学模型跑同一套 PS 应用，用于交叉核对（需 torch）。
- 每次运行的 `console.log`、`mmio_trace.txt` 写入 `sim/runs/<时间戳>/`（该目录不入库）。

参考结果在 `sim/reports/system_regression/final_*`，逐场景的输入、UART 日志、token 与数值对比齐全。

### 3. 上板复现

1. 将 `board/deployment/` 的**全部内容**复制到 FAT32 SD 卡根目录（`boot.scr`、`QWEN25.BIN`、`QWUART.ELF`、`QWEN25/` 子目录）。
2. SD 卡插入 KV260，连接 USB UART，上电。
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

启动链为 QSPI → FSBL 2020.2 → U-Boot 2020.01 → `boot.scr` → 配置 `QWEN25.BIN` → 加载 `QWUART.ELF` → 加载模型资源 → 自检 → `READY`。

实测证据：`board/measurements/`（boot / functional / functional_repeat / autoboot / length_scan 的原始串口日志与 JSON），
演示录像 `board/模型对话实际演示.mp4`。

### 4. 构建复现

成品与报告：

| 产物 | 路径 | 大小 |
| --- | --- | --- |
| bitstream | `build/hardware/qwen25_system_100mhz.bit` | 7,797,912 B |
| XSA | `build/hardware/qwen25_system_100mhz.xsa` | 6,274,719 B |
| 上板 PL 镜像 | `board/deployment/QWEN25.BIN` | 7,797,692 B |
| 时序/利用率/DRC | `build/timing/*.rpt` | — |

约束与结果：10.000 ns / 100 MHz，setup uncertainty 0.100 ns；Setup WNS **+1.700 ns**，Hold WHS **+0.010 ns**；
策略 `synth_1=Flow_AreaOptimized_medium`、`impl_1=Performance_ExplorePostRoutePhysOpt`。

本次实际执行的脚本已随仓库保留，可直接复用：

```powershell
vivado -mode batch -source build/timing/run_timing_regression_executed.tcl
vivado -mode batch -source build/timing/export_verified_hardware_executed.tcl
```

这两个 TCL 以 `hardware/vivado/qwen25_kv260.xpr`、`ip_repo/`、`board_files/` 为输入。这三个目录属于本地 Vivado 工程，
因体积与许可证原因未随包分发，**因此仓库内可直接复现的是报告与最终产物；完整重跑需自备上述 Vivado 工程目录**。
XSA 内的 bitstream 与独立 bitstream 已逐字节核对一致。

## 数据、报告与技能包

- `data/测试结果汇总.csv`：模型几何、RTL 仿真、上板功能、长度扫描、时序、数值误差的逐项汇总。
- `data/model-profile/`：Qwen2.5-0.5B-Instruct 参数与硬件成本，含原始 `model.safetensors` 的 SHA-256。
- `report/设计报告.md`：架构、数据通路、验证方法与结果。
- `report/大模型协作记录.md`：大模型在各阶段的参与范围与产出。
- `skill/README.md`：本项目沉淀的可复用方法。

## 关键产物校验

```text
bitstream qwen25_system_100mhz.bit  e1c3110a27d7ab07b4d4e88af222c42c4dc2c9bde4050eeaff5cd2f92e4de12a
PL 镜像   QWEN25.BIN                73a06211577e199bb4ff3735581899a50c48c2e4f94f81189d3469e8d9874d26
PS 固件   QWUART.ELF                0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c
```

## 许可

MIT License，见 [LICENSE](LICENSE)。
