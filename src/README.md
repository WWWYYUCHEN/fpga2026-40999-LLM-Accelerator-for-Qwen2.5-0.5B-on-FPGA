# 设计源码

candidate9 分支的全部设计输入，可据此重建 Vivado 工程与端到端仿真环境。

| 目录 | 内容 |
| --- | --- |
| `rtl/` | 加速器 RTL。`afu/`（MAC 阵列与 lane）、`attention/`、`rope/`、`storage/`（QKV 控制器、张量缓冲），根目录为顶层 `qwen25_accel_axi4_top.v`、MMU、DMA、GEMM 阵列与 **`kv8_*` KV-cache 量化模块族** |
| `ip/` | 厂商 IP 配置：7 个 IP 的 `.xci`（`floating_point_0`、`ip_cordic_sincos`、`ip_fixed_to_float`、`ip_fp_exp`、`ip_fp_mul`、`ip_fp_recip`、`ip_fp_rsqrt`）、`shim/`（RTL 包装）、`add/`（Full DSP 浮点加法 IP，EXP 用 Full BRAM + Full DSP） |
| `software/` | PS 侧裸机软件：`main.c`、`generation.c`、`tokenizer.c`、`utf8_stream.c`、`sd_entry.S`、`sd_prepare.c`、`uart_console_override.c`，以及 `build_a53.ps1` / `build_sim.ps1`、`tests/`、`vendor/utf8proc` |
| `BSP/` | A53 standalone 头文件与预编译库、`lscript.ld` |
| `verification/` | 验证基础设施：`models/`（Verilator 行为级模型：scalar math、sincos、XPM）、顶层 TB（`tb_qwen25_no_irq.sv`、`tb_array_pipeline.sv`、`tb_kchain_pipeline.sv`、`tb_kv8_quant_pipeline.sv`、`tb_result_no_irq.sv`）、`rtl_bridge.cpp`、`sim_dsp48e2_fast.v`、`qwen_gemm_array_reference.sv`、`board_parameters.h` |

## 说明

- RTL 里 `qwen25_qkv_controller.v` 是控制主体，`kv8_*` 系列实现 KV-cache 的 8 bit 分组量化/反量化。
- `ip/` 只保留 `.xci` 配置与 shim，生成物（`.dcp`、`.vho`、`.veo`、仿真网表）已剔除，
  Vivado 会按 `.xci` 重新生成。
- 顶层 IRQ 在本分支已移除，PS 侧等待方式与早期版本不同。
