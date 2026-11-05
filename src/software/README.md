# candidate9 固件（PS 侧软件）

本目录补齐了 candidate9 交付包缺失的**上板软件层**。原有 `fw/` 只够跑 Verilator 端到端仿真；本目录新增的部分让同一份固件能编译成 KV260 上真正运行的 A53 裸机镜像。

参照物是归档工程 `qwen25_kv260_2026_competition` 的 `src/software/`：同样的分层（平台层 + 应用层）、同样的构建断言、同样的链接顺序。差别只在 candidate9 自己的接口事实：CSR 基址 `0x8000_0000`（不是 `0xA000_0000`）、**顶层 IRQ 已移除**。

## 文件与职责

| 文件 | 层级 | 职责 |
| --- | --- | --- |
| `main.c` | 应用 | UART 交互主循环：SD 挂载、资产加载、分词、单次请求生成、CSV 记录 |
| `generation.c/.h` | 应用 | 生成状态机。只通过 `struct qgen_ops` 的 5 个回调访问硬件，不知道自己跑在 PL 还是仿真上 |
| `tokenizer.c/.h` | 应用 | BPE 分词，读 `TOKEN.BIN`，40 token 黄金自检 |
| `utf8_stream.c/.h` | 应用 | 增量 UTF-8 解码，处理被 token 边界切断的多字节字符 |
| `vendor/utf8proc/` | 依赖 | NFC 归一化 |
| **`sd_entry.S`** | **平台（新增）** | AArch64 入口 `sd_entry`，对应 `bsp/lscript.ld` 的 `ENTRY(sd_entry)` |
| **`sd_prepare.c`** | **平台（新增）** | `qwen_board_prepare()`：ID / VERSION / IDLE 三段握手 |
| **`uart_console_override.c`** | **平台（新增）** | newlib `_write` / `_read` 覆盖，`printf` 直达 UART FIFO |
| **`build_a53.ps1`** | **构建（新增）** | 上板固件构建，产物 `build_a53/QWENSD.ELF` |
| **`build_sim.ps1`** | **构建（新增）** | 宿主模拟器构建，产物 `build_sim/qwen_board_sim.exe` |
| `tests/board_sim.c` | 仿真 | 虚拟 UART / SD / MMIO，脚本注入 PL token |
| `tests/shim/` | 仿真 | 宿主编译用的 BSP 头文件桩 |
| `../bsp/` | 平台 | A53 standalone 头文件、预编译库、`lscript.ld`、`Xilinx.spec`（自归档工程复制，同器件 `xck26-sfvc784-2LV-c`） |

## 依赖关系

```text
                        main.c
                           |
        +------------------+------------------+---------------+
        |                  |                  |               |
  generation.c       tokenizer.c        utf8_stream.c   tests/board_sim.c (仅仿真)
        |                  |                  |               |
        |            vendor/utf8proc          |         虚拟 UART/SD/MMIO
        |                  |                  |         脚本化 PL token
        +---------+--------+------------------+
                  |
          ../board_parameters.h  ← 唯一地址真相：CSR 基址、7 个 DDR 窗口、资产表、KV 清零表
                  |
        ┌─────────┴──────────┐
   上板路径                仿真路径
        |                      |
   sd_prepare.c           board_sim.c 里的
   （CSR 握手）            qwen_board_prepare() 空实现
        |
   sd_entry.S（链接入口）
        |
   uart_console_override.o（printf 出口）
```

关键约束：`generation.c` 里**没有任何寄存器地址**。地址只出现在 `main.c` 的 `csr_write/csr_read` 回调和 `board_parameters.h`，所以同一份生成逻辑既能跑 PL，也能跑仿真，不会出现两套接口。

## 与 RTL 的接口契约

控制窗口：`QWEN_CSR_BASE = 0x8000_0000`，64 KiB，AXI4-Lite **64 位数据 / 16 位地址**。只能 64 位访问。

| 偏移 | 方向 | 语义 | 出处 |
| --- | --- | --- | --- |
| `0x0000` | RO | ID = `0x515732354b563236` | `qwen25_result_regs.v` |
| `0x0008` | RO | VERSION = `0x00010000` | 同上 |
| `0x0010` | WO | bit0 START，bit3 CLEAR；**读回恒 0**（IRQ 已移除） | 同上 |
| `0x0018` | RO | `{result_valid(bit6), status[5:0]}` | 同上 |
| `0x0020` | — | 读回 0，无状态 | 同上 |
| `0x0028` | — | 读回 0，无状态 | 同上 |
| `0x0030` | RO | ERROR_CODE = `status[95:64]` | 同上 |
| `0x01F8` | RO | BEST_TOKEN_ID（18 位） | 同上 |
| `0x0080` | RW | RUN_MODE，bit0：0=prefill，1=decode | `qwen25_compute_cfg.v` |
| `0x0088` | RW | SEQ_LEN = 本步行数 | 同上 |
| `0x0090` | RW | CONTEXT_LEN = position + rows | 同上 |
| `0x0098` | RW | TOKEN_POS = position | 同上 |
| `0x00B0` / `0x00B8` | RW | KV_CTRL / KV_VALID_LEN | 同上 |
| `0x00D0` | RW | RMS_EPS，复位 `0x3586` | 同上 |
| `0x0100` | RW | ACT_IN_ADDR | 同上 |
| `0x0108` | RW | ACT_OUT_ADDR | 同上 |
| `0x0110` | RW | MODEL_WEIGHT_BASE | 同上 |
| `0x0118` / `0x0120` | RW | K_CACHE_BASE / V_CACHE_BASE | 同上 |
| `0x0128` / `0x0130` | RW | Q_SCRATCH_BASE / P_SPILL_BASE | 同上 |
| `0x0140` | RW | AXI_BURST_CFG，复位 `0x0F`，固件写 255 | 同上 |

**IRQ 已移除是本交付包与归档工程最大的接口差异**：`qwen25_result_regs.v` 把 `abort_req` / `soft_reset_req` / `error_clear_req` 全部接 0，`0x0010` / `0x0020` / `0x0028` 读回都是 0。因此固件纯轮询，`sd_prepare.c` 不写 `0x0020` / `0x0028`，只写 `0x0010` 的 CLEAR 位。

DDR 窗口（`board_parameters.h`，与 `vivado/address_map.h` 一致）：

| 用途 | 基址 | 大小 |
| --- | --- | --- |
| K Cache ×24 | `0x1000_0000` | 每层 8 MiB |
| V Cache ×24 | `0x2000_0000` | 每层 8 MiB |
| DESC / SCALE / BIAS / GAMMA | `0x4000_0000` 起 | 6 KiB / 912128 / 55296 / 87808 |
| WEIGHTS | `0x4100_0000` | 357 826 560 |
| LMW | `0x5700_0000` | 136 134 656 |
| EMBED | `0x6000_0000` | 272 269 312 |
| ACT_IN | `0x7100_0000` | 458752 |
| Q Scratch | `0x7150_0000` | 458752 |
| CONTEXT（结果） | `0x7160_0000` | 458752 |

4 个 PL 主口直连 `S_AXI_HP0..HP3`，各自看到完整 2 GiB DDR_LOW，软件与 RTL 用同一套物理地址，无需换算。HP 口非一致，所以 PS 侧 D-cache 必须关（`main.c` 开头 `Xil_DCacheDisable()`）。

## 关键逻辑

**`sd_entry.S`：四步，顺序不能换。**
屏蔽 DAIF（未装向量表，且固件是轮询的）→ 读 `CurrentEL` 选栈顶（U-Boot 可能停在 EL2，`lscript.ld` 为三个 EL 各留了一段栈）→ 清 `.bss`（`NOLOAD`，加载器不搬，FATFS/FIL/XUartPs 对象和 4 MiB 分词表都在里面）→ 开 `CPACR_EL1.FPEN`（否则 newlib 的浮点路径会陷入）。

**不碰**时钟、复位、MMU、cache：PL 频率与 PL 复位由 FSBL/U-Boot 完成，U-Boot 跳转前已 `cleanup_before_linux()` 关掉 MMU 和两个 cache。重复配置会和启动链打架。

**`sd_prepare.c`：三段检查，逐级放行。**
ID 不匹配只有三种可能——PL 未配置、CSR 窗口未译码、`board_parameters.h` 的基址和 BD 地址编辑器不一致，所以带重试读；VERSION 挡住比固件更旧的 bitstream；IDLE 在 CLEAR 之后检查，避免上一轮残留任务被当成本次首个请求。

**`uart_console_override.c`：为什么覆盖 `_write` 而不是 `outbyte`。**
`nm` 实测：`outbyte` / `inbyte` 在 `libxil.a(xuartps_hw.c.obj)` 是 **T（强）**，而 `main.c` 必然拉这个目标文件（直接用 `XUartPs_*`），再定义一次就是重复定义；`_write` / `_read` 在 `libxilstandalone.a` 是 **W（弱）**，给出强定义即可接管。收益是 `printf` 不再依赖 `XUartPs_CfgInitialize()` 的成败——否则初始化失败时连报错信息都打不出来。

## 构建与运行

需要 `aarch64-none-elf-gcc`（上板）和 `gcc`（宿主）。工具路径通过环境变量或 `-Gcc` 传入。

```powershell
# 上板固件
$env:QWEN_A53_CC = 'D:\arm\bin\aarch64-none-elf-gcc.exe'
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_a53.ps1
# -> build_a53\QWENSD.ELF

# 宿主模拟器
$env:QWEN_HOST_CC = 'D:\mingw\bin\gcc.exe'
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_sim.ps1
# -> build_sim\qwen_board_sim.exe

# 跑一次（PL token 为脚本注入，不做模型计算）
$env:SIM_SD   = '<含 QWEN25\TOKEN.BIN 的目录>'
$env:SIM_PLAN = '<每行一串 token id>'
'hi' | .\build_sim\qwen_board_sim.exe
```

`build_a53.ps1` 会断言 `bsp/lscript.ld` 含 `ENTRY(sd_entry)` 与 `ORIGIN = 0x00100000`，再把栈/堆改成 1 MiB / 16 MiB，与归档工程的 `build.py a53` 完全一致。

## 验证状态

| 项 | 结果 |
| --- | --- |
| A53 固件构建 | **通过**，`-Wall -Wextra -Werror` 零警告 |
| 产物 | `build_a53/QWENSD.ELF`，1 349 440 B |
| SHA-256 | `8bc58c7ae04c32ee20a41469c7f707a7773081db0aab49d1e4b7a84d096aa944` |
| 类型 / 入口 | `EXEC` AArch64 / `0x00100000` = `sd_entry` |
| 符号归属 | `_write` / `_read` → `uart_console_override.o`；`outbyte` 仍来自 `libxil.a` |
| 宿主模拟器 | **通过**，一次完整请求：30 prompt token → 20 输出 token，`status=0` |
| 上板 | **未验证**。本交付包没有 candidate9 的完整 BD 布局布线结果，`QWEN25.BIN` 也未生成 |

上板前必须先有 candidate9 的 BD 与 bitstream，否则 `0x8000_0000` 这个窗口在真实器件上并不存在。这一点比软件本身更关键。
