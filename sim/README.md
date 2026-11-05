# 仿真与验证

| 目录 | 内容 |
| --- | --- |
| `tb/` | 18 个自检测试平台：GEMM 列/阵列流水、KV8 量化各阶段（`tb_kv8_*`）、EXP/ADD 对照（`tb_exp_compare.vhd`、`tb_add_compare.vhd`）、结果通路无 IRQ 版本 |
| `scripts/` | `build.ps1`、`run.ps1`、`run_custom.ps1`、`preflight.ps1` 驱动单元 TB；`verify_ab.py` 做 A/B 数值比对；`srcs_B.txt`、`segments.tsv` 为源清单与地址段 |
| `candidate9_chat_result/` | 已完成的一次 6 token chat 回归的完整证据 |
| `simulator/` | `qwen_candidate9.exe`，已构建好的端到端仿真程序 |

## candidate9_chat_result

| 文件 | 说明 |
| --- | --- |
| `hardware_tokens.csv` | 6 个 token 的 id 与周期数 |
| `hardware_logits_*.bin` | 每个 iteration 的 151936 维 logits（BF16 原始位，303,872 B/个） |
| `run.out` | 仿真完整控制台输出，含 `RTL_TOKEN_VERIFIED` |
| `in.txt`、`segments.tsv`、`timing.txt` | 输入、地址段映射、计时 |

token 序列：`9707, 0, 2585, 646, 358, 7789`；首轮 logits SHA-256 为
`BBA14D322C6F33940DBCEC531E7F0B3A97C38E650A0212880581A664AA39DE95`。

## 注意

这些结果产生于交付目录复制之前，**不代表重新构建后的新验证**。重跑请按根目录 README 的两条命令，
并使用新的 `-Tag` 与新的运行目录（脚本拒绝覆盖已存在的对象目录）。
