# 构建与实现

| 路径 | 内容 |
| --- | --- |
| `scripts/build_candidate9.ps1` | 端到端仿真构建（RTL + 行为级模型 + 固件 + 桥接），需新 `-Tag` |
| `scripts/run_candidate.ps1` | 运行仿真程序，指定模板、长度、prompt |
| `scripts/fw/`、`scripts/verification/` | 固件与验证的构建脚本副本 |
| `vivado/create_rtl_project.tcl` | 生成 `rtl_project/candidate9.xpr`（器件 `xck26-sfvc784-2LV-c`） |
| `vivado/ip_package.tcl` | 把加速器打包成 IP（输出 `package_candidate9/`） |
| `vivado/bd_build_ps.tcl` | 重建 KV260 PS Block Design |
| `vivado/clk.xdc` | 100 MHz 顶层 OOC 时钟约束 |
| `vivado/address_map.json` / `.h` | 寄存器与地址段定义 |
| `reports/timing_summary.rpt` | GEMM 局部综合时序（含 −5.238 ns 违规路径） |
| `reports/utilization.rpt` | 对应资源利用率 |
| `logs/create_rtl_project.log` | 工程创建日志 |

## 已知限制

- `clk.xdc` 的 100 MHz 是 **OOC 时钟约束，不是闭合保证**。
- 300 MHz 约束下 GEMM 局部综合 **WNS = −5.238 ns**，未收敛。
- **没有完整的 BD 布局布线结果**，200 MHz 与 300 MHz 均未验证通过。
- **没有 bitstream**，也没有 XSA。本分支不可直接上板。
- `bd_build_ps.tcl` 依赖本机 KV260 board preset 路径，换机器需先装好 board files。

## 工具路径

`build_candidate9.ps1` / `run_candidate.ps1` 默认使用：

```text
C:\Users\ZKHC\tools\oss-cad-suite
C:\Users\ZKHC\tools\ziglang_verilator\ziglang\zig.exe
```

换机器必须修改。
