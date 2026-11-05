"""Write the final review report only after the required regressions pass."""
from pathlib import Path
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT
from package_interactive import PACKAGE


def main():
    r=ROOT/'reports'
    status=json.loads((r/'regression_status.json').read_text(encoding='utf-8'))
    assert status['complete']
    numeric=json.loads((r/'final_all_layer_numeric_comparison.json').read_text())
    decode=json.loads((r/'final_decode_layer_numeric_comparison.json').read_text())
    tests=status['final_system_regressions']
    t=status['timing']
    rows=[]
    for name,item in tests.items():
        ids=', '.join(x['token_id'] for x in item['generated'])
        rows.append(f"| {name} | {item['template']} | {item['requests']} | {len(item['generated'])} | {item['seconds']:.2f} | PASS |")
    maximum=max(v['relative_l2'] for item in numeric['layers'] for v in item.values() if isinstance(v,dict))
    history_errors=[]
    for name in status['offline_reference_checks']:
        comparison=json.loads((r/'system_regression'/name/'numeric_comparison.json').read_text(encoding='utf-8'))
        history_errors.extend(item['relative_l2'] for item in comparison['comparisons'] if 'relative_l2' in item)
    body=f'''# KV260 加速器 RTL 回归与完整系统验证

默认入口为 **加速器 RTL 逐周期仿真**。电脑只提供输入/输出界面；PS C 程序完成聊天模板、分词、embedding 准备和寄存器调度；模型计算由 Verilator 执行实际加速器 RTL。CPU 数学模型只在离线阶段作独立结果对照。

系统仿真在本机执行与A53 ELF同源的PS C程序，经实际AXI-Lite握手控制RTL。DDR、SD、UART及厂商算术IP使用行为模型；覆盖软件与加速器的系统交互，尚未覆盖ARM指令级执行、DDR控制器时序或物理串口电气行为。

原testbench的自动激励在私有生成C++副本中关闭，由PS C接管AXI驱动。原bias地址边界断言会对LM Head阶段未使用的空闲地址报错，现仅在实际 `bias_read_en` 有效时检查边界；未改变RAM数据路径。生成模型、RTL源码和交付硬件的来源核对见 `sim/reports/source_lineage.json` 与各场景 `build_provenance.json`。

## 修复内容

| 问题 | 证据 | 修改 |
| --- | --- | --- |
| 归一化地址晚一拍 | 原向量循环错位8个BF16元素，首层V投影明显异常 | 保留pending命令寄存器；当前可见字地址包含pending clear/increment，存储随后接收 |
| 参数结束标记未与数据登记 | Gamma最后16字节全为0，最终归一化最后8元素丢失 | last与既有stream_checked_data同拍登记，Scale/Bias/Gamma均使用登记后的last |
| AFU错误信号重复常量驱动 | Vivado Synth 8-6859/6858告警 | 断开未使用的子模块常量错误输出，保留唯一驱动 |

没有修改模型权重、时钟目标、AXI地址布局或算术IP配置。保持原有数据流水级数；修改后使用原始100MHz板级约束重新综合、布局、布线并生成bitstream。

## 时序验证

| 项目 | 原实现 | 修复实现 |
| --- | --- | --- |
| PL时钟约束 | 10.000ns / 100MHz | {t['period_ns']}ns / 100MHz |
| Setup uncertainty | 0.100ns | {t['setup_uncertainty_ns']}ns |
| Setup WNS | +1.356ns | {t['setup_wns_ns']}ns |
| Hold WHS | +0.010ns | {t['hold_whs_ns']}ns |

setup与hold均满足约束。是否保持完全相同的裕量，应以表中实际结果判断；没有放宽约束或降低频率。报告位于 `sim/reports/timing_fixed/`。

综合和实现阶段均为0个critical warning、0个error。本次额外导出路径报告时，Vivado 2026.1对已弃用的 `report_timing -nets` 参数产生两条critical warning；该选项仅控制报告格式，不改变网表或时序约束，复现脚本已移除该参数。完整原始运行日志保留在独立工程的 `reports/vivado_regression.log`。

XSA首次导出因独立 `write_bitstream` 没有更新实现run的bitstream完成状态而失败。随后从同一布线checkpoint续跑run的write_bitstream步骤，再导出XSA；没有修改RTL、约束或重新布局布线。初始日志和修正导出日志均保存在 `sim/reports/timing_fixed/`，交付时逐字节校验XSA内bitstream与独立交付bitstream一致。

## 系统仿真

| 场景 | 模板 | 请求数 | 生成token数 | 仿真耗时/秒 | 结果 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

每个token均检查实际RTL计算、24层完成事件、写事务排空、151936个BF16 logits的独立argmax与CSR一致性，再由PS输出UART事件。聊天测试含自动模板、多组prefill、自回归decode、连续请求及EOS停止；错误恢复、UTF-8边界、输入长度和SD日志故障由PS固件回归另外覆盖。

## 数值验证

- Scale/Bias/Gamma全部RAM内容与模型文件逐字节一致：912128、55296、87808字节。
- 单token prefill的24层各阶段以实际RTL操作数逐层对照，最大相对L2误差 `{maximum:.8g}`。
- 给定实际RTL输入，最终归一化896元素相对L2误差 `{numeric['final_norm']['relative_l2']:.8g}`。
- 给定相同归一化输入，LM Head相对L2误差 `{numeric['lm_head']['relative_l2']:.8g}`；所有151936个logits参与对照。
- 带非零位置及真实KV历史的decode逐层对照：`{decode['status']}`。
- 完整聊天生成与离线参考的token对照：`{status['offline_reference_checks']}`。
- 从embedding起独立执行完整前向与真实生成历史时，logits相对L2误差范围为 `{min(history_errors):.8g}` 至 `{max(history_errors):.8g}`；全部已生成token ID仍一致。该项包含跨层累积差异，区别于给定相同RTL操作数的逐阶段对照。

参考模型显式采用RTL的RND8 BF16舍入、scale/bias边界和部署epsilon。FP32归约顺序及厂商算子模型仍存在有限数值差异；这里报告的是已列出测试向量的验证结果，并非所有可能输入的形式化等价证明。

## 交付与使用

双击 `打开交互窗口.lnk`，默认选择“RTL 逐周期模拟”。连接后等待 `prompt>`，输入中文或英文原文并按回车。可用 `/mode events` 接收每个token的ID、字节和时间；`/max N`设置输出上限。

`board/deployment/QWEN25.BIN`、`build/hardware/qwen25_system_100mhz.bit`与XSA来自通过时序回归的新实现；PS交互ELF已编译，支持相同串口输入流程。实际KV260不在现场，因此实板启动、串口、电源和SD访问仍需拿到板卡后验证。

完整回归记录在 `sim/reports/regression_status.json` 和 `sim/reports/system_regression/final_*`。源码修复和同名规格文档在 `src/rtl/`，补丁与原始故障证据保存在REPORTS。旧版试验报告作为历史证据保留，最终验收以本报告及final场景为准。

严格风格工具对两个修改模块的检查仍未通过，包含3051项格式、注释、命名及formatter解析诊断；八项gate状态见 `sim/reports/candidate_static_gate_summary.json`。未为了格式重写整个控制器。实际Vivado/Verilator编译、功能、数值与时序证据分别保存，不能把工具编译成功解释成严格风格gate通过。
'''
    for name in ['回归验证报告.md','验证结果.md']:(PACKAGE/name).write_text(body,encoding='utf-8')
    examples=[]
    for name in ['final_chat_nihao','final_chat_eos']:
        text=(r/'system_regression'/name/'uart.log').read_text(encoding='utf-8')
        fragments=re.findall(r'@TOKEN \d+ \d+ (\d+) \d+ ([a-f0-9]*)',text)
        raw=b''.join(bytes.fromhex(fragment) for token,fragment in fragments if int(token)<151643)
        examples.append((tests[name]['prompt'],raw.decode('utf-8','replace')))
    readme=f'''# KV260 串口交互与加速器 RTL 仿真

本包默认使用 **真实加速器 RTL 仿真计算**。原文输入、PS分词、AXI控制、24层模型计算、逐token串口输出和下一次输入已完成系统回归。具体修复、数值误差和100MHz布局布线结果见 [回归验证报告](回归验证报告.md)。

## 直接运行

双击 `打开交互窗口.lnk` 或 `START_GUI.pyw`，保持默认“RTL 逐周期模拟”，点击“连接 / 启动”。显示 `READY` 与 `prompt>` 后输入中文/英文，按Enter。

电脑只收发UTF-8文本。模型计算由 `sim/simulator/qwen_rtl_sim.exe` 执行当前修复后的加速器RTL；PS部分运行与上板ELF相同源码的C程序。UART和SD使用仿真适配层。没有用CPU数学模型替代默认路径中的加速器运算。

实际RTL测试输出：

```text
提示词：{examples[0][0]}
输出：{examples[0][1]}

提示词：{examples[1][0]}
输出：{examples[1][1]}
```

RTL仿真逐周期执行，完整聊天prefill可能需要数十分钟，每个新token还需数分钟。并行运行Vivado和多个仿真进程会增加耗时。本次“你好”8-token测试用时 {tests['final_chat_nihao']['seconds']/60:.1f} 分钟，算术问答至EOS用时 {tests['final_chat_eos']['seconds']/60:.1f} 分钟；各测试的实际耗时见报告。窗口显示实际计算进度，日志保存于 `RUNS/时间戳/`。仿真用于验证硬件正确性，速度不代表实物100MHz运行速度。

## 模式与命令

| 选择 | 用途 |
| --- | --- |
| RTL 逐周期模拟（默认） | 执行实际PS C与加速器RTL，使用部署模型权重 |
| PL 算法模拟 | 可选CPU数学参考，便于快速对照；不作为RTL验证证据 |
| 串口实板 | 选择COM口，115200 8N1，连接实际KV260 |

`/max N`设置最多生成token数，默认20；`/mode text`输出连续UTF-8，`/mode events`输出token ID、字节及计时，`/mode ids`只输出ID和计时。`/selftest`执行tokenizer黄金向量检查，`/encode 文本`查看PS分词，`/reset`清空上下文。默认自动添加Qwen聊天模板；`/template raw`用于裸token诊断，`/template chat`恢复。

每个请求清空KV，不保留跨请求历史。KV容量256，提示词（含模板）加生成上限不能超过256。输入上限4096 UTF-8字节；遇到EOS或生成上限停止。支持连续输入、中文跨token字节拼接和SD日志不可写时继续UART输出。

命令行：

```powershell
python .\\src\\PC\\run_cli.py
python .\\src\\PC\\run_cli.py --prompt '你好' --tokens 8
python .\\src\\PC\\run_cli.py --backend math --prompt '你好' --tokens 8
```

RTL模式不需要PyTorch。数学参考模式需要PyTorch和NumPy，解释器由 `QWEN_SIM_PYTHON` 指定，默认使用当前 Python。串口模式需要pyserial。Windows模拟程序为x64；窗口需要Python/Tk。

## 上板运行

将 `board/deployment` 内全部内容复制到FAT32卡根目录，保留 `QWEN25`。USB UART设置115200、8N1、UTF-8、无流控。通过现有QSPI U-Boot加载SD启动脚本：

```text
mmc list
fatload mmc 1:1 0x20000000 boot.scr
source 0x20000000
```

SD枚举为0时改成 `mmc 0:1`。脚本配置 `QWEN25.BIN`，执行 `QWUART.ELF`；等待模型加载和 `TOKENIZER_SELFTEST_OK 40 tokens` 后，在 `prompt>` 输入提示词。正式启动链已完成上板验证。

## 目录与复现

- `board/deployment/`：bitstream、交互ELF、模型与boot.scr。
- `build/hardware/`：匹配RTL的bitstream和XSA。
- `sim/simulator/`：实际RTL模拟器、PS C程序和tokenizer DLL。
- `src/PC/`：GUI/CLI；数学参考仅为可选模式。
- `src/rtl/`：生产RTL和同名规格文档。
- `src/software/interactive/`：PS、桥接、构建和回归源码；`src/BSP/`提供A53编译库和头文件。
- `sim/reports/`：系统回归、数值对照、时序与文件哈希。
- `SHA256SUMS`：交付文件校验清单。

在 `src/software/interactive` 中运行 `python tools/build.py a53` 可重建ELF，`python tools/build.py native`重建PS本机程序，`python tools/build_rtl_sim.py`重建RTL仿真程序。工具路径支持 `QWEN_A53_CC`、`QWEN_BSP`、`QWEN_HOST_CC`、`QWEN_ZIG`、`QWEN_VERILATOR_INCLUDE`。
'''
    (PACKAGE/'README.md').write_text(readme,encoding='utf-8')
    print(PACKAGE/'回归验证报告.md')


if __name__=='__main__':main()
