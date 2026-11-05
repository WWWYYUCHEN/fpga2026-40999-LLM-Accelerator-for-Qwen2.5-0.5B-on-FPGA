# 原生测试向量实板复测

当前状态：**自动启动已实测到 READY / prompt>**。U-Boot 2020.01 自动发现 /boot.scr，加载原 PL 和 QWUART.ELF，分词自检及全部模型加载通过；本段持续接收器 TX=0，没有主机启动命令或倒计时中断。固定证据见 [自动启动到 READY 日志](automatic-boot-to-ready.log) 和 [验收数据](automatic-boot-verification.json)。这是已观察重启路径的自动启动验收，未另做完整断电冷启动重复或稳定性测试。此前 U-Boot 2022.01 路径出现 SD 检测不稳定，仍保留为观察限制。

## 1. 变更范围

正式 SD 部署使用本目录对应的 `boot.cmd`、`boot.scr` 和已通过功能测试的 `QWUART.ELF`，并设置明确的自动启动标识。备份、说明和校验记录均集成在本目录。

| 项目 | 配置 |
| --- | --- |
| 启动链 | 板载既有 QSPI 引导 → U-Boot 自动扫描 SD → /boot.scr → PL 配置 → PS 裸机应用 → 加载模型 → READY |
| 自动启动标识 | Qwen25 KV260 AUTOBOOT QWUART（原始串口日志保留完整字符串） |
| PL | QWEN25.BIN；加载地址 0x10000000；fpga load |
| 应用 | QWUART.ELF；文件加载地址 0x18000000；bootelf -p；已知 ELF 入口 0x00110400 |
| SD 搜索 | 沿用交付脚本：先 mmc 1:1，失败后尝试 mmc 0:1 |
| 模型目录 | 0:/QWEN25/，保持原文件不变 |
| 交互界面 | COM11，115200、8N1、无流控；串口显示 READY 后输入提示词 |
| 上电后是否自动生成 | 不自动发送提示词或跑基准；自动加载完成后等待用户输入 |
| 不需要 | 手动 fatload / fpga load / bootelf、Linux、SSH；独立启动本身不依赖电脑 |

未修改 QSPI、U-Boot 永久环境、分区、模型或 PL，未重新编译应用或 BSP。正式部署包只包含 `QWUART.ELF`。生成 boot.scr 是启动脚本封装，不是重新编译固件。

## 2. 文件身份与校验

| 文件 | 新大小 / 字节 | 新 SHA-256 |
| --- | ---: | --- |
| boot.cmd | 545 | 1d6082144a8c659cdbb14f232b74455bc4a38e766bb037822a73835db3045cdc |
| boot.scr | 617 | 32c129004f62d2c00f7da829465beb3c9cf74d00d44b6b5841703c474d4881f5 |
| QWUART.ELF（内容未改） | 999656 | 0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c |

- boot.scr 沿用交付脚本的 U-Boot legacy image 格式及平台字段，重新计算长度表、数据 CRC32 和头部 CRC32；封装头的日期为固定复现时间，不代表实际部署时刻。
- 两个启动文件保留 FAT Archive 属性，避免旧 U-Boot 忽略文件。
- 写卡前后核验原部署清单、候选 ELF 及当前 CHAT.CSV：仅 boot.cmd、boot.scr 的内容变化，其余 13 个已记录文件内容不变。
- 目标卡身份及所有前后哈希见 [sd-deployment.json](sd-deployment.json)。写卡后的自动启动验收另见 automatic_boot_verification 字段；完整断电冷启动重复未独立确认。
- 原脚本来自当前 SD 卡的实际副本，位于 [original-sd-launcher/boot.cmd](original-sd-launcher/boot.cmd) 和 [original-sd-launcher/boot.scr](original-sd-launcher/boot.scr)，备份已与改动前哈希核对一致。

## 3. 每次正常使用

1. 板子断电时插好 SD 卡；如需操作界面，连接 USB 数据线并打开唯一一个串口终端。
2. 给 KV260 上电，不在 U-Boot 倒计时按键。
3. 观察自动执行 /boot.scr、上述 AUTOBOOT 标识、FPGA configured、应用标题、TOKENIZER_SELFTEST_OK 和模型 Loading 输出。
4. 等待全部模型资源加载完成，出现 READY / prompt> 后输入提示词。加载期间没有提示符不等于故障，应结合串口日志判断。

重要：原来的带 -InterruptAutoboot 监听器会自动发送空格，必须停用。此次新监听器未启用该参数，命令槽为空对象，不自动发送启动或推理命令。不要同时打开第二个程序占用 COM11。当前显示型监听窗口不把用户键盘输入转发到串口；要直接手动交互，应先关闭该监听再用串口终端连接。

## 4. 自动启动验收结果与范围

已在不发送主机启动命令、不打断自动倒计时的条件下观察到以下完整链路，成功段运行于 FSBL 2020.2 / U-Boot 2020.01：

- U-Boot 自动发现并执行 SD 上的 boot.scr。
- 新 AUTOBOOT 标识，随后 PL 配置成功并启动 QWUART.ELF（入口应为 0x00110400）。
- 应用输出预期 CSR ID / VERSION、分词自检通过、全部资源加载完成，最后进入 READY。

原监听证据保留于 [uart-live.log](uart-live.log)。新接收器证据见 [native-live.log](native-live.log) 和 [配置及心跳](native-live.json)。可见窗口标题为 KV260 COM11 native UART，不启用命令发送或倒计时中断。前置独立诊断曾发送一次 CR，单独记录在 native-com11-diagnostic.*；后续持续采集从新的原生句柄开始，TX=0，已捕获完整自动启动到 READY。

本次固定日志从原始字节的成功启动段提取，未经应用指令干预。原始片段 2,339 字节，SHA-256 为 4ec89fe0260d908536cbe47df1ee041c3d620daff2686f6545e6b8c2a7e861b1；字节偏移与全部检查点保存在 automatic-boot-verification.json。固定文本日志将换行规范化，乱码字节以转义表示，因此文本文件哈希不等于原始字节片段哈希。

观察限制：U-Boot 2022.01 路径曾报 Sdhci card detect state not stable / probe failed: -110 / no mmc device at slot 1 并回退网络启动；本次随后观察到的 U-Boot 2020.01 路径能正常识别 SD 并自动启动。未改写或切换 QSPI 固件，不据此宣称两种引导路径都可靠。串口启动阶段也记录到 BREAK/FRAME 标志，完整应用输出可读，未将其归因于特定硬件故障。监听器仅更换为原生接收与真实可见窗口，不能证明旧 .NET 监听曾异常的具体机制。

## 5. 回退与复现

原始启动脚本备份仅用于审计，不进入正常运行路径。正式复现应使用本目录 `../deployment/` 中的 `boot.cmd`、`boot.scr` 和 `QWUART.ELF`。

复现脚本在 `board/tools/`：

- `prepare_kv260_autoboot.py`：读取本目录中的原始启动脚本，生成 boot.cmd / boot.scr，并验证封装及 CRC；`--verify` 只校验生成产物。
- `deploy_kv260_autoboot.ps1`：识别目标卡、校验全部已有资源、备份原脚本、替换两文件并回读核验；已存在本轮备份或部署记录时拒绝再次执行，避免覆盖证据。

本目录属于启动配置记录，与报告一“实板功能验证”、报告二“decode 长度扫描”分别保留，不覆盖历史测试结论。任务未产生下载分片、安装包或固件构建中间文件；串口监听属于仍在使用的验证工具，保留。
