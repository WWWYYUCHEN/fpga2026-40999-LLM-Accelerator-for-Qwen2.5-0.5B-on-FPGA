# 上板工具

本目录包含项目内可直接使用的串口、SD 卡校验、回归和启动脚本复现工具。

| 文件 | 用途 |
| --- | --- |
| `start_kv260_terminal.bat` | 打开 COM11 交互终端 |
| `start_kv260_terminal.ps1` | 串口终端实现 |
| `kv260_uart_native.py` | 原生串口接收与命令槽 |
| `run_kv260_native_regression.py` | 顺序执行 6 类实板回归场景 |
| `summarize_kv260_board_regression.py` | 汇总串口记录与参考结果 |
| `verify_kv260_sd.ps1` | 核对 SD 卡与部署目录的路径、大小和 SHA-256 |
| `analyze_kv260_length_scan.py` | 复核 decode 长度扫描数据 |
| `prepare_kv260_autoboot.py` | 从包内原始脚本重新生成启动脚本 |
| `deploy_kv260_autoboot.ps1` | 校验并部署启动脚本到指定 FAT32 卡 |

所有默认路径均从脚本位置推导到项目根目录，不依赖开发机目录。
