# 上板资源

本分支**没有上板实测记录**（时序未闭合、无 bitstream），这里只提供 SD 卡所需的模型资源与 PS 固件。

| 路径 | 内容 |
| --- | --- |
| `sd/QWEN25/` | 模型资产：`WEIGHTS.BIN`(341 MB)、`EMBED.BIN`(260 MB)、`LMW.BIN`(130 MB)、`TOKEN.BIN`、`SCALE.BIN`、`GAMMA.BIN`、`BIAS.BIN`、`DESC.BIN` |
| `sd/CHAT.CSV` | 聊天模板/对照表 |
| `firmware/QWENSD.ELF` | PS 侧固件（1.29 MB），由 `src/software/build_a53.ps1` 生成 |

三个大文件由 **Git LFS** 托管，克隆后必须执行 `git lfs pull`，否则拿到的是指针文件。

## 使用前须知

- 顶层 IRQ 已移除，固件与 PL 的握手方式与早期交付不同；若使用旧版 bitstream 会不匹配。
- 模型资产的地址映射见 `build/vivado/address_map.json`。
- 要把这套资源真正跑起来，还需要一份时序闭合的 bitstream——本仓库没有。
