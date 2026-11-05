# qwen25_qkv_controller 归一化地址对齐修复

接口、时钟和复位沿用部署模块。`clk` 为共同计算时钟，`rst_n` 为同步低有效复位。

参数载入边界也作如下修复：`stream_checked_data` 已登记一拍，但 Scale/Bias/Gamma 的 `load_last` 原来直接使用未登记的 `rd_last`。标记可能落在倒数第二个数据字，提前宣布 loaded，Gamma 最后一字 8 个 BF16 丢失。新增一位 `stream_checked_last` 与已有 data/valid 同拍登记，并将三表的 last 输入接到该位。没有改变 payload 流水级数；终止标记与现有数据保持相同延迟。验证需比较参数 RAM 全部内容，尤其 Gamma 最后16字节，并检查最终归一化全部896元素及LM Head logits。

问题：`S_NORM_SEND` 退出时登记递增命令，下周期 `S_NORM_READ` 读 RAM；原 `norm_word` 寄存器到该读周期末尾才改变。因此实际读取旧字，而 AFU metadata 对应下一字。统计到应用阶段切换也会读到末尾旧字，导致归一化向量循环右移一个 128-bit 字（8 个 BF16 元素）。

修复：保留登记后的 `norm_word_cmd_pending`；由存储值与 pending clear/increment 计算当前可见的 `norm_word`。可见地址在 `S_NORM_READ` 开始时已经推进，存储在下个沿接收该可见值。clear 优先于 increment，最大边界仍由原控制事件限制。接口和阶段状态数不变。

周期契约：

| 周期 | 状态 | pending | 存储地址 | 对 RAM/metadata 可见地址 |
| --- | --- | --- | --- | --- |
| A | SEND 接受 | 无 | w | w |
| B | SEND 退出 | 无 | w | w |
| C | READ | increment | w | w+1 |
| D | WAIT | 无 | w+1 | w+1 |

验证：相同输入“你好”ID 108386；从实际 PS C 经 AXI-Lite 启动候选 RTL，捕获首层归一化、K/V/Q、Context、H/M/Y。与部署权重的独立数学模型比较。修复前归一化与参考右移8元素一致；修复后应直接对齐。另运行完整24层 prefill/decode、CSR argmax 校验和 UTF-8 事件输出。保留生产源备份与候选源，不把候选仿真解释为既有 bitstream 已更新。

```json5
{signal: [
  {name:'clk', wave:'p....'},
  {name:'state', wave:'===== ', data:['SEND accept','SEND exit','READ','WAIT','SEND']},
  {name:'pending_increment', wave:'0.10.'},
  {name:'stored_word', wave:'=..=.', data:['w','w+1']},
  {name:'visible_word', wave:'=.=..', data:['w','w+1']}
]}
```
