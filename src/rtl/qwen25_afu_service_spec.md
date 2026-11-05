# qwen25_afu_service 重复常量驱动修复

接口、时钟、复位、握手和数据路径沿用原模块。`internal_error` 在服务模块中固定为0，内部 AFU 的 `o_error` 也固定为0。原接线将这两个常量源接到同一输出，Vivado 报告 Synth 8-6859/6858 多驱动。

修复只断开内部 AFU 的未使用 `o_error` 输出，保留服务模块已有的唯一常量赋值。输出仍逐周期为0，没有增加寄存器、组合逻辑或数据延迟。重跑加速器综合确认该多驱动告警消除，并由系统回归检查相同运算结果。

```json5
{signal:[
  {name:'clk',wave:'p....'},
  {name:'rst_n',wave:'01...'},
  {name:'internal_error',wave:'0....'},
  {name:'request_valid',wave:'0.10.'},
  {name:'result_valid',wave:'0...1'}
]}
```
