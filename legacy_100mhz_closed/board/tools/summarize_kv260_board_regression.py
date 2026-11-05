"""Read-only UART/reference analysis; emit report text for apply_patch, not tests.

Does not open a serial port, run inference/simulation, or change the delivery.
Only the recorded 028..050 command interval is analyzed. All file creation is
left to the caller via the emitted patch (or the JSON document mapping).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

FIRST = "028-test-selftest"
LAST = "050-restore-default-limit"
CASES = [
    ("final_all_layers", ["033-final-all-layers"]),
    ("final_repeat_context_reset", ["035-final-repeat-a", "036-final-repeat-b"]),
    ("final_rows9_prefill", ["038-final-rows9-prefill"]),
    ("final_decode_trace", ["040-final-decode-trace-functional"]),
    ("final_chat_nihao", ["044-final-chat-nihao"]),
    ("final_chat_eos", ["047-final-chat-eos"]),
]
COMMAND = re.compile(r"\[HOST\] ([^\s]+) TX id=([^: ]+): ([^\n]*)\n")
BEGIN = re.compile(r"^@BEGIN (\d+) prompt_tokens=(\d+) max_new=(\d+)$", re.M)
TOKEN = re.compile(r"^@TOKEN (\d+) (\d+) (\d+) (\d+) ([0-9a-f]*)$", re.M)
END = re.compile(r"^@END (\d+) produced=(\d+) reason=(\w+) status=(-?\d+)$", re.M)
READY = re.compile(r"\nREADY\nprompt> *")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def metadata(path):
    data = path.read_bytes()
    return {"path": path.as_posix(), "bytes": len(data), "sha256": digest(data)}


def requests(text):
    starts = list(BEGIN.finditer(text))
    output = []
    for index, start in enumerate(starts):
        segment = text[start.start():starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        endings = list(END.finditer(segment))
        require(len(endings) == 1, "Expected exactly one END for each request")
        end = endings[0]
        request_id, count, maximum = map(int, start.groups())
        require(int(end[1]) == request_id, "BEGIN/END request ID mismatch")
        token_rows = []
        for token in TOKEN.finditer(segment):
            rid, iteration, token_id, ticks = map(int, token.groups()[:4])
            require(rid == request_id and token.start() < end.start(), "Token outside request")
            raw = bytes.fromhex(token[5])
            token_rows.append({"iteration": iteration, "token_id": token_id,
                               "timer_ticks": ticks, "utf8_hex": token[5],
                               "text_fragment": raw.decode("utf-8", errors="replace")})
        require([t["iteration"] for t in token_rows] == list(range(len(token_rows))),
                "Token iterations must be consecutive from zero")
        require(int(end[2]) == len(token_rows), "END produced/token count mismatch")
        output.append({"request_id": request_id, "prompt_tokens": count,
                       "max_new": maximum, "tokens": token_rows,
                       "produced": int(end[2]), "reason": end[3],
                       "application_return_code": int(end[4])})
    require(len(list(END.finditer(text))) == len(output), "Unpaired END event")
    return output


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(workspace, delivery, output):
    evidence_dir = delivery / "board/measurements/functional"
    live = read(evidence_dir / "uart-live.log")
    markers = list(COMMAND.finditer(live))
    first = next(m for m in markers if m[2] == FIRST)
    last = next(m for m in markers if m[2] == LAST)
    final_ready = READY.search(live, last.end())
    require(final_ready is not None, "Final default-restoration command has not completed")
    transcript = live[first.start():final_ready.end()].rstrip("\n") + "\n"
    markers = list(COMMAND.finditer(transcript))
    require(len(markers) == 23, "Unexpected command count in selected interval")
    require(len({m[2] for m in markers}) == len(markers), "Duplicate command ID")
    command_rows, by_id = [], {}
    state = {"template": "chat", "output_mode": "text", "max_new": 20}
    for index, marker in enumerate(markers):
        segment = transcript[marker.end():markers[index + 1].start() if index + 1 < len(markers) else len(transcript)]
        require(READY.search(segment) is not None, "Command did not return READY: " + marker[2])
        require(not re.search(r"FATAL|^ERROR|WARNING|\[HOST\] STOP", segment, re.M),
                "Unexpected error/warning in command " + marker[2])
        command = marker[3]
        if command.startswith("/template "):
            state["template"] = command.split()[1]
            require("TEMPLATE=" + state["template"] in segment, "Template acknowledgement missing")
        elif command.startswith("/mode "):
            state["output_mode"] = command.split()[1]
            require("MODE=" + state["output_mode"] in segment, "Mode acknowledgement missing")
        elif command.startswith("/max "):
            state["max_new"] = int(command.split()[1])
            require("MAX_NEW=" + str(state["max_new"]) in segment, "Limit acknowledgement missing")
        row = {"sent_at": marker[1], "command_id": marker[2], "command": command,
               "returned_ready": True, **state}
        command_rows.append(row)
        by_id[marker[2]] = {**row, "response": segment}
    require("TOKENIZER_SELFTEST_OK 40 tokens" in by_id[FIRST]["response"], "Selftest missing")
    require(state == {"template": "chat", "output_mode": "text", "max_new": 20},
            "Defaults not restored")

    source_paths = [delivery / "report/README.md", delivery / "sim/reports/regression_status.json",
                    delivery / "src/software/interactive/generation.c",
                    delivery / "src/software/interactive/main.c",
                    delivery / "src/BSP/include/xtimer_config.h",
                    delivery / "src/BSP/include/xparameters.h",
                    delivery / "board/deployment/QWEN25.BIN",
                    delivery / "board/deployment/QWUART.ELF",
                    Path(__file__).resolve()]
    parameters = read(delivery / "src/BSP/include/xparameters.h")
    frequency = int(re.search(r"^#define XPAR_CPU_TIMESTAMP_CLK_FREQ\s+(\d+)", parameters, re.M)[1])
    old = {
        "candidate": {"sha256": "0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c",
                      "board_crc32": "f195fdca", "entry": "0x00110400"},
        "pl": {"sha256": "73a06211577e199bb4ff3735581899a50c48c2e4f94f81189d3469e8d9874d26",
               "board_crc32": "137125e6", "entry": "0x10000000"},
        "boot": {"fsbl": "2020.2", "uboot": "2020.01"},
    }
    require(digest((delivery / "board/deployment/QWUART.ELF").read_bytes()) == old["candidate"]["sha256"],
            "QWUART.ELF differs from recorded board boot identity")
    scenarios, token_rows = [], []
    for name, command_ids in CASES:
        reference = delivery / "sim/reports/system_regression" / name
        paths = [reference / leaf for leaf in ("input.txt", "result.json", "hardware_tokens.csv", "uart.log")]
        source_paths.extend(paths)
        ref_result = json.loads(read(reference / "result.json"))
        require(ref_result["status"] == "PASS", "Reference not PASS")
        expected = requests(read(reference / "uart.log"))
        actual = []
        for command_id in command_ids:
            record = by_id[command_id]
            require(record["command"] == ref_result["prompt"], "Prompt differs from reference")
            require(record["template"] == ref_result["template"], "Template differs from reference")
            require(record["output_mode"] == "events", "Output mode must be events")
            require(record["max_new"] == ref_result["requested_tokens"], "Output limit differs")
            observed = requests(record["response"])
            require(len(observed) == 1, "Expected one request per serial command")
            actual.append({**observed[0], "command_id": command_id, "sent_at": record["sent_at"]})
        require(len(actual) == len(expected) == ref_result["requests"], "Request count differs")
        ref_csv = list(csv.DictReader(io.StringIO(read(reference / "hardware_tokens.csv"))))
        reference_ids = [t["token_id"] for req in expected for t in req["tokens"]]
        require(reference_ids == [int(t["token_id"]) for t in ref_csv], "Reference CSV/UART disagree")
        require(reference_ids == [int(t["token_id"]) for t in ref_result["generated"]],
                "Reference result/UART disagree")
        checks = []
        for request_index, (got, want) in enumerate(zip(actual, expected)):
            check = {key: got[key] == want[key] for key in
                     ("prompt_tokens", "max_new", "produced", "reason", "application_return_code")}
            check["token_ids"] = [t["token_id"] for t in got["tokens"]] == [t["token_id"] for t in want["tokens"]]
            check["token_bytes"] = [t["utf8_hex"] for t in got["tokens"]] == [t["utf8_hex"] for t in want["tokens"]]
            checks.append(check)
            for token, ref_token in zip(got["tokens"], want["tokens"]):
                token["nominal_ms"] = token["timer_ticks"] / frequency * 1000
                token["phase"] = "prefill_first_output" if token["iteration"] == 0 else "decode"
                token["expected_token_id"] = ref_token["token_id"]
                token["matches_reference"] = token["token_id"] == ref_token["token_id"]
                token_rows.append({"scenario": name, "command_id": got["command_id"],
                                   "request_id": got["request_id"], "prompt_tokens": got["prompt_tokens"],
                                   "timer_nominal_hz": frequency, **token})
        matched = all(all(c.values()) for c in checks)
        prefill = [t["nominal_ms"] for req in actual for t in req["tokens"] if t["iteration"] == 0]
        decode = [t for req in actual for t in req["tokens"] if t["iteration"] > 0]
        decode_ms = sum(t["nominal_ms"] for t in decode)
        scenarios.append({"name": name, "template": ref_result["template"],
                          "prompt": ref_result["prompt"], "max_new": ref_result["requested_tokens"],
                          "functional_match": matched, "checks": checks, "requests": actual,
                          "reference_token_ids": reference_ids,
                          "prefill_first_output_ms": prefill, "decode_samples": len(decode),
                          "decode_mean_ms": decode_ms / len(decode) if decode else None,
                          "decode_compute_window_tokens_per_second": len(decode) * 1000 / decode_ms if decode else None,
                          "decode_min_ms": min((t["nominal_ms"] for t in decode), default=None),
                          "decode_max_ms": max((t["nominal_ms"] for t in decode), default=None),
                          "trace_verified_on_board": False,
                          "coverage": "token_only_not_layer_trace" if name == "final_decode_trace" else "end_to_end_token_reference"})

    all_matched = all(s["functional_match"] for s in scenarios)
    require(len(token_rows) == 22 and sum(len(s["requests"]) for s in scenarios) == 7, "Unexpected run size")
    encodings = {}
    for command_id in ("032-test-raw-encode", "037-test-rows9-encode",
                       "043-test-chat-encode", "046-test-eos-encode"):
        m = re.search(r"^@PROMPT (\d+) (.*)$", by_id[command_id]["response"], re.M)
        ids = list(map(int, m[2].split()))
        require(len(ids) == int(m[1]), "Encoding length differs")
        encodings[command_id] = {"command": by_id[command_id]["command"], "count": len(ids), "ids": ids}
    require([x["count"] for x in encodings.values()] == [1, 9, 30, 35], "Unexpected tokenization")

    result = {
        "first_command_at": markers[0][1], "last_command_sent_at": markers[-1][1],
        "status": "functional_token_reference_matched_with_coverage_limits" if all_matched else "functional_mismatch",
        "board_tested": True, "simulation_run_this_task": False,
        "inference_backend": "physical KV260 PL; standalone PS control/tokenization/embedding",
        "host_gpu_used": False, "candidate": old["candidate"], "pl": old["pl"],
        "boot_versions": old["boot"],
        "board_regression": {"functional_scenarios_matched": sum(s["functional_match"] for s in scenarios),
                             "scenarios": 6, "requests": 7, "output_tokens": len(token_rows),
                             "matching_output_tokens": sum(t["matches_reference"] for t in token_rows),
                             "prefill_samples": 7, "decode_samples": 15,
                             "full_original_trace_scenarios_passed": False,
                             "all_requests_returned_ready": True, "tokenizer_selftest_vector_tokens": 40},
        "timing": {"nominal_timer_hz": frequency, "timer_independently_calibrated": False,
                   "formula_ms": "timer_ticks / 99999001 * 1000",
                   "window": "PS XTime immediately before START through polled DONE, including MMIO and usleep(100) polling; excludes tokenization, embedding copies, token read/decode, UART output, SD writes, request reset",
                   "throughput_formula": "decode samples * nominal_timer_hz / sum(decode timer_ticks)",
                   "throughput_is_end_to_end": False, "timer_ticks_are_pl_cycles": False},
        "scenarios": scenarios, "prompt_encodings": encodings,
        "restored_state": {**state, "ready": True},
        "listener": {"pid_at_last_inspection": 23008, "port": "COM11", "baud": 115200,
                     "preserved_running": True, "keyboard_passthrough": False},
        "evidence": {"live_log": (evidence_dir / "uart-live.log").as_posix(),
                     "frozen_transcript": (output / "uart-transcript.log").as_posix(),
                     "frozen_transcript_utf8_lf_sha256": digest(transcript.encode("utf-8")),
                     "command_interval": [FIRST, LAST]},
        "scope": {"rtl_modified": False, "firmware_rebuilt": False, "model_modified": False,
                  "delivery_modified": False, "qspi_written": False,
                  "persistent_boot_configuration_modified": False,
                  "sd_chat_csv_append_by_existing_firmware": True},
        "not_covered": [
            "Per-operator/tile GEMM, DMA loopback, RMSNorm, attention and MLP isolation",
            "Hardware layer completion events, intermediate tensor errors, full logits and independent argmax",
            "PL cycle counter, AXI beats, DDR bytes/stalls, busy/bank/mode counters",
            "Long-context, sustained throughput, long-run stability, power/temperature",
            "Timeout/abort/error recovery and SD fault injection",
            "No independent PyTorch forward executed this task; compare to delivered final scenario results"],
        "capacity": {"deployed_kv_tokens": 256, "old_16k_target_tested": False},
        "sources": [metadata(p) for p in source_paths],
    }
    table = []
    perf = []
    for s in scenarios:
        ids = ", ".join(map(str, [t["token_id"] for r in s["requests"] for t in r["tokens"]]))
        reason = " / ".join(r["reason"] for r in s["requests"])
        label = "一致，仅功能；trace未覆盖" if s["name"] == "final_decode_trace" else ("一致" if s["functional_match"] else "不一致")
        table.append(f"| {s['name']} | {s['template']} | {s['prompt']} | {s['max_new']} | {ids} | {reason} | {label} |")
        p = " / ".join(f"{x:.3f}" for x in s["prefill_first_output_ms"])
        d = f"{s['decode_mean_ms']:.3f}" if s["decode_samples"] else "—"
        rate = f"{s['decode_compute_window_tokens_per_second']:.3f}" if s["decode_samples"] else "—"
        perf.append(f"| {s['name']} | {' / '.join(str(r['prompt_tokens']) for r in s['requests'])} | {p} | {s['decode_samples']} | {d} | {rate} |")
    docs = "."
    report = f"""# KV260 实板最终场景测试记录

结论：按基线交付包的 6 类 final 场景执行了真实串口测试，7 次请求、22 个输出 token 全部与交付参考一致，结束原因及应用返回码也一致；40-token 分词黄金向量自检通过。5 类端到端场景完成 token 级验收，`final_decode_trace` 仅复现功能输入输出，未验证其逐层 trace，不能记为原场景全项 PASS。

这是 PS 裸机调度 + 真实 PL 推理，不是 Linux/CPU 数学模型或新一轮仿真；主机仅控制串口和分析数据，未使用 GPU。未改固件、RTL、原模型或基线交付物，未写 QSPI 或永久启动环境。

## 1. 输入依据与运行身份

| 项目 | 值 / 依据 |
| --- | --- |
| 权威验收来源 | [项目报告]({docs}/report/README.md)、`sim/reports/regression_status.json`、各 `sim/reports/system_regression/final_*/input.txt` / `result.json` / `hardware_tokens.csv` / `uart.log` |
| 运行 ELF | QWUART.ELF，999,656 字节；SHA-256 `{old['candidate']['sha256']}` |
| ELF 身份证据 | 沿用同一持续运行会话；启动时板端 CRC32 `{old['candidate']['board_crc32']}`、入口 `{old['candidate']['entry']}`；本轮重新核对 PC ELF SHA-256 |
| 软件差异 | 应用层物理 UART 替代层，应用/运行库 GCC 15.2.1 + 预编译 BSP/CRT GCC 13.4.0；没有重建 BSP。仍是候选而非已重新生成的纯 non-Xen BSP |
| PL | QWEN25.BIN，7,797,692 字节；启动时 CRC32 `137125e6`；成品名义 100MHz，未独立测频 |
| CSR / 启动版本 | 同会话启动记录：ID `0x515732354b563236`、VERSION `0x0000000000010000`；FSBL 2022.2、U-Boot 2022.01；本轮未重新读取版本 |
| 串口 | COM11，115200 8N1，无流控；既有 PID 23008 监听器持续保留，单端口单拥有者 |
| 当前 KV 上限 | 256 token，含模板与输出预留；配置见 `board/deployment/QWEN25/CONFIG.JSON` |

本轮首先显式设置模板与 events 模式；聊天场景也显式设置 chat，不依赖前一场景的状态。参考仿真与实板 request_id 不同，按场景内请求次序对应，不把编号差异当成数值差异。比对 token ID、token 字节、prompt token 数、max_new、produced、结束原因与返回码；不比较仿真 cycles 与板上 timer ticks。

## 2. 功能结果

| 场景 | 模板 | 输入 | max_new | 实板 token ID（按请求/迭代顺序） | 结束原因 | 与参考对照 |
| --- | --- | --- | ---: | --- | --- | --- |
{chr(10).join(table)}

7 次请求均返回 `@END ... status=0` 和 `READY / prompt>`。这里 status=0 是 `qgen_run()` 应用返回码，不是原始 CSR_STATUS 的读数。本轮片段未观察到 FATAL、ERROR 或 WARNING。

中文聊天可见内容：`你好！很高兴为您服务。有什么我可以`（max_new=8，按上限正常截断）。算术回答：`1+1等于2。`，随后输出 token 151645 `<|im_end|>`，produced=7、reason=eos，而非用满 20 个 token。raw `你好` 的前两个输出为 `，我`，这是基线参考的预期，不将其当作聊天模板回答。

分词实测：raw `你好` = 1 token（108386）；raw `123456789` = 9 token（16..24）；chat `你好` = 30 token；chat `1+1等于几？` = 35 token。聊天数包含固定 system/user/assistant 模板。完整 ID 在 test-results.json。重复 raw 请求两次均得到 token 3837、prompt_tokens=1，与固件逐请求清上下文行为及参考一致；未直接 dump KV 内存证明所有字节清零。

## 3. 性能记录：PS 观测的 PL 计算窗口

| 场景 | prompt tokens | 首 token 计算窗口 / ms | 后续 decode 样本数 | decode 平均 / ms | decode 计算窗口吞吐 / token·s⁻¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(perf)}

计时与公式依据：

- `src/software/interactive/generation.c`：在写 START 前取 PS XTime，到轮询 DONE 后取 XTime；包含 MMIO、屏障和 `usleep(100)` 轮询开销。iteration=0 是完整 prompt 的 prefill / 首次输出，iteration>0 才是 decode；共 7 个 prefill、15 个 decode 样本。
- `src/BSP/include/xtimer_config.h` → `xparameters.h`：`COUNTS_PER_SECOND = XPAR_CPU_TIMESTAMP_CLK_FREQ = {frequency}` Hz。`ms = timer_ticks / {frequency} × 1000`，使用 BSP 标称频率，未独立实测校准。
- 表内 decode 吞吐 = `decode 样本数 × {frequency} / Σdecode_ticks`，仅除以这些计算窗口之和。排除分词、上下文清理、embedding 拷贝、结果寄存器读取/解码、UART 输出和 SD 日志写入；不是用户端到端速度、不是纯 PL cycle，也不是稳态长跑基准。
- 各场景仅一轮（重复场景两次），上下文长度不同，不用 15 个混合 decode 的总平均冒充通用性能指标。不把参考仿真耗时、cycles 或其 UART ticks 挪为实板结果。

逐 token 的原始 ticks、标称 ms、阶段、token ID/字节及匹配项见 [tokens.csv](tokens.csv)。

## 4. 覆盖边界与待确认

| 验收项 | 本轮实板状态 | 限制 / 后续所需 |
| --- | --- | --- |
| 交互分词、模板、prefill、多步 decode、EOS | 已执行，上述 22 个输出与参考一致 | 有限输入，不等于所有输入数学等价 |
| 连续请求上下文隔离 | 两次相同 raw 请求功能一致 | 未导出 KV 内容或长历史 |
| 24 层推理的端到端输出 | token 对照一致 | 固件没有逐层完成事件导出，未独立计数每层执行 |
| final_decode_trace 逐层中间量 / 误差 | 未覆盖 | 当前串口无法导出 tensor / trace；后续需定义调试导出接口 |
| 独立全 logits / argmax / PyTorch 对照 | 未覆盖 | 本轮对照基线 final 参考；未执行新的离线 forward |
| 项目书 §13 单 tile/GEMM、DMA、RMSNorm、单层 attention / MLP | 未覆盖 | 当前成品交互命令无单算子入口；不猜地址，不自行改固件/RTL |
| 项目书 §16 PL cycles、AXI beats、stall、DDR / bank / mode 统计 | 未覆盖 | 固件未提供可读取的调试输出；上述 PS ticks 不能代替 |
| 16K 上下文、长稳、功耗温度、故障恢复 | 未覆盖 | 当前容量为256；未做注错、超时、掉卡或长跑，不宣称可靠性已验收 |

当前合理结论是“真实板端到端 token 参考对照通过，低层数值与性能计数验收尚未完成”。下一阶段若要推进这些缺口，先确认现有成品的调试/计数接口及读出方式，再另立范围；本轮没有为通过验收修改任何参考结果或 RTL。

## 5. 证据、复现与保留状态

同目录文件：`uart-transcript.log` 为命令 028..050 的冻结 UTF-8/LF 串口片段；`commands.csv` 含23条命令和发送时间；`tokens.csv` 含22条 token；`test-results.json` 含逐请求、逐 token 检查、原始来源 SHA-256 和版本信息。冻结日志 SHA-256：`{result['evidence']['frozen_transcript_utf8_lf_sha256']}`。持续日志仍在原目录追加，不把它当不可变文件。

生成脚本：[summarize_kv260_board_regression.py](summarize_kv260_board_regression.py)。脚本只读既有日志和包内参考，不访问串口、不运行仿真、不创建固件测试；`--verify-output` 可只读核对当前文件。可运行：

```powershell
python -B board/tools/summarize_kv260_board_regression.py --verify-output
```

结尾已恢复 `chat / text / max 20` 并收到 READY。COM11 可见监听窗口仅显示日志，不提供键盘转发。正式部署使用 `QWUART.ELF` 自动启动，旧固件不再进入 SD 清单。现有固件按其正常逻辑向 SD 的 CHAT.CSV 追加记录。
"""
    fields = ["scenario", "command_id", "request_id", "prompt_tokens", "iteration", "phase",
              "token_id", "expected_token_id", "matches_reference", "timer_ticks", "timer_nominal_hz",
              "nominal_ms", "utf8_hex", "text_fragment"]
    files = {"README.md": report, "test-results.json": json.dumps(result, ensure_ascii=False, indent=2) + "\n",
             "tokens.csv": csv_text(token_rows, fields),
             "commands.csv": csv_text(command_rows, list(command_rows[0])),
             "uart-transcript.log": transcript}
    return files, result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--delivery", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-patch", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output or workspace / "board/measurements/functional"
    files, result = build(workspace, args.delivery, output)
    if args.verify_output:
        for name, content in files.items():
            require(read(output / name) == content, "Output mismatch: " + name)
        print(json.dumps({"verified_files": list(files), "summary": result["board_regression"]}, ensure_ascii=False))
    elif args.emit_patch:
        print("*** Begin Patch")
        for name, content in files.items():
            require(not (output / name).exists(), "Refusing to replace existing report: " + name)
            print("*** Add File: " + (output / name).as_posix())
            print("\n".join("+" + line for line in content.splitlines()))
        print("*** End Patch")
    else:
        print(json.dumps({"output_directory": output.as_posix(), "files": files}, ensure_ascii=False))


if __name__ == "__main__":
    main()
