"""Read-only analysis of the 2026-10-25 board length scan. No board I/O.
Emit an apply_patch payload, or verify saved artifacts; never write directly.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'board/measurements/length_scan'
FREQUENCY = 99999001
ELF_SHA = '0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c'
STAGING = ROOT / 'src/deployment_firmware/source'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fit(rows):
    x = [r['n_history'] for r in rows]
    y = [r['milliseconds'] for r in rows]
    xm, ym = statistics.mean(x), statistics.mean(y)
    b = sum((u-xm)*(v-ym) for u,v in zip(x,y)) / sum((u-xm)**2 for u in x)
    a = ym-b*xm
    errors = [v-a-b*u for u,v in zip(x,y)]
    return {'a_ms': a, 'b_ms_per_history_token': b,
            'a_ticks': a*FREQUENCY/1000, 'b_ticks_per_history_token': b*FREQUENCY/1000,
            'r_squared': 1-sum(e*e for e in errors)/sum((v-ym)**2 for v in y),
            'rmse_ms': math.sqrt(statistics.mean(e*e for e in errors)),
            'max_abs_residual_ms': max(abs(e) for e in errors), 'sample_count': len(x)}

def build(source):
    raw = source.read_text(encoding='utf-8')
    first = raw.rfind('[HOST] ', 0, raw.index('TX id=length-013-raw:'))
    assert first >= 0
    # Stable test-only snapshot, independent of subsequent UART output.
    last = raw.index('TX id=length-026-restore-max:')
    prompt_end = re.search(r'prompt> *', raw[raw.index('READY', last):])
    assert prompt_end is not None
    last = raw.index('READY', last) + prompt_end.end()
    text = raw[first:last].rstrip() + '\n'
    assert 'FATAL' not in text and 'reason=error' not in text
    assert 'TEMPLATE=raw' in text and 'MODE=events' in text and 'MAX_NEW=8' in text
    assert 'TEMPLATE=chat' in text and 'MODE=text' in text and 'MAX_NEW=20' in text
    encoded = [(int(n), list(map(int, ids.split()))) for n,ids in re.findall(r'@PROMPT (\d+)((?: \d+)*)', text)]
    assert [n for n,_ in encoded] == [1,10,20,30]
    for n,ids in encoded:
        assert len(ids) == n and ids == ([16,17,18,19,20,21,22,23,24,15]*3)[:n]
    blocks = list(re.finditer(r'@BEGIN (\d+) prompt_tokens=(\d+) max_new=(\d+)(.*?)@END (\d+) produced=(\d+) reason=(\w+) status=(-?\d+)', text, re.S))
    assert len(blocks) == 4
    rows, groups = [], []
    for block, (expected_count, prompt_ids) in zip(blocks, encoded):
        request, count, maximum, body, end_id, produced, reason, status = block.groups()
        request, count = int(request), int(count)
        assert count == expected_count and int(maximum) == 8
        assert int(end_id) == request and int(produced) == 8 and reason == 'limit' and int(status) == 0
        tokens = re.findall(r'@TOKEN (\d+) (\d+) (\d+) (\d+) *([0-9a-f]*)', body)
        assert len(tokens) == 8 and [int(t[1]) for t in tokens] == list(range(8))
        group_rows = []
        for rid, iteration, token, ticks, hexbytes in tokens:
            iteration, ticks = int(iteration), int(ticks)
            assert int(rid) == request and ticks > 0
            row = {'request_id': request, 'prompt_tokens': count, 'iteration': iteration,
                   'phase': 'prefill_first_token' if iteration == 0 else 'decode',
                   'included_in_fit': iteration > 0,
                   'n_history': None if iteration == 0 else count+iteration-1,
                   'n_total': count if iteration == 0 else count+iteration,
                   'token_id': int(token), 'ticks': ticks, 'milliseconds': ticks*1000/FREQUENCY,
                   'utf8hex': hexbytes}
            rows.append(row)
            if iteration > 0:
                group_rows.append(row)
        groups.append({'request_id': request, 'prompt_tokens': count,
                       'prompt_text': ('1234567890'*3)[:count], 'prompt_token_ids': prompt_ids,
                       'generated_tokens': 8, 'decode_samples': len(group_rows),
                       'n_history_min': min(r['n_history'] for r in group_rows),
                       'n_history_max': max(r['n_history'] for r in group_rows),
                       'mean_n_history': statistics.mean(r['n_history'] for r in group_rows),
                       'mean_decode_ticks': statistics.mean(r['ticks'] for r in group_rows),
                       'mean_decode_ms': statistics.mean(r['milliseconds'] for r in group_rows),
                       'min_decode_ms': min(r['milliseconds'] for r in group_rows),
                       'max_decode_ms': max(r['milliseconds'] for r in group_rows)})
    decode = [r for r in rows if r['included_in_fit']]
    model = fit(decode)
    for row in decode:
        row['fit_ms'] = model['a_ms'] + model['b_ms_per_history_token']*row['n_history']
        row['residual_ms'] = row['milliseconds'] - row['fit_ms']
    elf = ROOT / 'board/deployment/QWUART.ELF'
    assert sha(elf.read_bytes()) == ELF_SHA
    params = STAGING / 'BSP/include/xparameters.h'
    assert re.search(r'#define\s+XPAR_CPU_TIMESTAMP_CLK_FREQ\s+99999001\b', params.read_text())
    sources = {}
    for relative in ['BSP/include/xparameters.h', 'BSP/include/xtimer_config.h', 'software/interactive/main.c', 'software/interactive/generation.c']:
        p = STAGING / relative
        sources[p.relative_to(ROOT).as_posix()] = sha(p.read_bytes())
    result = {'date': '2026-10-25', 'template': 'raw', 'max_new_tokens': 8,
              'runs_per_prompt_length': 1, 'timer_hz': FREQUENCY,
              'timer_frequency_source': 'BSP/include/xtimer_config.h: XSLEEPTIMER_FREQ -> XPAR_CPU_TIMESTAMP_CLK_FREQ; BSP/include/xparameters.h:1420',
              'ms_formula': 'ticks * 1000 / 99999001',
              'measurement_window': 'PS XTime just before CTRL.START through the DONE polling sample; includes MMIO/barriers and usleep(100) polling; excludes tokenization, context clear, embedding copy, token decode, UART and SD logging',
              'history_formula': 'decode iteration j >= 1: n_history = prompt_tokens + j - 1 (TOKEN_POS), n_total = n_history + 1 (CONTEXT_LEN)',
              'fit': model, 'equivalent_a_if_n_includes_current_token_ms': model['a_ms']-model['b_ms_per_history_token'],
              'equivalent_a_if_fitting_group_means_against_prompt_only_ms': model['a_ms']+3*model['b_ms_per_history_token'],
              'limitations': ['a is total fixed cost, not isolated weight DDR read latency', 'b includes length-dependent attention compute and KV activity, not isolated KV DDR read latency', 'one run per length; within-request tokens are not independent repeated runs', 'n=0 intercept extrapolated from measured n=1..36', 'timer frequency from BSP, not independently calibrated; not PL cycle timing or end-to-end timing', 'four excluded first tokens retained; no firmware recompilation, RTL/model/QSPI modification'],
              'firmware_sha256': ELF_SHA, 'board_elf_crc32': 'f195fdca',
              'board_pl_crc32': '137125e6', 'csr_id': '515732354b563236', 'csr_version': '0000000000010000',
              'kv_capacity_tokens': 256, 'source_sha256': sources,
              'transcript_sha256': sha(text.encode()), 'groups': groups, 'tokens': rows}
    buf = io.StringIO(newline='')
    fields = list(rows[-1])
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    a,b = model['a_ms'],model['b_ms_per_history_token']
    lines = ['# KV260 decode 长度扫描 — 2026-10-25', '',
             f'结论：`T_ms = {a:.6f} + {b:.6f} × n_history`。a = **{a:.6f} ms**；b = **{b:.6f} ms / 历史 token**。', '',
             f'4 档各运行 1 次，统一 raw / events / max_new=8；共 32 个输出 token，排除 4 个首 token，以 28 个 decode 样本做普通最小二乘。R² = {model["r_squared"]:.8f}，残差 RMSE = {model["rmse_ms"]:.6f} ms，最大绝对残差 = {model["max_abs_residual_ms"]:.6f} ms。', '',
             '重要：a 是总固定项，包含权重、计算、控制及 PS 轮询等；b 是每历史 token 的总增量，包含 KV 活动及长度相关 attention 计算。没有硬件分项接口，不能将两者直接认定为纯权重 DDR 读取和纯 KV DDR 读取耗时。', '',
             '## 1. 四档实测', '',
             '| prompt token 数 | decode 样本 | 实际历史 n 范围 | 平均 n | 平均 ticks | 平均 ms/token | 范围 ms/token |',
             '| ---: | ---: | --- | ---: | ---: | ---: | --- |']
    for g in groups:
        lines.append(f'| {g["prompt_tokens"]} | {g["decode_samples"]} | {g["n_history_min"]}–{g["n_history_max"]} | {g["mean_n_history"]:g} | {g["mean_decode_ticks"]:.3f} | {g["mean_decode_ms"]:.6f} | {g["min_decode_ms"]:.6f}–{g["max_decode_ms"]:.6f} |')
    lines += ['', '输入分别是 `1`、`1234567890`、`12345678901234567890`、`123456789012345678901234567890`。已逐条用板端 `/encode` 验证为 1/10/20/30 token，且与各请求 @BEGIN 相符。每个请求自动清上下文，均 @END status=0、produced=8、reason=limit，无 EOS 提前退出。', '',
              '## 2. 口径与公式', '',
              '- ticks 来源：当前候选固件 generation.c 的 PS XTime，START 前开始，到轮询观察到 DONE 的采样点结束，包含 MMIO、屏障和约 100μs 延时轮询。不是纯 PL 周期，也不是按回车到输出结束的端到端时延。',
              '- 窗口外：分词、上下文清零、embedding 拷贝、token 解码、UART 输出、SD 日志。SD 正常追加 CHAT.CSV 是既有固件行为。',
              '- `ms = ticks × 1000 / 99,999,001`。频率来自 BSP xtimer_config.h → xparameters.h:1420，未独立实测校准，不能直接用标称 PL 100MHz 换算。',
              '- `iteration=0` 为 prefill/首输出 token，全部排除；decode `j=1..7` 的 `n_history=S+j−1`，即 TOKEN_POS；attention 总长度 `n_total=S+j`，即 CONTEXT_LEN。',
              '- OLS：`b=Σ((n−均值n)(T−均值T))/Σ(n−均值n)²`，`a=均值T−b×均值n`，使用全部 28 个 decode 点，不将同一请求的每步历史长度都错误写成 prompt 长度。',
              f'- 若 n 定义为含当前输入 token 的总长度，等价截距是 {a-b:.6f} ms，b 不变。若把每档均值对 prompt 长度拟合，截距会变为 {a+3*b:.6f} ms（每档的平均历史比 prompt 多 3），不能将该值混作零历史固定项。',
              '- 只运行每档一次，无额外预热推理或重复；组内 7 个 decode 是连续生成而非 7 次独立重复。n=0 截距为外推，本轮不证明物理带宽或长上下文性能。', '',
              '## 3. 每个 decode 的原始记录', '',
              '| prompt tokens | iteration | n_history | token ID | 原始 ticks | ms |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in decode:
        lines.append(f'| {r["prompt_tokens"]} | {r["iteration"]} | {r["n_history"]} | {r["token_id"]} | {r["ticks"]} | {r["milliseconds"]:.6f} |')
    lines += ['', '4 个首 token 的原始 ticks 仍保存在 tokens.csv / scan-results.json，included_in_fit=false，不计入以上结果。', '',
              '## 4. 版本与证据', '',
              f'- ELF：QWUART.ELF，SHA-256 `{ELF_SHA}`；本轮板端 CRC32 `f195fdca`。未重编、未替换固件。',
              '- PL：QWEN25.BIN，本轮板端 CRC32 `137125e6`，加载返回 0。CSR ID `515732354b563236` / VERSION `0000000000010000`。仅加载已有 PL 文件，未改 RTL。',
              '- 使用现有 SD 模型资产，未改模型/QSPI/永久启动配置。当前固件 KV 容量 256，不把早期设计目标 16K 视为本轮可用容量。',
              '- uart-transcript.log：测试设置、4 次 /encode、4 次推理及恢复设置的固定日志快照；启动及 CRC 证据在同目录 uart-live.log。scan-results.json 保存快照及依据源文件 SHA-256。',
              '- tokens.csv：32 行，含 phase / included_in_fit / 实际历史长度 / 原始 ticks / ms / 拟合残差；scan-results.json：公式、全部原始数据、汇总和拟合指标。',
              '- 测试后已恢复 `/template chat`、`/mode text`、`/max 20` 并收到 READY；串口监听继续保留。', '',
              '## 5. 离线复核（CPU，只读，不触碰板卡）', '',
              '在仓库根目录执行：', '',
              '```powershell',
              'python -B board/tools/analyze_kv260_length_scan.py --verify-output',
              '```', '',
              '生成器默认只从固定快照复算；`--live --emit-patch` 可从持续日志中提取本轮固定片段并向标准输出发出 apply_patch 文本，不直接写文件，不重编固件。', '']
    return {'README.md': '\n'.join(lines), 'tokens.csv': buf.getvalue(),
            'scan-results.json': json.dumps(result, ensure_ascii=False, indent=2)+'\n',
            'uart-transcript.log': text}, result

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--emit-patch', action='store_true')
    group.add_argument('--verify-output', action='store_true')
    args = parser.parse_args()
    source = OUT / ('uart-live.log' if args.live else 'uart-transcript.log')
    payloads, result = build(source)
    if args.emit_patch:
        print('*** Begin Patch')
        for name, content in payloads.items():
            path = OUT / name
            if path.exists():
                raise SystemExit(f'Refusing to overwrite existing output: {path}')
            print(f'*** Add File: {path.as_posix()}')
            for line in content.splitlines():
                print('+'+line)
        print('*** End Patch')
    else:
        for name, expected in payloads.items():
            actual = (OUT / name).read_text(encoding='utf-8')
            if actual.rstrip('\n') != expected.rstrip('\n'):
                raise SystemExit(f'Verification failed: {name}')
        print(json.dumps({'verified_files': list(payloads), 'fit': result['fit']}, indent=2))

if __name__ == '__main__':
    main()
