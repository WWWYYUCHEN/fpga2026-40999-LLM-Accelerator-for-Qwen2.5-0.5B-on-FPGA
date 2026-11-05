"""Replay the delivery's final vectors through one existing UART owner.

No firmware build, RTL simulation, board reset, or direct serial-port open.
Each command waits for READY; no command retries after timeout or failure.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from summarize_kv260_board_regression import requests

CASES = ['final_all_layers','final_repeat_context_reset','final_rows9_prefill',
         'final_decode_trace','final_chat_nihao','final_chat_eos']
READY = re.compile(r'\nREADY\nprompt> *')


def read(path):
    return path.read_text(encoding='utf-8-sig').replace('\r\n','\n')


def stamp():
    return datetime.now().astimezone().isoformat()


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def meta(path):
    raw = path.read_bytes()
    return {'path':path.as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def csv_write(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--output',type=Path,default=root/'board/measurements/functional_repeat')
    parser.add_argument('--delivery',type=Path,default=root)
    parser.add_argument('--uart-log',type=Path,help='Existing UART owner log; does not open another serial port')
    parser.add_argument('--slot',type=Path,help='Existing UART owner command slot')
    parser.add_argument('--boot-evidence',type=Path,default=root/'board/measurements/autoboot/automatic-boot-verification.json')
    parser.add_argument('--boot-label',default='FSBL 2020.2 / U-Boot 2020.01')
    args = parser.parse_args()
    out, delivery = args.output.resolve(), args.delivery.resolve()
    log = args.uart_log.resolve() if args.uart_log else out/'uart-live.log'
    slot = args.slot.resolve() if args.slot else out/'uart.command.json'
    assert not (out/'run-state.json').exists(), 'Refusing to overwrite or replay an existing run'
    receiver = json.loads(read(log.with_suffix('.json')))
    assert receiver['status']=='listening', 'UART owner is not listening'
    heartbeat=datetime.fromisoformat(receiver['heartbeat_at'])
    assert 0 <= (datetime.now().astimezone()-heartbeat).total_seconds() < 15, 'Stale UART heartbeat'
    assert not json.loads(read(slot)), 'UART command slot is occupied'
    assert READY.search(read(log)[-1000:]), 'Current board is not at READY'
    assert args.boot_evidence.is_file(), 'Missing boot identity evidence'
    start_byte=log.stat().st_size
    out.mkdir(parents=True,exist_ok=True)
    state = {'started_at':stamp(),'status':'running','receiver_pid':receiver['pid'],
             'boot_label':args.boot_label,'source_log':log.as_posix(),
             'source_start_byte':start_byte,'commands':[]}
    dump(out/'run-state.json',state)
    prefix = uuid.uuid4().hex[:10]

    def send(command, scenario, expected=None):
        identity = f'{prefix}-{len(state["commands"])+1:03d}'
        marker = f'TX id={identity}: {command}\n'
        payload = {'id':identity,'command':command}
        temporary = slot.with_suffix('.pending')
        dump(temporary,payload)
        os.replace(temporary,slot)
        started = time.monotonic()
        while time.monotonic()-started < 90:
            transcript = read(log)
            if marker in transcript:
                response = transcript.split(marker,1)[1]
                if re.search(r'FATAL|^ERROR|^WARNING|\[HOST\].*ERROR',response,re.M):
                    raise RuntimeError('Board/receiver error after '+identity+': '+response)
                match = READY.search(response)
                if match:
                    response = response[:match.end()]
                    if expected and expected not in response:
                        raise RuntimeError('Missing acknowledgement '+expected)
                    row = {'id':identity,'scenario':scenario,'command':command,
                           'completed_at':stamp(),'response':response}
                    state['commands'].append(row)
                    dump(out/'run-state.json',state)
                    print(f'{identity} {scenario}: {command} -> READY',flush=True)
                    return response
            time.sleep(.05)
        raise TimeoutError('No READY after '+identity+'; no further commands sent')

    try:
        send('/help','interface','/selftest')
        send('/selftest','tokenizer','TOKENIZER_SELFTEST_OK 40 tokens')
        scenarios, tokens, sources = [], [], []
        param = delivery/'src/BSP/include/xparameters.h'
        frequency = int(re.search(r'^#define XPAR_CPU_TIMESTAMP_CLK_FREQ\s+(\d+)',read(param),re.M)[1])
        sources.extend([meta(param),meta(delivery/'report/README.md'),meta(Path(__file__).resolve()),
                        meta(Path(__file__).with_name('kv260_uart_native.py'))])
        for name in CASES:
            folder = delivery/'sim/reports/system_regression'/name
            ref = json.loads(read(folder/'result.json'))
            assert ref['status']=='PASS'
            reference = requests(read(folder/'uart.log'))
            source_csv = list(csv.DictReader(io.StringIO(read(folder/'hardware_tokens.csv'))))
            ids = [t['token_id'] for r in reference for t in r['tokens']]
            assert ids == [int(t['token_id']) for t in ref['generated']]
            assert ids == [int(t['token_id']) for t in source_csv]
            for leaf in ['input.txt','result.json','hardware_tokens.csv','uart.log']:
                sources.append(meta(folder/leaf))
            send('/template '+ref['template'],name,'TEMPLATE='+ref['template'])
            send('/mode events',name,'MODE=events')
            send('/max '+str(ref['requested_tokens']),name,'MAX_NEW='+str(ref['requested_tokens']))
            actual, checks = [], []
            for index, want in enumerate(reference):
                response = send(ref['prompt'],name)
                got_list = requests(response)
                assert len(got_list)==1, 'Expected exactly one request'
                got = got_list[0]
                check = {key:got[key]==want[key] for key in
                         ['prompt_tokens','max_new','produced','reason','application_return_code']}
                check['token_ids'] = [t['token_id'] for t in got['tokens']]==[t['token_id'] for t in want['tokens']]
                check['token_bytes'] = [t['utf8_hex'] for t in got['tokens']]==[t['utf8_hex'] for t in want['tokens']]
                actual.append(got)
                checks.append(check)
                for token in got['tokens']:
                    tokens.append({'scenario':name,'request_index':index,'request_id':got['request_id'],
                                   'prompt_tokens':got['prompt_tokens'],
                                   'phase':'prefill_first_output' if token['iteration']==0 else 'decode',
                                   'timer_nominal_hz':frequency,
                                   'nominal_ms':token['timer_ticks']*1000/frequency,**token})
                if not all(check.values()):
                    raise RuntimeError('Reference mismatch: '+name+' '+json.dumps(check))
            scenarios.append({'name':name,'template':ref['template'],'prompt':ref['prompt'],
                              'max_new':ref['requested_tokens'],'functional_match':all(all(c.values()) for c in checks),
                              'requests':actual,'checks':checks,'layer_trace_verified':False})
            print(name+': token reference MATCH',flush=True)
        send('/template chat','restore','TEMPLATE=chat')
        send('/mode text','restore','MODE=text')
        send('/max 20','restore','MAX_NEW=20')
        state.update(status='completed',finished_at=stamp(),defaults='chat / text / max 20')
        dump(out/'run-state.json',state)
        dump(slot,{})
        with log.open('rb') as stream:
            stream.seek(start_byte)
            transcript=stream.read().decode('utf-8-sig').replace('\r\n','\n')
        (out/'uart-transcript.log').write_text(transcript,encoding='utf-8',newline='')
        result = {'started_at':state['started_at'],'finished_at':state['finished_at'],
                  'functional_scenarios_matched':len(scenarios),'requests':sum(len(s['requests']) for s in scenarios),
                  'output_tokens':len(tokens),'tokenizer_selftest':'40 tokens OK',
                  'timer_nominal_hz':frequency,'timer_frequency_independently_calibrated':False,
                  'scenarios':scenarios,'sources':sources,
                  'boot_label':args.boot_label,'boot_identity_evidence':meta(args.boot_evidence),
                  'frozen_transcript':meta(out/'uart-transcript.log')}
        dump(out/'test-results.json',result)
        csv_write(out/'tokens.csv',tokens)
        csv_write(out/'commands.csv',[{k:v for k,v in row.items() if k!='response'} for row in state['commands']])
        table=[]
        for s in scenarios:
            r=s['requests']
            actual=' / '.join(','.join(str(t['token_id']) for t in x['tokens']) for x in r)
            table.append(f"| {s['name']} | {s['template']} | {s['prompt']} | {r[0]['prompt_tokens']} | {s['max_new']} | {actual} | token一致 |")
        performance=[]
        for s in scenarios:
            ts=[t for t in tokens if t['scenario']==s['name']]
            first=' / '.join(f"{t['nominal_ms']:.3f}" for t in ts if t['iteration']==0)
            dec=[t for t in ts if t['iteration']>0]
            avg=f"{sum(t['nominal_ms'] for t in dec)/len(dec):.3f}" if dec else '—'
            performance.append(f"| {s['name']} | {first} | {len(dec)} | {avg} |")
        report=f'''# KV260 原生测试向量实板复测

结论：在 {args.boot_label} 的 READY 状态下，在真实 KV260 上复跑基线交付包提供的 6 类 final 场景，共 7 次请求、22 个输出 token，全部与参考一致。40-token 分词自检通过。比对包含 prompt token 数、max_new、token ID/字节、输出数量、结束原因和 status；所有请求 status=0 并返回 READY。

本轮执行 PS 裸机 + PL 推理，主机只控制串口和整理数据。复用基线交付包测试输入、模板和输出上限；交付包的原生入口默认是 PC 上的 RTL 逐周期仿真，本轮没有重跑仿真。`final_decode_trace` 仅验证功能输出，未导出逐层 trace；不将端到端 token 一致写成该场景全项 PASS。

## 功能结果

| 场景 | 模板 | 输入 | prompt tokens | max_new | 实测 token IDs（不同请求用 / 分隔） | 对照 |
| --- | --- | --- | ---: | ---: | --- | --- |
{chr(10).join(table)}

raw「你好」前两 token 为「，我」，符合 raw 参考。chat「你好」输出「你好！很高兴为您服务。有什么我可以」，8 token 正常截断。算术回答「1+1等于2。」，第7个 token 151645 为 EOS，reason=eos；其余请求 reason=limit。重复 raw 请求两次均输出3837。

## 板内计时记录

| 场景 | 首输出窗口 / ms | decode 样本数 | decode 平均 / ms |
| --- | ---: | ---: | ---: |
{chr(10).join(performance)}

逐 token 原始 ticks、换算 ms 与阶段见 tokens.csv。频率来源为交付 BSP `XPAR_CPU_TIMESTAMP_CLK_FREQ={frequency}` Hz；公式 `ms=ticks×1000/{frequency}`，未独立校准。generation.c 通过 PS XTime 包围 START 至观察 DONE，包含 MMIO、屏障和100μs轮询；排除分词、embedding、UART和SD日志。iteration=0为prefill/首输出，后续才是decode。此数值不是纯PL周期或用户端到端耗时，不与参考仿真 cycles 比较，也不作为新优化效果。

## 版本、证据与边界

- 基线回归数据位于 `sim/reports/system_regression/final_*`，来源与 SHA-256 见 `test-results.json`。
- 沿用 QWUART.ELF；本轮未重编应用或 BSP，未替换模型或 PL。
- 启动标识：{args.boot_label}；证据 `../autoboot/automatic-boot-verification.json`，SHA-256 见 `test-results.json`。功能输出一致不等于从物理配置反解了完整网表。当前KV容量256。
- 串口单拥有者，COM11、115200 8N1，无流控；持续日志与冻结记录分别位于 `uart-live.log` 和 `uart-transcript.log`。
- 未发送闪存写入或启动配置变更命令，未使用GPU。启动选择器可自行更新Last Booted；固件按原逻辑向SD的CHAT.CSV追加请求记录。
- 当前固件不能导出逐层tensor/完成事件、全部logits或PL cycles/stall。本轮未独立复验基线 RTL 仿真的逐层数值误差、argmax与低层计数；也未做长上下文、功耗或长稳测试。
- 本目录为独立原生向量实板复测，未覆盖既有功能报告、长度扫描或其他启动版本的数据。
- 结束已恢复chat / text / max20并确认READY，命令槽清空，可见串口监听保留。

复现工具：`../board/tools/run_kv260_native_regression.py`，通过 `../board/tools/kv260_uart_native.py` 的显式命令槽顺序发送，一条命令收到 READY 才发下一条；错误、超时或参考不一致即停止。既有结果不覆盖、不自动重放。
'''
        (out/'README.md').write_text(report,encoding='utf-8')
        print(json.dumps({k:result[k] for k in ['functional_scenarios_matched','requests','output_tokens']},ensure_ascii=False))
    except Exception as error:
        state.update(status='failed',finished_at=stamp(),error=str(error))
        dump(out/'run-state.json',state)
        dump(slot,{})
        raise


if __name__=='__main__':
    main()
