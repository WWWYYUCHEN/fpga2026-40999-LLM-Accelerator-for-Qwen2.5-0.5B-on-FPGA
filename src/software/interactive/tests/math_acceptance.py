"""Acceptance: manual text -> production PS tokenizer -> numerical PL -> UART."""
import argparse
import csv
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time
import sys


def run(package, report):
    report.mkdir(parents=True, exist_ok=True)
    runtime = os.environ.get('QWEN_SIM_PYTHON', sys.executable)
    text = '/max 64\n请介绍厦门大学\n1+1等于几？\n/mode events\n/max 4\n你好\n'
    (report / 'input.txt').write_text(text, encoding='utf-8')
    start = time.time()
    result = subprocess.run([runtime, str(package / 'src/PC/math_simulator.py'), '--package', str(package), '--run', str(report)],
                            input=text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    transcript = result.stdout.decode('utf-8', errors='strict')
    (report / 'console.log').write_bytes(result.stdout)
    assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr, transcript)
    assert transcript.count('@BEGIN ') == 3 and transcript.count('@END ') == 3
    assert 'answer> 厦门大学' in transcript and '1+1等于2' in transcript, transcript
    transactions = [json.loads(line) for line in (report / 'model_transactions.jsonl').read_text().splitlines()]
    assert all(x['error'] == 0 for x in transactions)
    starts = [x for x in transactions if x['clear']]
    assert len(starts) == 3 and all(x['position'] == 0 for x in starts)
    # Prove the IDs handed to the numerical accelerator are the IDs from PS C.
    lib = C.CDLL(str(package / 'sim/simulator/qwen_ps.dll'))
    lib.qt_init.argtypes = [C.c_void_p, C.c_size_t]
    lib.qt_chat_encode.argtypes = [C.c_char_p, C.POINTER(C.c_uint32), C.c_uint, C.POINTER(C.c_uint)]
    data = C.create_string_buffer((package / 'board/deployment/QWEN25/TOKEN.BIN').read_bytes())
    assert lib.qt_init(data, C.sizeof(data)-1) == 0
    for original, request in zip(['请介绍厦门大学', '1+1等于几？', '你好'], starts):
        out, count = (C.c_uint32 * 256)(), C.c_uint()
        assert lib.qt_chat_encode(original.encode(), out, 256, C.byref(count)) == 0
        assert list(out[:count.value]) == request['ids']
    # Decode the transaction stream with the production tokenizer, not a reference on PC.
    lib.qt_decode.argtypes = [C.c_uint32, C.POINTER(C.POINTER(C.c_ubyte)), C.POINTER(C.c_uint), C.POINTER(C.c_int)]
    answers, current = [], bytearray()
    for item in transactions:
        if item['clear'] and current: answers.append(current.decode('utf-8', 'replace')); current = bytearray()
        ptr, length, special = C.POINTER(C.c_ubyte)(), C.c_uint(), C.c_int()
        assert lib.qt_decode(item['token'], C.byref(ptr), C.byref(length), C.byref(special)) == 0
        if not special.value: current += C.string_at(ptr, length.value)
    answers.append(current.decode('utf-8', 'replace'))
    verification = dict(status='PASS', requests=3, produced=len(transactions), answers=answers,
                        ps_implementation='production C main/tokenizer/generation/UTF-8 decoder',
                        pl_substitute='mathematical Qwen model with exact deployed packed INT8 weights and BF16 assets',
                        scripted_answers=False, tokenizer_on_pc=False, seconds=round(time.time()-start, 2),
                        rtl_cycle_equivalence=False, board_tested=False)
    (report / 'result.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('package', type=Path)
    ap.add_argument('report', type=Path)
    args = ap.parse_args(); run(args.package, args.report)
