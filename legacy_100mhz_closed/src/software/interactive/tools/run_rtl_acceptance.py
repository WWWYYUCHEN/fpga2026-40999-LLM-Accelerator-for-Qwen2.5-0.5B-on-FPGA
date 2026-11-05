"""Run new production PS C against real compiled RTL with local model assets."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import sys
import argparse
import shutil
sys.path.insert(0, str(Path(__file__).parent))
from build import ROOT, DEPLOY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', action='store_true')
    args = ap.parse_args()
    out = ROOT / 'reports' / ('rtl_current_raw_acceptance' if args.raw else 'rtl_current_acceptance')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SIM_SD=str(DEPLOY / 'sd_card'), SIM_TRACE=str(out / 'mmio_trace.txt'))
    # TOKEN.BIN is the sole additional SD asset the interactive firmware requires.
    (DEPLOY / 'sd_card/QWEN25/TOKEN.BIN').write_bytes((ROOT / 'TOKEN.BIN').read_bytes())
    input_text = ('/template raw\n' if args.raw else '') + '/mode events\n/max 2\n你好\n'
    (out / 'input.txt').write_text(input_text, encoding='utf-8')
    started = time.time()
    with (out / 'uart_transcript.txt').open('wb') as log:
        shutil.copy2(ROOT / 'build/rtl_current/segments.tsv', out / 'segments.tsv')
        process = subprocess.Popen([str(ROOT / 'build/rtl_current/qwen_rtl_sim.exe')], cwd=out,
                                   env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        process.stdin.write(input_text.encode()); process.stdin.close()
        (out / 'process.json').write_text(json.dumps(dict(pid=process.pid, started=started)) + '\n')
        rc = process.wait()
    text = (out / 'uart_transcript.txt').read_text(encoding='utf-8', errors='replace')
    rows = list(csv.DictReader((out / 'hardware_tokens.csv').open()))
    ok = rc == 0 and '@END 1 produced=2 reason=limit status=0' in text and len(rows) == 2
    result = dict(status='PASS' if ok else 'FAIL', returncode=rc, generated=rows,
                  seconds=round(time.time() - started, 2), prompt='你好', template='raw' if args.raw else 'chat',
                  real_ps_firmware=True, actual_compiled_rtl=True, scripted_tokens=False,
                  vendor_operators='existing behavioral math/XPM/DSP simulation models',
                  numeric_model_signoff=False, board_tested=False)
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not ok: raise SystemExit(1)


if __name__ == '__main__': main()
