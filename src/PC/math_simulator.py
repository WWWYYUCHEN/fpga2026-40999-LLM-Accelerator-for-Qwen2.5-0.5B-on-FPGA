"""Bridge the unmodified PS main loop to a numerical PL model via CSR requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from model_backend import Model


def serve(model, folder, process, journal):
    request = folder / 'request.json'
    while process.poll() is None:
        if not request.exists(): time.sleep(0.002); continue
        data = json.loads(request.read_text())
        request.unlink()
        start = time.perf_counter()
        try:
            if data['clear']: model.clear()
            token = model.forward(data['ids'], data['position'])
            error = 0
        except Exception as exception:
            print('MODEL_ERROR', str(exception), file=sys.stderr, flush=True)
            token, error = 0, 1
        record = dict(**data, token=token, error=error, seconds=round(time.perf_counter()-start, 6))
        with journal.open('a', encoding='utf-8') as f: f.write(json.dumps(record) + '\n')
        temp = folder / 'response.tmp'
        temp.write_text(f"{data['serial']} {token} {error}", encoding='ascii')
        temp.replace(folder / 'response.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', write_through=True)
    print('SIMULATION: numerical PL model; executes deployed weights on PC CPU; not RTL or a real KV260.', flush=True)
    model = Model(args.package / 'board/deployment')
    args.run.mkdir(parents=True, exist_ok=True)
    io = args.run / 'model_io'; io.mkdir(exist_ok=True)
    env = dict(os.environ, SIM_SD=str(args.package / 'board/deployment'), MODEL_IO=str(io), SIM_TRACE=str(args.run / 'mmio_trace.txt'))
    process = subprocess.Popen([str(args.package / 'sim/simulator/qwen_math_sim.exe')], env=env,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    def input_loop():
        try:
            while True:
                data = sys.stdin.buffer.read1(4096)
                if not data: process.stdin.close(); return
                process.stdin.write(data); process.stdin.flush()
        except (OSError, ValueError): return
    threading.Thread(target=input_loop, daemon=True).start()
    threading.Thread(target=serve, args=(model, io, process, args.run / 'model_transactions.jsonl'), daemon=True).start()
    try:
        with (args.run / 'uart.log').open('wb') as log:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk: break
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
        raise SystemExit(process.wait())
    finally:
        if process.poll() is None: process.terminate()


if __name__ == '__main__': main()
