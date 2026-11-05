"""Console entry. The PC transports UTF-8 only; PS C performs tokenization."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import datetime
import shutil
import threading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--backend', choices=['rtl', 'math'], default='rtl')
    ap.add_argument('--prompt')
    ap.add_argument('--tokens', type=int, default=20)
    args = ap.parse_args()
    package = args.package
    run = package / 'sim/runs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    if args.backend == 'math':
        runtime = os.environ.get('QWEN_SIM_PYTHON', sys.executable)
        command = [runtime, str(package / 'src/PC/math_simulator.py'), '--package', str(package), '--run', str(run)]
        env = os.environ.copy()
    else:
        shutil.copy2(package / 'sim/simulator/segments.tsv', run / 'segments.tsv')
        command = [str(package / 'sim/simulator/qwen_rtl_sim.exe')]
        env = dict(os.environ, SIM_SD=str(package / 'board/deployment'), SIM_TRACE=str(run / 'mmio_trace.txt'))
    process = subprocess.Popen(command, cwd=run, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if args.prompt is not None:
        if not 1 <= args.tokens <= 255 or '\n' in args.prompt or '\r' in args.prompt: ap.error('invalid tokens or multiline prompt')
        process.stdin.write((f'/max {args.tokens}\n' + args.prompt + '\n').encode('utf-8'))
        process.stdin.close()
    else:
        def send():
            try:
                for line in sys.stdin:
                    process.stdin.write(line.encode('utf-8')); process.stdin.flush()
                process.stdin.close()
            except (OSError, ValueError): pass
        threading.Thread(target=send, daemon=True).start()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', write_through=True)
    try:
        with (run / 'console.log').open('wb') as log:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk: break
                log.write(chunk); log.flush()
                sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
        return process.wait()
    finally:
        if process.poll() is None:
            if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else: process.terminate()


if __name__ == '__main__': raise SystemExit(main())
