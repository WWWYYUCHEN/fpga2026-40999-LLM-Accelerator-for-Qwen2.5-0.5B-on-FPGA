"""Clone the implemented board project; keep the original checkout untouched."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[4]
DEPLOY = Path(os.environ.get('QWEN_SOURCE_ROOT', PROJECT))
ROOT = PROJECT / 'src/software/interactive'
TARGET = Path(os.environ.get('QWEN_TIMING_WORKDIR', PROJECT / 'build/repro_timing'))


def main():
    if not (TARGET / 'CLONE_COMPLETE.json').exists():
        TARGET.mkdir(exist_ok=True)
        for folder in ['hardware/vivado', 'ip_repo', 'board_files']:
            shutil.copytree(DEPLOY / folder, TARGET / folder, dirs_exist_ok=True)
        for name in ['qwen25_pl0_100mhz.xdc', 'qwen25_system.tcl']:
            shutil.copy2(DEPLOY / 'hardware' / name, TARGET / 'hardware' / name)
        (TARGET / 'reports').mkdir(exist_ok=True)
        (TARGET / 'scripts').mkdir(exist_ok=True)
        (TARGET / 'CLONE_COMPLETE.json').write_text(json.dumps(dict(original=str(DEPLOY), clone=str(TARGET)))+'\n')
    for candidate in (ROOT/'rtl_candidate').glob('*.v'):
        shutil.copy2(candidate,TARGET/'ip_repo/qwen25_accel/src'/candidate.name)
    shutil.copy2(ROOT / 'tools/run_timing_regression.tcl', TARGET / 'scripts/run_timing_regression.tcl')
    shutil.copy2(ROOT / 'tools/export_verified_hardware.tcl', TARGET / 'scripts/export_verified_hardware.tcl')
    # Short paths are required by Vivado's generated subprocesses on Windows.
    mapped = subprocess.run(['subst'], text=True, stdout=subprocess.PIPE, check=True).stdout
    existing=[line for line in mapped.splitlines() if line.startswith('R:')]
    if existing:
        if str(TARGET).lower() not in existing[0].lower():
            raise RuntimeError('R: is occupied; choose another short-path mapping')
    else:subprocess.run(['subst', 'R:', str(TARGET)], check=True)
    print('ISOLATED_TIMING_PROJECT', TARGET, flush=True)


if __name__ == '__main__': main()
