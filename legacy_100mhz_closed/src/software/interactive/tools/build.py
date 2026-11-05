"""Build the real A53 ELF and a native board-interface simulator from the same C."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parents[1]


def run(args, log):
    log.append(subprocess.list2cmdline([str(x) for x in args]))
    p = subprocess.run([str(x) for x in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding='utf-8', errors='replace')
    log.append(p.stdout)
    if p.returncode:
        raise RuntimeError(p.stdout)


def build_native():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    cc = Path(os.environ.get('QWEN_HOST_CC') or shutil.which('gcc') or 'gcc')
    if not cc.exists():
        cc = Path(shutil.which('gcc') or 'gcc')
    out = ROOT / 'build/native'
    out.mkdir(parents=True, exist_ok=True)
    log = []
    common = [cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-DUTF8PROC_STATIC', '-static-libgcc']
    core = [ROOT / p for p in ['tokenizer.c', 'utf8_stream.c', 'generation.c', 'vendor/utf8proc/utf8proc.c']]
    run([*common, '-shared', *core, '-o', out / 'qwen_ps.dll'], log)
    run([*common, '-DQWEN_HOST_SIM', '-I', ROOT / 'tests/shim', '-I', ROOT,
         ROOT / 'main.c', ROOT / 'tests/board_sim.c', *core, '-o', out / 'qwen_board_sim.exe'], log)
    run([*common, '-Wno-unused-variable', '-DQWEN_HOST_SIM', '-DQWEN_MODEL_SIM', '-I', ROOT / 'tests/shim', '-I', ROOT,
         ROOT / 'main.c', ROOT / 'tests/board_sim.c', *core, '-o', out / 'qwen_math_sim.exe'], log)
    (ROOT / 'reports/native_build.log').write_text('\n'.join(log), encoding='utf-8')
    print(out / 'qwen_board_sim.exe')


def build_a53():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    cc = Path(os.environ.get('QWEN_A53_CC') or shutil.which('aarch64-none-elf-gcc') or 'aarch64-none-elf-gcc')
    default_bsp = DEPLOY / 'BSP' if (DEPLOY / 'BSP').exists() else DEPLOY / 'sw_100mhz/kv260_platform_100mhz/export/kv260_platform_100mhz/sw/standalone_sd'
    bsp = Path(os.environ.get('QWEN_BSP', str(default_bsp)))
    linker_source = bsp / 'lscript.ld' if (bsp / 'lscript.ld').exists() else DEPLOY / 'sw_100mhz/qwen25_sd_100mhz/src/lscript.ld'
    out = ROOT / 'build/a53'
    out.mkdir(parents=True, exist_ok=True)
    linker = linker_source.read_text()
    linker = re.sub(r'_STACK_SIZE = .*?;', '_STACK_SIZE = 0x100000;', linker, count=1)
    linker = re.sub(r'_HEAP_SIZE = .*?;', '_HEAP_SIZE = 0x1000000;', linker, count=1)
    assert 'ENTRY(sd_entry)' in linker and 'ORIGIN = 0x00100000' in linker
    (out / 'lscript.ld').write_text(linker, encoding='ascii')
    common = [cc, '-DSDT', '-DUTF8PROC_STATIC', '-O2', '-g', '-std=c11', '-Wall', '-Wextra', '-Werror',
              '-isystem', bsp / 'include', '-specs=' + str(bsp / 'Xilinx.spec')]
    sources = [ROOT / p for p in ['main.c', 'tokenizer.c', 'utf8_stream.c', 'generation.c', 'vendor/utf8proc/utf8proc.c']]
    sources += [DEPLOY / 'software/sd_prepare.c', DEPLOY / 'software/sd_entry.S']
    log, objects = [], []
    try:
        for source in sources:
            obj = out / (source.stem + '.o')
            run([*common, '-c', source, '-o', obj], log)
            objects.append(obj)
        elf = out / 'QWENSD.ELF'
        run([*common, *objects, '-o', elf, '-Wl,-T,' + str(out / 'lscript.ld'),
             '-Wl,-Map,' + str(out / 'qwen_uart.map'), '-L', bsp / 'lib',
             '-Wl,--start-group', '-lxilstandalone', '-lxiltimer', '-lxilffs', '-lxil', '-lgcc', '-lc', '-Wl,--end-group'], log)
        prefix = str(cc).removesuffix('gcc.exe')
        run([prefix + 'size.exe', elf], log)
        run([prefix + 'readelf.exe', '-h', '-l', elf], log)
        import struct
        data = elf.read_bytes()
        phoff, = struct.unpack_from('<Q', data, 32)
        phsize, phcount = struct.unpack_from('<HH', data, 54)
        loads = []
        for i in range(phcount):
            typ, flags, off, virtual, physical, file_bytes, memory_bytes, align = struct.unpack_from('<II6Q', data, phoff+i*phsize)
            if typ == 1:
                assert 0x00100000 <= physical < physical + memory_bytes <= 0x04000000
                loads.append(dict(start=hex(physical), end=hex(physical+memory_bytes), file_bytes=file_bytes, memory_bytes=memory_bytes))
        assert loads
        report = dict(elf=str(elf), sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                      bsp=str(bsp), build='AArch64 EL1 baremetal, 100 MHz XSA',
                      heap_bytes=0x1000000, stack_bytes=0x100000, board_tested=False)
        report['load_regions'] = loads
        (ROOT / 'reports/a53_build.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(elf)
    finally:
        (ROOT / 'reports/a53_build.log').write_text('\n'.join(log), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=['native', 'a53', 'all'])
    args = ap.parse_args()
    if args.target in ['native', 'all']: build_native()
    if args.target in ['a53', 'all']: build_a53()
