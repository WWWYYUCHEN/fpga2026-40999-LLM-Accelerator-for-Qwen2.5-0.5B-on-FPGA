"""Hash the delivered immutable files, verify them, and produce a portable zip."""
from pathlib import Path
import hashlib
import json
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).parent))
from package_interactive import PACKAGE


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def selected():
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file(): continue
        parts = path.relative_to(PACKAGE).parts
        if path.name == 'SHA256SUMS' or '__pycache__' in parts or parts[0] == 'RUNS': continue
        if parts[:4] == ('SOURCE', 'software', 'interactive', 'build'): continue
        if parts[:4] == ('SOURCE', 'software', 'interactive', 'reports'): continue
        yield path


def main():
    config = json.loads((PACKAGE / 'board_config.json').read_text(encoding='utf-8'))
    for item in config['model_assets']:
        path = PACKAGE / 'SD_CARD' / item['file']
        assert path.stat().st_size == item['size'] and digest(path) == item['sha256'], path
    for name in ['QWENSD.ELF', 'QWEN25.BIN', 'boot.scr', 'QWEN25/TOKEN.BIN']:
        assert (PACKAGE / 'SD_CARD' / name).is_file()
    for name in ['tests/rtl_bridge.cpp','tests/board_sim.c','tools/build_rtl_sim.py','tools/run_timing_regression.tcl','tools/export_verified_hardware.tcl']:
        source=PACKAGE/'src/software/interactive'/name
        assert source.is_file(),source
    status=json.loads((PACKAGE/'report/PROJECT_STATUS.json').read_text(encoding='utf-8'))
    assert status.get('rtl_regression',{}).get('complete') and status['hardware_bitstream_matches_simulator']
    sim=digest(PACKAGE/'sim/simulator/qwen_rtl_sim.exe')
    gui=json.loads((PACKAGE/'sim/reports/gui_rtl_acceptance.json').read_text(encoding='utf-8'))
    assert gui['status']=='PASS' and gui['exe_sha256']==sim
    promoted=json.loads((PACKAGE/'sim/reports/promoted_rtl.json').read_text(encoding='utf-8'))
    for name,expected in promoted['hardware_sha256'].items():assert digest(PACKAGE/name)==expected,name
    a53=json.loads((PACKAGE/'sim/reports/a53_build.json').read_text(encoding='utf-8'))
    assert digest(PACKAGE/'board/deployment/QWENSD.ELF')==a53['sha256']
    bit_hash=digest(PACKAGE/'build/hardware/qwen25_system_100mhz.bit')
    with zipfile.ZipFile(PACKAGE/'build/hardware/qwen25_system_100mhz.xsa') as platform:
        embedded=[name for name in platform.namelist() if name.endswith('.bit')]
        assert embedded and any(hashlib.sha256(platform.read(name)).hexdigest()==bit_hash for name in embedded)
    files = list(selected())
    lines = [f"{digest(p)}  {p.relative_to(PACKAGE).as_posix()}\n" for p in files]
    (PACKAGE / 'SHA256SUMS').write_text(''.join(lines), encoding='utf-8')
    for p in files: assert digest(p) == next(line.split('  ')[0] for line in lines if line.split('  ', 1)[1].rstrip('\n') == p.relative_to(PACKAGE).as_posix())
    archive = PACKAGE.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for p in [*files, PACKAGE / 'SHA256SUMS']:
            output.write(p, PACKAGE.name + '/' + p.relative_to(PACKAGE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    result = dict(status='PASS', immutable_files=len(files), model_asset_hashes='all matched existing 100 MHz assets',
                  archive=str(archive), archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
                  runtime_logs_excluded=True, source_rebuild_outputs_excluded=True)
    archive.with_suffix('.zip.sha256').write_text(result['archive_sha256'] + '  ' + archive.name + '\n', encoding='ascii')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
