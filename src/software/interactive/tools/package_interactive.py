"""Assemble the runnable simulation and the matching interactive SD payload."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build import ROOT, DEPLOY

PROJECT = Path(__file__).resolve().parents[4]
PACKAGE = Path(os.environ.get('QWEN_PACKAGE_ROOT', PROJECT / 'build/package'))
BASE = Path(os.environ.get('QWEN_BASE_ROOT', PROJECT))


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.stat().st_size == target.stat().st_size and source.stat().st_mtime_ns == target.stat().st_mtime_ns:
        return
    shutil.copy2(source, target)


def main():
    for name in ['DESC.BIN', 'SCALE.BIN', 'BIAS.BIN', 'GAMMA.BIN', 'WEIGHTS.BIN', 'LMW.BIN', 'EMBED.BIN']:
        copy(DEPLOY / 'sd_card/QWEN25' / name, PACKAGE / 'board/deployment/QWEN25' / name)
    copy(ROOT / 'TOKEN.BIN', PACKAGE / 'board/deployment/QWEN25/TOKEN.BIN')
    copy(ROOT / 'build/a53/QWENSD.ELF', PACKAGE / 'board/deployment/QWENSD.ELF')
    for name in ['QWEN25.BIN', 'boot.cmd', 'boot.scr']:
        copy(BASE / 'SD_CARD' / name, PACKAGE / 'SD_CARD' / name)
    copy(BASE / 'build/hardware/qwen25_system_100mhz.xsa', PACKAGE / 'build/hardware/qwen25_system_100mhz.xsa')
    for name in ['qwen_board_sim.exe', 'qwen_math_sim.exe', 'qwen_ps.dll']:
        copy(ROOT / 'build/native' / name, PACKAGE / 'SIMULATOR' / name)
    for name in ['qwen_rtl_sim.exe', 'segments.tsv']:
        copy(ROOT / 'build/rtl_current' / name, PACKAGE / 'SIMULATOR' / name)
    for name in ['chat_gui.py', 'model_backend.py', 'math_simulator.py', 'run_cli.py']:
        copy(ROOT / 'tools' / name, PACKAGE / 'PC' / name)
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ['.c', '.cpp', '.h', '.py', '.md', '.v', '.sv', '.tcl'] and not any(x in ['build', 'reports', '__pycache__'] for x in p.relative_to(ROOT).parts):
            copy(p, PACKAGE / 'src/software/interactive' / p.relative_to(ROOT))
    for name in ['board_parameters.h', 'sd_prepare.c', 'sd_entry.S']:
        copy(DEPLOY / 'software' / name, PACKAGE / 'src/software' / name)
    copy(ROOT / 'TOKEN.BIN', PACKAGE / 'src/software/interactive/TOKEN.BIN')
    copy(ROOT / 'TOKEN.json', PACKAGE / 'src/software/interactive/TOKEN.json')
    copy(DEPLOY / 'board_config.json', PACKAGE / 'src/board_config.json')
    copy(Path(r'src/reference/tokenizer.json'), PACKAGE / 'src/reference/tokenizer.json')
    bsp = DEPLOY / 'sw_100mhz/kv260_platform_100mhz/export/kv260_platform_100mhz/sw/standalone_sd'
    for name in ['include', 'lib']:
        for p in (bsp / name).rglob('*'):
            if p.is_file(): copy(p, PACKAGE / 'src/BSP' / name / p.relative_to(bsp / name))
    copy(bsp / 'Xilinx.spec', PACKAGE / 'src/BSP/Xilinx.spec')
    copy(DEPLOY / 'sw_100mhz/qwen25_sd_100mhz/src/lscript.ld', PACKAGE / 'src/BSP/lscript.ld')
    for p in (DEPLOY / 'source/rtl').rglob('*'):
        if p.is_file(): copy(p, PACKAGE / 'src/rtl' / p.relative_to(DEPLOY / 'source/rtl'))
    project = DEPLOY.parents[1]
    models = Path('synth/operators_400mhz/verification/regression/models')
    for p in (project / models).iterdir():
        if p.is_file() and p.suffix in ['.v', '.sv', '.cpp']:
            copy(p, PACKAGE / 'SOURCE' / models / p.name)
    bench = Path('verification/verification/tb_qwen25_chat_generation.sv')
    copy(project / bench, PACKAGE / 'SOURCE' / bench)
    config = json.loads((BASE / 'board_config.json').read_text(encoding='utf-8'))
    config.update(prompt_source='UART UTF-8 input', tokenizer='QWEN25/TOKEN.BIN',
                  application='interactive; each request clears KV context', generate_tokens_default=20)
    for name in ['prompt', 'prompt_ids', 'generate_tokens']: config.pop(name, None)
    for target in [PACKAGE / 'board_config.json', PACKAGE / 'board/deployment/QWEN25/CONFIG.JSON']:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for name in ['a53_build.json', 'a53_build.log', 'native_build.log', 'verification.json', 'rtl_bridge_build.json', 'TOKEN.json', 'rtl_gate_summary.json', 'candidate_static_gate_summary.json', 'model_weight_validation.json', 'gui_acceptance.json', 'gui_acceptance.txt', 'rtl_math_comparison.json',
                 'regression_status.json','regression_fixes.patch','norm_word_issue.json','parameter_last_issue.json','final_all_layer_numeric_comparison.json','final_decode_layer_numeric_comparison.json','gui_rtl_acceptance.json','gui_rtl_acceptance.txt','hardware_export_validation.json']:
        p = ROOT / 'reports' / name
        if name == 'TOKEN.json': p = ROOT / name
        if p.exists(): copy(p, PACKAGE / 'REPORTS' / name)
    for folder in ['simulation', 'rtl_current_acceptance', 'rtl_current_raw_acceptance', 'math_acceptance','parameter_fixed','system_regression']:
        if not (ROOT / 'reports' / folder).exists(): continue
        for p in (ROOT / 'reports' / folder).rglob('*'):
            if p.is_file() and p.suffix!='.exe' and not any(x in ['sd', 'model_io'] for x in p.relative_to(ROOT / 'reports' / folder).parts):
                copy(p, PACKAGE / 'REPORTS' / folder / p.relative_to(ROOT / 'reports' / folder))
    status = dict(ps_interactive_implemented=True, tokenizer_on_ps=True, utf8_streaming=True,
                  a53_elf_built=True, mathematical_simulation='deployed INT8/BF16 assets, CPU numerical PL substitute',
                  rtl_simulation='compiled current deployed RTL, behavioral vendor IP models',
                  rtl_signoff=False, numeric_model_signoff=False, board_tested=False)
    status['default_backend']='accelerator RTL simulation'
    status['hardware_bitstream_matches_simulator']=False
    status['rtl_candidate']='src/software/interactive/rtl_candidate'
    result = ROOT / 'reports/rtl_current_acceptance/result.json'
    if result.exists(): status['rtl_integration'] = json.loads(result.read_text(encoding='utf-8'))
    result = ROOT / 'reports/math_acceptance/result.json'
    if result.exists(): status['mathematical_integration'] = json.loads(result.read_text(encoding='utf-8'))
    result = ROOT / 'reports/rtl_current_raw_acceptance/result.json'
    if result.exists(): status['rtl_raw_integration'] = json.loads(result.read_text(encoding='utf-8'))
    result = ROOT / 'reports/gui_acceptance.json'
    if result.exists(): status['gui_integration'] = json.loads(result.read_text(encoding='utf-8'))
    result = ROOT / 'reports/rtl_math_comparison.json'
    if result.exists(): status['rtl_math_comparison'] = json.loads(result.read_text(encoding='utf-8'))
    destination = PACKAGE / 'report/PROJECT_STATUS.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(status, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (PACKAGE / 'START_GUI.pyw').write_text("from pathlib import Path\nimport runpy, sys\np=Path(__file__).resolve().parent\nsys.path.insert(0,str(p/'PC'))\nsys.argv=[str(p/'PC/chat_gui.py'),'--package',str(p)]\nrunpy.run_path(str(p/'PC/chat_gui.py'),run_name='__main__')\n", encoding='utf-8')
    (PACKAGE / 'PC/requirements_math.txt').write_text('torch>=2.6\nnumpy>=1.24\npyserial>=3.5\n', encoding='ascii')
    (PACKAGE / 'PC/requirements_verify.txt').write_text('tokenizers==0.22.1\n', encoding='ascii')
    print(PACKAGE, flush=True)


if __name__ == '__main__': main()
