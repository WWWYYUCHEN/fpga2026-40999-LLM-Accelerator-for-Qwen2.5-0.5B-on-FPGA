"""Promote only RTL and hardware whose complete regression artifacts passed."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import zipfile
import sys
sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT,DEPLOY
from package_interactive import PACKAGE


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_source_lineage(status,timing):
    """Bind every passing executable and implemented RTL source to this delivery."""
    candidates={p.name:p for p in (ROOT/'rtl_candidate').glob('*.v')}
    assert {name:sha(p) for name,p in candidates.items()}==status['candidate_sha256']
    records=[]
    for name,result in status['final_system_regressions'].items():
        folder=ROOT/'reports/system_regression'/name
        assert sha(folder/'qwen_rtl_sim.exe')==result['exe_sha256'],name
        provenance=json.loads((folder/'build_provenance.json').read_text(encoding='utf-8'))
        assert provenance['exe_sha256']==result['exe_sha256'],name
        for item in provenance['sources']:
            source=Path(item['path'])
            assert source.is_file() and sha(source)==item['frozen_sha256'],source
        records.append(dict(scenario=name,exe_sha256=result['exe_sha256'],sources_verified=True))
    rtl_records=[]
    for original in (DEPLOY/'source/rtl').rglob('*'):
        if original.suffix not in ['.v','.sv']:continue
        source=candidates.get(original.name,original)
        expected=sha(source)
        ip_source=timing/'ip_repo/qwen25_accel/src'/original.name
        assert ip_source.is_file() and sha(ip_source)==expected,ip_source
        generated=list((timing/'hardware/vivado/qwen25_kv260.gen/sources_1/bd/qwen25_system/ipshared').rglob(original.name))
        assert generated and all(sha(p)==expected for p in generated),original.name
        rtl_records.append(dict(file=str(original.relative_to(DEPLOY/'source/rtl')),sha256=expected))
    result=dict(status='PASS',scenarios=records,rtl_sources=rtl_records)
    (ROOT/'reports/source_lineage.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    status=json.loads((ROOT/'reports/regression_status.json').read_text(encoding='utf-8'))
    assert status['complete'],'Complete functional, numeric and timing regression is required before promotion'
    timing=Path(os.environ.get('QWEN_BUILD_ROOT', Path(__file__).resolve().parents[4] / 'build'))
    for name in ['qwen25_system_fixed_100mhz.bit','qwen25_system_fixed_100mhz.bin','qwen25_system_fixed_100mhz.xsa']:
        assert (timing/'hardware'/name).is_file(),name
    verify_source_lineage(status,timing)
    backup=DEPLOY/'backups/before_full_system_fixes_20261020'
    backup.mkdir(exist_ok=True)
    modifications=[]
    for candidate in (ROOT/'rtl_candidate').glob('*.v'):
        relative=next((DEPLOY/'source/rtl').rglob(candidate.name)).relative_to(DEPLOY/'source/rtl')
        destinations=[DEPLOY/'source/rtl'/relative,DEPLOY.parents[1]/'rtl_2lane_bram'/relative,
                      DEPLOY/'ip_repo/qwen25_accel/src'/candidate.name]
        original=ROOT/'reports'/('norm_word_original.v' if candidate.name=='qwen25_qkv_controller.v' else 'afu_service_original.v')
        for target in destinations:
            if not target.is_file():raise FileNotFoundError(target)
            previous=sha(target)
            assert previous in [sha(original),sha(candidate)],f'Concurrent source edit requires reconciliation: {target}'
            saved=backup/str(len(modifications))/target.name;saved.parent.mkdir(parents=True,exist_ok=True)
            if not saved.exists():shutil.copy2(target,saved)
            shutil.copy2(candidate,target)
            shutil.copy2(ROOT/'rtl_candidate'/(candidate.stem+'_spec.md'),target.with_name(candidate.stem+'_spec.md'))
            modifications.append(dict(path=str(target),before_sha256=previous,after_sha256=sha(target),backup=str(saved)))
        package_rtl=PACKAGE/'src/rtl'/relative
        package_rtl.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(candidate,package_rtl)
        shutil.copy2(ROOT/'rtl_candidate'/(candidate.stem+'_spec.md'),package_rtl.with_name(candidate.stem+'_spec.md'))
    for original,name in [('qwen25_system_fixed_100mhz.bin','board/deployment/QWEN25.BIN'),
                          ('qwen25_system_fixed_100mhz.bit','build/hardware/qwen25_system_100mhz.bit'),
                          ('qwen25_system_fixed_100mhz.xsa','build/hardware/qwen25_system_100mhz.xsa')]:
        destination=PACKAGE/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(timing/'hardware'/original,destination)
    # IP files retain relative paths; promote the checksummed metadata with its RTL.
    shutil.copy2(timing/'ip_repo/qwen25_accel/component.xml',DEPLOY/'ip_repo/qwen25_accel/component.xml')
    ip_folder=timing/'ip_repo/qwen25_accel'
    with zipfile.ZipFile(PACKAGE/'build/hardware/qwen25_accel_fixed_ip.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in ip_folder.rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(ip_folder).as_posix())
    for target_name,source_name in [('qwen25_system_fixed_100mhz.bit','qwen25_system_fixed_100mhz.bit'),
                                    ('qwen25_system_fixed_100mhz.bin','qwen25_system_fixed_100mhz.bin'),
                                    ('qwen25_system_fixed_100mhz.xsa','qwen25_system_fixed_100mhz.xsa')]:
        shutil.copy2(timing/'hardware'/source_name,DEPLOY/'hardware'/target_name)
    for manifest_path in [DEPLOY/'reports/rtl_source_manifest.json']:
        if manifest_path.exists():
            saved=backup/manifest_path.name
            if not saved.exists():shutil.copy2(manifest_path,saved)
            manifest=json.loads(manifest_path.read_text())
            for relative in list(manifest):
                source=DEPLOY/'source/rtl'/relative
                if not source.exists():
                    found=list((DEPLOY/'source/rtl').rglob(relative))
                    if len(found)==1:source=found[0]
                if source.exists():manifest[relative]=sha(source)
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for p in [PACKAGE/'board_config.json',PACKAGE/'board/deployment/QWEN25/CONFIG.JSON']:
        cfg=json.loads(p.read_text(encoding='utf-8'))
        cfg['validation_status'].update(setup_wns_ns=float(status['timing']['setup_wns_ns']),
                                        hold_whs_ns=float(status['timing']['hold_whs_ns']),
                                        generated_text='RTL regression vectors passed',board_tested=False)
        cfg['validated_rtl']='source fixes from 2026-10-20 system regression'
        p.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report=PACKAGE/'sim/reports/timing_fixed';report.mkdir(parents=True,exist_ok=True)
    for p in (timing/'reports').iterdir():
        if p.is_file() and p.suffix in ['.rpt','.txt']:shutil.copy2(p,report/p.name)
    shutil.copy2(timing/'reports/vivado_regression.log',report/'vivado_regression.log')
    shutil.copy2(timing/'scripts/run_timing_regression.tcl',report/'run_timing_regression_executed.tcl')
    if (timing/'reports/hardware_export.log').exists():
        shutil.copy2(timing/'reports/hardware_export.log',report/'hardware_export.log')
        shutil.copy2(timing/'scripts/export_verified_hardware.tcl',report/'export_verified_hardware_executed.tcl')
    p=PACKAGE/'report/PROJECT_STATUS.json';pkg=json.loads(p.read_text(encoding='utf-8'))
    pkg['historical_experiments']={name:pkg.pop(name) for name in ['rtl_integration','rtl_raw_integration','rtl_math_comparison','gui_integration'] if name in pkg}
    pkg.update(default_backend='accelerator RTL simulation',hardware_bitstream_matches_simulator=True,
               rtl_regression=status,rtl_signoff=True,strict_style_signoff=False,
               gui_integration=json.loads((ROOT/'reports/gui_rtl_acceptance.json').read_text(encoding='utf-8')),
               numeric_model_signoff='passed defined regression vectors; not a formal bit-exact proof',board_tested=False)
    p.write_text(json.dumps(pkg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    result=dict(status='PROMOTED',source_changes=modifications,hardware_sha256={name:sha(PACKAGE/name) for name in ['board/deployment/QWEN25.BIN','build/hardware/qwen25_system_100mhz.bit','build/hardware/qwen25_system_100mhz.xsa']})
    (ROOT/'reports/promoted_rtl.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(ROOT/'reports/promoted_rtl.json',PACKAGE/'sim/reports/promoted_rtl.json')
    shutil.copy2(ROOT/'reports/source_lineage.json',PACKAGE/'sim/reports/source_lineage.json')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
