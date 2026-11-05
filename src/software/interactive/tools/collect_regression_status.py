"""Create a reviewable status from actual completed regression artifacts."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT, DEPLOY


def main():
    project=ROOT.parents[2]
    build=Path(os.environ.get('QWEN_BUILD_ROOT', project/'build'))
    timing=build/'timing'
    hardware=build/'hardware'
    report=project/'sim/reports'
    final_names=['final_all_layers','final_chat_nihao','final_chat_eos','final_repeat_context_reset','final_rows9_prefill','final_decode_trace']
    scenarios={}
    for name in final_names:
        folder=report/'system_regression'/name
        p=folder/'result.json'
        if p.exists():scenarios[name]=json.loads(p.read_text(encoding='utf-8'))
        else:
            log=folder/'uart.log'
            text=log.read_text(encoding='utf-8',errors='replace') if log.exists() else ''
            cycles=re.findall(r'PROGRESS cycle=(\d+)',text)
            tokens=re.findall(r'@TOKEN (\d+) (\d+) (\d+)',text)
            scenarios[name]=dict(status='RUNNING',last_reported_cycle=int(cycles[-1]) if cycles else 0,committed_tokens=len(tokens))
    params=report/'parameter_fixed/parameter_ram_validation.json'
    numeric=report/'final_all_layer_numeric_comparison.json'
    timing_result=timing/'timing_result.txt'
    result=dict(default_backend='accelerator RTL simulation',cpu_model_substitution=False,
                bugs=[dict(id='normalization_word',cause='RAM sampled pre-increment word; one-word cyclic shift',fixed=True),
                      dict(id='parameter_last',cause='TLAST bypassed the registered payload; last Gamma word lost',fixed=True),
                      dict(id='duplicate_error_driver',cause='AFU error output had redundant constant drivers',fixed=True)],
                candidate_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'rtl_candidate').glob('*.v')},
                final_system_regressions=scenarios,board_tested=False)
    if params.exists():result['parameter_ram']=json.loads(params.read_text())
    if numeric.exists():
        p=json.loads(numeric.read_text());result['numeric_summary']=dict(status=p['status'],lm_head=p['lm_head'],final_norm=p['final_norm'])
    result['timing']=dict(status='RUNNING')
    if timing_result.exists():
        values=dict(line.split('=',1) for line in timing_result.read_text().splitlines() if '=' in line)
        result['timing']=dict(status='PASS' if float(values['setup_wns_ns'])>=0 and float(values['hold_whs_ns'])>=0 else 'FAIL',**values)
    decode=report/'final_decode_layer_numeric_comparison.json'
    if decode.exists():
        p=json.loads(decode.read_text());result['decode_numeric_summary']=dict(status=p['status'],lm_head=p['lm_head'],final_norm=p['final_norm'])
    offline={}
    for name in ['final_all_layers','final_chat_nihao','final_chat_eos','final_rows9_prefill']:
        p=report/'system_regression'/name/'numeric_comparison.json'
        if p.exists():offline[name]=json.loads(p.read_text(encoding='utf-8'))['status']
    result['offline_reference_checks']=offline
    exports={}
    for name in ['qwen25_system_fixed_100mhz.bit','qwen25_system_fixed_100mhz.bin','qwen25_system_fixed_100mhz.xsa']:
        p=hardware/name
        fresh=p.is_file() and timing_result.is_file() and p.stat().st_mtime>=timing_result.stat().st_mtime
        exports[name]=dict(status='PASS' if fresh else 'PENDING',bytes=p.stat().st_size if p.is_file() else 0)
    result['hardware_exports']=exports
    implementation_logs=[timing/name for name in ['vivado_regression.log','hardware_export.log']]
    implemented=any(bool(re.search(r'^REGRESSION_IMPLEMENTATION_PASS setup=',p.read_text(errors='replace'),re.M)) for p in implementation_logs if p.exists())
    result['implementation_export_complete']=implemented
    result['complete']=all(x.get('status')=='PASS' for x in scenarios.values()) and result['timing']['status']=='PASS' and result.get('numeric_summary',{}).get('status')=='PASS' and result.get('decode_numeric_summary',{}).get('status')=='PASS' and len(offline)==4 and all(x=='PASS' for x in offline.values()) and all(x['status']=='PASS' for x in exports.values()) and implemented
    (report/'regression_status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=result['complete'],timing=result['timing'],
                          system={k:dict(status=v['status'],tokens=len(v.get('generated',[])) if 'generated' in v else v.get('committed_tokens',0),cycle=v.get('last_reported_cycle')) for k,v in scenarios.items()}),indent=2,ensure_ascii=False))


if __name__=='__main__':main()
