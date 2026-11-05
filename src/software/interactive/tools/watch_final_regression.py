"""Finish dependent reference checks as RTL scenarios complete in this run."""
from pathlib import Path
import json
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT,DEPLOY

PROJECT=ROOT.parents[2]


def run(command,label):
    log_dir=PROJECT/'sim/reports'
    log_dir.mkdir(parents=True,exist_ok=True)
    with (log_dir/f'{label}.log').open('wb') as log:
        p=subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(label+' failed; see its log')


def main():
    reference_python=Path(sys.executable)
    names=['final_all_layers','final_chat_nihao','final_chat_eos','final_rows9_prefill']
    while True:
        for name in names:
            folder=PROJECT/'sim/reports/system_regression'/name
            p=folder/'result.json'
            if not p.exists():continue
            result=json.loads(p.read_text(encoding='utf-8'))
            if result['status']!='PASS':raise RuntimeError('RTL system scenario failed: '+name)
            if (folder/'numeric_comparison.json').exists():continue
            cmd=[reference_python,ROOT/'tests/compare_system_numeric.py',ROOT,PROJECT/'board/deployment',folder]
            if name=='final_chat_eos':cmd+=['--require-eos']
            run(cmd,'offline_'+name)
            print('OFFLINE_NUMERIC_PASS',name,flush=True)
        run([sys.executable,ROOT/'tools/collect_regression_status.py'],'current_status')
        state=json.loads((PROJECT/'sim/reports/regression_status.json').read_text(encoding='utf-8'))
        if state['complete']:
            print('REQUIRED_REGRESSION_COMPLETE',flush=True)
            break
        if state['timing']['status']=='FAIL':raise RuntimeError('Post-route timing failed; correction required')
        time.sleep(20)


if __name__=='__main__':main()
