"""Assemble a matching delivery only after all required regression gates finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    while True:
        path=ROOT/'reports/regression_status.json'
        try:
            status=json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):
            status={}
        failed=[name for name,item in status.get('final_system_regressions',{}).items() if item.get('status')=='FAIL']
        failed += [name for name,value in status.get('offline_reference_checks',{}).items() if value=='FAIL']
        if status.get('timing',{}).get('status')=='FAIL':failed.append('post-route timing')
        if failed:raise RuntimeError('Regression failed: '+', '.join(failed))
        if status.get('complete'):break
        if not args.wait:raise RuntimeError('Regression is incomplete; use --wait to wait for the active tests')
        time.sleep(20)
    for name in ['package_interactive.py','publish_verified_rtl.py','write_regression_report.py','finalize_interactive.py']:
        print('DELIVERY_STEP',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'tools'/name)],check=True)
    print('VERIFIED_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':main()
