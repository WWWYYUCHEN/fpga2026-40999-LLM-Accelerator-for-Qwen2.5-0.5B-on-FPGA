"""Execute production PS firmware plus the RTL accelerator as one simulated board."""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import sys
sys.path.insert(0,str(Path(__file__).parent))
from build import ROOT, DEPLOY


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--prompt',default='你好')
    ap.add_argument('--tokens',type=int,default=2)
    ap.add_argument('--raw',action='store_true')
    ap.add_argument('--requests',type=int,default=1)
    ap.add_argument('--dump-layers',action='store_true')
    args=ap.parse_args()
    assert args.name.replace('_','').replace('-','').isalnum()
    out=ROOT/'reports/system_regression'/args.name
    out.mkdir(parents=True,exist_ok=True)
    exe=ROOT/'build/rtl_current/qwen_rtl_sim.exe'
    shutil.copy2(exe,out/'qwen_rtl_sim.exe')
    shutil.copy2(ROOT/'build/rtl_current/segments.tsv',out/'segments.tsv')
    shutil.copy2(ROOT/'reports/rtl_bridge_build.json',out/'build_provenance.json')
    command=('/template raw\n' if args.raw else '')+'/mode events\n'+f'/max {args.tokens}\n'+(args.prompt+'\n')*args.requests
    (out/'input.txt').write_text(command,encoding='utf-8')
    env=dict(os.environ,SIM_SD=str(DEPLOY/'sd_card'),SIM_TRACE=str(out/'mmio_trace.txt'))
    if args.dump_layers:env['RTL_DUMP_LAYERS']='1'
    start=time.time()
    with (out/'uart.log').open('wb') as log:
        p=subprocess.Popen([str(out/'qwen_rtl_sim.exe')],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env)
        (out/'process.json').write_text(json.dumps(dict(pid=p.pid,started=start,name=args.name))+'\n')
        p.stdin.write(command.encode('utf-8'));p.stdin.close()
        rc=p.wait()
    text=(out/'uart.log').read_text(encoding='utf-8',errors='replace')
    tokens=list(csv.DictReader((out/'hardware_tokens.csv').open()))
    ends=[line for line in text.splitlines() if line.startswith('@END ')]
    requests=re.findall(r'@BEGIN (\d+) prompt_tokens=(\d+).*?@END \1 produced=(\d+) reason=(\w+) status=(-?\d+)',text,re.S)
    expected_layers=sum(24*((int(n)+7)//8)+24*max(int(produced)-1,0) for _,n,produced,_,_ in requests)
    observed_layers=len(re.findall(r'LAYER_DONE iteration=',text))
    verified_logits=len(re.findall(r'RTL_TOKEN_VERIFIED iteration=',text))
    passed=rc==0 and len(ends)==args.requests and all('status=0' in line for line in ends) and len(tokens)>0 and observed_layers==expected_layers and verified_logits==len(tokens)
    result=dict(status='PASS' if passed else 'FAIL',returncode=rc,prompt=args.prompt,
                template='raw' if args.raw else 'chat',requests=args.requests,requested_tokens=args.tokens,
                generated=tokens,end_events=ends,seconds=round(time.time()-start,2),
                expected_layer_completions=expected_layers,observed_layer_completions=observed_layers,
                independent_logit_verifications=verified_logits,
                exe_sha256=hashlib.sha256((out/'qwen_rtl_sim.exe').read_bytes()).hexdigest(),
                real_ps_c=True,actual_rtl_computation=True,cpu_model_substitution=False,board_tested=False)
    (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
