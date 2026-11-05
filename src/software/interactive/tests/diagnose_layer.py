"""Find earliest numerical divergence using real RTL intermediate tensors."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_backend import Model
from rtl_numeric_reference import configure_reference,qkv_projection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('sd', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--prompt', default='你好')
    ap.add_argument('--reuse',action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.root / 'build/rtl_current/segments.tsv', args.output / 'segments.tsv')
    env = dict(os.environ, SIM_SD=str(args.sd), RTL_LAYER_DIAGNOSTIC='1')
    if not args.reuse:
        with (args.output / 'uart.log').open('wb') as log:
            p = subprocess.run([str(args.root / 'build/rtl_current/qwen_rtl_sim.exe')],
                               input=('/template raw\n/max 1\n'+args.prompt+'\n').encode('utf-8'), cwd=args.output,
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert p.returncode==0
    assert 'LAYER_DIAGNOSTIC_COMPLETE' in (args.output/'uart.log').read_text(encoding='utf-8')
    m = configure_reference(Model(args.sd))
    all_ids=list(map(int,(args.output/'prompt_ids.txt').read_text().split()))
    ids=all_ids[:8]
    assert 1<=len(ids)<=8, 'Layer diagnostic compares the first group of up to 8 rows'
    x = m.embedding[ids]; w = m.weights[0]
    n = m.norm(x, m.gamma[:896])
    q = qkv_projection(m,n,w['q'],m.bias[:896])
    k = qkv_projection(m,n,w['k'],m.bias[896:1024])
    v = qkv_projection(m,n,w['v'],m.bias[1024:1152])
    rows=len(ids)
    positions=torch.arange(rows).float()
    q=m.rope(q.reshape(rows,14,64),positions).reshape(rows,896)
    k=m.rope(k.reshape(rows,2,64),positions).reshape(rows,128)
    keys=k.reshape(rows,2,64).repeat_interleave(7,dim=1)
    values=v.reshape(rows,2,64).repeat_interleave(7,dim=1)
    scores=m.bf(torch.einsum('rhd,shd->hrs',q.reshape(rows,14,64),keys)/8.0)
    scores.masked_fill_(torch.arange(rows)[None,:]>positions[:,None],-float('inf'))
    probs=m.bf(torch.softmax(scores,dim=-1))
    ctx=m.bf(torch.einsum('hrs,shd->rhd',probs,values)).reshape(rows,896)
    h = m.bf(x + m.linear(ctx, w['o']))
    z = m.norm(h, m.gamma[896:1792])
    gate, up = m.linear(z, w['gate']), m.linear(z, w['up'])
    mid = m.bf(m.bf(torch.nn.functional.silu(gate)) * up)
    y = m.bf(h + m.linear(mid, w['down']))
    result = {}
    def compare(key, actual, expected):
        a, g = actual.flatten(), expected.flatten()
        result[key] = dict(max_abs=float((a-g).abs().max()), mean_abs=float((a-g).abs().mean()),
                           cosine=float(torch.nn.functional.cosine_similarity(a, g, dim=0)),
                           relative_l2=float(torch.linalg.vector_norm(a-g)/torch.linalg.vector_norm(g)))
    def load(p): return torch.from_numpy((np.fromfile(p,dtype='<u2').astype(np.uint32)<<16).view(np.float32))
    actual_norm = load(args.output / 'norm_probe.bin')[:896*rows]
    compare('norm', actual_norm, n)
    for key, value in [('input',x),('q',q),('k',k),('v',v),('context',ctx),('h',h),('m',mid),('y',y)]:
        compare(key,load(args.output / ('layer0_'+key+'.bin')),value)
    aq=load(args.output/'layer0_q.bin').reshape(rows,14,64)
    ak=load(args.output/'layer0_k.bin').reshape(rows,2,64).repeat_interleave(7,dim=1)
    av=load(args.output/'layer0_v.bin').reshape(rows,2,64).repeat_interleave(7,dim=1)
    actual_scores=m.bf(torch.einsum('rhd,shd->hrs',aq,ak)/8.0)
    actual_scores.masked_fill_(torch.arange(rows)[None,:]>positions[:,None],-float('inf'))
    reference_context=m.bf(torch.einsum('hrs,shd->rhd',m.bf(torch.softmax(actual_scores,dim=-1)),av)).reshape(rows,896)
    compare('attention_given_rtl_qkv',load(args.output/'layer0_context.bin'),reference_context)
    ah=load(args.output/'layer0_h.bin').reshape(rows,896)
    az=m.norm(ah,m.gamma[896:1792])
    ag,au=m.linear(az,w['gate']),m.linear(az,w['up'])
    reference_mid=m.bf(m.bf(torch.nn.functional.silu(ag))*au)
    compare('ffn_given_rtl_hidden',load(args.output/'layer0_m.bin'),reference_mid)
    assert result['norm']['relative_l2']<0.005
    assert result['v']['relative_l2']<0.01
    assert result['attention_given_rtl_qkv']['relative_l2']<0.01
    assert result['ffn_given_rtl_hidden']['relative_l2']<0.01
    (args.output / 'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
