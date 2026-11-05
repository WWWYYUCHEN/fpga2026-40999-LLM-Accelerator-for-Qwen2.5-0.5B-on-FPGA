"""Validate every transformer layer at actual RTL tensor boundaries, one token."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_backend import Model
from rtl_numeric_reference import configure_reference,qkv_projection


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('sd',type=Path)
    ap.add_argument('rtl_run',type=Path)
    ap.add_argument('report',type=Path)
    ap.add_argument('--iteration',type=int,default=0)
    args=ap.parse_args()
    m=configure_reference(Model(args.sd))
    def load(path):return torch.from_numpy((np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view(np.float32))
    def metric(actual,expected):
        a,g=actual.flatten(),expected.flatten()
        return dict(relative_l2=float(torch.norm(a-g)/torch.norm(g)),mean_abs=float((a-g).abs().mean()),max_abs=float((a-g).abs().max()))
    iteration=args.iteration
    outputs=list(csv.DictReader((args.rtl_run/'hardware_tokens.csv').open()))
    token=108386 if iteration==0 else int(outputs[iteration-1]['token_id'])
    x=m.embedding[[token]]
    rows=[]
    for layer,w in enumerate(m.weights):
        def tensor(key):return load(args.rtl_run/f'iteration_{iteration}_layer_{layer}_{key}.bin').reshape(1,-1)
        actual={key:tensor(key) for key in ['q','k','v','context','h','m','y']}
        norm=m.norm(x,m.gamma[layer*1792:layer*1792+896])
        result=dict(layer=layer)
        for key,start,end in [('q',0,896),('k',896,1024),('v',1024,1152)]:
            expected=qkv_projection(m,norm,w[key],m.bias[layer*1152+start:layer*1152+end])
            if key in ['q','k']:
                heads=14 if key=='q' else 2
                expected=m.rope(expected.reshape(1,heads,64),torch.tensor([float(iteration)])).reshape(1,-1)
            result[key]=metric(actual[key],expected)
        cached_keys=torch.cat([load(args.rtl_run/f'iteration_{i}_layer_{layer}_k.bin').reshape(1,2,64) for i in range(iteration+1)],dim=0).repeat_interleave(7,dim=1)
        cached_values=torch.cat([load(args.rtl_run/f'iteration_{i}_layer_{layer}_v.bin').reshape(1,2,64) for i in range(iteration+1)],dim=0).repeat_interleave(7,dim=1)
        scores=m.bf(torch.einsum('rhd,shd->hrs',actual['q'].reshape(1,14,64),cached_keys)/8.0)
        expected_context=m.bf(torch.einsum('hrs,shd->rhd',m.bf(torch.softmax(scores,dim=-1)),cached_values)).reshape(1,896)
        result['attention_given_rtl_qkv']=metric(actual['context'],expected_context)
        expected_h=m.bf(x+m.linear(actual['context'],w['o']))
        result['residual_given_rtl_context']=metric(actual['h'],expected_h)
        z=m.norm(actual['h'],m.gamma[layer*1792+896:(layer+1)*1792])
        g,u=m.linear(z,w['gate']),m.linear(z,w['up'])
        expected_mid=m.bf(m.bf(torch.nn.functional.silu(g))*u)
        result['ffn_given_rtl_hidden']=metric(actual['m'],expected_mid)
        expected_y=m.bf(actual['h']+m.linear(actual['m'],w['down']))
        result['down_given_rtl_mid']=metric(actual['y'],expected_y)
        result['pass']=all(item['relative_l2']<0.01 for item in result.values() if isinstance(item,dict))
        rows.append(result)
        x=actual['y']
    reference_logits=m.linear(m.norm(x,m.gamma[43008:]),m.lm).flatten()
    actual_logits=load(args.rtl_run/f'hardware_logits_{iteration}.bin')
    lm=metric(actual_logits,reference_logits)
    lm['rtl_token']=int(actual_logits.numel()-1-torch.argmax(actual_logits.flip(0)))
    lm['reference_token']=int(reference_logits.numel()-1-torch.argmax(reference_logits.flip(0)))
    norm_path=args.rtl_run/'lm_norm_probe.bin'
    final_norm=None
    if norm_path.exists() and norm_path.stat().st_size>=1792*(iteration+1):
        actual_norm=load(norm_path)[896*iteration:896*(iteration+1)].reshape(1,896)
        final_norm=metric(actual_norm,m.norm(x,m.gamma[43008:]))
        lm_given_actual_norm=metric(actual_logits,m.linear(actual_norm,m.lm))
        lm['given_rtl_normalized_input']=lm_given_actual_norm
    result=dict(status='PASS' if all(x['pass'] for x in rows) and lm['rtl_token']==lm['reference_token'] and lm['relative_l2']<0.01 else 'FAIL',
                iteration=iteration,layers=rows,lm_head=lm,final_norm=final_norm,checks='Actual RTL operands; independent RND8 tensor-boundary reference; 1% relative-L2 maximum per stage',
                bit_exact_reference=False,board_tested=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],failing_layers=[x['layer'] for x in rows if not x['pass']],lm_head=lm),indent=2))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
