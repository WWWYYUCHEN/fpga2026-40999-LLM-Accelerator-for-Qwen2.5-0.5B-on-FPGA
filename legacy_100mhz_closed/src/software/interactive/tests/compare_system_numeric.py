"""Check RTL generation against the independent numerical reference offline."""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from model_backend import Model
from rtl_numeric_reference import configure_reference


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('sd',type=Path)
    ap.add_argument('run',type=Path)
    ap.add_argument('--require-eos',action='store_true')
    args=ap.parse_args()
    scenario=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    assert scenario['status']=='PASS'
    lib=C.CDLL(str(args.root/'build/native/qwen_ps.dll'))
    lib.qt_init.argtypes=[C.c_void_p,C.c_size_t]
    lib.qt_chat_encode.argtypes=lib.qt_encode.argtypes=[C.c_char_p,C.POINTER(C.c_uint32),C.c_uint,C.POINTER(C.c_uint)]
    data=C.create_string_buffer((args.root/'TOKEN.BIN').read_bytes())
    assert lib.qt_init(data,C.sizeof(data)-1)==0
    out,count=(C.c_uint32*256)(),C.c_uint()
    encode=lib.qt_encode if scenario['template']=='raw' else lib.qt_chat_encode
    assert encode(scenario['prompt'].encode('utf-8'),out,256,C.byref(count))==0
    prompt=list(out[:count.value])
    model=configure_reference(Model(args.sd))
    model.clear()
    transcript=(args.run/'uart.log').read_text(encoding='utf-8')
    actual=[(int(req),int(it),int(tok)) for req,it,tok in re.findall(r'@TOKEN (\d+) (\d+) (\d+) ',transcript)]
    rows=[]
    position=0;ids=prompt
    for request,iteration,token in actual:
        if iteration==0:model.clear();position=0;ids=prompt
        expected=model.forward(ids,position)
        entry=dict(request=request,iteration=iteration,rtl_token=token,reference_token=expected,token_matches=token==expected)
        # Existing fixtures overwrite logits at a new request, so only compare the last request's files.
        logit_file=args.run/f'hardware_logits_{iteration}.bin'
        if request==actual[-1][0] and logit_file.exists():
            bits=np.fromfile(logit_file,dtype='<u2').astype(np.uint32)<<16
            a=torch.from_numpy(bits.view(np.float32));g=model.last_logits
            entry['relative_l2']=float(torch.norm(a-g)/torch.norm(g))
            entry['max_abs']=float((a-g).abs().max())
        rows.append(entry)
        position+=len(ids);ids=[token]
    eos=any(token in [151643,151645] for _,_,token in actual)
    passed=all(x['token_matches'] for x in rows) and (eos or not args.require_eos)
    result=dict(status='PASS' if passed else 'FAIL',prompt=scenario['prompt'],template=scenario['template'],
                prompt_tokens=len(prompt),comparisons=rows,eos_seen=eos,
                reference='independent mathematical operations with hardware RND8 and scale/bias boundaries',
                reference_is_offline=True,bit_exact_reference=False)
    (args.run/'numeric_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
