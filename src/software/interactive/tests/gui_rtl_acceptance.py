"""Exercise the GUI, PS tokenizer and real accelerator RTL without a CPU model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tkinter as tk


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('package',type=Path)
    ap.add_argument('report',type=Path)
    args=ap.parse_args()
    sys.path.insert(0,str(args.package/'PC'))
    from chat_gui import Chat
    window=tk.Tk();window.withdraw()
    app=Chat(window,args.package)
    assert app.kind.get()=='RTL 逐周期模拟'
    app.limit.set('1')
    commands=['/mode events','/template raw','你好']
    state=dict(sent=0,complete=False)
    started=time.time()
    app.start()
    def poll():
        text=app.output.get('1.0','end')
        if app.ready and state['sent']<len(commands):
            app.entry.insert(0,commands[state['sent']]);state['sent']+=1;app.send()
        elif app.ready and state['sent']==len(commands) and '@END 1 ' in text:
            state['complete']='@TOKEN 1 0 3837 ' in text and 'status=0' in text and 'scripted PL tokens' not in text
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.with_suffix('.txt').write_text(text,encoding='utf-8')
            app.stop();window.destroy();return
        if time.time()-started>900:
            app.stop();window.destroy();return
        window.after(100,poll)
    window.after(100,poll);window.mainloop()
    result=dict(status='PASS' if state['complete'] else 'FAIL',backend='accelerator RTL simulation',
                production_ps_c=True,cpu_model_substitution=False,prompt='你好',token=3837,
                seconds=round(time.time()-started,2),
                exe_sha256=hashlib.sha256((args.package/'sim/simulator/qwen_rtl_sim.exe').read_bytes()).hexdigest())
    args.report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if not state['complete']:raise SystemExit(1)


if __name__=='__main__':main()
