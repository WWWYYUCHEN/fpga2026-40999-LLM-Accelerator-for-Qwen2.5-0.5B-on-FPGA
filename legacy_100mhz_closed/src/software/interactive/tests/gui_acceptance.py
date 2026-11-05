"""Drive the actual PC GUI through numerical model chat, without a board."""
import argparse
import json
from pathlib import Path
import sys
import time
import tkinter as tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('package', type=Path)
    ap.add_argument('report', type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.package / 'PC'))
    from chat_gui import Chat
    root = tk.Tk(); root.withdraw()
    app = Chat(root, args.package)
    app.kind.set('PL 算法模拟')
    app.limit.set('12')
    start = time.time()
    state = dict(sent=False, complete=False)
    app.start()
    def tick():
        if app.ready and not state['sent']:
            state['sent'] = True
            app.entry.insert(0, '1+1等于几？'); app.send()
        transcript = app.output.get('1.0', 'end')
        if state['sent'] and '@END 1 ' in transcript and app.ready:
            state['complete'] = 'answer> 1+1等于2。' in transcript
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.with_suffix('.txt').write_text(transcript, encoding='utf-8')
            app.stop(); root.destroy(); return
        if time.time()-start > 45:
            app.stop(); root.destroy(); return
        root.after(80, tick)
    root.after(80, tick); root.mainloop()
    result = dict(status='PASS' if state['complete'] else 'FAIL',
                  input='1+1等于几？', output='1+1等于2。', seconds=round(time.time()-start, 2),
                  interface='actual Tk GUI', backend='mathematical PL substitute, production PS C')
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not state['complete']: raise SystemExit(1)


if __name__ == '__main__': main()
