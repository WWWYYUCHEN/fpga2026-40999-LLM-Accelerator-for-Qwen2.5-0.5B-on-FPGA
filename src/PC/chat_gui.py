"""PC input/output terminal for either the real-RTL simulator or the board UART."""
import argparse
import codecs
import datetime
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox


class Chat:
    def __init__(self, window, package):
        self.window, self.package = window, package
        self.transport = self.process = None
        self.pending = queue.Queue()
        self.ready = self.busy = False
        self.stopping = False
        window.title('KV260 串口交互与 RTL 模拟')
        window.geometry('1000x720')
        bar = ttk.Frame(window); bar.pack(fill='x', padx=12, pady=10)
        ttk.Label(bar, text='连接方式').pack(side='left')
        self.kind = ttk.Combobox(bar, values=['RTL 逐周期模拟', 'PL 算法模拟', '串口实板'], state='readonly', width=16)
        self.kind.set('RTL 逐周期模拟'); self.kind.pack(side='left', padx=8)
        self.port = ttk.Entry(bar, width=10); self.port.insert(0, 'COM11'); self.port.pack(side='left')
        ttk.Label(bar, text='生成上限').pack(side='left', padx=(15, 4))
        self.limit = tk.StringVar(value='20')
        ttk.Spinbox(bar, from_=1, to=255, width=5, textvariable=self.limit).pack(side='left')
        self.start_button = ttk.Button(bar, text='连接 / 启动', command=self.start)
        self.start_button.pack(side='left', padx=12)
        ttk.Button(bar, text='停止', command=self.stop).pack(side='left')
        self.status = tk.StringVar(value='默认运行加速器 RTL 仿真，包含逐周期计算；CPU 数学模型仅作参考对照。')
        ttk.Label(window, textvariable=self.status).pack(anchor='w', padx=12)
        ttk.Label(window, text='分词始终在同一套 PS C 程序内完成。算法模拟以 CPU 计算代替 PL；RTL 模拟执行硬件逻辑。').pack(anchor='w', padx=12, pady=4)
        self.output = tk.Text(window, wrap='word', font=('Microsoft YaHei UI', 12), state='disabled')
        self.output.pack(fill='both', expand=True, padx=12, pady=8)
        entrybar = ttk.Frame(window); entrybar.pack(fill='x', padx=12, pady=8)
        self.entry = ttk.Entry(entrybar, font=('Microsoft YaHei UI', 12)); self.entry.pack(side='left', fill='x', expand=True)
        self.entry.bind('<Return>', lambda event: self.send())
        self.send_button = ttk.Button(entrybar, text='发送', command=self.send, state='disabled')
        self.send_button.pack(side='left', padx=8)
        ttk.Label(window, text='直接输入中文/英文，按 Enter。也可发送 /help、/mode ids、/selftest。每次提问独立清空 KV。').pack(anchor='w', padx=12, pady=(0, 10))
        window.protocol('WM_DELETE_WINDOW', self.close)
        window.after(80, self.poll)

    def append(self, value):
        self.output.configure(state='normal'); self.output.insert('end', value)
        self.output.see('end'); self.output.configure(state='disabled')

    def start(self):
        if self.transport or self.process: return
        self.stopping = False; self.ready = False
        self.pending_prompt = None
        try:
            if self.kind.get() != '串口实板':
                run = self.package / 'sim/runs' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                run.mkdir(parents=True)
                shutil.copy2(self.package / 'sim/simulator/segments.tsv', run / 'segments.tsv')
                env = dict(os.environ, SIM_SD=str(self.package / 'board/deployment'), SIM_TRACE=str(run / 'mmio_trace.txt'))
                if self.kind.get() == 'PL 算法模拟':
                    runtime = os.environ.get('QWEN_SIM_PYTHON', sys.executable)
                    command = [runtime, str(self.package / 'src/PC/math_simulator.py'), '--package', str(self.package), '--run', str(run)]
                else:
                    command = [str(self.package / 'sim/simulator/qwen_rtl_sim.exe')]
                self.process = subprocess.Popen(command, cwd=run,
                                                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.transport = self.process
                self.read_handle = self.process.stdout
                self.log = (run / 'console.log').open('wb')
                self.status.set('正在加载模型。')
            else:
                try:
                    import serial
                except ImportError:
                    messagebox.showerror('缺少串口组件', '请安装 pyserial：py -3 -m pip install pyserial'); return
                self.transport = serial.Serial(self.port.get().strip(), 115200, timeout=0.2,
                                               xonxoff=False, rtscts=False, dsrdtr=False)
                self.read_handle = self.transport
                self.log = None
                self.status.set('串口已连接；等待板卡 READY，或按 Enter 获取提示符。')
                self.write(b'\r')
            self.start_button.configure(state='disabled')
            threading.Thread(target=self.reader, daemon=True).start()
        except Exception as error:
            self.stop(); messagebox.showerror('连接失败', str(error))

    def write(self, data):
        if self.process: self.process.stdin.write(data); self.process.stdin.flush()
        elif self.transport:
            # UART has no flow control; pacing also leaves time for PS echo/commands.
            for at in range(0, len(data), 16):
                self.transport.write(data[at:at+16]); self.transport.flush(); time.sleep(0.02)

    def reader(self):
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        partial = ''
        try:
            while not self.stopping:
                chunk = self.read_handle.read(1 if self.process else 1024)
                if not chunk:
                    if self.process: break
                    continue
                if self.log: self.log.write(chunk); self.log.flush()
                decoded = decoder.decode(chunk)
                if not decoded: continue
                self.pending.put(('text', decoded))
                partial = (partial + decoded)[-8192:]
                if 'prompt> ' in partial:
                    partial = partial.split('prompt> ')[-1]
                    self.pending.put(('ready', None))
                if '\nPROGRESS cycle=' in partial and '\n' in partial[partial.rfind('\nPROGRESS cycle=')+1:]:
                    line = partial[partial.rfind('\nPROGRESS cycle=')+1:].splitlines()[0]
                    self.pending.put(('progress', line)); partial = ''
            self.pending.put(('text', decoder.decode(b'', final=True)))
        except Exception as error:
            if not self.stopping: self.pending.put(('text', '\n连接中断：' + str(error) + '\n'))
        self.pending.put(('closed', None))

    def send(self):
        text = self.entry.get()
        if not text or not self.ready or not self.transport: return
        try:
            limit = int(self.limit.get())
            if limit < 1 or limit > 255: raise ValueError('生成上限须为 1..255')
            if '\r' in text or '\n' in text: raise ValueError('一次发送一行提示词')
            if len(text.encode()) > 4096: raise ValueError('提示词超过 4096 UTF-8 字节')
            self.ready = False; self.busy = True; self.send_button.configure(state='disabled')
            # This client only transmits text and controls. Tokenization stays in PS C.
            self.pending_prompt = (text + '\n').encode('utf-8') if not text.startswith('/') else None
            payload = f'/max {limit}\n'.encode('utf-8') if self.pending_prompt else (text + '\n').encode('utf-8')
            self.write(payload); self.entry.delete(0, 'end')
            self.status.set('正在处理提示词并等待生成 token。')
        except Exception as error: messagebox.showerror('发送失败', str(error))

    def poll(self):
        try:
            while True:
                kind, value = self.pending.get_nowait()
                if kind == 'text': self.append(value)
                elif kind == 'ready':
                    if getattr(self, 'pending_prompt', None):
                        payload, self.pending_prompt = self.pending_prompt, None
                        self.write(payload)
                    else:
                        self.ready = True; self.busy = False
                        self.send_button.configure(state='normal'); self.status.set('就绪，可以输入下一条提示词。')
                        self.entry.focus_set()
                elif kind == 'progress': self.status.set('RTL 正在计算：' + value)
                elif kind == 'closed':
                    self.ready = False; self.send_button.configure(state='disabled'); self.start_button.configure(state='normal')
            if not self.stopping: self.status.set('程序已停止；完整日志保存在 sim/runs 中。')
        except queue.Empty: pass
        self.window.after(80, self.poll)

    def stop(self):
        self.stopping = True; self.ready = False
        self.pending_prompt = None
        if self.process:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            else: self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill()
        elif self.transport: self.transport.close()
        if getattr(self, 'log', None):
            try: self.log.close()
            except (OSError, ValueError): pass
        self.transport = self.process = None
        self.start_button.configure(state='normal'); self.send_button.configure(state='disabled')
        self.status.set('已停止。')

    def close(self): self.stop(); self.window.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    window = tk.Tk(); Chat(window, args.package); window.mainloop()
