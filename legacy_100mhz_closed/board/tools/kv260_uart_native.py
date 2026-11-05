"""Windows native UART diagnostic/monitor, independent of .NET SerialPort.
Passive by default; --probe-enter sends exactly one CR after 3 quiet seconds.
Raw bytes and a readable log are kept on the PC. Never resets the board.
"""
import argparse
import codecs
import ctypes as ct
from ctypes import wintypes as wt
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

class DCB(ct.Structure):
    _fields_ = [('DCBlength',wt.DWORD),('BaudRate',wt.DWORD),('flags',wt.DWORD),
                ('wReserved',wt.WORD),('XonLim',wt.WORD),('XoffLim',wt.WORD),
                ('ByteSize',wt.BYTE),('Parity',wt.BYTE),('StopBits',wt.BYTE),
                ('XonChar',ct.c_char),('XoffChar',ct.c_char),('ErrorChar',ct.c_char),
                ('EofChar',ct.c_char),('EvtChar',ct.c_char),('wReserved1',wt.WORD)]

class Timeouts(ct.Structure):
    _fields_ = [(name,wt.DWORD) for name in
                ['ReadIntervalTimeout','ReadTotalTimeoutMultiplier','ReadTotalTimeoutConstant',
                 'WriteTotalTimeoutMultiplier','WriteTotalTimeoutConstant']]

class CommStat(ct.Structure):
    _fields_ = [('flags',wt.DWORD),('in_queue',wt.DWORD),('out_queue',wt.DWORD)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='COM11',choices=['COM11','COM12'])
    p.add_argument('--seconds',type=float,default=12,help='0: continuous')
    p.add_argument('--log',required=True)
    p.add_argument('--probe-enter',action='store_true')
    p.add_argument('--command-file',type=Path,help='Optional explicit JSON {id,command} slot; existing ID is not replayed')
    args = p.parse_args()
    seen_command = None
    if args.command_file and args.command_file.exists():
        seen_command = json.loads(args.command_file.read_text(encoding='utf-8-sig')).get('id')
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    kernel = ct.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes = [wt.LPCWSTR,wt.DWORD,wt.DWORD,ct.c_void_p,wt.DWORD,wt.DWORD,wt.HANDLE]
    kernel.CreateFileW.restype = wt.HANDLE
    for name, signature in {
        'GetCommState':[wt.HANDLE,ct.POINTER(DCB)],'SetCommState':[wt.HANDLE,ct.POINTER(DCB)],
        'SetCommTimeouts':[wt.HANDLE,ct.POINTER(Timeouts)],
        'ReadFile':[wt.HANDLE,ct.c_void_p,wt.DWORD,ct.POINTER(wt.DWORD),ct.c_void_p],
        'WriteFile':[wt.HANDLE,ct.c_void_p,wt.DWORD,ct.POINTER(wt.DWORD),ct.c_void_p],
        'ClearCommError':[wt.HANDLE,ct.POINTER(wt.DWORD),ct.POINTER(CommStat)],
        'CloseHandle':[wt.HANDLE],
    }.items():
        fn = getattr(kernel,name)
        fn.argtypes = signature
        fn.restype = wt.BOOL
    def checked(value):
        if not value:
            raise ct.WinError(ct.get_last_error())
    log = Path(args.log).resolve()
    log.parent.mkdir(parents=True,exist_ok=True)
    report = {'pid':os.getpid(),'port':args.port,'interface':'Win32 CreateFileW/ReadFile',
              'baud':115200,'format':'8N1','dtr_rts':False,'probe_enter':args.probe_enter,
              'started_at':datetime.now().astimezone().isoformat(),'rx_bytes':0,'tx_bytes':0,
              'uart_errors':[],'status':'starting'}
    decoder = codecs.getincrementaldecoder('utf-8')(errors='backslashreplace')
    handle = None
    with log.open('x',encoding='utf-8',buffering=1,newline='') as textlog, log.with_suffix('.bin').open('xb',buffering=0) as rawlog:
        def note(message):
            line = f'\n[HOST] {datetime.now().astimezone().isoformat()} {message}\n'
            textlog.write(line)
            print(line,end='',flush=True)
        def heartbeat():
            report['heartbeat_at'] = datetime.now().astimezone().isoformat()
            log.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding='utf-8')
        try:
            handle = kernel.CreateFileW('\\\\.\\'+args.port,0xc0000000,0,None,3,0,None)
            if handle == ct.c_void_p(-1).value:
                handle = None
                raise ct.WinError(ct.get_last_error())
            dcb = DCB()
            dcb.DCBlength = ct.sizeof(DCB)
            checked(kernel.GetCommState(handle,ct.byref(dcb)))
            dcb.BaudRate,dcb.flags,dcb.ByteSize,dcb.Parity,dcb.StopBits = 115200,1,8,0,0
            checked(kernel.SetCommState(handle,ct.byref(dcb)))
            checked(kernel.GetCommState(handle,ct.byref(dcb)))
            report['effective_dcb'] = {'baud':dcb.BaudRate,'flags':dcb.flags,
                                       'data_bits':dcb.ByteSize,'parity':dcb.Parity,
                                       'stop_bits':dcb.StopBits}
            assert (dcb.BaudRate,dcb.ByteSize,dcb.Parity,dcb.StopBits) == (115200,8,0,0)
            checked(kernel.SetCommTimeouts(handle,ct.byref(Timeouts(0xffffffff,0,100,0,1000))))
            report['status'] = 'listening'
            note(f'OPEN {args.port} 115200 8N1, native ReadFile, DTR/RTS off; no automatic boot interruption')
            report['console_window_handle'] = int(kernel.GetConsoleWindow() or 0)
            heartbeat()
            start,last_heartbeat = time.monotonic(),time.monotonic()
            probe_sent = False
            buffer = ct.create_string_buffer(8192)
            while args.seconds == 0 or time.monotonic()-start < args.seconds:
                errors,stat = wt.DWORD(),CommStat()
                checked(kernel.ClearCommError(handle,ct.byref(errors),ct.byref(stat)))
                if errors.value:
                    report['uart_errors'].append({'at':datetime.now().astimezone().isoformat(),'mask':errors.value})
                    note(f'UART_ERROR mask=0x{errors.value:x} (RXOVER=1, OVERRUN=2, PARITY=4, FRAME=8, BREAK=16)')
                count = wt.DWORD()
                checked(kernel.ReadFile(handle,buffer,len(buffer),ct.byref(count),None))
                if count.value:
                    data = buffer.raw[:count.value]
                    rawlog.write(data)
                    report['rx_bytes'] += len(data)
                    rendered = decoder.decode(data)
                    textlog.write(rendered)
                    print(rendered,end='',flush=True)
                if args.command_file and args.command_file.exists():
                    try:
                        command = json.loads(args.command_file.read_text(encoding='utf-8-sig'))
                    except (ValueError, PermissionError):
                        command = {}
                    identity = command.get('id')
                    if identity and identity != seen_command:
                        seen_command = identity
                        value = command.get('command')
                        if not isinstance(value,str) or not value or any(c in value for c in '\r\n\0'):
                            raise ValueError('Command must be one nonempty line')
                        payload = (value+'\r').encode('utf-8')
                        if len(payload)>1024:
                            raise ValueError('Command exceeds UART input limit')
                        note(f'TX id={identity}: {value}')
                        sent = wt.DWORD()
                        checked(kernel.WriteFile(handle,payload,len(payload),ct.byref(sent),None))
                        report['tx_bytes'] += sent.value
                        if sent.value != len(payload):
                            raise OSError('Partial UART write; stopping without retry')
                        report['last_command_id'] = identity
                        heartbeat()
                if args.probe_enter and not probe_sent and time.monotonic()-start >= 3:
                    probe_sent = True
                    if report['rx_bytes'] == 0:
                        sent = wt.DWORD()
                        checked(kernel.WriteFile(handle,b'\r',1,ct.byref(sent),None))
                        report['tx_bytes'] += sent.value
                        note('TX one CR to request an idle prompt; not a boot or inference command')
                    else:
                        note('Skipped CR probe because UART output was received')
                if time.monotonic()-last_heartbeat >= 2:
                    heartbeat()
                    last_heartbeat = time.monotonic()
            report['status'] = 'completed'
        except KeyboardInterrupt:
            report['status'] = 'stopped'
        except Exception as error:
            report['status'],report['error'] = 'error',str(error)
            note(f'ERROR: {error}')
        finally:
            if handle is not None:
                kernel.CloseHandle(handle)
            report['finished_at'] = datetime.now().astimezone().isoformat()
            heartbeat()
            note(f'CLOSE rx_bytes={report["rx_bytes"]} tx_bytes={report["tx_bytes"]} status={report["status"]}')
    return 1 if report['status'] == 'error' else 0

if __name__ == '__main__':
    raise SystemExit(main())
