"""Derive an SD launcher from the delivered U-Boot legacy script.
No application compilation or source-package writes. Outputs stay in the
current task's workspace result directory.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'board/measurements/autoboot'
SOURCE = OUT / 'original-sd-launcher'
FMT = '>7I4B32s'
EXPECTED_ORIGINAL = '09d7b17dfffb24804b4dca13f764ce9050e03d973040337ea3bee1b9b2c25644'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def unpack_script(blob):
    fields = list(struct.unpack(FMT, blob[:64]))
    assert fields[0] == 0x27051956, 'Not a U-Boot legacy image'
    assert fields[3] == len(blob)-64, 'Wrong payload length'
    assert fields[1] == zlib.crc32(blob[:4]+bytes(4)+blob[8:64]), 'Header CRC mismatch'
    assert fields[6] == zlib.crc32(blob[64:]), 'Payload CRC mismatch'
    assert fields[7:11] == [17,22,6,0], 'Unexpected legacy image metadata'
    length, end = struct.unpack('>2I', blob[64:72])
    assert end == 0 and length == len(blob)-72, 'Unexpected single-script size table'
    return fields, blob[72:]

def expected():
    old = (src/'boot.scr').read_bytes()
    assert sha(old) == EXPECTED_ORIGINAL
    fields, payload = unpack_script(old)
    old_cmd = (src/'boot.cmd').read_bytes().replace(b'\r\n',b'\n').rstrip(b'\n')+b'\n'
    assert payload == old_cmd
    text = payload.decode('ascii')
    assert text.count('QWENSD.ELF') == 2
    text = text.replace('QWENSD.ELF','QWUART.ELF').replace(
        'Qwen25 KV260 100 MHz bare-metal launcher',
        'Qwen25 KV260 AUTOBOOT QWUART 2026-10-25')
    command = text.encode('ascii')
    data = struct.pack('>2I',len(command),0)+command
    fields[1] = 0
    # Fixed build stamp, for byte-for-byte reproducibility; not deployment time.
    fields[2] = int(datetime.fromisoformat('2026-10-25T00:00:00+08:00').timestamp())
    fields[3] = len(data)
    fields[6] = zlib.crc32(data)
    fields[-1] = b'Qwen25 KV260 UART autoboot'.ljust(32,b'\x00')
    fields[1] = zlib.crc32(struct.pack(FMT,*fields))
    script = struct.pack(FMT,*fields)+data
    _, checked_command = unpack_script(script)
    assert checked_command == command
    assert b'QWENSD.ELF' not in script and b'QWUART.ELF' in script
    return {'boot.cmd':command,'boot.scr':script},old

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    outputs,old = expected()
    if not args.verify:
        OUT.mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():
            path = OUT/name
            if path.exists():
                assert path.read_bytes() == data, 'Refusing to replace a different artifact'
            else:
                # Generated image artifact; delivered inputs are strictly read-only.
                with path.open('xb') as stream:
                    stream.write(data)
    for name,data in outputs.items():
        assert (OUT/name).read_bytes() == data, 'Read-back mismatch: '+name
    print(json.dumps({'status':'verified',
        'method':'reuse original launcher and legacy image metadata; recalculate script length table, payload CRC32 and header CRC32',
        'source_boot_scr_sha256':sha(old),
        'outputs':[{'path':name,'bytes':len(data),'sha256':sha(data),'crc32':f'{zlib.crc32(data):08x}'} for name,data in outputs.items()],
        'changes':['launcher banner','QWENSD.ELF -> QWUART.ELF','legacy image timestamp/name/length/checksums'],
        'unchanged':['SD search order and partition','PL/ELF DDR addresses','fpga load and bootelf -p','firmware','model','RTL','QSPI','permanent U-Boot environment']},indent=2))

if __name__ == '__main__':
    main()
