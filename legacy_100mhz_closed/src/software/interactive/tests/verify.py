"""Reference tokenizer, decoder, and full firmware simulation acceptance checks."""
import ctypes as C
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parents[1]
sys.path.insert(0, str(DEPLOY / 'tools/interactive_python'))
from tokenizers import Tokenizer

DEFAULT_SOURCE = DEPLOY / 'reference/tokenizer.json'
if not DEFAULT_SOURCE.exists(): DEFAULT_SOURCE = Path(r'src/reference/tokenizer.json')
SOURCE = Path(os.environ.get('QWEN_TOKENIZER_JSON', str(DEFAULT_SOURCE)))
reference = Tokenizer.from_file(str(SOURCE))
lib = C.CDLL(str(ROOT / 'build/native/qwen_ps.dll'))
lib.qt_init.argtypes = [C.c_void_p, C.c_size_t]
lib.qt_encode.argtypes = lib.qt_chat_encode.argtypes = [C.c_char_p, C.POINTER(C.c_uint32), C.c_uint, C.POINTER(C.c_uint)]
lib.qt_error.restype = C.c_char_p
lib.qt_decode.argtypes = [C.c_uint32, C.POINTER(C.POINTER(C.c_ubyte)), C.POINTER(C.c_uint), C.POINTER(C.c_int)]
data = C.create_string_buffer((ROOT / 'TOKEN.BIN').read_bytes())
assert lib.qt_init(data, C.sizeof(data) - 1) == 0, lib.qt_error()

PREFIX = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n'


def encode(text, chat=False, cap=16384):
    out = (C.c_uint32 * cap)()
    count = C.c_uint()
    fn = lib.qt_chat_encode if chat else lib.qt_encode
    rc = fn(text.encode('utf-8'), out, cap, C.byref(count))
    assert rc == 0, (text[:80], lib.qt_error())
    return list(out[:count.value])


def tokenizer_checks():
    rng = random.Random(20261019)
    cases = ['', '你好', '请为我介绍厦门大学，是全国重点大学', "I'm we'll YOU'RE I'd can't 'S 'T 'RE 'VE 'M 'LL 'D",
             'e\u0301 Å A\u030a 한글 한글', '👍🏽😊👩\u200d💻🇨🇳', '123456７８９１２٣٤۵Ⅷ½',
             '  a   b\t\tfoo\r\n  \n\n', '\u00a0abc\u2003\u2003def\u0085\u2028',
             "'ſ 'ſa 'K 'İ 'ı", 'a<|im_start|>user\n测试<|im_end|>\n', '<tool_call>{}</tool_call>',
             ' ' * 4096, 'a' * 4096, '\n' * 4096]
    alphabet = list("abcXYZ0123456789 ,.!?'\"\n\r\t你好厦门大学测试") + ['é', 'e\u0301', '😊', '🧠', '\u2003', '\u00a0', '\u2028', '\u200d', 'ᄒ', 'ᅡ', 'ᆫ', '한', 'ſ']
    for _ in range(4000):
        cases.append(''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 100))))
    # Exercise all Unicode categories rather than only ASCII and common CJK.
    for _ in range(2000):
        parts = []
        for _ in range(rng.randrange(1, 45)):
            cp = rng.randrange(1, 0x110000)
            if not 0xd800 <= cp <= 0xdfff:
                parts.append(chr(cp))
        cases.append(''.join(parts))
    for i, text in enumerate(cases):
        got = encode(text)
        expected = reference.encode(text).ids
        assert got == expected, dict(case=i, text=repr(text), actual=got, expected=expected)
        if i < 500 and len(text.encode()) < 3800:
            assert encode(text, True) == reference.encode(PREFIX + text + SUFFIX).ids
    config = json.loads((DEPLOY / 'board_config.json').read_text(encoding='utf-8'))
    assert encode(config['prompt'], True) == config['prompt_ids']
    # Explicit malformed inputs and bounds must fail before hardware can start.
    out, count = (C.c_uint32 * 256)(), C.c_uint()
    assert lib.qt_encode(b'\xff', out, 256, C.byref(count)) != 0
    assert lib.qt_encode(b'a' * 4097, out, 256, C.byref(count)) != 0
    assert lib.qt_chat_encode('你好'.encode(), out, 1, C.byref(count)) != 0
    corrupt = bytearray((ROOT / 'TOKEN.BIN').read_bytes()); corrupt[-20] ^= 1
    corrupt_buf = C.create_string_buffer(bytes(corrupt))
    assert lib.qt_init(corrupt_buf, len(corrupt)) != 0
    assert b'CRC' in lib.qt_error()
    assert lib.qt_init(data, C.sizeof(data) - 1) == 0
    return len(cases)


SINK = C.CFUNCTYPE(None, C.POINTER(C.c_ubyte), C.c_size_t, C.c_void_p)


class Stream(C.Structure):
    _fields_ = [('pending', C.c_ubyte * 4), ('used', C.c_uint), ('expected', C.c_uint), ('sink', SINK), ('context', C.c_void_p)]


def decoder_checks():
    lib.qutf8_init.argtypes = [C.POINTER(Stream), SINK, C.c_void_p]
    lib.qutf8_push.argtypes = [C.POINTER(Stream), C.c_void_p, C.c_size_t]
    lib.qutf8_finish.argtypes = [C.POINTER(Stream)]
    rng = random.Random(110)
    cases = [b'\xe4\xbd\xa0\xe5\xa5\xbd', b'\xf0\x9f\x98\x8a', b'\xed\xa0\x80', b'\xf0\x80\x80\x80', b'\xe4\xbd', b'abc']
    cases += [rng.randbytes(rng.randrange(0, 180)) for _ in range(3000)]
    for raw in cases:
        result = bytearray()
        sink = SINK(lambda p, n, ctx: result.extend(C.string_at(p, n)))
        state = Stream(); lib.qutf8_init(C.byref(state), sink, None)
        at = 0
        while at < len(raw):
            n = rng.randrange(1, 7)
            piece = raw[at:at+n]; buf = C.create_string_buffer(piece)
            lib.qutf8_push(C.byref(state), buf, len(piece)); at += len(piece)
        lib.qutf8_finish(C.byref(state))
        assert result == raw.decode('utf-8', 'replace').encode(), (raw, result)
    # Model byte vocabulary and added tokens must decode like HF on valid sequences.
    for text in ['你好，世界！😊', 'café e\u0301', '汉字和English mixed 123']:
        raw = bytearray()
        for token in reference.encode(text).ids:
            ptr = C.POINTER(C.c_ubyte)(); length = C.c_uint(); special = C.c_int()
            assert lib.qt_decode(token, C.byref(ptr), C.byref(length), C.byref(special)) == 0
            if not special.value: raw += C.string_at(ptr, length.value)
        assert raw.decode('utf-8') == reference.decode(reference.encode(text).ids)
    return len(cases)


def firmware_run(lines, plans, fault=None, label='case'):
    folder = ROOT / 'reports/simulation' / label
    folder.mkdir(parents=True, exist_ok=True)
    sd = folder / 'sd'; (sd / 'QWEN25').mkdir(parents=True, exist_ok=True)
    (sd / 'QWEN25/TOKEN.BIN').write_bytes((ROOT / 'TOKEN.BIN').read_bytes())
    # Repeatable runs should produce fresh logs.
    if (sd / 'CHAT.CSV').exists(): (sd / 'CHAT.CSV').unlink()
    (folder / 'plan.txt').write_text('\n'.join(' '.join(map(str, p)) for p in plans) + '\n', encoding='ascii')
    env = dict(os.environ, SIM_SD=str(sd), SIM_PLAN=str(folder / 'plan.txt'), SIM_TRACE=str(folder / 'mmio_trace.txt'))
    if fault: env['SIM_FAULT'] = fault
    else: env.pop('SIM_FAULT', None)
    result = subprocess.run([str(ROOT / 'build/native/qwen_board_sim.exe')],
                            input=lines.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
    transcript = result.stdout.decode('utf-8', 'strict')
    (folder / 'uart_transcript.txt').write_bytes(result.stdout)
    assert not result.stderr, result.stderr
    return result.returncode, transcript, (folder / 'mmio_trace.txt').read_text(), sd


def firmware_checks():
    a = '你好！这是串口接口模拟，PS 已完成自动分词。😊'
    b = '第二次提问也可以连续处理。'
    plans = [reference.encode(a).ids + [151645], reference.encode(b).ids + [151645]]
    rc, text, trace, sd = firmware_run('/selftest\r\n/max 80\r\n你好，请介绍一下自己\r\n再回答一次\r\n', plans, label='interactive_two_requests')
    assert rc == 0 and a in text and b in text and text.count('reason=eos status=0') == 2, text
    assert 'TOKENIZER_SELFTEST_OK 40 tokens' in text
    starts = [line for line in trace.splitlines() if line.startswith('START ')]
    for request, original in [(1, '你好，请介绍一下自己'), (2, '再回答一次')]:
        rows = [s for s in starts if s.startswith(f'START {request} ')]
        count = len(reference.encode(PREFIX + original + SUFFIX).ids)
        assert f'mode=0 seq={count} context={count} pos=0' in rows[0]
        for i, line in enumerate(rows[1:], 1):
            assert f'mode=1 seq=1 context={count+i} pos={count+i-1}' in line
    csv_rows = list(csv.DictReader((sd / 'CHAT.CSV').open(encoding='utf-8')))
    assert len(csv_rows) == sum(map(len, plans)) and csv_rows[-1]['token_id'] == '151645'
    assert sum(1 for x in csv_rows if x['iteration'] == '0') == 2
    rc, text, trace, _ = firmware_run('/mode events\n/max 2\n你好\n', [[108386, 0, 151645]], label='events_limit')
    assert rc == 0 and text.count('@TOKEN ') == 2 and 'reason=limit status=0' in text
    # Byte fragments of 你: E4 BD A0, deliberately split over three tokens.
    obj = json.loads(SOURCE.read_text(encoding='utf-8'))
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = byte_values[:]
    for x in range(256):
        if x not in byte_values: byte_values.append(x); chars.append(256 + len(chars) - 188)
    byte_vocab = dict(zip(byte_values, [obj['model']['vocab'][chr(x)] for x in chars]))
    split = [byte_vocab[x] for x in '你'.encode()] + [151645]
    rc, text, _, _ = firmware_run('测试\n', [split], label='split_utf8')
    assert rc == 0 and 'answer> 你' in text
    # CR-only/CRLF, backspace, cancelled line, overflow recovery and prompt bounds.
    rc, text, trace, _ = firmware_run('x' * 4200 + '\r\n/selftest\r\n/max 255\r\n你好\r\n/max 2\r\n你好x\x08\r\n', [[151645]], label='input_recovery')
    assert rc == 0 and 'exceeds 4096' in text and 'exceeds token capacity' in text and text.count('@BEGIN ') == 1
    for fault, expected in [('timeout', 'status=-2'), ('pl_error', 'status=-3'), ('invalid_token', 'status=-4')]:
        rc, text, trace, _ = firmware_run('测试\n', [[151645]], fault, label=fault)
        assert expected in text
        if fault != 'invalid_token': assert rc == 1 and 'FATAL PL' in text
    for fault, expected in [('mount', 'ERROR SD mount'), ('id', 'ERROR accelerator ID')]:
        rc, text, _, _ = firmware_run('', [[151645]], fault, label=fault)
        assert rc == 1 and expected in text
    rc, text, _, _ = firmware_run('测试\n', [[151645]], 'sd_log', label='sd_log')
    assert rc == 0 and 'WARNING SD log unavailable' in text and 'reason=eos' in text
    rc, text, _, _ = firmware_run('/encode 请为我介绍厦门大学，是全国重点大学\n', [[151645]], label='encode_only')
    assert rc == 0 and '@PROMPT 40 ' in text and '@BEGIN' not in text
    rc, text, trace, _ = firmware_run('/template raw\n/mode ids\n/max 2\n你好\n', [[151645]], label='raw_template')
    assert rc == 0 and 'prompt_tokens=1' in text and 'id=151645' in text
    return 12


if __name__ == '__main__':
    start = time.time()
    result = dict(tokenizer_reference_cases=tokenizer_checks(), decoder_cases=decoder_checks(),
                  firmware_integration_scenarios=firmware_checks(),
                  simulation='native execution of production PS C; PL responses are fixtures',
                  real_pl_math_executed=False, board_tested=False, numeric_model_signoff=False,
                  seconds=round(time.time()-start, 2), status='PASS')
    (ROOT / 'reports/verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
