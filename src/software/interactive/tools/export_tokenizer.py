"""Export the model's byte BPE tables. No tokenizer runs on the PC at inference."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


def export(source, output):
    obj = json.loads(source.read_text(encoding='utf-8'))
    assert obj['normalizer'] == {'type': 'NFC'}
    assert obj['model']['type'] == 'BPE'
    expected = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    assert obj['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'] == expected
    assert not obj['pre_tokenizer']['pretokenizers'][1]['use_regex']
    vocab = obj['model']['vocab']
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            chars.append(256 + len(chars) - 188)
    # GPT-2 bytes_to_unicode: 188 bytes are represented by themselves.
    inverse = dict(zip(map(chr, chars), values))
    assert len(inverse) == 256
    records = [None] * (max([*vocab.values(), *[x['id'] for x in obj['added_tokens']]]) + 1)
    blob = bytearray()
    for token, index in vocab.items():
        raw = bytes(inverse[c] for c in token)
        records[index] = (len(blob), len(raw), 0)
        blob += raw
    for entry in obj['added_tokens']:
        assert not entry.get('lstrip') and not entry.get('rstrip') and not entry.get('single_word')
        assert not entry.get('normalized')
        raw = entry['content'].encode('utf-8')
        records[entry['id']] = (len(blob), len(raw), 1 | (2 if entry.get('special') else 0))
        blob += raw
    assert all(x is not None for x in records)
    pairs = []
    for merge in obj['model']['merges']:
        left, right = merge.split(' ') if isinstance(merge, str) else merge
        pairs.append((vocab[left], vocab[right], vocab[left + right]))
    blob += b'\0' * (-len(blob) % 4)
    payload = b''.join(struct.pack('<IHH', *x) for x in records)
    payload += bytes(blob)
    payload += b''.join(struct.pack('<III', *x) for x in pairs)
    data = struct.pack('<8s6I', b'QWNBPE01', len(records), len(blob), len(pairs),
                       len(payload), zlib.crc32(payload), 1) + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  sha256=hashlib.sha256(data).hexdigest(), vocab_records=len(records),
                  merge_count=len(pairs), bytes=len(data), normalization='NFC', format='QWNBPE01')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tokenizer', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    export(args.tokenizer, args.output)
