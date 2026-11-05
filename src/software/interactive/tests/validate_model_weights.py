"""Independently compare unpacked deployment tensors with the source checkpoint."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_backend import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sd', type=Path)
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('report', type=Path)
    args = ap.parse_args()
    started = time.time()
    model = Model(args.sd)
    with args.checkpoint.open('rb') as file:
        header_bytes, = struct.unpack('<Q', file.read(8))
        index = json.loads(file.read(header_bytes))
    def tensor(name, start=None, stop=None):
        entry = index[name]
        assert entry['dtype'] == 'BF16'
        raw = np.memmap(args.checkpoint, dtype='<u2', mode='r',
                        offset=8+header_bytes+entry['data_offsets'][0], shape=tuple(entry['shape']))
        if start is not None: raw = raw[start:stop]
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).float()
    checked = 0
    for layer in range(24):
        for short, name in [('q', 'self_attn.q_proj'), ('k', 'self_attn.k_proj'), ('v', 'self_attn.v_proj'),
                            ('o', 'self_attn.o_proj'), ('gate', 'mlp.gate_proj'), ('up', 'mlp.up_proj'), ('down', 'mlp.down_proj')]:
            source = tensor(f'model.layers.{layer}.{name}.weight')
            scales = (source.abs().amax(dim=1).clamp_min(1e-12)/127).to(torch.bfloat16).float()
            q = torch.round(source / scales[:, None]).clamp(-127, 127)
            actual, actual_scales = model.weights[layer][short]
            assert torch.equal(q, actual), (layer, short, 'INT8 layout/values')
            assert torch.equal(scales, actual_scales), (layer, short, 'BF16 scales')
            checked += 1
        for short, off, count in [('q', 0, 896), ('k', 896, 128), ('v', 1024, 128)]:
            source = tensor(f'model.layers.{layer}.self_attn.{short}_proj.bias')
            assert torch.equal(source, model.bias[layer*1152+off:layer*1152+off+count])
            checked += 1
        for name, off in [('input_layernorm', 0), ('post_attention_layernorm', 896)]:
            assert torch.equal(tensor(f'model.layers.{layer}.{name}.weight'), model.gamma[layer*1792+off:layer*1792+off+896])
            checked += 1
    assert torch.equal(tensor('model.norm.weight'), model.gamma[43008:])
    for begin in range(0, 151936, 2048):
        end = min(begin+2048, 151936)
        source = tensor('model.embed_tokens.weight', begin, end)
        assert torch.equal(source, model.embedding[begin:end])
        scales = (source.abs().amax(dim=1).clamp_min(1e-12)/127).to(torch.bfloat16).float()
        q = torch.round(source/scales[:, None]).clamp(-127, 127)
        assert torch.equal(q, model.lm[0][begin:end])
        assert torch.equal(scales, model.lm[1][begin:end])
    report = dict(status='PASS', layer_tensors=checked, final_norm=True, all_embedding_rows=151936,
                  all_lm_head_rows=151936, source=str(args.checkpoint),
                  checkpoint_sha256=hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest(),
                  seconds=round(time.time()-started, 2),
                  meaning='packed deployment INT8/BF16 assets exactly match original checkpoint quantization',
                  rtl_numerical_equivalence=False)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
