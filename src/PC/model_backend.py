"""Mathematical PL substitute using the exact deployed INT8/BF16 assets.

This models Qwen operators and KV cache, not RTL cycles or bit-exact reduction.
It exists to exercise the complete PS application without a physical board.
"""
import json
import math
from pathlib import Path
import struct
import numpy as np
import torch


class Model:
    def __init__(self, sd, threads=4):
        torch.set_num_threads(threads)
        self.sd = Path(sd) / 'QWEN25'
        self.config = json.loads((self.sd / 'CONFIG.JSON').read_text(encoding='utf-8'))
        assert (self.config['hidden'], self.config['layers'], self.config['vocab']) == (896, 24, 151936)
        self.scale = self.bf16_file('SCALE.BIN')
        self.bias = self.bf16_file('BIAS.BIN')
        self.gamma = self.bf16_file('GAMMA.BIN')
        self.embedding = self.bf16_file('EMBED.BIN').reshape(151936, 896)
        self.weights = []
        descriptors = (self.sd / 'DESC.BIN').read_bytes()
        packed = np.memmap(self.sd / 'WEIGHTS.BIN', mode='r', dtype=np.int8)
        for layer in range(24):
            items = {}
            for key, field, columns, inner, scale_off in [
                ('q', 40, 896, 896, 0), ('k', 48, 128, 896, 896), ('v', 56, 128, 896, 1024),
                ('o', 80, 896, 896, 1152), ('gate', 96, 4864, 896, 2048),
                ('up', 104, 4864, 896, 6912), ('down', 120, 896, 4864, 11776)]:
                address, = struct.unpack_from('<Q', descriptors, layer * 256 + field)
                off = address - 0x41000000
                assert 0 <= off < packed.size
                q = self.unpack(packed[off:off + columns * inner], columns, inner)
                items[key] = (q, self.scale[layer * 12672 + scale_off:layer * 12672 + scale_off + columns])
            self.weights.append(items)
        lm = np.memmap(self.sd / 'LMW.BIN', mode='r', dtype=np.int8)
        self.lm = (self.unpack(lm, 151936, 896), self.scale[304128:])
        self.inverse_frequency = 1.0 / (1000000.0 ** (torch.arange(0, 64, 2).float() / 64))
        self.clear()

    @staticmethod
    def bf(x): return x.to(torch.bfloat16).float()

    def bf16_file(self, name):
        raw = np.fromfile(self.sd / name, dtype='<u2').astype(np.uint32) << 16
        return torch.from_numpy(raw.view(np.float32))

    @staticmethod
    def unpack(raw, columns, inner):
        physical = np.asarray(raw).reshape(columns // 128, inner // 128, 4, 128, 2, 16)
        logical = physical.transpose(0, 4, 2, 5, 1, 3).copy().reshape(columns, inner)
        return torch.from_numpy(logical).float()

    def clear(self):
        self.keys = [None] * 24
        self.values = [None] * 24
        self.position = 0

    def linear(self, x, weight):
        q, scale = weight
        return self.bf(self.bf(torch.nn.functional.linear(x, q)) * scale)

    def project_bias(self,x,weight,bias):
        return self.bf(self.linear(x,weight)+bias)

    def norm(self, x, gamma):
        mean = torch.mean(x * x, dim=-1, keepdim=True)
        return self.bf(self.bf(x * torch.rsqrt(mean + 1e-6)) * gamma)

    def rope(self, x, positions):
        phase = positions[:, None] * self.inverse_frequency[None, :]
        cosine = torch.cat((phase.cos(), phase.cos()), dim=-1)[:, None, :]
        sine = torch.cat((phase.sin(), phase.sin()), dim=-1)[:, None, :]
        rotated = torch.cat((-x[..., 32:], x[..., :32]), dim=-1)
        return self.bf(x * cosine + rotated * sine)

    @torch.inference_mode()
    def forward(self, ids, position):
        if position != self.position or not ids or position + len(ids) > 256:
            raise ValueError(f'KV position mismatch pos={position} cached={self.position} ids={len(ids)}')
        x = self.embedding[ids]
        rows = len(ids)
        positions = torch.arange(position, position + rows).float()
        context = position + rows
        causal = torch.arange(context)[None, :] > positions[:, None]
        for layer, weight in enumerate(self.weights):
            h = self.norm(x, self.gamma[layer * 1792:layer * 1792 + 896])
            q = self.project_bias(h,weight['q'],self.bias[layer*1152:layer*1152+896]).reshape(rows,14,64)
            k = self.project_bias(h,weight['k'],self.bias[layer*1152+896:layer*1152+1024]).reshape(rows,2,64)
            v = self.project_bias(h,weight['v'],self.bias[layer*1152+1024:layer*1152+1152]).reshape(rows,2,64)
            q, k = self.rope(q, positions), self.rope(k, positions)
            self.keys[layer] = k if self.keys[layer] is None else torch.cat((self.keys[layer], k), dim=0)
            self.values[layer] = v if self.values[layer] is None else torch.cat((self.values[layer], v), dim=0)
            keys = self.keys[layer].repeat_interleave(7, dim=1)
            values = self.values[layer].repeat_interleave(7, dim=1)
            score = self.bf(torch.einsum('rhd,shd->hrs', q, keys) / 8.0)
            score.masked_fill_(causal[None, :, :], -float('inf'))
            probability = self.bf(torch.softmax(score, dim=-1))
            attn = self.bf(torch.einsum('hrs,shd->rhd', probability, values)).reshape(rows, 896)
            x = self.bf(x + self.linear(attn, weight['o']))
            h = self.norm(x, self.gamma[layer * 1792 + 896:(layer + 1) * 1792])
            gate, up = self.linear(h, weight['gate']), self.linear(h, weight['up'])
            mid = self.bf(self.bf(torch.nn.functional.silu(gate)) * up)
            x = self.bf(x + self.linear(mid, weight['down']))
        h = self.norm(x[-1:], self.gamma[43008:])
        logits = self.linear(h, self.lm).flatten()
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite mathematical model logits')
        self.last_logits = logits
        self.position = context
        # Match the accelerator's highest-ID tie-break behavior.
        token = logits.numel() - 1 - int(torch.argmax(logits.flip(0)))
        return token


if __name__ == '__main__':
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument('sd', type=Path)
    args = ap.parse_args()
    model = Model(args.sd)
    print('MODEL_READY', flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('clear'): model.clear(); print('{}', flush=True)
        else: print(json.dumps(dict(token=model.forward(req['ids'], req['position']))), flush=True)
