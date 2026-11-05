"""Record agreement and divergence; never mistake protocol success for math signoff."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_backend import Model
from rtl_numeric_reference import configure_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sd', type=Path)
    ap.add_argument('rtl_run', type=Path)
    ap.add_argument('report', type=Path)
    ap.add_argument('--rtl-boundaries',action='store_true')
    args = ap.parse_args()
    model = Model(args.sd)
    if args.rtl_boundaries:configure_reference(model)
    rows = list(csv.DictReader((args.rtl_run / 'hardware_tokens.csv').open()))
    ids, position = [108386], 0  # qt_encode("你好") in raw diagnostic mode.
    items = []
    for row in rows:
        expected = model.forward(ids, position)
        index, actual = int(row['iteration']), int(row['token_id'])
        raw = np.fromfile(args.rtl_run / f'hardware_logits_{index}.bin', dtype='<u2').astype(np.uint32) << 16
        rtl = torch.from_numpy(raw.view(np.float32))
        math_logits = model.last_logits
        diff = (rtl - math_logits).abs()
        items.append(dict(iteration=index, rtl_token=actual, mathematical_token=expected,
                          token_matches=actual == expected, max_abs_logit_difference=float(diff.max()),
                          mean_abs_logit_difference=float(diff.mean()),
                          rtl_min=float(rtl.min()), rtl_max=float(rtl.max()),
                          mathematical_min=float(math_logits.min()), mathematical_max=float(math_logits.max())))
        position += len(ids); ids = [actual]  # Compare identical autoregressive histories.
    result = dict(status='MATCH' if all(x['token_matches'] for x in items) else 'DIVERGENCE',
                  prompt='你好', template='raw', comparisons=items,
                  numerical_signoff=False,reference_rounding='RND8 and fused scale+bias boundary' if args.rtl_boundaries else 'generic tensor BF16',
                  qualification='Token IDs agree for the tested autoregressive history; BF16 logits retain rounding/reduction differences.' if all(x['token_matches'] for x in items) else 'Token ID divergence requires diagnosis; mathematical model does not reproduce RTL per-cycle summation/rounding.')
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
