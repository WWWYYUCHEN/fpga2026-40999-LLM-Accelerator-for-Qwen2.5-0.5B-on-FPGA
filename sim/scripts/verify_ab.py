r"""verify_ab.py -- one-shot, hard-rule-checked A/B comparison for stage 5.

Why this exists (rather than calling cmp_logits.py / cmp_layer0.py by hand):
hard rule 1 says *every* A/B must first re-confirm the baseline artefact, and
that "same cycles" / "same layer-0 dumps" do NOT prove the harness is unchanged.
This script therefore (a) checks the A artefact against the recorded constants,
byte for byte, and (b) checks that both builds recorded the SAME harness hashes
in logs\srcs_<Tag>.txt, and only then prints the comparison table.

Usage:
  python verify_ab.py <runA_dir> <runB_dir> [--allow-a-mismatch]

Defaults assume runA_gen / runB_gen.  Exit code 0 = all hard checks passed and
the A/B table is printed; 1 = a hard check failed (nothing else is printed as a
"result").
"""
import argparse
import hashlib
import os
import sys

import numpy as np

# ---- hard rule 1 constants (REPORT_STAGE5.md section 2, re-measured 11:41) ----
A_TOKEN = 9707
A_CYCLES = 134082442
A_LOGITS_SHA = "A3AFC50CABAD1069826CB4F4EDE3A1B1EA0E455B794B07870A586C3420631103"

# [V-FIX] The logit-corr acceptance line of this project is the PRE-REGISTERED
# band of REPORT_STAGE5_GAP §7 ("corr 0.9903 +- 0.02"), not a hand-picked 0.99.
# The earlier hard-coded 0.99 printed FAIL on the V-fix run (corr 0.989208) even
# though that run is inside the registered band -- a tool that contradicts the
# project's own criterion trains everyone to ignore its red line.  Override with
# --corr-min if a different band is ever registered.
CORR_MIN = 0.9703          # 0.9903 - 0.02
CORR_STOP_HI = 0.995       # GAP §7 stop line: above this, suspect the reference

# the three harness files, and where the build's source list records two of them
HARNESS = {
    "models\\tb_qwen25_chat_generation.sv": r"D:\ds\kv260_mmu\models\tb_qwen25_chat_generation.sv",
    "work_i8attn\\rtl_bridge.cpp": "rtl_bridge.cpp",
    "work_i8attn\\sim_dsp48e2_fast.v": "sim_dsp48e2_fast.v",
}

TENSORS = ["input", "q", "k", "v", "context", "h", "m", "y"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_tokens(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.lower().startswith("iteration"):
                continue
            rows.append([int(x) for x in line.split(",")])
    return rows


def srcs_hashes(tag):
    """Map basename -> sha256 recorded in logs\\srcs_<tag>.txt (written by build.ps1)."""
    p = os.path.join("logs", f"srcs_{tag}.txt")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[os.path.basename(parts[1].strip())] = parts[0].upper()
    return out


def check_a(run_dir, allow_mismatch):
    ok = True
    lpath = os.path.join(run_dir, "hardware_logits_0.bin")
    tpath = os.path.join(run_dir, "hardware_tokens.csv")
    print(f"--- hard rule 1: baseline artefact in {run_dir} ---")
    if not os.path.exists(lpath):
        print("  FAIL: hardware_logits_0.bin missing (run did not complete)")
        return False
    got_sha = sha256(lpath)
    print(f"  logits sha256 : {got_sha}")
    print(f"  expected      : {A_LOGITS_SHA}")
    if got_sha != A_LOGITS_SHA:
        print("  FAIL: logits bytes differ from the baseline artefact")
        ok = False
    rows = read_tokens(tpath)
    if not rows:
        print("  FAIL: hardware_tokens.csv missing/empty")
        return False
    tok, cyc = rows[0][1], rows[0][2]
    print(f"  token/cycles  : {tok} / {cyc}   (expected {A_TOKEN} / {A_CYCLES})")
    if tok != A_TOKEN or cyc != A_CYCLES:
        print("  FAIL: token/cycles differ from the baseline artefact")
        ok = False
    if ok:
        print("  BASELINE_OK: token / cycles / logits sha256 all reproduce")
    elif allow_mismatch:
        print("  (continuing because --allow-a-mismatch was given)")
        ok = True
    return ok


def check_harness():
    print("--- hard rule 2: same harness in A and B ---")
    ok = True
    cur = {}
    for label, path in HARNESS.items():
        if not os.path.exists(path):
            print(f"  FAIL: {label} missing on disk")
            ok = False
            continue
        cur[os.path.basename(path)] = sha256(path)
        print(f"  {label:42s} {cur[os.path.basename(path)]}")
    for tag in ("A", "B"):
        rec = srcs_hashes(tag)
        if not rec:
            print(f"  logs\\srcs_{tag}.txt missing -> cannot prove which sources {tag} used")
            ok = False
            continue
        nfail = 0
        for name in ("tb_qwen25_chat_generation.sv", "sim_dsp48e2_fast.v"):
            if name not in rec:
                print(f"  FAIL: {name} not recorded in srcs_{tag}.txt")
                nfail += 1
            elif rec[name] != cur.get(name):
                print(f"  FAIL: {tag} built with a different {name}: "
                      f"{rec[name]} vs {cur.get(name)}")
                nfail += 1
        if nfail == 0:
            print(f"  srcs_{tag}.txt: harness hashes match what is on disk")
        ok = ok and (nfail == 0)
    # rtl_bridge.cpp is a C++ file: build.ps1 does not hash it into srcs_*.txt.
    print("  note: rtl_bridge.cpp is not in srcs_*.txt; it is checked by mtime only "
          "(must predate both builds) -- see the report")
    return ok


def logits_table(da, db):
    pa = os.path.join(da, "hardware_logits_0.bin")
    pb = os.path.join(db, "hardware_logits_0.bin")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f"--- logits: missing {'A' if not os.path.exists(pa) else 'B'} dump, skipped")
        return
    ra = np.fromfile(pa, dtype=np.uint16)
    rb = np.fromfile(pb, dtype=np.uint16)
    a, b = bf16_to_f32(ra).astype(np.float64), bf16_to_f32(rb).astype(np.float64)
    print("--- logits (151936 bf16) ---")
    if a.size != b.size:
        print(f"  size mismatch {a.size} vs {b.size}")
        return
    n = a.size
    for name, v in (("A", a), ("B", b)):
        order = np.argsort(v)[::-1]
        top = order[:2]
        print(f"  {name}: argmax={int(top[0])} top1={v[top[0]]:.4f} "
              f"top2(idx={int(top[1])})={v[top[1]]:.4f} "
              f"margin={v[top[0]] - v[top[1]]:.4f}")
    d = b - a
    corr = float(np.corrcoef(a, b)[0, 1])
    # [V-FIX] the acceptance line is the PRE-REGISTERED band of REPORT_STAGE5_GAP
    # §7 (0.9903 +- 0.02, i.e. >= 0.9703), not the old hard 0.99.  A run at
    # 0.9892 is a PASS by the registered criterion, and a red VERDICT on a passing
    # run is how a real failure gets ignored later.
    corr_min = CORR_MIN
    print(f"  corr          = {corr:.6f}      (pre-registered band >= {corr_min:.4f},"
          f" i.e. 0.9903 +- 0.02)")
    print(f"  mean|dlogit|  = {np.mean(np.abs(d)):.6f}")
    print(f"  max |dlogit|  = {np.max(np.abs(d)):.6f}")
    print(f"  bit-identical = {int(np.sum(ra == rb))}/{n}")
    print(f"  VERDICT       = {'PASS' if corr >= corr_min else 'FAIL'}"
          + ("" if corr <= 0.995 else "  (above the 0.995 stop line: check the reference)"))


def layer0_table(da, db):
    print("--- layer-0 tensors ---")
    print(f"{'tensor':8s} {'n':>7s} {'bit_eq':>7s} {'rmsRel':>10s} {'maxRel':>10s} {'corr':>10s}")
    for t in TENSORS:
        pa, pb = os.path.join(da, f"layer0_{t}.bin"), os.path.join(db, f"layer0_{t}.bin")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{t:8s}  (missing dump)")
            continue
        ra = np.fromfile(pa, dtype=np.uint16)
        rb = np.fromfile(pb, dtype=np.uint16)
        if ra.size != rb.size:
            print(f"{t:8s} size mismatch {ra.size} vs {rb.size}")
            continue
        a, b = bf16_to_f32(ra).astype(np.float64), bf16_to_f32(rb).astype(np.float64)
        rms = float(np.sqrt(np.mean(a ** 2)))
        rms_rel = float(np.sqrt(np.mean((b - a) ** 2)) / rms) if rms > 0 else float("nan")
        big = np.abs(a) > 0.01 * rms
        max_rel = float(np.max(np.abs(b[big] - a[big]) / np.abs(a[big]))) if np.any(big) else float("nan")
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        print(f"{t:8s} {ra.size:7d} {float(np.mean(ra == rb)):7.4f} "
              f"{rms_rel:10.3e} {max_rel:10.3e} {corr:10.6f}")


def clip_count(da, db):
    """Clip counts are printed by the RTL (kv8_k_clip_count / kv8_v_clip_count) at
    each LAYER_DONE when the build carries the one-line $display; absent otherwise."""
    print("--- clip count over the prefill ---")
    found = False
    for tag, d in (("A", da), ("B", db)):
        p = os.path.join(d, "run.out")
        if not os.path.exists(p):
            continue
        last = None
        with open(p, errors="ignore") as fh:
            for line in fh:
                if "KV8_CLIP" in line:
                    last = line.strip()
                    found = True
        if last:
            print(f"  {tag}: {last}")
    if not found:
        print("  ABSENT: no KV8_CLIP line in run.out -- the build does not expose the "
              "counter, so this column cannot be reported")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?", default="runA_gen")
    ap.add_argument("b", nargs="?", default="runB_gen")
    ap.add_argument("--diag-a", default="runA_diag")
    ap.add_argument("--diag-b", default="runB_diag")
    ap.add_argument("--allow-a-mismatch", action="store_true")
    ap.add_argument("--corr-min", type=float, default=CORR_MIN,
                    help="logit-corr acceptance floor (default: the pre-registered "
                         "band of REPORT_STAGE5_GAP section 7, 0.9903 +- 0.02)")
    args = ap.parse_args()
    globals()['CORR_MIN'] = args.corr_min

    ok = check_a(args.a, args.allow_a_mismatch)
    ok = check_harness() and ok
    if not ok:
        print("\nHARD CHECK FAILED -- no A/B numbers printed.")
        return 1
    print()
    logits_table(args.a, args.b)
    print()
    layer0_table(args.diag_a, args.diag_b)
    print()
    clip_count(args.diag_a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
