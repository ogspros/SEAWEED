#!/usr/bin/env python3
"""
viking_coherence_vs_truth.py -- recompute the correlation between stack
coherence and agreement with the recorded signature, over the Viking runs.

The paper quotes this number three times and two of them disagree, at
$r = -0.62$ and $r = -0.63$. Rather than hunting for whichever file it was
first printed from, this recomputes it from the runs themselves, so the value
comes with its sample size and its per-family breakdown attached.

The coherence is read from each _sel.npz, under the key make_paper_figures_vik.py
uses. The agreement is recomputed through the same reference chain as that
script: recording delay stripped, receiver ghost applied, low-passed, decimated
to the extraction interval, band-passed like the extraction, and slid over a
bounded lag.

    python3 validation/viking/viking_coherence_vs_truth.py FarField.dat viking_blocks

Add --fam C to restrict to one family, or list files explicitly. The default
uses every run in the directory, which is what the paper's "over the Viking
runs" should mean.
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.signal import butter, sosfiltfilt

PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)'
                 r'_wavelet\.npy$')
COH_KEYS = ("coherence", "coh", "stack_coherence")


# ------------------------------------------------------------------ reference
# Copied from make_paper_figures_vik.py so this runs standalone. A change there
# must be mirrored here or the two will disagree without saying so.
def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low',
                              output='sos'), x)


def far_field(path, delay, z_rec, vw, dt_ff):
    ff = np.loadtxt(path)
    if ff.ndim != 1:
        raise SystemExit("%s does not have exactly one column" % path)
    ff = ff[delay:]
    k = int(round((2.0 * z_rec / vw) / dt_ff))
    g = np.zeros(k + 1)
    g[0], g[k] = 1.0, -1.0
    return np.convolve(ff, g)


def reference(ff_g, lag_ms, cfg):
    k = int(round(lag_ms))
    w = np.concatenate([np.zeros(k), ff_g]) if k >= 0 else ff_g[-k:]
    w = lp(np.pad(w, (0, 400)), cfg["dt_ff"], cfg["lp_hz"])
    w4 = w[::cfg["up"]][:cfg["nw"]]
    if w4.size < cfg["nw"]:
        w4 = np.pad(w4, (0, cfg["nw"] - w4.size))
    return bp(w4, cfg["dt"], cfg["lo"], cfg["hi"])


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best_lag(ex, ff_g, cfg):
    return max((ncc(ex, reference(ff_g, L, cfg)), L)
               for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("runs", nargs="+", help="directory, or wavelet files")
    ap.add_argument("--fam", default=None)
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--dt_ff", type=float, default=1e-3)
    ap.add_argument("--delay", type=int, default=27)
    ap.add_argument("--z_rec", type=float, default=10.0)
    ap.add_argument("--vw", type=float, default=1448.0)
    ap.add_argument("--lp_hz", type=float, default=115.0)
    ap.add_argument("--nw", type=int, default=50)
    ap.add_argument("--maxlag_ms", type=int, default=30)
    args = ap.parse_args()

    lo, hi = args.band
    cfg = dict(dt=args.dt, dt_ff=args.dt_ff,
               up=int(round(args.dt / args.dt_ff)), nw=args.nw,
               lp_hz=args.lp_hz, lo=lo, hi=hi, maxlag_ms=args.maxlag_ms)
    ff_g = far_field(args.farfield, args.delay, args.z_rec, args.vw, args.dt_ff)

    paths = []
    for p in args.runs:
        paths.extend(sorted(glob.glob(os.path.join(p, "*_wavelet.npy")))
                     if os.path.isdir(p) else [p])

    rows, skipped = [], []
    for p in paths:
        m = PAT.match(os.path.basename(p))
        fam = m.group("fam") if m else "?"
        if args.fam and fam != args.fam:
            continue
        w = np.load(p).astype(float)
        if w.size != args.nw:
            skipped.append((os.path.basename(p), "%d samples" % w.size))
            continue
        sel = p.replace("_wavelet.npy", "_sel.npz")
        if not os.path.exists(sel):
            skipped.append((os.path.basename(p), "no _sel.npz"))
            continue
        z = np.load(sel, allow_pickle=True)
        key = next((k for k in COH_KEYS if k in z), None)
        if key is None:
            skipped.append((os.path.basename(p),
                            "no coherence key; has %s" % list(z.keys())))
            continue
        rows.append((os.path.basename(p).replace("_wavelet.npy", ""), fam,
                     float(z[key]), best_lag(bp(w, args.dt, lo, hi), ff_g, cfg)[0]))

    if skipped:
        print("skipped %d run(s):" % len(skipped))
        for n, why in skipped:
            print("   %-30s %s" % (n[:30], why))
        print()

    if len(rows) < 3:
        raise SystemExit("only %d usable runs" % len(rows))

    coh = np.array([r[2] for r in rows])
    acc = np.array([r[3] for r in rows])
    print("%-30s %3s %10s %10s" % ("run", "fam", "coherence", "NCC"))
    print("-" * 58)
    for n, f, c, a in rows:
        print("%-30s %3s %10.4f %10.4f" % (n[:30], f, c, a))
    print("-" * 58)
    print("n = %d" % len(rows))
    print("Pearson  r = %+.4f" % np.corrcoef(coh, acc)[0, 1])
    print("Spearman r = %+.4f" % spearman(coh, acc))

    fams = sorted({r[1] for r in rows})
    if len(fams) > 1:
        print("\nby family, since the paper's statement is about the pooled set")
        print("%-4s %4s %10s %10s %10s" % ("fam", "n", "coh mean",
                                           "NCC mean", "r"))
        for f in fams:
            c = np.array([r[2] for r in rows if r[1] == f])
            a = np.array([r[3] for r in rows if r[1] == f])
            r = np.corrcoef(c, a)[0, 1] if c.size > 2 else float("nan")
            print("%-4s %4d %10.4f %10.4f %+10.4f"
                  % (f, c.size, c.mean(), a.mean(), r))

    print("""
The paper quotes this as "over the Viking runs, stack coherence and agreement
with the recorded signature correlate at r = ...". Use the Pearson value above,
and check that n matches the 41 the text claims. If it does not, the text was
computed on a different subset and the subset should be named, not the number
changed.""")


if __name__ == "__main__":
    main()
