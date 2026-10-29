#!/usr/bin/env python3
"""
viking_spread_vs_accuracy.py -- the reference-free spread, computed properly.

Panel (b) of the Viking control figure plots the mean accuracy against the
scatter of that same accuracy, and both quantities need the recorded signature.
It therefore cannot support the claim it is used for, that a spread measurable
WITHOUT a reference tracks the accuracy.

This computes the quantity that claim requires: the agreement among the
extracted wavelets themselves, within each span family. Nothing but the
extractions enters it, so it is available on a survey where no signature
survives, which is the situation the paper addresses. The accuracy is computed
alongside, through the same reference chain as make_paper_figures_vik.py, only
so that the two can be plotted against each other.

    python3 validation/viking/viking_spread_vs_accuracy.py FarField.dat viking_blocks

Families are read from the file names: A is the 2.5 km blocks, D the 12.5 km
cells, C the 25 km combs. Pass --fam to change the set.
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np
from scipy.signal import butter, sosfiltfilt

PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)'
                 r'_wavelet\.npy$')
SPAN = {'A': 2.5, 'D': 12.4, 'C': 24.8}


# ---------------------------------------------------------------- reference
def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low',
                              output='sos'), x)


def far_field(path, delay, z_rec, vw, dt_ff):
    ff = np.loadtxt(path)[delay:]
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


def ncc_best(a, b, max_lag):
    """NCC maximised over a bounded shift, as everywhere else in the paper."""
    n = a.size
    best = -2.0
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            u, v = a[L:], b[:n - L]
        else:
            u, v = a[:n + L], b[-L:]
        if u.size > 3:
            best = max(best, ncc(u, v))
    return best


def best_lag(ex, ff_g, cfg):
    return max((ncc(ex, reference(ff_g, L, cfg)), L)
               for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("runs")
    ap.add_argument("--fam", nargs="+", default=["A", "D", "C"])
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--dt_ff", type=float, default=1e-3)
    ap.add_argument("--delay", type=int, default=27)
    ap.add_argument("--z_rec", type=float, default=10.0)
    ap.add_argument("--vw", type=float, default=1448.0)
    ap.add_argument("--lp_hz", type=float, default=115.0)
    ap.add_argument("--nw", type=int, default=50)
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--pair_lag", type=int, default=5,
                    help="shift allowed when comparing two extractions, in "
                         "samples; they share a time origin, so this is small")
    ap.add_argument("--save_npz", default=None)
    args = ap.parse_args()

    lo, hi = args.band
    cfg = dict(dt=args.dt, dt_ff=args.dt_ff,
               up=int(round(args.dt / args.dt_ff)), nw=args.nw,
               lp_hz=args.lp_hz, lo=lo, hi=hi, maxlag_ms=args.maxlag_ms)
    ff_g = far_field(args.farfield, args.delay, args.z_rec, args.vw, args.dt_ff)

    out = {}
    print("%-4s %5s %8s %10s %10s   %10s"
          % ("fam", "n", "span km", "pairs", "spread", "accuracy"))
    print("-" * 60)
    for fam in args.fam:
        W, names = [], []
        for p in sorted(glob.glob(os.path.join(args.runs, "*_wavelet.npy"))):
            m = PAT.match(os.path.basename(p))
            if not m or m.group("fam") != fam:
                continue
            w = np.load(p).astype(float)
            if w.size != args.nw:
                continue
            W.append(bp(w, args.dt, lo, hi))
            names.append(os.path.basename(p))
        if len(W) < 3:
            print("%-4s  too few extractions" % fam)
            continue

        # the reference-free quantity: how much the extractions differ from
        # one another. Nothing here uses the recorded signature.
        pair = [ncc_best(W[i], W[j], args.pair_lag)
                for i, j in itertools.combinations(range(len(W)), 2)]
        spread = 1.0 - float(np.mean(pair))

        acc = [best_lag(w, ff_g, cfg)[0] for w in W]
        print("%-4s %5d %8.1f %10d %10.4f   %6.4f +- %.4f"
              % (fam, len(W), SPAN.get(fam, float('nan')), len(pair),
                 spread, float(np.mean(acc)), float(np.std(acc, ddof=1))))
        out[fam] = dict(spread=spread, acc_mean=float(np.mean(acc)),
                        acc_sd=float(np.std(acc, ddof=1)), n=len(W),
                        span=SPAN.get(fam, float('nan')),
                        pair=np.array(pair), acc=np.array(acc))

    if len(out) > 2:
        s = np.array([out[f]["spread"] for f in out])
        a = np.array([out[f]["acc_mean"] for f in out])
        print("\nPearson correlation between the reference-free spread and the "
              "accuracy: %+.3f over %d spans" % (np.corrcoef(s, a)[0, 1], s.size))
        print("A negative value is what the claim needs: extractions that agree "
              "with each other\nare the ones that agree with the truth. Three "
              "points cannot establish a law,\nso report it as the direction "
              "and not as a fit.")

    if args.save_npz:
        np.savez(args.save_npz, **{f"{f}_{k}": v for f, d in out.items()
                                   for k, v in d.items()})
        print("\nwrote %s" % args.save_npz)


if __name__ == "__main__":
    main()
