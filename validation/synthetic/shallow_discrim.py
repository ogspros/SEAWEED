#!/usr/bin/env python3
"""
shallow_discrim.py -- how much discrimination the moveout test has left, depth
by depth.

The picker separates the seafloor from a sub-bottom reflector at the same time
by asking whether the event moves out at the velocity of water.  That test
needs offset, and the stretch limit caps the offset at
x_max = vw * t0 * sqrt((1+s)^2 - 1), which is proportional to t0: at 400 m of
water it leaves 22 channels, at 100 m it leaves two, and the code then takes
the three nearest anyway because a semblance on one or two channels is close to
1 by construction.  So in shallow water the discriminant may be exhausted, and
the failures there would not be a matter of tuning.

This measures it where the answer is known.  For every record it regenerates
the model, runs Stage 1 as the workflow does (AGC first, then the candidates
and the scoring), and prints:

  how many channels the score actually used, and the stretch that cost;
  the semblance at the TRUE seafloor;
  the best semblance among the candidates that are NOT the seafloor;
  the margin between the two, which is what the pick rests on.

A margin of a few hundredths means any coherent event passes and the physical
test has stopped working, whatever the window length.

    python3 validation/synthetic/shallow_discrim.py --config validation/synthetic/matrix_shallow.yaml --n 3
    python3 validation/synthetic/shallow_discrim.py --config validation/synthetic/matrix_v2.yaml --n 2 --axis grid_D_contam
"""
import argparse
import json
import os

import numpy as np

import make_fig_synthetic as F
from seaweed import pick_consensus as PC
from seaweed.wavelet_extractor_v2 import estimate_vw_direct


# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_shallow.yaml"))
    ap.add_argument("--n", type=int, default=3, help="records per depth")
    ap.add_argument("--axis", default=None)
    ap.add_argument("--near_ms", type=float, default=40.0,
                    help="a candidate within this of the true seafloor counts "
                         "as the seafloor and not as a rival")
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config))
    root = cfg["paths"]["out_root"]
    runs = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    if a.axis:
        runs = [r for r in runs if r["axis"] == a.axis]
    by = {}
    for r in runs:
        by.setdefault(r["params"]["D_m"], []).append(r)

    print("%7s %7s %5s %8s %9s %9s %9s %9s %8s"
          % ("D_m", "D_true", "nch", "stretch", "sem true", "sem rival",
             "rival/t0", "margin", "pick err"))
    for D in sorted(by, key=float):
        for run in by[D][:a.n]:
            try:
                g, dt, offs, w_bare, Dts = F.regenerate(run, cfg)
            except Exception as exc:
                print("%7.0f  regeneration failed: %s" % (D, str(exc)[:50]))
                continue
            m = cfg["meta"]
            Dt = float(np.median(np.atleast_1d(np.asarray(Dts, float))))
            t0_true = 2.0 * Dt / m["v_water"]
            vw, _x, _n = estimate_vw_direct(g, dt, offs, D_guess=Dt)
            gg = PC.agc(np.asarray(g, float), dt)

            ch, st = PC.usable_channels(t0_true, np.asarray(offs, float), vw)
            sem_true, nch = PC.semblance_at(gg, dt, offs, t0_true, vw)

            hi = min(1.50, (gg.shape[1] * dt) / 2.2)
            cands = sorted(set(np.round(
                PC.autocorr_candidates(gg, dt, 0.10, hi)
                + PC.semblance_candidates(gg, dt, np.asarray(offs, float), vw,
                                          0.10, hi), 4)))
            rival, t_riv = -1.0, float("nan")
            for t in cands:
                if abs(t - t0_true) * 1e3 <= a.near_ms:
                    continue
                v = PC.semblance_at(gg, dt, offs, t, vw)[0]
                if v > rival:
                    rival, t_riv = v, t
            try:
                t0s, _rep = PC.pick_line(np.asarray(g, float), dt, offs, vw,
                                         group=10, tol_ms=120.0)
                err = (float(np.median(t0s)) - t0_true) * 1e3
            except Exception:
                err = float("nan")
            # the ratio names the rival: 2.00 is the first multiple, which
            # has exactly the moveout of water and is therefore not separable
            # from a deeper seafloor by this test at all
            print("%7.0f %7.0f %5d %8.2f %9.3f %9.3f %9.2f %+9.3f %+8.0f"
                  % (D, Dt, nch, st, sem_true, rival, t_riv / t0_true,
                     sem_true - rival, err))
    print("\n  'sem rival' is the best candidate more than %.0f ms from the "
          "true seafloor." % a.near_ms)
    print("  A margin near zero means the moveout test no longer separates "
          "them, and the pick\n  is then decided by the multiple, the polarity "
          "and the neighbours instead.")
    print("  'rival/t0' names it: 2.0 is the first water-layer multiple, which "
          "moves out\n  exactly like water and cannot be separated from a "
          "deeper seafloor by moveout.")


if __name__ == "__main__":
    main()
