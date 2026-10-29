#!/usr/bin/env python3
"""
tables_v5_extra.py -- the threshold and towing-depth tables, from the v5 reruns.

Same conventions as tables_v5.py: the score is the agreement between the
returned wavelet and the truth, at the best shift within +-8 ms of the origin
bias; a record whose returned window holds less than 80 per cent of the true
wavelet gets no score and is counted as a failure instead of being averaged in.

    python3 validation/synthetic/tables_v5_extra.py wavelet_matrix_shallow_v5 wavelet_matrix_z_v5
"""
import glob
import json
import os
import sys

import numpy as np

SCORE = "ncc_shape"


def load(root):
    p = os.path.join(root, "results.jsonl")
    if os.path.exists(p):
        return [json.loads(l) for l in open(p)]
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "results_shard*.jsonl")))
            for l in open(f)]


def stat(g):
    v = np.array([r[SCORE] for r in g if r.get(SCORE) is not None
                  and r[SCORE] == r[SCORE]], float)
    off = sum(1 for r in g if r.get("origin_off_wavelet"))
    return len(g), off, v


def main():
    roots = sys.argv[1:] or ["wavelet_matrix_shallow_v5", "wavelet_matrix_z_v5"]
    for root in roots:
        try:
            R = [r for r in load(root) if r.get("ok")]
        except Exception as e:
            print("== %s: %s" % (root, e)); continue
        print("=" * 74)
        print("%s: %d usable records" % (root, len(R)))
        print("=" * 74)
        # the towing-depth branch only where the towing depth actually varies:
        # every record carries z_src_m, so testing that it is present put the
        # whole shallow matrix into one row of the wrong table
        pairs = {(r.get("z_src_m"), r.get("z_rec_m")) for r in R}
        if len(pairs) > 1:
            print("%7s %7s %14s %6s %8s %18s %9s"
                  % ("z_src", "z_rec", "notches (Hz)", "n", "no score", "NCC",
                     "coherence"))
            for zs, zr in sorted({(r["z_src_m"], r["z_rec_m"]) for r in R}):
                g = [r for r in R if r["z_src_m"] == zs and r["z_rec_m"] == zr]
                n, off, v = stat(g)
                print("%7.0f %7.0f %7.0f,%6.0f %6d %7.0f%% %10.3f +- %.3f %9.3f"
                      % (zs, zr, 1500 / (2 * zs), 1500 / (2 * zr), n,
                         100.0 * off / max(n, 1),
                         v.mean() if v.size else float("nan"),
                         v.std() if v.size else float("nan"),
                         np.median([r["coh_prim"] for r in g])))
            continue
        print("%7s %7s %7s %6s %8s %18s %9s %9s"
              % ("D_m", "D_true", "margin", "n", "no score", "NCC", "sep (ms)",
                 "coherence"))
        for D in sorted({r["D_m"] for r in R}, key=float):
            g = [r for r in R if r["D_m"] == D]
            n, off, v = stat(g)
            print("%7.0f %7.0f %7.2f %6d %7.0f%% %10.3f +- %.3f %9.0f %9.3f"
                  % (D, np.median([r.get("D_true", np.nan) for r in g]),
                     np.median([r["sep_margin"] for r in g]), n,
                     100.0 * off / max(n, 1),
                     v.mean() if v.size else float("nan"),
                     v.std() if v.size else float("nan"),
                     np.median([r.get("sep_direct_primary_ms", np.nan) for r in g]),
                     np.median([r["coh_prim"] for r in g])))
        print("\n  D_true is what the generator actually built: it does not go")
        print("  below 78 m, so the cells requested at 40, 55 and 70 m all hold")
        print("  the same model and belong in one row.")


if __name__ == "__main__":
    main()
