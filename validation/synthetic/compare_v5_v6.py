#!/usr/bin/env python3
"""
compare_v5_v6.py -- the extraction with the seafloor known, against the whole
chain with the seafloor picked.

The first is the diagnostic: it says what the extraction can do when the window
is where it should be.  The second is what a user receives.  The difference
between them is the cost of the automatic picking, and it is the number the
paper needs in order to claim an automatic workflow.

Both are read at zero deliberate bias and 30 dB, over the same applicable
domain, so the two differ in the seafloor and in nothing else.

    python3 validation/synthetic/compare_v5_v6.py
"""
import glob
import json
import os

import numpy as np

V5, V6 = "wavelet_matrix_v5_bias", "wavelet_matrix_v6"
SCORE = "ncc_shape"


def load(root, bias=0.0):
    p = os.path.join(root, "results.jsonl")
    R = ([json.loads(l) for l in open(p)] if os.path.exists(p) else
         [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "results_shard*.jsonl")))
          for l in open(f)])
    return [r for r in R if r.get("ok") and r.get("D_m", 0) >= 220
            and not str(r.get("run_id", "")).startswith("ref_")
            and (r.get("t0_bias_ms") or 0.0) == bias]


def block(lab, g):
    v = np.array([r[SCORE] for r in g if r.get(SCORE) is not None
                  and r[SCORE] == r[SCORE]], float)
    off = sum(1 for r in g if r.get("origin_off_wavelet"))
    bad = sum(1 for r in g if r.get("pick_failed"))
    print("%-28s n=%4d | no score %4d (%4.1f%%) | picker failed %3d | "
          "median %s | mean %s"
          % (lab, len(g), off, 100.0 * off / max(len(g), 1), bad,
             "%.3f" % np.median(v) if v.size else "  -  ",
             "%.3f +- %.3f" % (v.mean(), v.std()) if v.size else "  -  "))
    return v


def main():
    A, B = load(V5), load(V6)
    print("=" * 78)
    print("THE EXTRACTION, AND THE CHAIN")
    print("=" * 78)
    a = block("seafloor known (v5)", A)
    b = block("seafloor picked (v6)", B)
    if a.size and b.size:
        print("\ncost of the automatic picking: median %+.3f, "
              "failures %+.1f points"
              % (np.median(b) - np.median(a),
                 100.0 * (sum(1 for r in B if r.get("origin_off_wavelet")) / max(len(B), 1)
                          - sum(1 for r in A if r.get("origin_off_wavelet")) / max(len(A), 1))))

    # paired, which removes the scatter between models
    ka = {r["run_id"]: r for r in A}
    pairs = [(ka[r["run_id"]], r) for r in B if r["run_id"] in ka]
    d = np.array([q[SCORE] - p[SCORE] for p, q in pairs
                  if p.get(SCORE) == p.get(SCORE) and q.get(SCORE) == q.get(SCORE)
                  and p.get(SCORE) is not None and q.get(SCORE) is not None], float)
    if d.size:
        print("\npaired on the same model, %d of %d pairs scored in both:"
              % (d.size, len(pairs)))
        print("  difference median %+.3f, mean %+.3f, worse in %.0f%% of them"
              % (np.median(d), d.mean(), 100 * np.mean(d < 0)))
        lost = sum(1 for p, q in pairs
                   if not p.get("origin_off_wavelet") and q.get("origin_off_wavelet"))
        gained = sum(1 for p, q in pairs
                     if p.get("origin_off_wavelet") and not q.get("origin_off_wavelet"))
        print("  records that the picking loses: %d; that it recovers: %d"
              % (lost, gained))

    t = np.array([r["track_pick_err_ms"] for r in B
                  if isinstance(r.get("track_pick_err_ms"), (int, float))
                  and r["track_pick_err_ms"] == r["track_pick_err_ms"]], float)
    if t.size:
        print("\npicking error against the true seafloor, %d records:" % t.size)
        print("  median %+.0f ms | within -100..+50 ms (what the extraction "
              "tolerates) %.0f%% | within +-80 ms %.0f%%"
              % (np.median(t), 100 * np.mean((t >= -100) & (t <= 50)),
                 100 * np.mean(np.abs(t) <= 80)))
        for D in sorted({r["D_m"] for r in B}):
            e = np.array([r["track_pick_err_ms"] for r in B if r["D_m"] == D
                          and isinstance(r.get("track_pick_err_ms"), (int, float))
                          and r["track_pick_err_ms"] == r["track_pick_err_ms"]], float)
            q = [r for r in B if r["D_m"] == D]
            vv = np.array([r[SCORE] for r in q if r.get(SCORE) is not None
                           and r[SCORE] == r[SCORE]], float)
            if e.size:
                print("   D %6.0f m  n=%4d  pick %+7.0f ms  tolerated %3.0f%%  "
                      "median NCC %s"
                      % (D, e.size, np.median(e),
                         100 * np.mean((e >= -100) & (e <= 50)),
                         "%.3f" % np.median(vv) if vv.size else "  -  "))


if __name__ == "__main__":
    main()
