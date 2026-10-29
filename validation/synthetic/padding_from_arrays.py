#!/usr/bin/env python3
"""
padding_from_arrays.py -- how many scores of the canonical matrix were computed
over a zero-filled sample, read from the archived arrays.

wavelet_matrix_v2 does not record frac_padded or n_invented_kept, so the
padding numbers in the paper cannot come from it.  One of the two can be
recovered anyway.  The extractor pads the stack with exact zeros AFTER
stacking, and the archived segment is the one the score was computed over, cut
from that stack with no filtering afterwards.  A compared segment that ends in
exact zeros therefore contains filled samples, and their count is
n_invented_kept.  frac_padded, the padding of the whole window, is not
recoverable: the part the slide stepped clear of was never archived.

The count is checked against wavelet_matrix_v2_TAIL250, which records
n_invented_kept, before it is trusted on the canonical matrix.

    python3 validation/synthetic/padding_from_arrays.py wavelet_matrix_v2_TAIL250 wavelet_matrix_v2
"""
import glob
import json
import os
import sys

import numpy as np


def trailing_zeros(w):
    nz = np.nonzero(w)[0]
    return int(w.size - nz[-1] - 1) if nz.size else int(w.size)


def run(root):
    rp = os.path.join(root, "results.jsonl")
    files = [rp] if os.path.exists(rp) else sorted(
        glob.glob(os.path.join(root, "results_shard*.jsonl")))
    R = [json.loads(l) for f in files for l in open(f)]
    W = {}
    for f in sorted(glob.glob(os.path.join(root, "results_shard*_wavelets.npz"))):
        with np.load(f) as z:
            for k in z.files:
                if not k.endswith("|ref"):
                    W[k] = z[k]
    d = [x for x in R if x.get("ok") and x.get("D_m", 0) >= 220
         and not str(x.get("run_id", "")).startswith("ref_")]
    rows, miss, agree, recorded = [], 0, 0, 0
    for x in d:
        k = "%s|%s|%s" % (x["run_id"], x.get("snr_db"), x.get("t0_bias_ms"))
        if k not in W:
            miss += 1
            continue
        n = trailing_zeros(W[k])
        rows.append((n, x["ncc_prim_true"]))
        if x.get("n_invented_kept") is not None:
            recorded += 1
            agree += int(x["n_invented_kept"] == n)
    print("== %s: domain %d, with arrays %d, without %d" % (root, len(d), len(rows), miss))
    if not rows:
        return
    if recorded:
        print("  check against the recorded n_invented_kept: %d of %d agree%s"
              % (agree, recorded, "" if agree == recorded else "   <-- DO NOT TRUST"))
    n = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    f = n > 0
    print("  scored over a filled sample: %d of %d (%.1f%%)"
          % (f.sum(), n.size, 100 * f.mean()))
    print("  median NCC: all %.3f | clean %.3f | filled %s"
          % (np.median(s), np.median(s[~f]),
             "%.3f" % np.median(s[f]) if f.any() else "-"))
    if f.any():
        print("  filled samples when present: median %d, max %d of %d"
              % (np.median(n[f]), n.max(), W[next(iter(W))].size))


for root in sys.argv[1:] or ["wavelet_matrix_v2_TAIL250", "wavelet_matrix_v2"]:
    run(root)
