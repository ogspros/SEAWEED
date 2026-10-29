#!/usr/bin/env python3
"""
provenance_padding.py -- which results file holds the padding numbers the paper
quotes (29% padded, 8.8% scored over a filled sample, 1637 clean of 2480,
median 0.885 clean against 0.876 overall)?

wavelet_matrix_v2/results.jsonl does not record frac_padded or n_invented_kept
at all.  This reads, for every directory given, both the merged file and the
shards separately, and prints what each one holds, so the number can be traced
to a file before it is kept.

    python3 validation/synthetic/provenance_padding.py wavelet_matrix_v2 wavelet_matrix_v2_TAIL250 \
        wavelet_matrix_v2_before_ref
"""
import glob
import json
import os
import sys

import numpy as np
from datetime import datetime


def when(f):
    return datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")


def report(label, R):
    d = [x for x in R if x.get("ok") and x.get("D_m", 0) >= 220
         and not str(x.get("run_id", "")).startswith("ref_")]
    if not d:
        print("  %-28s no applicable records" % label)
        return
    have = [x for x in d if x.get("frac_padded") is not None]
    line = "  %-28s domain %4d  frac_padded in %4d" % (label, len(d), len(have))
    if have:
        pad = [x for x in have if x["frac_padded"] > 0]
        kept = [x for x in have if (x.get("n_invented_kept") or 0) > 0]
        clean = [x["ncc_prim_true"] for x in have if x["frac_padded"] == 0]
        line += ("  padded %4d (%.1f%%)  scored over a filled sample %4d (%.1f%%)"
                 "  clean %4d  median clean %.3f  all %.3f"
                 % (len(pad), 100. * len(pad) / len(have), len(kept),
                    100. * len(kept) / len(have), len(clean), np.median(clean),
                    np.median([x["ncc_prim_true"] for x in have])))
    else:
        line += "  (no padding fields)  median all %.3f" % np.median(
            [x["ncc_prim_true"] for x in d])
    print(line)


for root in sys.argv[1:] or ["wavelet_matrix_v2"]:
    print("== %s" % root)
    m = os.path.join(root, "results.jsonl")
    if os.path.exists(m):
        print("  results.jsonl modified %s" % when(m))
        report("results.jsonl", [json.loads(l) for l in open(m)])
    sh = sorted(glob.glob(os.path.join(root, "results_shard*.jsonl")))
    if sh:
        print("  %d shards, newest modified %s"
              % (len(sh), max(when(f) for f in sh)))
        report("shards", [json.loads(l) for f in sh for l in open(f)])
