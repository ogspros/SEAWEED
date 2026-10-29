#!/usr/bin/env python3
"""
check_rev2.py -- the numbers the second review asks about, read from the
results files instead of reconstructed.

    python3 validation/synthetic/check_rev2.py wavelet_matrix_v2 <threshold matrix dir> <noise matrix dir>

Each argument is a directory holding results.jsonl (or results_shard*.jsonl).
Every block below prints only if the file has the fields it needs, so the same
command can be pointed at all three matrices.

1. MODELLED DEPTH.  sep_direct_primary_ms is computed in run_matrix_v2.py as
   (hypot(2 D_nom, x) - x) / vw_est, with D_nom the median of the depths
   actually present in the velocity model.  It is geometric: nothing clips it.
   So 51 ms at three nominal depths means either D_nom or vw_est was the same
   in those three cells.  51 ms at 125 m offset is D ~ 79 m at 1500 m/s, and
   would need vw ~ 460 m/s if the model really were 40 m deep.  The block
   prints, per nominal depth, what the model held.

2. EFFECTIVE SNR.  run_matrix_v2.py does not measure it on the stack.  It is
   20 log10(rms of the NOISE-FREE near-channel window / rms of the added
   noise) + 10 log10(number of shots), on the unfiltered gather.  The offset
   from the nominal SNR is therefore the same at every level by construction,
   and splits into a window term and a stacking term.  The block prints both.

3. MATCHED-FILTER EDGE.  The multiple is searched over +-80 ms about its
   predicted arrival.  A lag at the edge means the maximum lay outside the
   window and the depth it implies is a boundary, not a measurement.  The
   multiple's own offset from its lead is dt_matched_ms + origin_minus_lead_ms.

4. CHANNEL COUNT.  The axis gives the highest mean of Table 2, while the paper
   argues for one channel.  The block prints it level by level against the
   reference-cell controls, which use one channel.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

EDGE_MS = 80.0


def load(root):
    rp = os.path.join(root, "results.jsonl")
    files = [rp] if os.path.exists(rp) else sorted(
        glob.glob(os.path.join(root, "results_shard*.jsonl")))
    return [json.loads(l) for f in files for l in open(f)]


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def block_depth(R):
    if not any("sep_direct_primary_ms" in r for r in R):
        return
    print("\n[1] what the model held, per nominal water depth")
    print("  %6s %5s %9s %8s %8s %8s  %s" % ("D_m", "n", "D_true", "vw_est",
                                          "sep_ms", "geom_ms", "median NCC"))
    by = defaultdict(list)
    for r in R:
        if r.get("ok") and r.get("D_m") is not None:
            by[r["D_m"]].append(r)
    for D in sorted(by, key=float):
        g = by[D]
        x = 125.0
        geo = (math.hypot(2 * float(D), x) - x) / 1500.0 * 1e3
        print("  %6s %5d %9.1f %8.1f %8.1f %8.1f  %.3f"
              % (D, len(g), med([r.get("D_true") for r in g]),
                 med([r.get("vw_est") for r in g]),
                 med([r.get("sep_direct_primary_ms") for r in g]), geo,
                 med([r.get("ncc_prim_true") for r in g])))
    print("  geom_ms is the nominal depth at 1500 m/s and 125 m.  Where D_true "
          "departs from D_m,\n  the row was modelled at D_true, and its m and "
          "its NCC belong to that depth.")


def block_snr(R):
    g = [r for r in R if r.get("ok") and r.get("snr_eff_db") not in (None, float("inf"))
         and isinstance(r.get("snr_eff_db"), (int, float)) and r["snr_eff_db"] < 1e9]
    if not g:
        return
    print("\n[2] effective against nominal SNR")
    by = defaultdict(list)
    for r in g:
        by[r["snr_db"]].append(r)
    for s in sorted(by, key=float):
        rr = by[s]
        off = med([r["snr_eff_db"] - r["snr_db"] for r in rr])
        nsh = med([r.get("n_win_prim", float("nan")) / max(1, r.get("n_ch_used", 1))
                   for r in rr])
        stk = 10 * math.log10(nsh) if nsh == nsh and nsh > 0 else float("nan")
        print("  nominal %+6.1f dB  offset %+5.1f dB  = window %+5.1f dB + stack "
              "%+5.1f dB (%.0f shots)" % (float(s), off, off - stk, stk, nsh))
    print("  the shot count is n_win_prim / n_ch_used, i.e. the windows actually "
          "stacked;\n  the code uses the number of shots in the gather, so a few "
          "rejected windows\n  would show here as a small difference")


def nominal(r):
    # the domain is defined on the NOMINAL depth, as in the paper.  D_true is
    # what the 3 m grid holds (219 m in the 220 m cell), and filtering on it
    # would drop that whole cell
    return r.get("D_m", r.get("D_true", 0))


def block_edge(R, dmin=220.0):
    dom = [r for r in R if r.get("ok") and nominal(r) >= dmin
           and not str(r.get("run_id", "")).startswith("ref_")]
    g = [r for r in dom if r.get("dt_matched_ms") is not None]
    bad = [r for r in dom if r.get("dt_matched_ms") is None]
    if not g:
        return
    print("\n[3a] multiple branch, applicable domain: %d records, result in %d, "
          "none in %d" % (len(dom), len(g), len(bad)))
    if bad:
        def summ(rr, lab):
            # an absent field is not a zero: a file that does not record
            # padding would otherwise read as 0% padded
            have = [r for r in rr if r.get("frac_padded") is not None]
            if not have:
                print("  %-22s n=%4d  frac_padded NOT RECORDED in this file"
                      % (lab, len(rr)))
                return
            pad = np.mean([r["frac_padded"] > 0 for r in have])
            oml = med([r.get("origin_minus_lead_ms") for r in rr])
            print("  %-22s n=%4d  primary padded %3.0f%%  primary origin past "
                  "lead, median %+.0f ms" % (lab, len(rr), 100 * pad, oml))
        summ(g, "with a multiple")
        summ(bad, "without")
        print("  if the failures are the padded ones, the multiple failed because "
              "the\n  primary origin had already locked onto a later event and "
              "pushed the\n  multiple window off the record")
    q = np.array([r["dt_matched_ms"] + r["origin_minus_lead_ms"] for r in g])
    edge = np.abs(q) >= EDGE_MS - 2.5
    sg = [r.get("validator_sign") for r in g]
    print("\n[3b] matched filter on the multiple, %d records" % len(g))
    print("  multiple offset from its predicted arrival: median %+.1f ms, "
          "|.| p95 %.1f ms, max %.1f ms" % (np.median(q), np.percentile(np.abs(q), 95),
                                           np.abs(q).max()))
    print("  at the +-%.0f ms edge: %d of %d (%.1f%%)"
          % (EDGE_MS, edge.sum(), len(g), 100 * edge.mean()))
    print("  sign of the selected correlation: negative %d, positive %d"
          % (sum(1 for s in sg if s == -1), sum(1 for s in sg if s == 1)))
    # NOT a depth error.  q cannot be converted into a depth: in
    # run_matrix_v2.py the template is the TRIMMED primary segment, whose start
    # is not the primary origin, so q carries that trim offset as well.  In
    # run_real.py the template is the primary wavelet itself and the depth is
    # computed from the difference of the two origins.  The two runners do not
    # measure the same thing here, and only the edge count above is common.


def block_channels(R):
    g = [r for r in R if r.get("ok") and "n_ch" in str(r.get("axis", ""))]
    if not g:
        return
    ctl = [r["ncc_prim_true"] for r in R if r.get("ok")
           and str(r.get("run_id", "")).startswith("ref_") and r.get("ncc_prim_true") is not None]
    print("\n[4] channel-count axis against the one-channel controls")
    if ctl:
        print("  controls, 1 channel     n=%3d  mean %.3f  median %.3f"
              % (len(ctl), np.mean(ctl), np.median(ctl)))
    by = defaultdict(list)
    for r in g:
        by[r.get("n_ch_used")].append(r["ncc_prim_true"])
    for k in sorted(by, key=lambda v: float(v)):
        v = by[k]
        print("  %2s channels              n=%3d  mean %.3f  median %.3f"
              % (k, len(v), np.mean(v), np.median(v)))


def main():
    roots = sys.argv[1:] or ["wavelet_matrix_v2"]
    for root in roots:
        R = load(root)
        print("=" * 72)
        print("%s: %d records" % (root, len(R)))
        block_depth(R)
        block_snr(R)
        block_edge(R)
        block_channels(R)


if __name__ == "__main__":
    main()
