#!/usr/bin/env python3
"""The three criteria of the autocorrelation test of the operating domain,
fixed on 26 October 2026 before its run (docs/experiments/operating_domain.md,
rule_overrides.md):

  1. the 4 off-cycle edge records of the long-direct class (D140_L150_r019,
     D180_L150_r000, D220_L300_r009, D220_L80_r000, long direct) are "outside"
  2. among the records with true separation >= 2 max(5 T_dom, end of the
     direct), at most 5 % flagged "outside" because of the fourth separation
     (decided_by = water-layer autocorrelation)
  3. none of the four real lines "outside"

    python validation/synthetic/domain_criteria.py RECORDS PICKER LONG [LINES]

RECORDS: picker_records.py; PICKER: picker_test.py run; LONG: long_direct.py
run; LINES: validation/real/lines.sh (criterion 3, optional).

First, the classes behind the 5 T_dom limit: the 216 records by true
direct-seafloor separation at the near trace, in multiples of the true T_dom,
on the edges 0, 2, 3, 4, 5, 6, 8 T_dom, every class with fewer than ten records
merged into the one above (the top one into the one below); per class, the
share of the onset anchors of the complete method within T_dom/2 of the true
seafloor (stage-1 anchors where the onset tracking is not applicable).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT]
from seaweed.wavelet_extractor_v2 import arrival_time                  # noqa: E402

EDGES_T = [0, 2, 3, 4, 5, 6, 8, np.inf]
MIN_RECORDS = 10
FOUR = ("grid_D140_L150_r019", "grid_D180_L150_r000", "grid_D220_L300_r009", "grid_D220_L80_r000")


def truth(GEN, uid):
    tr = np.load(os.path.join(GEN, uid + "_truth.npz"), allow_pickle=True)
    m = np.load(os.path.join(GEN, uid + "_cube.npy.meta.npz"), allow_pickle=True)
    x0 = float(np.asarray(m["off_hdr"])[0]); v = float(tr["vw_true"])
    return float(np.median(arrival_time(np.asarray(tr["t0_true"], float), x0, v) - x0 / v)) * 1e3, float(tr["tdom_true_ms"])


def classes(GEN, PK):
    rec = []
    for p in sorted(glob.glob(os.path.join(PK, "*", "p", "track.npz"))):
        uid = p.split(os.sep)[-3]
        sep, T = truth(GEN, uid)
        z = np.load(p, allow_pickle=True)
        t0t = np.asarray(np.load(os.path.join(GEN, uid + "_truth.npz"), allow_pickle=True)["t0_true"], float)
        if "anchors_i" in z.files:
            i, t = np.asarray(z["anchors_i"], int), np.asarray(z["anchors_t"], float)
        else:
            a = np.load(os.path.join(os.path.dirname(p), "stage1_anchors.npz"), allow_pickle=True)
            i, t = np.asarray(a["i"], int), np.asarray(a["t"], float)
        rec.append(dict(sepT=sep / T, ok=list(np.abs(t - t0t[i]) * 1e3 < T / 2)))
    e = list(EDGES_T)
    while True:
        cnt = [sum(1 for m in rec if e[k] <= m["sepT"] < e[k + 1]) for k in range(len(e) - 1)]
        small = [k for k, c in enumerate(cnt) if c < MIN_RECORDS]
        if not small or len(e) <= 2:
            break
        k = small[0]
        del e[k + 1 if k < len(e) - 2 else k]
    print("## Separation classes of the 5 T_dom limit (%d records)\n" % len(rec))
    print("| separation / T_dom | records | anchors on the correct cycle |\n|---|---|---|")
    for k in range(len(e) - 1):
        L = [m for m in rec if e[k] <= m["sepT"] < e[k + 1]]
        xs = [x for m in L for x in m["ok"]]
        print("| [%g, %s) | %d | %.3f |" % (e[k], "inf" if not np.isfinite(e[k + 1]) else "%g" % e[k + 1], len(L),
                                          np.mean(xs) if xs else np.nan))
    print()


def main():
    GEN, PK, LD = sys.argv[1:4]; LN = sys.argv[4] if len(sys.argv) > 4 else None; ok = True
    print("# Criteria of the autocorrelation test (`%s`, `%s`)\n" % (PK, LD))
    classes(GEN, PK)
    print("## 1. The four off-cycle edge records, long direct\n")
    print("| record | complete error (ms) | flag | decided by | active bound |\n|---|---|---|---|---|")
    c1 = True
    recs = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(LD, "records.jsonl"))}
    for u in FOUR:
        p = os.path.join(LD, u, "long", "summary.json")
        if not os.path.exists(p):
            print("| %s | pipeline failed | - | - | - |" % u); c1 = False; continue
        s = json.load(open(p))["domain"]
        out = s["flag"].startswith("outside")
        c1 &= out
        print("| %s | %+.0f | %s | %s | %s |" % (u, recs[u]["long"]["complete_err_ms"], s["flag"], s.get("decided_by"), s.get("active")))
    print("\n**criterion 1: %s**\n" % ("pass" if c1 else "FAIL")); ok &= c1
    print("## 2. Records with true separation >= 2 max(5 T_dom, end of the direct)\n")
    rows = []
    for p in glob.glob(os.path.join(PK, "*", "p", "summary.json")):
        rows.append(("picker", p.split("/")[-3], p))
    for p in glob.glob(os.path.join(LD, "*", "*", "summary.json")):
        rows.append(("longdirect " + p.split("/")[-2], p.split("/")[-3], p))
    el, flagged4, flagged_any = [], [], []
    for src, uid, p in rows:
        s = json.load(open(p))["domain"]
        sep, T = truth(GEN, uid)
        dms = s.get("direct_ms")
        bound = max(5 * T, dms if dms is not None and np.isfinite(dms) else 0.0)
        if sep >= 2 * bound:
            el.append((src, uid))
            if s["flag"].startswith("outside") and s.get("decided_by") == "water-layer autocorrelation":
                flagged4.append((src, uid, sep, bound, s.get("separation_autocorr_median_ms")))
            if s["flag"].startswith("outside"):
                flagged_any.append((src, uid, s.get("decided_by")))
    frac = len(flagged4) / max(len(el), 1)
    c2 = frac <= 0.05
    print("eligible %d (picker %d, long-direct class %d); outside because of the fourth separation %d (%.1f %%); outside for "
          "any reason %d\n" % (len(el), sum(s == "picker" for s, _ in el), sum(s != "picker" for s, _ in el), len(flagged4),
                                100 * frac, len(flagged_any)))
    for x in flagged4[:30]:
        print("- %s %s: true separation %.0f ms, bound %.0f ms, autocorrelation separation %s ms" % (
            x[0], x[1], x[2], x[3], "%.0f" % x[4] if x[4] is not None else "-"))
    print("\n**criterion 2: %s**\n" % ("pass" if c2 else "FAIL")); ok &= c2
    if LN is None:
        print("**criteria 1 and 2: %s**" % ("pass" if ok else "FAIL")); return
    print("## 3. The four real lines\n")
    c3 = True
    for L in ("viking", "it065", "mepa", "ms29"):
        p = os.path.join(LN, L, "summary.json")
        if not os.path.exists(p):
            print("- %s: missing" % L); c3 = False; continue
        s = json.load(open(p))["domain"]
        c3 &= not s["flag"].startswith("outside")
        print("- %s: %s (decided by %s; active bound %s; separation %s ms from the %s; autocorrelation %s ms)" % (
            L, s["flag"], s.get("decided_by"), s.get("active"), "%.0f" % s["separation_median_ms"], s.get("separation_active"),
            "%.0f" % s["separation_autocorr_median_ms"] if s.get("separation_autocorr_median_ms") == s.get("separation_autocorr_median_ms") else "-"))
    print("\n**criterion 3: %s**\n" % ("pass" if c3 else "FAIL")); ok &= c3
    print("**%s**" % ("All three criteria pass" if ok else "At least one criterion fails: stop"))


if __name__ == "__main__":
    main()
