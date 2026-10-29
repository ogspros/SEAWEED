#!/usr/bin/env python3
"""
picker_tolerance.py -- the picker scored against the margin the extraction
actually has, which is not symmetric.

The picker was reported as landing within +-80 ms of the true seafloor.  That
is the wrong window.  The bias sweep of the v5 matrix shows the extraction
unaffected from -100 to +50 ms and undefined at +100: tracking early is safe,
because the window leads the arrival by 80 ms and still contains the
reflection, while tracking late eats that lead.  A pick at +70 ms passes a
+-80 ms test and fails the extraction, so the symmetric figure overstates the
result.

This recomputes the share inside several windows, from the files eval_picker.py
already wrote, and reports the asymmetric one alongside the symmetric one so
that the difference between them can be read rather than argued.

    python3 validation/synthetic/picker_tolerance.py                     # main grid
    python3 validation/synthetic/picker_tolerance.py --prefix eval_sec   # held-out secondary axes
"""
import argparse
import glob
import json

import numpy as np

WINDOWS = [("symmetric +-25 ms", -25.0, 25.0),
           ("symmetric +-80 ms", -80.0, 80.0),
           ("as the extraction tolerates, -100 to +50 ms", -100.0, 50.0),
           ("conservative, -80 to +40 ms", -80.0, 40.0)]
PICKERS = [("semblance", "semblance_err_ms"),
           ("autocorrelation", "autocorr_err_ms"),
           ("consensus", "consensus_err_ms")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="eval_picker")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="convention offset already subtracted inside the "
                         "picker, in ms; the stored errors were measured "
                         "before it was introduced, so pass 9 to score the "
                         "picker as it now stands")
    a = ap.parse_args()

    R = [json.loads(l) for f in sorted(glob.glob("%s_*.jsonl" % a.prefix))
         for l in open(f)]
    if not R:
        raise SystemExit("no %s_*.jsonl found" % a.prefix)
    print("%s: %d records%s\n"
          % (a.prefix, len(R),
             "" if not a.offset else
             ", errors shifted by %+.0f ms for the convention offset" % -a.offset))

    print("%-14s %5s %s" % ("picker", "n",
                            "  ".join("%26s" % w[0] for w in WINDOWS)))
    for lab, key in PICKERS:
        e = np.array([r[key] for r in R if r.get(key) is not None], float)
        if not e.size:
            print("%-14s   none" % lab); continue
        e = e - a.offset
        cells = []
        for _, lo, hi in WINDOWS:
            cells.append("%25.0f%%" % (100 * np.mean((e >= lo) & (e <= hi))))
        print("%-14s %5d %s" % (lab, e.size, "  ".join(cells)))

    print("\nwhere the two windows disagree, per picker:")
    for lab, key in PICKERS:
        e = np.array([r[key] for r in R if r.get(key) is not None], float) - a.offset
        if not e.size:
            continue
        sym = (e >= -80) & (e <= 80)
        asym = (e >= -100) & (e <= 50)
        print("  %-14s inside +-80 but not tolerated: %3d (%.0f%%); tolerated "
              "but outside +-80: %3d"
              % (lab, int((sym & ~asym).sum()), 100 * (sym & ~asym).mean(),
                 int((asym & ~sym).sum())))
    print("\n  The first column of that line is what the symmetric figure adds"
          "\n  and the extraction does not get.")


if __name__ == "__main__":
    main()
