#!/usr/bin/env python3
"""
track_from_consensus.py -- write a track file from the consensus picker, so the
extraction can be run twice on the same line and the two compared.

The consensus picker lands within the tolerated margin on all four archive
lines, but that is a statement about the SEAFLOOR TIME.  Whether the extracted
wavelet is the same is a separate question, and on Viking it can be answered
against the recorded signature rather than argued.  This writes a track in the
format run_real.py loads, with t0 from the picker and everything else copied
from the existing track, so the only thing that differs between the two runs is
the seafloor.

    python3 validation/archive/track_from_consensus.py viking_pick/viking_cube.npy \\
        viking_pick/viking_track.npz viking_pick/viking_track_consensus.npz

The picker returns a time at the envelope-peak convention, some tens of
milliseconds later than the stored picks; --shift_ms applies a constant
correction if the two conventions are to be matched, and defaults to none so
that what is compared is the picker as it stands.
"""
import argparse

import numpy as np

from seaweed import pick_consensus as PC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("track_in")
    ap.add_argument("track_out")
    ap.add_argument("--group", type=int, default=60,
                    help="shots per picking group: the picker returns one "
                         "candidate per group, which is what limits it on a "
                         "seafloor that changes along the line")
    ap.add_argument("--shift_ms", type=float, default=0.0)
    ap.add_argument("--tol_ms", type=float, default=100.0,
                    help="how far a group may sit from the line reference")
    a = ap.parse_args()

    g = np.load(a.cube, mmap_mode="r")
    z = np.load(a.track_in, allow_pickle=True)
    dt = float(z["dt"]); offs = np.asarray(z["offs"], float); vw = float(z["vw"])
    t0, rep = PC.pick_line(np.asarray(g, float), dt, offs, vw, group=a.group,
                           tol_ms=a.tol_ms)
    t0 = t0 + a.shift_ms * 1e-3
    reports = rep["groups"]
    n_ref = sum(1 for q in reports if q["source"] == "reference")
    n_2nd = sum(1 for q in reports if q["source"] == "second choice")
    print("reference %.3f s | %d groups: %d took their own best, %d a second "
          "choice, %d the reference"
          % (rep["reference"], len(reports), len(reports) - n_ref - n_2nd,
             n_2nd, n_ref))

    old = np.asarray(z["t0"], float)
    d = (t0 - old) * 1e3
    print("%d shots in %d groups | consensus minus stored: median %+.0f ms, "
          "p10 %+.0f, p90 %+.0f, max |.| %.0f"
          % (t0.size, len(reports), np.median(d), np.percentile(d, 10),
             np.percentile(d, 90), np.abs(d).max()))
    print("group picks:", ", ".join("%.3f" % q["t0"] for q in reports))
    unc = np.array([q["unconstrained"] for q in reports], float)
    if np.isfinite(unc).any():
        print("without the continuity constraint the groups would have spanned "
              "%.3f to %.3f s" % (np.nanmin(unc), np.nanmax(unc)))

    out = {k: z[k] for k in z.files}
    out["t0"] = t0
    np.savez(a.track_out, **out)
    print("wrote %s" % a.track_out)


if __name__ == "__main__":
    main()
