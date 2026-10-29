#!/usr/bin/env python3
"""
eval_picker_real.py -- the consensus picker on the four real lines.

On the synthetic models the consensus picker lands within the 80 ms the
extraction tolerates on 98 per cent of records, against 48 for the semblance
chain and 41 for the autocorrelation picker.  That is measured on models whose
failure modes were known while the picker was being written, so it is the
weakest kind of evidence there is.  The real lines are not: their seafloor was
picked months ago, by hand where it mattered, and the picker has never seen
them.

For each line this loads the cube and the track that were saved at the time,
runs pick_consensus on the same gathers, and reports the difference.  The
stored track is not ground truth -- it is a pick that was inspected and used --
so the question is not whether the two agree exactly but whether they agree
within the margin the extraction tolerates.

    python3 validation/archive/eval_picker_real.py \\
        viking:viking_pick/viking_cube.npy:viking_pick/viking_track.npz \\
        ms29:span_experiment/ms29_cube.npy:span_experiment/ms29_track.npz \\
        it065:span_experiment/it065_cube.npy:span_experiment/it065_track.npz \\
        mepa:span_experiment/mepa_cube.npy:span_experiment/mepa_track.npz
"""
import sys

import numpy as np

from seaweed import pick_consensus as PC


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    print("%-8s %6s %8s %9s %11s %9s %8s %7s"
          % ("line", "shots", "dt ms", "t0 stored", "t0 consensus", "diff ms",
             "|d|<80", "stretch"))
    for spec in args:
        name, cube_p, track_p = spec.split(":")
        try:
            g = np.load(cube_p, mmap_mode="r")
            z = np.load(track_p, allow_pickle=True)
        except Exception as e:
            print("%-8s  could not load: %s" % (name, str(e)[:60]))
            continue
        t0_stored = np.asarray(z["t0"], float)
        dt = float(z["dt"])
        offs = np.asarray(z["offs"], float)
        vw = float(z["vw"])
        # the cube can be large; the picker only needs a stretch of line, and
        # taking the middle avoids the ends where the tracking is least sure
        S = g.shape[0]
        n = min(60, S)
        a = max(0, S // 2 - n // 2)
        gg = np.asarray(g[a:a + n]).astype(float)
        try:
            t0s, rep = PC.pick_t0(gg, dt, offs, vw)
        except Exception as e:
            print("%-8s  picker failed: %s" % (name, str(e)[:60]))
            continue
        d = (np.median(t0s) - np.median(t0_stored[a:a + n])) * 1e3
        best = [q for q in rep["candidates"] if abs(q["t0"] - rep["t0"]) < 1e-9][0]
        print("%-8s %6d %8.1f %9.3f %11.3f %+9.0f %8s %7.2f"
              % (name, n, dt * 1e3, np.median(t0_stored[a:a + n]),
                 np.median(t0s), d, "yes" if abs(d) < 80 else "NO",
                 best["stretch"]))
        print("         semblance %.3f | multiple %.3f | polarity %+.2f | "
              "%d candidates, runner-up %.0f ms away"
              % (best["semblance"], best["semblance_multiple"],
                 best["multiple_polarity"], len(rep["candidates"]),
                 rep["agree_ms"]))


if __name__ == "__main__":
    main()
