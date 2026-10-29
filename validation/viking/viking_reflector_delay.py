#!/usr/bin/env python3
"""Viking: the delay of the sub-bottom reflector below the seafloor, in the
two halves of the line (Sect. 3.5).

The near traces are band-passed and aligned on their raw onset, which needs no
tracking: the envelope peak between 300 and 850 ms, walked back until the
envelope falls below max(0.08 peak, 4 x the RMS noise of 300-360 ms).  Per block
of 50 shots the aligned traces are stacked; the seafloor trough is the most
negative sample 0-50 ms after the onset, the reflector trough the most negative
sample 12-70 ms after it.  The delay of each half is the median over its blocks
(west: shots 0-499, east: 500 on), with the range.  The same is printed with the
traces aligned on a tracking, for comparison.

    python validation/viking/viking_reflector_delay.py CUBE TRACK

CUBE: viking_pick/viking_cube.npy; TRACK: a tracking file (t0, offs, vw).
"""
import sys

import numpy as np

from seaweed.run_real import BAND
from seaweed.wavelet_extractor_v2 import arrival_time, bandpass, envelope

DT = 4.0                                        # ms, the Viking cube
REL = np.arange(-10, 51)                        # samples around the alignment, -40..+200 ms


def onset(x, t_lo=300., t_hi=850., noise=(300., 360.), frac=0.08, k=4.0):
    a, b = int(t_lo / DT), int(t_hi / DT)
    env = envelope(x[a:b]); ipk = int(np.argmax(env))
    n = x[int(noise[0] / DT):int(noise[1] / DT)]
    thr = max(frac * env[ipk], k * float(np.sqrt(np.mean(n ** 2))))
    i = ipk
    while i > 0 and env[i] > thr:
        i -= 1
    return (a + i) * DT


def main():
    cube = np.load(sys.argv[1], mmap_mode="r"); S = cube.shape[0]
    z = np.load(sys.argv[2], allow_pickle=True)
    x0 = float(np.asarray(z["offs"])[0]); vw = float(z["vw"])
    tr = np.array([bandpass(np.asarray(cube[s][:, 0], float), DT * 1e-3, BAND) for s in range(S)])
    align = {"raw onset": np.array([onset(tr[s]) for s in range(S)]),
             "tracking": np.array([arrival_time(float(t), x0, vw, 0) for t in np.asarray(z["t0"])[:S]]) * 1e3}
    t = REL * DT
    print("| shots | " + " | ".join("%s: seafloor trough / delay / relative amplitude" % k for k in align) + " |")
    print("|---|" + "---|" * len(align))
    D = {k: [] for k in align}
    for b0 in range(0, S, 50):
        cells = []
        for lab, ref in align.items():
            k0 = np.round(ref / DT).astype(int)
            st = np.array([tr[s][k0[s] + REL] for s in range(b0, min(S, b0 + 50))]).mean(0)
            m = (t >= 0) & (t <= 50); i_sf = int(np.flatnonzero(m)[np.argmin(st[m])])
            m2 = (t >= t[i_sf] + 12) & (t <= t[i_sf] + 70); i_r = int(np.flatnonzero(m2)[np.argmin(st[m2])])
            D[lab].append((b0, t[i_r] - t[i_sf]))
            cells.append("%+.0f / %.0f / %.2f" % (t[i_sf], t[i_r] - t[i_sf], st[i_r] / st[i_sf]))
        print("| %d-%d | %s |" % (b0, min(S, b0 + 50) - 1, " | ".join(cells)))
    for lab in align:
        w = [d for b, d in D[lab] if b < 500]; e = [d for b, d in D[lab] if b >= 500]
        print("\n%s: west median %.0f ms (%.0f-%.0f), east median %.0f ms (%.0f-%.0f)"
              % (lab, np.median(w), min(w), max(w), np.median(e), min(e), max(e)))


if __name__ == "__main__":
    main()
