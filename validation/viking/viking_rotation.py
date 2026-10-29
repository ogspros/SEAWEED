#!/usr/bin/env python3
"""
viking_rotation.py -- is the lobe imbalance of the Viking extraction a constant
phase rotation?

viking_lobes.py showed that the first positive lobe of the extraction is about
twice that of the recorded signature on every channel (0.88-0.92 of the trough
against 0.47), that the second is smaller (0.36-0.55 against 0.64), and that a
receiver ghost at the true angle makes the agreement worse, not better.  The
ghost geometry is therefore not the cause.

A precursor lobe that grows while the follower shrinks is what a phase rotation
does to a wavelet.  A constant rotation is also what two plausible causes would
produce to first order: a difference between the recording filters of the
streamer and of the far-field measurement, and a seafloor whose reflection
coefficient has a phase, as a gradational or thinly layered top does.  And it
would fit the phase test, where the minimum-phase equivalent beat the
extracted wavelet by 0.030.

Test: rotate every comb wavelet by phi (w cos(phi) - H[w] sin(phi)), score it
against the vertical reference exactly as viking_phase_test.py does, and find
the phi that maximises the agreement.

Reading it
  one phi, the same on every channel, that lifts the agreement well above its
      value at phi = 0 -> a constant rotation between data and reference
      explains the main lobes; the coda is a separate matter
  best phi near 0, or no real gain -> not a rotation

    python3 validation/viking/viking_rotation.py FarField.dat viking_blocks viking_ch1 viking_ch2 \\
        viking_ch4 viking_ch8
"""
import argparse
import glob
import os

import numpy as np
from scipy.signal import hilbert

from viking_phase_test import PAT, bp, far_field, reference, ncc


def rotate(w, phi_deg):
    p = np.radians(phi_deg)
    return np.cos(p) * w - np.sin(p) * np.imag(hilbert(w))


def best(ex, ff_g, cfg):
    return max((ncc(ex, reference(ff_g, L, cfg)), L)
               for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--fam", default="C")
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--step", type=float, default=5.0)
    a = ap.parse_args()

    cfg = dict(dt=4e-3, dt_ff=1e-3, up=4, nw=50, lp_hz=115.0,
               lo=a.band[0], hi=a.band[1], maxlag_ms=a.maxlag_ms)
    ff_g = far_field(a.farfield, 27, 10.0, 1448.0, 1e-3)
    # the full circle: the first run scanned +-90 and its optimum sat against
    # the +90 edge, so the range must not be what decides the answer
    phis = np.arange(-180.0, 180.0 - 1e-9, a.step)

    print("band %g-%g Hz, shift limit +-%d ms, phi step %g deg"
          % (a.band[0], a.band[1], a.maxlag_ms, a.step))
    print("%-16s %3s %9s %9s %8s %22s"
          % ("dir", "n", "phi = 0", "best", "phi", "phi range within 0.005"))
    for d in a.dirs:
        ws = []
        for p in sorted(glob.glob(os.path.join(d, "*_wavelet.npy"))):
            m = PAT.match(os.path.basename(p))
            if m and m.group("fam") == a.fam:
                w = np.load(p).astype(float)
                if w.size == cfg["nw"]:
                    ws.append(bp(w, cfg["dt"], *a.band))
        if not ws:
            print("%-16s no %s wavelets" % (d, a.fam))
            continue
        # the rotation is applied to each comb and the scores averaged, so the
        # number at phi = 0 is the one viking_phase_test reports
        S = np.array([[best(rotate(w, f), ff_g, cfg)[0] for w in ws]
                      for f in phis]).mean(1)
        k0 = int(np.argmin(np.abs(phis)))
        kb = int(np.argmax(S))
        near = phis[S >= S[kb] - 0.005]
        print("%-16s %3d %9.4f %9.4f %+8.0f %12.0f to %+.0f"
              % (d[:16], len(ws), S[k0], S[kb], phis[kb], near.min(), near.max()))


if __name__ == "__main__":
    main()
