#!/usr/bin/env python3
"""
viking_zones.py -- the same line, where the sub-bottom layer is there and where
it is not.

A deconvolved display of the Viking near channel shows a reflector about 50 ms
below the seafloor, strong at low shot numbers and closing out near SP 800.  If
that reflector is what puts the minimum near 18 Hz into the extracted wavelet,
then a wavelet extracted where the layer is absent should not have it, and one
extracted where the layer is strong should.  This is the test the modelling of
viking_thin_layer.py could not settle: there the delay and the amplitude were
free, here the geology does the separating and nothing is fitted.

It compares two wavelets already extracted over different shot ranges, on their
spectra and on their lobes, and against the recorded signature.

    python3 validation/viking/viking_zones.py FarField.dat \\
        layer:viking_zones/A_wavelet.npy  clear:viking_zones/B_wavelet.npy

Read it as follows.  A minimum near 18 Hz in the first and not in the second is
the layer.  The same minimum in both is not: it would then belong to the source,
to the receiver array or to the recording, and the wavelet would be right to
carry it.  Note that the two stretches also differ in everything else the
geology does, so a difference here is evidence and not proof; what would make it
proof is the minimum moving with the delay of the reflector, which requires the
layer to change depth along the line.
"""
import argparse
import os

import numpy as np

from viking_phase_test import bp, far_field, reference, ncc


def taper_to(w, dt, tmax_ms):
    if tmax_ms is None:
        return w
    n = w.size
    t = np.arange(n) * dt * 1e3
    k = np.ones(n)
    ramp = (t > tmax_ms - 10.0) & (t <= tmax_ms)
    k[ramp] = 0.5 * (1 + np.cos(np.pi * (t[ramp] - (tmax_ms - 10.0)) / 10.0))
    k[t > tmax_ms] = 0.0
    return w * k


def minima(f, A, lo, hi):
    m = (f >= lo) & (f <= hi)
    ff, AA = f[m], A[m]
    k = np.flatnonzero((AA[1:-1] < AA[:-2]) & (AA[1:-1] < AA[2:])) + 1
    return [(ff[i], AA[i] / AA.max()) for i in k]


def lobes(w, dt):
    i = int(np.argmin(w))
    T = -w[i]
    if T <= 0:
        return float("nan"), float("nan")
    k = max(1, int(round(0.020 / dt)))
    pre, post = w[max(0, i - k):i], w[i + 1:i + 1 + 2 * k]
    return (pre.max() / T if pre.size else float("nan"),
            post.max() / T if post.size else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("items", nargs="+", help="label:path_to_wavelet.npy")
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--tmax_ms", type=float, default=None,
                    help="compare only the first tmax ms of the wavelet, with "
                         "a 10 ms cosine taper, applied to the extraction and "
                         "to the reference alike.  An event entering the late "
                         "part of the window lowers the agreement there and "
                         "not in the first half, so the two readings together "
                         "say where along the wavelet the difference sits")
    a = ap.parse_args()

    lo, hi = a.band
    cfg = dict(dt=a.dt, dt_ff=1e-3, up=4, nw=50, lp_hz=115.0, lo=lo, hi=hi,
               maxlag_ms=a.maxlag_ms)
    ff_g = far_field(a.farfield, 27, 10.0, 1448.0, 1e-3)
    lags = []

    print("%-10s %9s %7s %7s   %s"
          % ("zone", "vs signature", "lobe 1", "lobe 2", "minima 10-30 Hz "
             "(Hz, depth relative to the peak)"))
    for it in a.items:
        lab, path = it.split(":", 1)
        if not os.path.exists(path):
            print("%-10s  missing: %s" % (lab, path)); continue
        w = taper_to(bp(np.load(path).astype(float), a.dt, lo, hi), a.dt,
                     a.tmax_ms)
        best = max((ncc(w, taper_to(reference(ff_g, L, cfg), a.dt, a.tmax_ms)), L)
                   for L in range(-a.maxlag_ms, a.maxlag_ms + 1))
        lags.append(best[1])
        f = np.fft.rfftfreq(4096, a.dt)
        A = np.abs(np.fft.rfft(w, 4096))
        mm = minima(f, A, 10.0, 30.0)
        p1, p2 = lobes(w, a.dt)
        print("%-10s %9.4f %7.2f %7.2f   %s"
              % (lab, best[0], p1, p2,
                 ", ".join("%.1f (%.2f)" % m for m in mm) or "none"))
    if lags:
        # The signature is drawn at the median lag of all the zones.  It used
        # to take the lag of the first zone listed, so the row changed with
        # the order of the arguments.
        lag_ref = float(np.median(lags))
        ref = reference(ff_g, lag_ref, cfg)
        f = np.fft.rfftfreq(4096, a.dt)
        A = np.abs(np.fft.rfft(ref, 4096))
        p1, p2 = lobes(ref, a.dt)
        print("%-10s %9s %7.2f %7.2f   %s"
              % ("signature", "-", p1, p2,
                 ", ".join("%.1f (%.2f)" % m for m in minima(f, A, 10.0, 30.0))
                 or "none"))
        print("  (signature drawn at %+.0f ms, the median lag of the zones)"
              % lag_ref)
    print("\n  A 50 ms delay puts minima at 20 and 40 Hz if the layer has the "
          "opposite\n  polarity to the seafloor, and at 10, 30 and 50 Hz if it "
          "has the same one.\n  The observed minimum near 18 Hz is close to the "
          "first of those.")


if __name__ == "__main__":
    main()
