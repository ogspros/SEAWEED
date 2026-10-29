#!/usr/bin/env python3
"""
viking_thin_layer.py -- can one sub-bottom reflector account for the notch near
18 Hz in the Viking extraction, and for the shape difference as well?

What is to be explained.  The extracted wavelet on Viking has a minimum near
17-18 Hz that the recorded signature does not, its first positive lobe is about
twice the reference's relative to the trough, and its second is smaller.  We
have shown the lobe imbalance is not the receiver-ghost angle, and that a
constant phase rotation of about 70 degrees fits it; a rotation is a description
and not a cause.  A reflection from a few tens of metres below the seafloor
would be a cause, and it predicts the notch and the lobes together.

The model is two reflections carrying the same wavelet:

    s(t) = w_ref(t) + a w_ref(t - dt)

so that |S(f)|^2 = |W(f)|^2 [1 + a^2 + 2 a cos(2 pi f dt)], with minima at
(k + 1/2)/dt for a > 0 and at k/dt for a < 0.  A notch at 18 Hz therefore wants
dt about 28 ms with the same polarity, or about 56 ms with the opposite one,
which at 1800 m/s is 25 m or 50 m below the seafloor.

The test is whether ONE pair (dt, a) explains the notch AND the waveform at the
same time.  A fit that matches the spectrum while leaving the lobes wrong would
say the notch is coincidence.  So both are reported: the agreement of the
waveform, and where the predicted minima fall against the observed ones.  The
comparison is made in the band and with the same shift limit the paper uses, so
that the number is on the same footing as the 0.827 quoted there.

    python3 validation/viking/viking_thin_layer.py FarField.dat viking_blocks --fam C
"""
import argparse
import glob
import os

import numpy as np

from viking_phase_test import PAT, bp, far_field, reference, ncc


def add_reflector(w, dt, delay_s, a):
    """w + a w(t - delay), delay by Fourier shift so it is not tied to a sample."""
    n = w.size
    nf = 1 << int(np.ceil(np.log2(4 * n)))
    W = np.fft.rfft(w, nf)
    f = np.fft.rfftfreq(nf, dt)
    d = np.fft.irfft(W * np.exp(-2j * np.pi * f * delay_s), nf)[:n]
    return w + a * d


def spectrum(w, dt, nf=4096):
    f = np.fft.rfftfreq(nf, dt)
    return f, np.abs(np.fft.rfft(w, nf))


def lobes(w, dt):
    """First positive lobe and second positive lobe, relative to the main
    trough.  The notch and the lobe imbalance are two different symptoms and a
    reflector that explains one need not explain the other."""
    i = int(np.argmin(w))
    T = -w[i]
    if T <= 0:
        return float("nan"), float("nan")
    k = max(1, int(round(0.020 / dt)))
    pre = w[max(0, i - k):i]
    post = w[i + 1:i + 1 + 2 * k]
    return (pre.max() / T if pre.size else float("nan"),
            post.max() / T if post.size else float("nan"))


def minima(f, A, lo, hi):
    m = (f >= lo) & (f <= hi)
    ff, AA = f[m], A[m]
    k = np.flatnonzero((AA[1:-1] < AA[:-2]) & (AA[1:-1] < AA[2:])) + 1
    return ff[k], AA[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("wavelets", nargs="+")
    ap.add_argument("--fam", default="C")
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--vsed", type=float, default=1800.0)
    a = ap.parse_args()

    lo, hi = a.band
    cfg = dict(dt=a.dt, dt_ff=1e-3, up=4, nw=50, lp_hz=115.0, lo=lo, hi=hi,
               maxlag_ms=a.maxlag_ms)
    ff_g = far_field(a.farfield, 27, 10.0, 1448.0, 1e-3)

    paths = []
    for p in a.wavelets:
        paths.extend(sorted(glob.glob(os.path.join(p, "*_wavelet.npy")))
                     if os.path.isdir(p) else [p])
    W = []
    for p in paths:
        m = PAT.match(os.path.basename(p))
        if m and m.group("fam") == a.fam:
            w = np.load(p).astype(float)
            if w.size == cfg["nw"]:
                W.append(bp(w, a.dt, lo, hi))
    if not W:
        raise SystemExit("no wavelets selected")
    obs = np.mean([w / np.linalg.norm(w) for w in W], axis=0)

    # the reference at the lag the paper uses, so the two are on one footing
    best = max((ncc(obs, reference(ff_g, L, cfg)), L)
               for L in range(-a.maxlag_ms, a.maxlag_ms + 1))
    ref = reference(ff_g, best[1], cfg)
    print("%d combs | reference at %+d ms | agreement as it stands: %.4f"
          % (len(W), best[1], best[0]))

    f, Ao = spectrum(obs, a.dt)
    fo, _ = minima(f, Ao, 10.0, 30.0)
    fr, _ = minima(f, np.abs(np.fft.rfft(ref, 4096)), 10.0, 30.0)
    print("minima between 10 and 30 Hz: extraction %s | reference %s"
          % (", ".join("%.1f" % x for x in fo) or "none",
             ", ".join("%.1f" % x for x in fr) or "none"))

    print("\n%8s %7s %9s %9s %11s %s"
          % ("dt (ms)", "a", "depth (m)", "agreement", "gain", "first minima (Hz)"))
    rows = []
    for delay_ms in np.arange(10.0, 70.1, 2.0):
        for amp in np.arange(-0.9, 0.91, 0.05):
            if abs(amp) < 0.05:
                continue
            syn = bp(add_reflector(ref, a.dt, delay_ms * 1e-3, amp), a.dt, lo, hi)
            v = max(ncc(obs, np.roll(syn, L)) for L in (-1, 0, 1))
            rows.append((v, delay_ms, amp))
    rows.sort(reverse=True)
    for v, delay_ms, amp in rows[:5]:
        syn = bp(add_reflector(ref, a.dt, delay_ms * 1e-3, amp), a.dt, lo, hi)
        fm, _ = minima(*spectrum(syn, a.dt), 10.0, 30.0)
        h = a.vsed * delay_ms * 1e-3 / 2.0
        print("%8.0f %7.2f %9.0f %9.4f %+11.4f %s"
              % (delay_ms, amp, h, v, v - best[0],
                 ", ".join("%.1f" % x for x in fm) or "none"))

    # Can ANY (dt, a) put a minimum where the extraction has one?  The
    # waveform fit and the notch are scored separately on purpose: a delay that
    # improves the shape while leaving the notch where the reference already
    # had it has not explained the notch.
    target = fo[0] if len(fo) else float("nan")
    if target == target:
        best_n = None
        for delay_ms in np.arange(10.0, 70.1, 1.0):
            for amp in np.arange(-0.9, 0.91, 0.05):
                if abs(amp) < 0.05:
                    continue
                syn = bp(add_reflector(ref, a.dt, delay_ms * 1e-3, amp),
                         a.dt, lo, hi)
                fm, _ = minima(*spectrum(syn, a.dt), 10.0, 30.0)
                if not len(fm):
                    continue
                d = abs(fm[0] - target)
                if best_n is None or d < best_n[0]:
                    best_n = (d, delay_ms, amp, fm[0],
                              max(ncc(obs, np.roll(syn, L)) for L in (-1, 0, 1)))
        if best_n is not None:
            print("\nclosest any reflector gets to the observed minimum at "
                  "%.1f Hz: %.1f Hz\n  at dt = %.0f ms, a = %+.2f, and that fit "
                  "agrees with the extraction at %.4f"
                  % (target, best_n[3], best_n[1], best_n[2], best_n[4]))

    # the lobes, for the best waveform fit and for the two ends of the question
    v, dl, am = rows[0]
    syn = bp(add_reflector(ref, a.dt, dl * 1e-3, am), a.dt, lo, hi)
    print("\nlobes relative to the main trough (first positive, second "
          "positive)")
    for lab, w in (("extraction", obs), ("reference", ref),
                   ("reference + reflector", syn)):
        p1, p2 = lobes(w, a.dt)
        print("  %-24s %5.2f %5.2f" % (lab, p1, p2))

    print("\n  'gain' is the change in agreement with the recorded signature "
          "once the\n  reflector is added to it.  A positive gain of a few "
          "hundredths, at a delay whose\n  predicted minima land on the "
          "observed ones, is the case where one reflector\n  explains both the "
          "notch and the shape.  A fit that gains nothing, or that needs\n  an "
          "amplitude near one, is not an explanation: the second reflection "
          "would then\n  be as strong as the seafloor.")
    print("  Depths assume %.0f m/s in the sediment and normal incidence."
          % a.vsed)


if __name__ == "__main__":
    main()
