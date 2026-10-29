r"""
test_minphase.py — are the extracted wavelets minimum phase?

WHAT IT MEASURES, AND WHAT IT DOES NOT
--------------------------------------
This measures how far each extracted wavelet is from the minimum-phase wavelet
with the same amplitude spectrum.  It does not sort sources into minimum and
mixed phase, and it cannot: a free-surface ghost 1 - r z^-k has its zeros
at |z| = |r|^(1/k), inside the circle for any |r| < 1, and a bubble 1 + a z^-k
likewise for |a| < 1.  A bubble weaker than the primary and a ghost from a real
sea surface are therefore both minimum phase, and an airgun signature need not
be mixed phase at all.  What makes a marine signature depart from minimum phase
is a reflection coefficient at the surface indistinguishable from -1, which puts
the ghost zeros on the circle rather than inside it.

The quantity is still worth having.  The paper's independent spectral check
constrains the modulus only, so nothing in the workflow says anything about
phase; a wavelet can carry the right amplitude spectrum and the wrong phase and
go unnoticed.  A number that says how far each extraction sits from minimum
phase is the first phase statement the workflow can make, and comparing it
across an explosive source and two airgun configurations is informative whether
or not they separate.

TWO MEASURES, BOTH ON THE WAVELET ALONE
---------------------------------------
Given the amplitude spectrum, the minimum-phase wavelet is the one whose phase
is the Hilbert transform of the log spectrum.  Compare the extracted wavelet
with that reconstruction:

    correlation, after allowing a small shift, since the reconstruction sets
    its own time origin;

    energy build-up.  Among all wavelets sharing an amplitude spectrum, the
    minimum-phase one concentrates energy earliest: its partial energy sum is
    the largest at every sample.  The gap between the two curves is a measure
    that does not depend on aligning anything.

Neither needs a reference wavelet, so both work on archive data.

    python3 validation/viking/test_minphase.py w1.npy w2.npy ... --dt 0.004 --band 8 65
"""
import argparse
import os

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def minimum_phase(w, nf=8192, floor_db=-60.0):
    """The minimum-phase wavelet with the same amplitude spectrum as w.

    Its phase is the Hilbert transform of the log amplitude spectrum.  The
    spectrum is floored before the logarithm: a band-limited wavelet has
    near-zeros outside its band, and log(0) would dominate the transform with
    values that carry no information about the wavelet.
    """
    A = np.abs(np.fft.rfft(w, nf))
    A = np.maximum(A, A.max() * 10 ** (floor_db / 20.0))
    logA = np.log(A)
    # phase = -Im{Hilbert(log|A|)} on the full spectrum
    full = np.concatenate([logA, logA[-2:0:-1]])
    ph = -np.imag(hilbert(full))[:len(logA)]
    return np.fft.irfft(A * np.exp(1j * ph), nf)[:len(w)]


def ncc_shift(a, b, maxlag):
    best, bl = -2.0, 0
    for L in range(-maxlag, maxlag + 1):
        x = a[L:] if L > 0 else a
        y = b[:len(b) - L] if L > 0 else b[-L:]
        n = min(len(x), len(y))
        if n < 8:
            continue
        u = x[:n] - x[:n].mean()
        v = y[:n] - y[:n].mean()
        d = np.sqrt((u @ u) * (v @ v))
        if d > 0 and u @ v / d > best:
            best, bl = float(u @ v / d), L
    return best, bl


def buildup(w):
    e = np.cumsum(w ** 2)
    return e / (e[-1] + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavelets", nargs="+")
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--band", type=float, nargs=2, default=[8., 65.])
    ap.add_argument("--maxlag_ms", type=float, default=40.)
    a = ap.parse_args()

    ml = max(1, int(round(a.maxlag_ms * 1e-3 / a.dt)))
    print("%-32s %8s %10s %10s" % ("wavelet", "NCC", "buildup gap", "verdict"))
    for p in a.wavelets:
        if not os.path.exists(p):
            print("  %-30s missing" % os.path.basename(p))
            continue
        w = bp(np.load(p).astype(float), a.dt, *a.band)
        mp = bp(minimum_phase(w), a.dt, *a.band)
        v, lag = ncc_shift(w, mp, ml)
        # largest shortfall of the wavelet's energy build-up against the
        # minimum-phase one: zero means it concentrates energy as early as any
        # wavelet with this spectrum can, which is what minimum phase means
        gap = float(np.max(buildup(mp) - buildup(w)))
        # The verdict rests on the build-up gap.  The correlation is reported
        # because it answers the same question independently, but it is the
        # fragile one: on a wavelet that IS minimum phase it read 0.78, because
        # the reconstruction sets its own origin and its tail is longer than
        # the window it is compared over.
        if gap < 0.05:
            verdict = "at minimum phase"
        elif gap < 0.20:
            verdict = "near minimum phase"
        elif gap < 0.50:
            verdict = "departs"
        else:
            verdict = "far from minimum phase"
        print("%-32s %8.3f %10.3f   %s"
              % (os.path.basename(p).replace("_wavelet.npy", ""), v, gap,
                 verdict))
    print("""
The two measures answer the same question differently and should agree.  A high
correlation with the minimum-phase reconstruction and a small build-up gap mean
the wavelet is minimum phase; a low correlation with a large gap means energy
arrives later than the spectrum requires, which is the signature of zeros
outside the unit circle.

The thresholds are conventions.  What carries weight is the number itself and
the contrast between surveys, since neither the extraction nor these measures
were told what the source was.  A wavelet reading "at minimum phase" is not
proof that the source was: it means the extraction returned something whose
energy arrives as early as its own spectrum permits.""")


if __name__ == "__main__":
    main()
