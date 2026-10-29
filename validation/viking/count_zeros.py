r"""
count_zeros.py — where do the zeros of the extracted wavelet lie?

WHY ZEROS
---------
Comparing a wavelet with the minimum-phase reconstruction of its own amplitude
spectrum depends on an alignment, which can move energy from the front to the
tail, and on a build-up measure that can read the window lead rather than the
phase.  Counting the zeros of the z-transform is direct.  A wavelet is minimum phase
when every zero lies inside the unit circle, maximum phase when every zero lies
outside, and mixed otherwise.  Nothing is reconstructed and nothing is aligned,
so the failures above cannot recur.

WHAT MAKES IT DELICATE
----------------------
Two things, and both are reported rather than hidden.

Band limitation puts zeros ON the circle.  A wavelet that has been band-passed
has no energy outside its band, and the polynomial must vanish there: those
zeros sit at |z| = 1 and say nothing about the source.  They are counted
separately, and if they dominate, the verdict is withheld.

Root finding on a long polynomial is ill-conditioned.  The wavelet is trimmed to
its significant support first, and the self-test reports the accuracy achieved
on polynomials whose roots are known by construction.

    python3 validation/viking/count_zeros.py --selftest
    python3 validation/viking/count_zeros.py w1.npy w2.npy --dt 0.002 --band 8 65
"""
import argparse
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def trim(w, frac=1e-3):
    """Cut the wavelet to where it has energy.

    Leading and trailing samples below `frac` of the peak contribute zeros of
    no physical meaning and make the polynomial longer than it needs to be,
    which is where the root finding loses accuracy.
    """
    a = np.abs(w)
    k = np.where(a > frac * a.max())[0]
    return w[k[0]:k[-1] + 1] if k.size else w


def zeros_of(w):
    """Zeros of the z-transform, in z.

    W(z) = sum w[n] z^-n, so the polynomial in z has coefficients w in the
    order given: np.roots takes the highest power first, which for
    w[0] z^(N-1) + ... + w[N-1] is w itself.
    """
    w = np.asarray(w, float)
    w = w / (np.max(np.abs(w)) + 1e-30)
    return np.roots(w)


def classify(r, tol=0.05):
    m = np.abs(r)
    return dict(n=len(r),
                inside=int(np.sum(m < 1 - tol)),
                outside=int(np.sum(m > 1 + tol)),
                onedge=int(np.sum((m >= 1 - tol) & (m <= 1 + tol))))


def verdict(c):
    decided = c["inside"] + c["outside"]
    if decided < 0.4 * c["n"]:
        return "undecided: too many zeros on the circle"
    f = c["outside"] / max(decided, 1)
    if f < 0.05:
        return "minimum phase"
    if f > 0.95:
        return "maximum phase"
    return "mixed phase (%.0f%% of decided zeros outside)" % (100 * f)


def selftest():
    """Build wavelets from prescribed roots and check they are recognised.

    This is the step the previous attempt skipped: its control case was a
    wavelet assumed to be minimum phase which the root count later showed to be
    maximum phase, so the tool was validated against the wrong answer.
    """
    rng = np.random.default_rng(0)
    ok = True
    print("self-test: wavelets built from known roots\n")
    print("  %-26s %5s %7s %8s %7s   %s"
          % ("case", "n", "inside", "outside", "edge", "verdict"))
    for name, radii in (
            ("all inside (min phase)", rng.uniform(0.3, 0.9, 20)),
            ("all outside (max phase)", 1.0 / rng.uniform(0.3, 0.9, 20)),
            ("half and half (mixed)",
             np.r_[rng.uniform(0.3, 0.9, 10), 1.0 / rng.uniform(0.3, 0.9, 10)]),
    ):
        ang = rng.uniform(0, np.pi, len(radii))
        roots = np.r_[radii * np.exp(1j * ang), radii * np.exp(-1j * ang)]
        w = np.real(np.poly(roots))
        c = classify(zeros_of(w))
        v = verdict(c)
        expect = ("minimum" if "inside" in name else
                  "maximum" if "outside" in name else "mixed")
        good = v.startswith(expect)
        ok &= good
        print("  %-26s %5d %7d %8d %7d   %s%s"
              % (name, c["n"], c["inside"], c["outside"], c["onedge"], v,
                 "" if good else "   <-- WRONG"))
    print("\n  %s" % ("all three recognised" if ok else
                      "SELF-TEST FAILED: do not use on data"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavelets", nargs="*")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--band", type=float, nargs=2, default=None,
                    help="band-pass before counting.  Leave unset: filtering "
                         "adds zeros on the circle at the band edges, which "
                         "are the ones that make the count undecidable")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="how far from |z|=1 a zero must be to be counted")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest or not a.wavelets:
        selftest()
        if not a.wavelets:
            return
        print()
    elif not selftest():
        raise SystemExit("self-test failed")
    else:
        print()

    print("  %-30s %5s %7s %8s %7s   %s"
          % ("wavelet", "n", "inside", "outside", "edge", "verdict"))
    for p in a.wavelets:
        if not os.path.exists(p):
            print("  %-30s missing" % os.path.basename(p))
            continue
        w = np.load(p).astype(float)
        if a.band:
            w = bp(w, a.dt, *a.band)
        w = trim(w)
        c = classify(zeros_of(w), a.tol)
        print("  %-30s %5d %7d %8d %7d   %s"
              % (os.path.basename(p).replace("_wavelet.npy", ""),
                 c["n"], c["inside"], c["outside"], c["onedge"], verdict(c)))
    print("""
Zeros within %.2f of the unit circle are counted as being on it and excluded
from the verdict, because a band-limited wavelet must vanish outside its band
and those zeros belong to the band limitation rather than to the source.  Where
they dominate, no verdict is given: that is a statement about what the data can
support, not a failure of the measurement.""" % a.tol)


if __name__ == "__main__":
    main()
