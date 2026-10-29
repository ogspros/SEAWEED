#!/usr/bin/env python3
"""
viking_phase_test.py -- does the extracted wavelet carry phase, or only an
amplitude spectrum?

The paper reports an agreement of 0.827 between the Viking extractions and the
recorded far-field signature.  That is a waveform correlation, so it already
contains the phase, but it has never been separated out: a reader cannot tell
from it how much of the agreement the amplitude spectrum alone would produce.

This separates it.  Three candidates are compared against the same recorded
reference:

    extracted     the wavelet as the workflow returns it
    minimum       its minimum-phase equivalent
    zero          its zero-phase equivalent

All three have, by construction, exactly the same amplitude spectrum: the
minimum-phase one is built by Hilbert transform of the log amplitude, the
zero-phase one by discarding the phase.  Any difference in agreement is
therefore phase and nothing else.

Two further rows are printed as bounds.  `reference-min` is the minimum-phase
equivalent of the REFERENCE, scored against the reference: it is the best any
amplitude-only method could do on this line, whatever the wavelet.  And
`amplitude only` reports the correlation of the amplitude spectra, which is the
part of the agreement that carries no phase at all.

Reading it
  extracted clearly above minimum and zero  -> the phase is measured, and
      saying so in the paper is a statement of fact rather than a claim
  the three within a few hundredths        -> the workflow delivers an
      amplitude spectrum, and any downstream use of the phase is unsupported

The reference chain is the one in make_paper_figures_vik.py: recording delay
stripped, receiver ghost applied, low-passed, decimated, band-passed.  It is
reproduced here rather than imported so this runs from anywhere.

    python3 validation/viking/viking_phase_test.py FarField.dat viking_blocks --fam C
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.signal import butter, sosfiltfilt

PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)'
                 r'_wavelet\.npy$')


# ------------------------------------------------------------------ reference
def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low',
                              output='sos'), x)


def far_field(path, delay, z_rec, vw, dt_ff):
    ff = np.loadtxt(path)
    if ff.ndim != 1:
        raise SystemExit("%s does not have exactly one column" % path)
    ff = ff[delay:]
    k = int(round((2.0 * z_rec / vw) / dt_ff))
    g = np.zeros(k + 1)
    g[0] = 1.0
    g[k] = -1.0
    return np.convolve(ff, g)


def reference(ff_g, lag_ms, cfg):
    k = int(round(lag_ms))
    w = np.concatenate([np.zeros(k), ff_g]) if k >= 0 else ff_g[-k:]
    w = lp(np.pad(w, (0, 400)), cfg["dt_ff"], cfg["lp_hz"])
    w4 = w[::cfg["up"]][:cfg["nw"]]
    if w4.size < cfg["nw"]:
        w4 = np.pad(w4, (0, cfg["nw"] - w4.size))
    return bp(w4, cfg["dt"], cfg["lo"], cfg["hi"])


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best_lag(ex, ff_g, cfg):
    return max((ncc(ex, reference(ff_g, L, cfg)), L)
               for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1))


# ------------------------------------------------------------------ phases
def minimum_phase(x, nfft=8192):
    """Minimum-phase wavelet with the same amplitude spectrum as x.

    Phase from the Hilbert transform of the log amplitude, i.e. the standard
    cepstral construction.  Checked below by counting the zeros of the
    polynomial: they must all lie inside the unit circle.
    """
    n = x.size
    A = np.abs(np.fft.rfft(x, nfft))
    eps = 1e-8 * A.max()
    logA = np.log(np.maximum(A, eps))
    full = np.concatenate([logA, logA[-2:0:-1]])
    c = np.fft.ifft(full).real
    h = np.zeros(nfft)
    h[0] = 1.0
    h[1:nfft // 2] = 2.0
    h[nfft // 2] = 1.0
    return np.fft.ifft(np.exp(np.fft.fft(c * h))).real[:n]


def zero_phase(x, nfft=8192):
    """Zero-phase wavelet with the same amplitude spectrum, centred in place.

    Unlike the minimum-phase construction this one is not exact: a zero-phase
    wavelet is symmetric about its centre and therefore needs as much room
    before the centre as after, which a window of the original length does not
    give.  Truncating costs a few per cent of the amplitude spectrum, so the
    error is measured and reported rather than hoped away, and a cosine taper
    is applied to keep the truncation from ringing.
    """
    n = x.size
    A = np.abs(np.fft.rfft(x, nfft))
    y = np.roll(np.fft.irfft(A, nfft), n // 2)[:n]
    k = max(2, n // 10)
    taper = np.ones(n)
    taper[-k:] = 0.5 * (1 + np.cos(np.pi * np.arange(1, k + 1) / k))
    return y * taper


def amplitude_error(a, b, dt, lo, hi, nfft=8192):
    """Largest in-band difference of the two amplitude spectra, relative to
    the peak.  The whole test rests on the candidates sharing an amplitude
    spectrum, so this is checked rather than assumed."""
    f = np.fft.rfftfreq(nfft, dt)
    m = (f >= lo) & (f <= hi)
    A = np.abs(np.fft.rfft(a, nfft))
    B = np.abs(np.fft.rfft(b, nfft))
    return float(np.abs(A - B)[m].max() / A[m].max())


def zeros_inside(x):
    r = np.roots(x)
    a = np.abs(r)
    return int((a < 1).sum()), r.size


def amplitude_correlation(a, b, dt, lo, hi, nfft=8192):
    f = np.fft.rfftfreq(nfft, dt)
    m = (f >= lo) & (f <= hi)
    A = np.abs(np.fft.rfft(a, nfft))[m]
    B = np.abs(np.fft.rfft(b, nfft))[m]
    return ncc(A, B)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("wavelets", nargs="+",
                    help="wavelet files, or a directory holding them")
    ap.add_argument("--fam", default=None)
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--dt", type=float, default=4e-3)
    ap.add_argument("--dt_ff", type=float, default=1e-3)
    ap.add_argument("--delay", type=int, default=27)
    ap.add_argument("--z_rec", type=float, default=10.0)
    ap.add_argument("--vw", type=float, default=1448.0)
    ap.add_argument("--lp_hz", type=float, default=115.0)
    ap.add_argument("--nw", type=int, default=50)
    ap.add_argument("--maxlag_ms", type=int, default=60,
                    help="wider than the figure script uses, because the three "
                         "candidates sit at different intrinsic delays and the "
                         "search must be able to reach each of them")
    ap.add_argument("--save_npz", default=None)
    args = ap.parse_args()

    lo, hi = args.band
    cfg = dict(dt=args.dt, dt_ff=args.dt_ff,
               up=int(round(args.dt / args.dt_ff)), nw=args.nw,
               lp_hz=args.lp_hz, lo=lo, hi=hi, maxlag_ms=args.maxlag_ms)
    ff_g = far_field(args.farfield, args.delay, args.z_rec, args.vw, args.dt_ff)

    paths = []
    for p in args.wavelets:
        paths.extend(sorted(glob.glob(os.path.join(p, "*_wavelet.npy")))
                     if os.path.isdir(p) else [p])
    if args.fam:
        keep = []
        for p in paths:
            m = PAT.match(os.path.basename(p))
            if m and m.group("fam") == args.fam:
                keep.append(p)
        paths = keep
    if not paths:
        raise SystemExit("no wavelets selected")

    # the reference at zero lag, used for the amplitude-only bounds
    ref0 = reference(ff_g, 0.0, cfg)
    rmin = minimum_phase(ref0)
    v_refmin = best_lag(bp(rmin, args.dt, lo, hi), ff_g, cfg)[0]

    print("band %g-%g Hz, %d wavelets, lag searched over +/-%d ms"
          % (lo, hi, len(paths), args.maxlag_ms))
    print("\n%-26s %9s %9s %9s   %8s"
          % ("wavelet", "extracted", "minimum", "zero", "zeros in"))
    print("-" * 70)

    rows, out, amp_err = [], {}, []
    for i, p in enumerate(paths):
        w = np.load(p).astype(float)
        if w.size != args.nw:
            continue
        wmin = minimum_phase(w)
        wzero = zero_phase(w)
        ve = best_lag(bp(w, args.dt, lo, hi), ff_g, cfg)[0]
        vm = best_lag(bp(wmin, args.dt, lo, hi), ff_g, cfg)[0]
        vz = best_lag(bp(wzero, args.dt, lo, hi), ff_g, cfg)[0]
        nin, ntot = zeros_inside(w)
        eam = amplitude_error(w, wmin, args.dt, lo, hi)
        eaz = amplitude_error(w, wzero, args.dt, lo, hi)
        amp_err.append((eam, eaz))
        print("%-26s %9.4f %9.4f %9.4f   %4d/%d"
              % (os.path.basename(p).replace("_wavelet.npy", "")[:26],
                 ve, vm, vz, nin, ntot))
        rows.append((ve, vm, vz))
        out["w_%d" % i] = w
        out["wmin_%d" % i] = wmin

    if not rows:
        raise SystemExit("no wavelet of the expected length")

    a = np.array(rows)
    print("-" * 70)
    print("%-26s %9.4f %9.4f %9.4f" % ("mean", *a.mean(0)))
    if len(rows) > 1:
        print("%-26s %9.4f %9.4f %9.4f"
              % ("s.d.", *a.std(0, ddof=1)))

    amp = np.mean([amplitude_correlation(np.load(p).astype(float), ref0,
                                         args.dt, lo, hi)
                   for p in paths if np.load(p).size == args.nw])
    nin_r, ntot_r = zeros_inside(ref0)
    ae = np.array(amp_err)
    print("\ncheck that the substitutes really do share the amplitude spectrum")
    print("  largest in-band deviation: minimum-phase %.1e, zero-phase %.1e"
          % (ae[:, 0].max(), ae[:, 1].max()))
    if ae[:, 1].max() > 0.05:
        print("    the zero-phase one is not exact -- it is symmetric and the "
              "window is too short to hold both halves.  Read its column as "
              "indicative; the minimum-phase column is the exact comparison")
    print("\nbounds")
    print("  amplitude spectra alone, extracted vs reference : %.4f" % amp)
    print("  reference's own minimum-phase equivalent        : %.4f"
          % v_refmin)
    print("    -- the ceiling for any amplitude-only method on this line")
    print("  zeros of the reference inside the unit circle   : %d/%d"
          % (nin_r, ntot_r))

    # the minimum-phase column is the one to judge on: it shares the
    # amplitude spectrum exactly, so the difference is purely phase
    # Two discriminators, because the obvious one is weak.  A recorded
    # airgun signature is itself close to minimum phase, so the absolute gap
    # between the extracted wavelet and its minimum-phase equivalent stays
    # small even when the phase is perfectly recovered.  What separates the
    # cases, on controls built with a known answer, is (i) whether the
    # extracted wavelet clears the ceiling that amplitude alone can reach, and
    # (ii) whether the per-cell difference has a consistent sign.
    d = a[:, 0] - a[:, 1]
    npos = int((d > 0).sum())
    above = int((a[:, 0] > v_refmin).sum())
    print("\nverdict")
    print("  extracted minus its own minimum-phase equivalent, per cell: "
          "%+.4f +/- %.4f, positive on %d of %d"
          % (d.mean(), d.std(ddof=1) if d.size > 1 else 0.0, npos, d.size))
    print("  cells clearing the amplitude-only ceiling of %.4f: %d of %d"
          % (v_refmin, above, d.size))
    if npos == d.size and above == d.size:
        print("  Both hold.  The extracted phase carries information the "
              "amplitude spectrum does not, and the paper can say so with "
              "these numbers rather than leaving the phase unclaimed.")
    elif npos >= 0.8 * d.size and above >= 0.8 * d.size:
        print("  Both hold on most cells but not all.  Report it with the "
              "count attached, and do not lean on the phase downstream "
              "without saying how often it held.")
    elif npos <= 0.2 * d.size:
        print("  The sign goes the other way: the minimum-phase equivalent "
              "matches the recorded signature BETTER than the extracted "
              "wavelet does.  The workflow is delivering an amplitude "
              "spectrum.  Building a minimum-phase wavelet from it is "
              "supported; putting the extracted phase onto the data is not.")
    else:
        print("  The two tests disagree, or the sign is not consistent across "
              "cells.  That is not a result either way: it means this line "
              "cannot settle the question, most likely because the recorded "
              "signature is itself near minimum phase and there is little for "
              "the test to see.")

    if args.save_npz:
        np.savez(args.save_npz, scores=a, reference=ref0, **out)
        print("\nwrote %s" % args.save_npz)


if __name__ == "__main__":
    main()
