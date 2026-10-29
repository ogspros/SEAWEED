#!/usr/bin/env python3
"""
viking_stacking_filter.py -- test, against the recorded signature, whether the
high-frequency deficit of an extracted wavelet is the low-pass filter induced
by residual misalignment in the stack.

Aligned stacking of windows whose residual timing error has standard deviation
sigma multiplies the amplitude spectrum by exp(-omega^2 sigma^2 / 2).  This is
already asserted in wavelet_extractor_v2.py, beside align_scatter_ms, but it
has never been measured: measuring it needs a wavelet whose truth is known, and
Viking Graben is the only line where that is the case.

THE REFERENCE IS NOT FarField.dat.  It is that file with the recording delay
stripped, convolved with the receiver ghost of the streamer, low-passed,
decimated to the extraction interval and band-passed exactly as the extraction
is -- the chain in make_paper_figures_vik.py, reproduced here so that the NCC
this script prints is the same number as the one in the figure.  Comparing
against the raw file instead puts the receiver-ghost shaping, which spans the
whole comparison band, into the ratio the fit is trying to read.

What is measured

  [A] sigma, from the log ratio of the extracted and reference amplitude
      spectra over the comparison band.

  [B] the same fit on the upper half of the band alone.  A stacking filter
      returns the same sigma there; a band limit or a spectral notch does not.

  [C] the decisive test: the fitted filter is undone on the extracted wavelet
      and the agreement with the reference recomputed.  A rise confirms the
      mechanism, a fall refutes it, and sigma never saw the waveform -- it was
      fitted on the amplitude spectrum alone.

Nothing is regenerated; it reads wavelets already on disk.

    python3 validation/viking/viking_stacking_filter.py FarField.dat viking_blocks --fam C
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt

PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)'
                 r'_wavelet\.npy$')


# ------------------------------------------------------------------ reference
# These functions reproduce make_paper_figures_vik.py.  They are copied rather
# than imported so this runs from anywhere, but a change there must be mirrored
# here or the two will disagree silently.
def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    sos = butter(order, [f1 / ny, min(f2 / ny, 0.98)], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low',
                              output='sos'), x)


def far_field(path, delay, z_rec, vw, dt_ff):
    """Recorded signature, delay stripped, with the receiver ghost applied."""
    ff = np.loadtxt(path)
    if ff.ndim != 1:
        raise SystemExit("%s does not have exactly one column" % path)
    ff = ff[delay:]
    k = int(round((2.0 * z_rec / vw) / dt_ff))
    g = np.zeros(k + 1)
    g[0] = 1.0
    g[k] = -1.0
    return np.convolve(ff, g), k


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
    v = [(ncc(ex, reference(ff_g, L, cfg)), L)
         for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1)]
    return max(v)


# ------------------------------------------------------------------ fitting
def spectrum(x, dt, nfft):
    return np.fft.rfftfreq(nfft, dt), np.abs(np.fft.rfft(x - np.mean(x), nfft))


def fit_sigma(f, aw, at, lo, hi, floor_db=20.0):
    """sigma from log|W| - log|REF| = a - omega^2 sigma^2 / 2 over [lo, hi].

    Two precautions, both learned on controls with a known answer.
    Frequencies where either spectrum falls more than floor_db below its
    in-band peak are dropped: there the ratio is a ratio of noise, which is
    flat and biases sigma down by up to a factor of two.  And the loss is
    robust, because a spectral zero in either wavelet makes outliers in the
    log ratio that a least-squares fit follows.
    """
    band = (f >= lo) & (f <= hi)
    m = band.copy()
    if floor_db is not None:
        g = 10.0 ** (-abs(floor_db) / 20.0)
        m &= (aw > aw[band].max() * g) & (at > at[band].max() * g)
    if m.sum() < 20:
        return None
    y = np.log(aw[m]) - np.log(at[m])
    w = 2.0 * np.pi * f[m]

    def resid(p):
        return (p[0] - 0.5 * (w * p[1]) ** 2) - y

    res = least_squares(resid, [0.0, 0.004], bounds=([-50.0, 0.0], [50.0, 0.05]),
                        loss="soft_l1", f_scale=0.3)
    a, sig = res.x
    pred = a - 0.5 * (w * sig) ** 2
    return dict(a=a, sigma=sig, n=int(m.sum()),
                f_lo=float(f[m].min()), f_hi=float(f[m].max()),
                r=float(np.corrcoef(pred, y)[0, 1]),
                rms_db=float(8.686 * np.std(pred - y)))


def deblur(w, dt, sigma, lo, hi, taper=0.15):
    """Undo exp(-omega^2 sigma^2 / 2), inside the fitted band only.

    The inverse grows without bound, so the boost is confined between lo and hi
    and tapered to unity outside.  Applying it beyond the band, where the
    wavelet is at the noise floor, cost 0.08 of agreement on a control.
    """
    nfft = int(2 ** np.ceil(np.log2(max(w.size * 8, 512))))
    W = np.fft.rfft(w, nfft)
    f = np.fft.rfftfreq(nfft, dt)
    boost = np.exp(0.5 * (2 * np.pi * f * sigma) ** 2)

    ramp = np.ones_like(f)
    dl = max(taper * (hi - lo), 1e-6)
    lo_in = (f >= lo - dl) & (f < lo)
    hi_in = (f > hi) & (f <= hi + dl)
    ramp[f < lo - dl] = 0.0
    ramp[f > hi + dl] = 0.0
    ramp[lo_in] = 0.5 * (1 - np.cos(np.pi * (f[lo_in] - (lo - dl)) / dl))
    ramp[hi_in] = 0.5 * (1 + np.cos(np.pi * (f[hi_in] - hi) / dl))

    return np.fft.irfft(W * (1.0 + ramp * (boost - 1.0)), nfft)[:w.size]


def read_applied_scatter(path):
    """align_scatter_ms from the matching _sel.npz, if there is one."""
    sel = path.replace("_wavelet.npy", "_sel.npz")
    if not os.path.exists(sel):
        return float("nan")
    try:
        z = np.load(sel, allow_pickle=True)
        for k in ("align_scatter_ms", "align_residual_ms_mean"):
            if k in z:
                return float(z[k])
    except Exception:
        pass
    return float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield", help="FarField.dat, one column, sampled dt_ff")
    ap.add_argument("wavelets", nargs="+",
                    help="wavelet files, or a directory holding them")
    ap.add_argument("--fam", default=None,
                    help="keep only this family letter (A, B, C, D, Z)")
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--floor_db", type=float, default=20.0)
    # the reference chain; defaults are the CONFIG of make_paper_figures_vik.py
    ap.add_argument("--dt", type=float, default=4e-3, help="extraction interval")
    ap.add_argument("--dt_ff", type=float, default=1e-3, help="FarField interval")
    ap.add_argument("--delay", type=int, default=27,
                    help="recording delay of FarField.dat, in its own samples")
    ap.add_argument("--z_rec", type=float, default=10.0, help="streamer depth")
    ap.add_argument("--vw", type=float, default=1448.0)
    ap.add_argument("--lp_hz", type=float, default=115.0)
    ap.add_argument("--nw", type=int, default=50, help="wavelet length, samples")
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--save_npz", default=None)
    args = ap.parse_args()

    lo, hi = args.band
    up = int(round(args.dt / args.dt_ff))
    cfg = dict(dt=args.dt, dt_ff=args.dt_ff, up=up, nw=args.nw,
               lp_hz=args.lp_hz, lo=lo, hi=hi, maxlag_ms=args.maxlag_ms)

    ff_g, kg = far_field(args.farfield, args.delay, args.z_rec, args.vw,
                         args.dt_ff)
    print("reference: %s, %d samples of delay stripped, receiver ghost at "
          "%.1f m (%d samples), low-pass %.0f Hz, decimated %d:1 to %.1f ms, "
          "band-pass %g-%g Hz"
          % (os.path.basename(args.farfield), args.delay, args.z_rec, kg,
             args.lp_hz, up, args.dt * 1e3, lo, hi))
    print("receiver-ghost notch %.1f Hz, its first maximum %.1f Hz"
          % (args.vw / (2 * args.z_rec), args.vw / (4 * args.z_rec)))

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

    nfft = 4096
    out = dict(band=[lo, hi])
    rows = []

    print("\n%-28s %6s %7s %8s %6s %8s  %7s %8s %8s"
          % ("wavelet", "sigma", "sig_hi", "applied", "fit r", "band",
             "NCC", "NCC_corr", "delta"))
    print("-" * 104)

    for i, path in enumerate(paths):
        w = np.load(path).astype(float)
        if w.size != args.nw:
            print("%-28s skipped: %d samples, expected %d"
                  % (os.path.basename(path)[:28], w.size, args.nw))
            continue
        ex = bp(w, args.dt, lo, hi)
        v0, L = best_lag(ex, ff_g, cfg)
        ref = reference(ff_g, L, cfg)

        f, AW = spectrum(ex, args.dt, nfft)
        _, AR = spectrum(ref, args.dt, nfft)

        fit = fit_sigma(f, AW, AR, lo, hi, args.floor_db)
        if fit is None:
            print("%-28s too few usable frequencies"
                  % os.path.basename(path)[:28])
            continue
        mid = 0.5 * (lo + hi)
        fh = fit_sigma(f, AW, AR, mid, hi, args.floor_db)
        sig_hi = fh["sigma"] if fh else float("nan")

        exc = deblur(ex, args.dt, fit["sigma"], fit["f_lo"], fit["f_hi"])
        v1, _ = best_lag(exc, ff_g, cfg)
        applied = read_applied_scatter(path)

        print("%-28s %6.2f %7.2f %8.2f %6.3f %3.0f-%-4.0f %7.4f %8.4f %+8.4f"
              % (os.path.basename(path).replace("_wavelet.npy", "")[:28],
                 fit["sigma"] * 1e3, sig_hi * 1e3, applied, fit["r"],
                 fit["f_lo"], fit["f_hi"], v0, v1, v1 - v0))

        rows.append((fit["sigma"] * 1e3, sig_hi * 1e3, applied, fit["r"],
                     v0, v1))
        out["ex_%d" % i] = ex
        out["exc_%d" % i] = exc
        out["ref_%d" % i] = ref
        out["sigma_%d" % i] = fit["sigma"]
        out["name_%d" % i] = os.path.basename(path)

    if len(rows) > 1:
        a = np.array(rows)
        print("-" * 104)
        print("%-28s %6.2f %7.2f %8.2f %6.3f %8s %7.4f %8.4f %+8.4f"
              % ("mean of %d" % len(rows), a[:, 0].mean(), a[:, 1].mean(),
                 np.nanmean(a[:, 2]), a[:, 3].mean(), "", a[:, 4].mean(),
                 a[:, 5].mean(), (a[:, 5] - a[:, 4]).mean()))
        print("%-28s %6.2f %7.2f %8.2f %6.3f %8s %7.4f %8.4f"
              % ("s.d.", a[:, 0].std(ddof=1), a[:, 1].std(ddof=1),
                 np.nanstd(a[:, 2], ddof=1), a[:, 3].std(ddof=1), "",
                 a[:, 4].std(ddof=1), a[:, 5].std(ddof=1)))

    print("""
reading the table
  NCC          must reproduce the accuracy already known for these cells.  If
               it does not, the reference chain is wrong and nothing else on
               the line means anything
  sigma        residual timing scatter the stack would need for the deficit to
               be a stacking filter
  applied      scatter of the shifts the aligner applied, from the _sel.npz.
               sigma must come out SMALLER: the filter is caused by what the
               alignment left, not by what it removed
  sig_hi       the same fit on the upper half of the band.  On controls with a
               known answer this agreed within 15 per cent for a genuine blur
               and disagreed by a factor 3 for a sharp band cut
  fit r        about 0.99 for a genuine Gaussian blur on those controls, 0.64
               for a sharp cut
  delta        the test.  sigma was fitted on the amplitude spectrum and never
               saw the waveform, so a rise here is a prediction confirmed""")

    if args.save_npz:
        np.savez(args.save_npz, **out)
        print("\ncurves written to %s" % args.save_npz)


if __name__ == "__main__":
    main()
