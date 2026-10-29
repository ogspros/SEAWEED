#!/usr/bin/env python3
"""
Does the 200 ms extraction window alone move the far-field bubble minima?

The far-field signature is resampled to the data interval, cut with the same
window length as the extraction, placed where it best matches each extracted
wavelet, and its spectral minima are compared with those of the wavelet.
Spectra and minima are computed by the same routine for every trace, so the
comparison does not depend on how viking_zones.py defines them.

Two scans show how sensitive the result is: window length at the best
position, and window position at the nominal length.

usage:
  python3 validation/viking/farfield_window_test.py FarField.dat \
      clear:viking_zones/clear_wavelet.npy layer:viking_zones/layer_wavelet.npy \
      layer5:viking_zones2/layer5_wavelet.npy [--ff_dt 0.0001] [--out ff_window]
  python3 validation/viking/farfield_window_test.py --selftest
"""
import argparse
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly, resample

DT = 0.004


def load_farfield(path, ff_dt=None, t_unit=None):
    raw = np.genfromtxt(path, comments='#', invalid_raise=False)
    raw = raw[~np.isnan(raw).any(axis=1)] if raw.ndim == 2 else raw[~np.isnan(raw)]
    if raw.ndim == 2 and raw.shape[1] >= 2:
        t, a = raw[:, 0], raw[:, 1]
        dt = float(np.median(np.diff(t)))
        unit = t_unit or ('ms' if dt > 0.05 else 's')
        if unit == 'ms':
            dt *= 1e-3
        src = f"2 columns, time in {unit} ({'forced' if t_unit else 'guessed'})"
    else:
        if ff_dt is None:
            raise SystemExit("single-column far field: give --ff_dt in seconds")
        a, dt, src = raw.ravel(), ff_dt, "1 column, dt from --ff_dt"
    if ff_dt is not None:
        dt = ff_dt
    return a.astype(float), dt, src


def to_data_dt(a, dt_in, dt_out=DT):
    fr = Fraction(dt_in / dt_out).limit_denominator(1000)
    err = abs(float(fr) - dt_in / dt_out) / (dt_in / dt_out)
    y = resample_poly(a, fr.numerator, fr.denominator)
    return y, fr, err


def spectrum(x, dt, nfft=4096, band=(8., 90.)):
    X = np.abs(np.fft.rfft(x, nfft))
    f = np.fft.rfftfreq(nfft, dt)
    m = (f >= band[0]) & (f <= band[1])
    return f, X / (X[m].max() + 1e-30)


def minima(f, A, lo=10., hi=30.):
    i = np.where((A[1:-1] < A[:-2]) & (A[1:-1] <= A[2:]))[0] + 1
    i = i[(f[i] >= lo) & (f[i] <= hi)]
    return [(float(f[k]), float(A[k])) for k in i]


def fmt(mins):
    return ', '.join(f"{fm:.1f} ({d:.2f})" for fm, d in mins) or '-'


def taper(x, dt, t_ms):
    if t_ms <= 0:
        return x
    k = max(1, int(round(t_ms * 1e-3 / dt)))
    w = np.ones(len(x))
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    w[:k] = ramp; w[-k:] = ramp[::-1]
    return x * w


class Windower:
    """Cuts windows at sub-sample positions from a band-limited 4 ms signal."""
    def __init__(self, sig4, pad, up=8):
        s = np.pad(sig4, (pad, pad))
        self.fine = resample(s, up * len(s))
        self.up, self.pad = up, pad

    def cut(self, start_s, n):          # start in samples of the 4 ms signal (padded coords)
        k0 = int(round(start_s * self.up))
        idx = k0 + self.up * np.arange(n)
        if idx[0] < 0 or idx[-1] >= len(self.fine):
            return None
        return self.fine[idx]


def best_position(win, w, step_s=1 / 8):
    n = len(w)
    best = (-1., None, 1)
    starts = np.arange(0, (len(win.fine) / win.up) - n, step_s)
    wn = w / (np.linalg.norm(w) + 1e-30)
    for st in starts:
        seg = win.cut(st, n)
        if seg is None:
            continue
        c = seg @ wn / (np.linalg.norm(seg) + 1e-30)
        if abs(c) > best[0]:
            best = (abs(c), st, int(np.sign(c)))
    return best


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, dt_ff, src = load_farfield(args.farfield, args.ff_dt, args.t_unit)
    print(f"far field: {len(a)} samples, dt {dt_ff*1e3:.3f} ms, {len(a)*dt_ff*1e3:.0f} ms long ({src})")
    ipk = int(np.argmax(np.abs(a)))
    lag0 = int(round(0.060 / dt_ff))
    if ipk + lag0 < len(a):
        ib = ipk + lag0 + int(np.argmax(a[ipk + lag0:] * np.sign(a[ipk])))
        print(f"  main peak at {ipk*dt_ff*1e3:.1f} ms, largest same-sign peak after +60 ms "
              f"at +{(ib-ipk)*dt_ff*1e3:.1f} ms (bubble period check)")
    sig4, fr, err = to_data_dt(a, dt_ff)
    print(f"  resampled to {DT*1e3:.0f} ms with ratio {fr} (error {err:.1e})")
    n_nom = int(round(args.wl_ms * 1e-3 / DT))
    pad = 4 * n_nom
    win = Windower(sig4, pad)
    full = np.pad(sig4, (0, max(0, 2048 - len(sig4))))
    fF, AF = spectrum(full, DT)
    print(f"\nfull signature (no window)          minima 10-30 Hz: {fmt(minima(fF, AF))}")
    print("  depth = amplitude at the minimum / in-band peak (8-90 Hz); same routine for all traces")

    items = [s.split(':', 1) for s in args.wavelets]
    fig, ax = plt.subplots(len(items), 2, figsize=(12, 3.2 * len(items)), squeeze=False)
    for r, (name, path) in enumerate(items):
        w = np.load(path).astype(float)
        n = len(w)
        c, st, sg = best_position(win, w)
        seg = taper(sg * win.cut(st, n), DT, args.taper_ms)
        fw, Aw = spectrum(w, DT)
        fs, As = spectrum(seg, DT)
        lead = (st - pad) * DT * 1e3 - ipk * dt_ff * 1e3
        print(f"\n[{name}] n {n} ({n*DT*1e3:.0f} ms) | best match NCC {c:.3f} sign {sg:+d} | "
              f"window starts {lead:+.1f} ms from the signature main peak")
        print(f"  extracted wavelet                  minima: {fmt(minima(fw, Aw))}")
        print(f"  signature, same window             minima: {fmt(minima(fs, As))}")
        print("  window length scan at the same start:")
        lens = [150, 200, 250, 300, 400, 600]
        for L in lens:
            nL = int(round(L * 1e-3 / DT))
            sL = win.cut(st, nL)
            if sL is None:
                continue
            fL, AL = spectrum(taper(sg * sL, DT, args.taper_ms), DT)
            print(f"    {L:4d} ms  {fmt(minima(fL, AL))}")
            ax[r, 1].plot(fL, AL, lw=0.9, label=f'{L} ms')
        print("  window start scan at the nominal length (ms relative to best):")
        for dstart in (-20, -10, -5, 0, 5, 10, 20):
            sP = win.cut(st + dstart / (DT * 1e3), n)
            if sP is None:
                continue
            fP, AP = spectrum(taper(sg * sP, DT, args.taper_ms), DT)
            print(f"    {dstart:+4d} ms  {fmt(minima(fP, AP))}")
        ax[r, 0].plot(fF, AF, color='0.6', lw=0.8, label='signature, full')
        ax[r, 0].plot(fs, As, 'k--', lw=1.2, label=f'signature, {n*DT*1e3:.0f} ms window')
        ax[r, 0].plot(fw, Aw, 'b', lw=1.4, label=f'extracted ({name})')
        for axx in ax[r]:
            axx.set_xlim(0, 60); axx.axvspan(10, 30, color='0.93', zorder=-1)
            axx.set_xlabel('Hz'); axx.legend(fontsize=7)
        ax[r, 0].set_title(f'{name}: extracted vs windowed signature (NCC {c:.2f})', fontsize=9)
        ax[r, 1].set_title(f'{name}: signature, window length scan', fontsize=9)
    plt.tight_layout(); plt.savefig(args.out + '.png', dpi=135)
    print(f"\nfigure: {args.out}.png")


def selftest():
    dtf = 0.00025
    t = np.arange(int(1.0 / dtf)) * dtf

    def pulse(t0, fp=40.):
        u = (np.pi * fp * (t - t0)) ** 2
        return (1 - 2 * u) * np.exp(-u)

    # 1) resampling and minimum finder: a copy at 25 ms, same sign -> notch at 20 Hz
    s = pulse(0.1) + pulse(0.125)
    s4, fr, err = to_data_dt(s, dtf)
    assert abs(len(s4) * DT - len(s) * dtf) < 2 * DT, "resampled length"
    f, A = spectrum(s4, DT)
    m = minima(f, A)
    print(f"notch expected 20.0 Hz, found {fmt(m)}  (ratio {fr})")
    assert any(abs(fm - 20.0) < 0.3 and d < 0.05 for fm, d in m)
    # 2) sub-sample cut: shifting the start by 1/8 sample must shift the waveform by 0.5 ms
    win = Windower(s4, 100)
    a0, a1 = win.cut(100 + 20, 50), win.cut(100 + 20 + 1 / 8, 50)
    X0, X1 = np.fft.rfft(a0, 4096), np.fft.rfft(a1, 4096)
    ff = np.fft.rfftfreq(4096, DT)
    k = np.argmin(abs(ff - 30))
    ph = np.angle(X1[k] / X0[k])
    print(f"phase at 30 Hz for a 0.5 ms advance: expected {2*np.pi*30*0.0005:.3f} rad, found {ph:.3f}")
    assert abs(ph - 2 * np.pi * 30 * 0.0005) < 0.02
    # 3) a long window keeps the notch of the full trace; a short one that cuts
    #    the delayed copy loses it
    sq = pulse(0.1) + pulse(0.25)          # copy at 150 ms -> minima every 6.7 Hz
    win = Windower(to_data_dt(sq, dtf)[0], 100)
    st = 100 + int(0.08 / DT)
    for L in (400, 100):
        seg = win.cut(st, int(L * 1e-3 / DT))
        f, A = spectrum(seg, DT)
        print(f"copy at 150 ms, window {L} ms: minima {fmt(minima(f, A))}")
    print("selftest passed")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('farfield', nargs='?')
    ap.add_argument('wavelets', nargs='*', help='name:path.npy')
    ap.add_argument('--ff_dt', type=float, default=None, help='far-field interval in s (overrides the file)')
    ap.add_argument('--t_unit', choices=['s', 'ms'], default=None, help='unit of the time column')
    ap.add_argument('--wl_ms', type=float, default=200.)
    ap.add_argument('--taper_ms', type=float, default=0., help='cosine edge taper, as in extract_wavelet')
    ap.add_argument('--out', default='ff_window')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        if not (a.farfield and a.wavelets):
            ap.error('far field file and at least one name:wavelet.npy are required')
        main(a)
