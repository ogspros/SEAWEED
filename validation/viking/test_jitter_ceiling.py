"""
test_jitter_ceiling.py — is the alignment residual actually costing anything?

THE QUESTION
------------
The residual grows with span: 1.95 ms at 2.5 km, 4.03 at 12.4, 6.01 at 24.8.
One reading takes that as stacking jitter acting on the wavelet as

    A(f) -> A(f) exp(-2 pi^2 f^2 sigma^2)

which at sigma = 6 ms leaves 0.32 of the amplitude at 40 Hz and 0.08 at 60 Hz.
If that reading is right, the full-span extraction is throwing away most of its
own bandwidth and there is a large prize in fixing the alignment.

But it may not be right, and the distinction is definitional: if
`align_residual_ms_mean` is the scatter of the lags the cross-correlation
APPLIES, the windows end up aligned and nothing is attenuated -- the number is
then a measure of how far the data depart from the kinematic model, not a loss.
If it is the misfit REMAINING after alignment, it does attenuate.

THE TEST
--------
Smear the recorded far-field signature by exp(-2 pi^2 f^2 sigma^2) and see which
sigma matches the extraction best.

  * optimum near the measured residual, with a clear gain over sigma = 0
        -> the extraction IS jitter-smeared. The gain is an upper bound on what
           better alignment could recover.
  * optimum at sigma = 0, match degrading monotonically
        -> the residual is corrected, not lost. Nothing to fix here, and the
           exp(-2 pi^2 f^2 sigma^2) reading in the paper needs qualifying.

A second, independent route: fit the same Gaussian to the ratio of the two
amplitude spectra.  On the three original runs (measured residual ~2 ms) that
fit returned 0-3 ms, so it is calibrated; if it returns 5-7 ms on the combs the
two routes agree.

USAGE
-----
    python3 validation/viking/test_jitter_ceiling.py FarField.dat  wavelet1.npy [wavelet2.npy ...]

Each wavelet is matched with its own _sel.npz when present, so the measured
residual is printed next to the fitted one.
"""
import os
import sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

DT_FF, DT_EX, UP = 1e-3, 4e-3, 4
VW, Z_REC, I_DELAY = 1448.0, 10.0, 27
NW, MAXLAG_MS = 50, 30
BAND = (8.0, 65.0)


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low',
                              output='sos'), x)


def far_field(path):
    ff = np.loadtxt(path)[I_DELAY:]
    k = int(round((2.0 * Z_REC / VW) / DT_FF))
    g = np.zeros(k + 1); g[0] = 1.0; g[k] = -1.0        # receiver ghost
    return np.convolve(ff, g)


def reference(ff_g, lag_ms, sigma_ms, f1=BAND[0], f2=BAND[1], n=NW):
    w = ff_g
    if sigma_ms > 0:
        nf = 4096
        W = np.fft.rfft(w, nf)
        f = np.fft.rfftfreq(nf, DT_FF)
        W *= np.exp(-2.0 * np.pi ** 2 * (f * sigma_ms * 1e-3) ** 2)
        w = np.fft.irfft(W, nf)[:len(w) + 200]
    k = int(round(lag_ms))
    w = np.concatenate([np.zeros(k), w]) if k >= 0 else w[-k:]
    w = lp(np.pad(w, (0, 400)), DT_FF, 115.0)
    w4 = w[::UP][:n]
    if w4.size < n:
        w4 = np.pad(w4, (0, n - w4.size))
    return bp(w4, DT_EX, f1, f2)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best(ex, ff_g, sigma):
    v = [(ncc(ex, reference(ff_g, L, sigma)), L)
         for L in range(-MAXLAG_MS, MAXLAG_MS + 1)]
    return max(v)


def fitted_sigma(ex_raw, ff_g):
    """Gaussian fit to the ratio of the amplitude spectra, 12-62 Hz."""
    nf = 1024
    R = np.abs(np.fft.rfft(reference(ff_g, 20, 0.0, 1, 120), nf))
    A = np.abs(np.fft.rfft(ex_raw, nf))
    f = np.fft.rfftfreq(nf, DT_EX)
    m = (f >= 12) & (f <= 62)
    ratio = A / (R + 1e-12)
    ratio = ratio / ratio[m][0]
    y = np.log(np.clip(ratio[m], 1e-3, None))
    a = np.polyfit(f[m] ** 2, y, 1)[0]
    return np.sqrt(max(-a, 0) / (2 * np.pi ** 2)) * 1e3


def main(ffp, paths):
    ff_g = far_field(ffp)
    sigmas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    print('sigma applied to the REFERENCE (ms):')
    print('%-30s %8s  ' % ('wavelet', 'measured') + '  '.join('%5.0f' % s for s in sigmas)
          + '   | best   fitted')
    print('-' * 116)
    for p in paths:
        ex = bp(np.load(p).astype(float), DT_EX, *BAND)
        selp = p.replace('_wavelet.npy', '_sel.npz')
        meas = float('nan')
        if os.path.exists(selp):
            meas = float(np.load(selp, allow_pickle=True)['align_residual_ms'])
        vals = [best(ex, ff_g, s)[0] for s in sigmas]
        k = int(np.argmax(vals))
        print('%-30s %7.2f   ' % (os.path.basename(p).replace('_wavelet.npy', ''), meas)
              + '  '.join('%5.3f' % v for v in vals)
              + '   | %.0f ms  %.1f ms' % (sigmas[k], fitted_sigma(np.load(p).astype(float), ff_g)))
    print()
    print('Reading: a maximum at sigma = 0 with the row falling monotonically means')
    print('the residual is corrected and costs nothing.  A maximum near the measured')
    print('value, clearly above the sigma = 0 entry, means the extraction is smeared')
    print('and that gain is roughly what better alignment could recover.')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
