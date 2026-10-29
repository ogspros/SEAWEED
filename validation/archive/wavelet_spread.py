"""
wavelet_spread.py — how well do extractions from disjoint parts of the same line
agree with each other?

This is the reference-free quality measure calibrated on the Viking Graben line,
where a recorded signature exists.  There the spread across ten disjoint
interleaved subsets tracked the accuracy exactly:

    span      spread (s.d.)   agreement with the recorded signature
    2.5 km        0.093                    0.693
    12.4 km       0.029                    0.740
    24.8 km       0.005                    0.827

so on a dataset with no ground truth --- which is every dataset this workflow
was written for --- the agreement among disjoint subsets is the thing to look
at.  Two extractions that agree tell you the answer no longer depends on which
shots you happened to pick; two that disagree tell you it does, however high the
stack coherence of either.

The Viking experiment also showed that the disagreement lives at the EDGES of
the band: between 28 and 48 Hz every selection agreed, and what separated a good
extraction from a bad one was 18-28 Hz and 48-65 Hz.  So the per-band agreement
below is more diagnostic than the broadband number, and it needs no reference
either.

USAGE
-----
    python3 validation/archive/wavelet_spread.py --dt 0.002 a_wavelet.npy b_wavelet.npy [...]
    python3 validation/archive/wavelet_spread.py --dt 0.004 --band 8 65 viking_blocks/C_*_wavelet.npy

With two files it answers "do these two agree"; with ten it gives the spread.
Sanity check before trusting it on a new dataset: run it on the Viking families
and confirm it separates them (C combs should agree far more closely than A
blocks).
"""
import argparse
import itertools
import sys
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_BANDS = [(8, 18), (18, 28), (28, 38), (38, 48), (48, 65)]


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    f2 = min(f2, 0.98 * ny)
    if f1 >= f2:
        return None
    return sosfiltfilt(butter(order, [f1 / ny, f2 / ny], btype='band',
                              output='sos'), x)


def ncc_lag(a, b, maxlag):
    """Best normalised cross-correlation over lags, and the lag.

    The two extractions carry their own time origin, so a bounded slide is
    needed; it is bounded because an unbounded one lets the comparison latch
    onto a different lobe of the bubble train.
    """
    best, bl = -2.0, 0
    for L in range(-maxlag, maxlag + 1):
        if L >= 0:
            x, y = a[L:], b[:len(b) - L] if L else b
        else:
            x, y = a[:len(a) + L], b[-L:]
        n = min(len(x), len(y))
        if n < 8:
            continue
        x = x[:n] - x[:n].mean(); y = y[:n] - y[:n].mean()
        d = np.sqrt((x @ x) * (y @ y))
        if d <= 0:
            continue
        v = float(x @ y / d)
        if v > best:
            best, bl = v, L
    return best, bl


def pair_classes(files):
    """Which pairs sample the same stretch of line and which do not.

    The distinction is the whole point.  Subsets interleaved over one stretch
    agree because they see the same geology a few metres apart; subsets from
    different stretches agree only if the extraction no longer depends on which
    geology it saw.  Only the second is evidence of convergence, and averaging
    the two together produces a number that means neither.
    """
    iv = []
    for f in files:
        sel = f.replace('_wavelet.npy', '_sel.npz')
        if not os.path.exists(sel):
            return None
        z = np.load(sel, allow_pickle=True)
        iv.append((int(z['sp_first']), int(z['sp_last'])))
    same = []
    for i, j in itertools.combinations(range(len(files)), 2):
        lo = max(iv[i][0], iv[j][0]); hi = min(iv[i][1], iv[j][1])
        ov = max(0, hi - lo) / max(1, min(iv[i][1] - iv[i][0], iv[j][1] - iv[j][0]))
        same.append(ov > 0.5)
    return np.array(same)


def pair_separation(files, dxsp):
    """Distance between the centres of the stretches of each pair, in km.

    Disjoint is not the same as independent.  Two blocks that happen to be
    adjacent sample neighbouring geology and agree more than two drawn from
    opposite ends of the line, so a set of contiguous blocks gives an
    optimistic picture of convergence.  On MS29 the two adjacent stretches
    agreed at 0.992 while the same blocks agreed with the distant ones at
    0.80-0.93.
    """
    if dxsp is None:
        return None
    cen = []
    for f in files:
        sel = f.replace('_wavelet.npy', '_sel.npz')
        if not os.path.exists(sel):
            return None
        z = np.load(sel, allow_pickle=True)
        cen.append(0.5 * (int(z['sp_first']) + int(z['sp_last'])) * dxsp / 1000.0)
    return np.array([abs(cen[i] - cen[j])
                     for i, j in itertools.combinations(range(len(files)), 2)])


def pairwise(ws, dt, maxlag, f1=None, f2=None):
    if f1 is not None:
        ws = [bp(w, dt, f1, f2) for w in ws]
        if any(w is None for w in ws):
            return None, None
    n = len(ws)
    M = np.eye(n)
    lags = []
    for i, j in itertools.combinations(range(n), 2):
        v, L = ncc_lag(ws[i], ws[j], maxlag)
        M[i, j] = M[j, i] = v
        lags.append(L)
    off = M[np.triu_indices(n, 1)]
    return off, np.array(lags)


def effective_duration(w, dt, lo=0.05, hi=0.95):
    """Time between the 5th and 95th percentile of the cumulative energy.

    A shorter wavelet means a thinner window in which sub-bottom reflectivity
    can contaminate the extraction, so this is the natural scale against which
    the required span should be measured.
    """
    e = np.cumsum(w ** 2)
    e /= (e[-1] + 1e-30)
    i0 = int(np.searchsorted(e, lo))
    i1 = int(np.searchsorted(e, hi))
    return (i1 - i0) * dt * 1e3


def stretch_overlap(files):
    """Do these subsets sample DIFFERENT parts of the line, or the same part?

    This is not a detail.  Ten combs interleaved over the same 25 km share no
    shots but sample the same geology 25 m apart, so they agree with each other
    almost by construction; ten contiguous blocks at different positions do not.
    Mutual agreement means completely different things in the two cases, and
    comparing the two constructions on one axis is meaningless.

    Returns (mean shot overlap, mean stretch overlap) or None if the runs did
    not write a _sel.npz.
    """
    iv, sets = [], []
    for f in files:
        sel = f.replace('_wavelet.npy', '_sel.npz')
        if not os.path.exists(sel):
            return None
        z = np.load(sel, allow_pickle=True)
        a, b, k = int(z['sp_first']), int(z['sp_last']), int(z['sp_step'])
        iv.append((a, b))
        sets.append(set(range(a, b, k)))
    sh, st = [], []
    for i, j in itertools.combinations(range(len(files)), 2):
        sh.append(len(sets[i] & sets[j]) / max(1, min(len(sets[i]), len(sets[j]))))
        lo = max(iv[i][0], iv[j][0]); hi = min(iv[i][1], iv[j][1])
        inter = max(0, hi - lo)
        st.append(inter / max(1, min(iv[i][1] - iv[i][0], iv[j][1] - iv[j][0])))
    return float(np.mean(sh)), float(np.mean(st))


def band_energy_check(ws, dt, f1, f2):
    """How much of the wavelets' energy the comparison band actually contains.

    The band is a per-survey parameter and getting it wrong changes every number
    without announcing itself.  It happened repeatedly on this project: MEPA
    compared over 8-65 Hz when its energy peaks near 55 and runs to 90, MS29
    compared over 8-65 when it dies at 50, and twice --band was passed on the
    same command line, where argparse silently keeps the last one.  Each time
    the output looked perfectly ordinary.

    So the check is made on the data.  A band holding little of the energy is
    comparing noise with noise over part of its range.  Returns the fraction
    inside the band and the wavelets' own 2-98 per cent range.
    """
    nf = 4096
    f = np.fft.rfftfreq(nf, dt)
    A = np.zeros(f.size)
    for w in ws:
        S = np.abs(np.fft.rfft(w, nf)) ** 2
        A += S / (S.sum() + 1e-30)
    A /= len(ws)
    inside = float(A[(f >= f1) & (f <= f2)].sum() / (A.sum() + 1e-30))
    c = np.cumsum(A) / (A.sum() + 1e-30)
    lo = float(f[int(np.searchsorted(c, 0.02))])
    hi = float(f[int(np.searchsorted(c, 0.98))])
    return inside, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--dt', type=float, default=0.004,
                    help='sampling interval of the wavelets, seconds')
    ap.add_argument('--band', type=float, nargs=2, default=[8.0, 65.0],
                    help='broadband comparison band, Hz')
    ap.add_argument('--maxlag_ms', type=float, default=30.0)
    ap.add_argument('--bands', type=float, nargs='+', default=None,
                    help='edges of the sub-bands, e.g. --bands 15 30 45 60 75 100. '
                         'The default 8/18/28/38/48/65 suits a conventional '
                         'survey; a high-resolution line whose energy sits above '
                         '50 Hz needs its own, or the analysis is run outside the '
                         'band where the data actually live')
    ap.add_argument('--dxsp', type=float, default=None,
                    help='shot-point spacing in metres, to convert the span into '
                         'kilometres and form the span/duration ratio')
    ap.add_argument('--force_band', action='store_true',
                    help='proceed even if the comparison band holds less than '
                         '80 per cent of the energy')
    ap.add_argument('--fig', default=None, help='write an overlay figure here')
    a = ap.parse_args()
    if sum(1 for v in sys.argv if v == '--band') > 1:
        print('NOTE: --band was given more than once; argparse keeps the LAST '
              'one, which is\n      the band reported below.  This has already '
              'produced numbers measured in\n      the wrong band on this '
              'project.')

    ws, names = [], []
    for f in a.files:
        w = np.load(f).astype(float)
        ws.append(w)
        names.append(os.path.basename(f).replace('_wavelet.npy', ''))
    single = len(ws) == 1
    L = {len(w) for w in ws}
    if len(L) != 1:
        raise SystemExit('wavelets have different lengths: %s' % sorted(L))
    n = len(ws[0])
    maxlag = int(a.maxlag_ms * 1e-3 / a.dt)
    print('%d wavelets, %d samples, dt = %.4f s (%.0f ms window), band %g-%g Hz'
          % (len(ws), n, a.dt, n * a.dt * 1e3, a.band[0], a.band[1]))

    bands = DEFAULT_BANDS
    if a.bands:
        e = sorted(a.bands)
        bands = list(zip(e[:-1], e[1:]))
    frac, sug_lo, sug_hi = band_energy_check(ws, a.dt, *a.band)
    print('    the band holds %.0f%% of the energy of these wavelets; '
          'their own 2-98%%\n    range is %.0f-%.0f Hz'
          % (100 * frac, sug_lo, sug_hi))
    # Refusing applies only where an agreement is computed.  With a single
    # wavelet the useful outputs are its duration and peak frequency, and those
    # are measured below on the wavelet's OWN band rather than on the supplied
    # one: a wrong --band would otherwise return a duration and a peak
    # frequency belonging to whatever noise happened to lie in it.
    if frac < 0.80 and not a.force_band and not single:
        raise SystemExit(
            '\nREFUSING: the comparison band holds only %.0f%% of the energy.\n'
            'Part of it is noise against noise, and the agreement it returns is\n'
            'not a property of the wavelets.  Use a band matched to this survey,\n'
            'for instance:\n'
            '    --band %.0f %.0f --bands %s\n'
            'Pass --force_band to override, but then say in the caption which\n'
            'band was used and why.'
            % (100 * frac, sug_lo, sug_hi,
               ' '.join('%.0f' % v for v in
                        np.linspace(sug_lo, sug_hi, 6))))
    if frac > 0.995 and (a.band[1] - a.band[0]) > 2.5 * (sug_hi - sug_lo):
        print('    NOTE: the band is much wider than the signal, so part of the')
        print('    correlation is computed on samples that carry nothing.')

    wsb = [bp(w, a.dt, *a.band) for w in ws]

    if single:
        # One wavelet: no agreement to measure, but the effective duration is
        # still worth having, since it is the scale against which the required
        # span should be compared across surveys of different bandwidth.
        # Measured on the wavelet's own 2-98 per cent band, so that the answer
        # does not depend on --band.
        own = bp(ws[0], a.dt, max(1.0, sug_lo), min(sug_hi, 0.49 / a.dt))
        print('    duration and peak measured over %.0f-%.0f Hz, the band of '
              'this wavelet' % (sug_lo, sug_hi))
        d = effective_duration(own, a.dt)
        print('\neffective wavelet duration (5-95%% of the energy): %.0f ms' % d)
        A = np.abs(np.fft.rfft(own, 4096))
        f = np.fft.rfftfreq(4096, a.dt)
        fp = f[int(np.argmax(A))]
        print('peak frequency %.1f Hz  ->  %.0f ms period, i.e. %.1f periods long'
              % (fp, 1e3 / max(fp, 1e-9), d / (1e3 / max(fp, 1e-9))))
        print('\nNote that a wavelet extracted from a short stretch of line carries')
        print('contamination in its coda, which inflates this figure: on Viking the')
        print('extracted duration was 128-144 ms against 121 ms for the recorded')
        print('signature.  Treat it as an upper bound, and measure it on the')
        print('longest-span extraction available.')
        return

    # ---------------- pairwise matrix -------------------------------------
    if len(ws) <= 8:
        print('\npairwise agreement (best lag):')
        print('  %-22s' % '' + ' '.join('%7s' % s[:7] for s in names))
        for i in range(len(ws)):
            row = []
            for j in range(len(ws)):
                row.append(1.0 if i == j else ncc_lag(wsb[i], wsb[j], maxlag)[0])
            print('  %-22s' % names[i][:22] + ' '.join('%7.3f' % v for v in row))

    off, lags = pairwise(wsb, a.dt, maxlag)
    print('\nbroadband:  mean pairwise NCC = %.3f +- %.3f   (min %.3f, %d pairs)'
          % (off.mean(), off.std(ddof=1) if off.size > 1 else 0.0,
             off.min(), off.size))
    print('lag between extractions: %.1f +- %.1f ms  (max |lag| %.0f ms)'
          % (lags.mean() * a.dt * 1e3,
             lags.std(ddof=1) * a.dt * 1e3 if lags.size > 1 else 0.0,
             np.abs(lags).max() * a.dt * 1e3))

    sep = pair_separation(a.files, a.dxsp)
    if sep is not None and sep.size > 3 and np.ptp(sep) > 1e-6:
        rho = float(np.corrcoef(sep, off)[0, 1])
        near_m = off[sep <= np.median(sep)].mean()
        far_m = off[sep > np.median(sep)].mean()
        print('  agreement against separation between stretches:')
        print('    adjacent half  %.3f   distant half  %.3f   (Pearson %+.2f)'
              % (near_m, far_m, rho))
        print('    A strong negative correlation means the set is dominated by')
        print('    neighbouring stretches and the mean above is optimistic; the')
        print('    distant half is the honest figure.')

    cls = pair_classes(a.files)
    if cls is not None and cls.any() and (~cls).any():
        print('  split by construction:')
        print('    same stretch      %.3f +- %.3f  (%d pairs)  reproducibility'
              % (off[cls].mean(), off[cls].std(ddof=1) if cls.sum() > 1 else 0.0,
                 int(cls.sum())))
        print('    different stretch %.3f +- %.3f  (%d pairs)  CONVERGENCE'
              % (off[~cls].mean(), off[~cls].std(ddof=1) if (~cls).sum() > 1 else 0.0,
                 int((~cls).sum())))
        print('    Read the second line.  The first only says the extraction is')
        print('    repeatable on the geology it saw.')

    # ---------------- construction and scale ------------------------------
    dur = np.array([effective_duration(w, a.dt) for w in wsb])
    print('effective wavelet duration (5-95%% of the energy): %.0f +- %.0f ms'
          % (dur.mean(), dur.std(ddof=1) if dur.size > 1 else 0.0))

    ov = stretch_overlap(a.files)
    if ov is not None:
        sh, st = ov
        print('subsets share %.0f%% of their shots and %.0f%% of their stretch of line'
              % (100 * sh, 100 * st))
        if st > 0.5:
            print('  WARNING: these subsets sample the SAME stretch of line.  They')
            print('  agree because they see the same geology a few metres apart, not')
            print('  because the extraction has converged, and this number must not')
            print('  be put on the same axis as one measured between different')
            print('  stretches.')
        sel0 = a.files[0].replace('_wavelet.npy', '_sel.npz')
        z = np.load(sel0, allow_pickle=True)
        nsh = int(z['sp_last']) - int(z['sp_first'])
        print('  each subset spans %d shot positions; span/duration is the ratio to'
              % nsh)
        print('  compare across surveys of different bandwidth (pass --dxsp to get it)')
        if a.dxsp:
            span_km = nsh * a.dxsp / 1000.0
            print('  span %.1f km, duration %.0f ms  ->  span/duration = %.2f km/ms'
                  % (span_km, dur.mean(), span_km / max(dur.mean(), 1e-9)))

    # ---------------- per band --------------------------------------------
    print('\nagreement by band (this is the diagnostic part):')
    print('  %-12s %8s %8s' % ('band (Hz)', 'mean', 's.d.'))
    ny = 0.5 / a.dt
    for (f1, f2) in bands:
        if f1 >= 0.98 * ny:
            continue
        o, _ = pairwise(ws, a.dt, maxlag, f1, min(f2, 0.95 * ny))
        if o is None or o.size == 0:
            continue
        print('  %-12s %8.3f %8.3f'
              % ('%g-%g' % (f1, min(f2, 0.95 * ny)), o.mean(),
                 o.std(ddof=1) if o.size > 1 else 0.0))
    print('\n  On Viking the core 28-48 Hz agreed for every selection and the edges')
    print('  did not; an edge band that disagrees here means the extraction has not')
    print('  converged there, whatever the broadband number says.')

    # ---------------- optional figure -------------------------------------
    if a.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ref = wsb[0]
        t = np.arange(n) * a.dt * 1e3
        A = []
        for w in wsb:
            _, L0 = ncc_lag(w, ref, maxlag)
            s = np.roll(w, -L0)
            A.append(s / (np.abs(s).max() + 1e-30))
        A = np.array(A)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].fill_between(t, A.min(0), A.max(0), color='#4c72b0', alpha=0.3, lw=0)
        ax[0].plot(t, np.median(A, 0), color='#4c72b0', lw=1.4)
        ax[0].set_xlabel('time (ms)'); ax[0].set_ylabel('normalised amplitude')
        ax[0].set_title('aligned extractions, range across subsets', loc='left')
        nf = 2048
        f = np.fft.rfftfreq(nf, a.dt)
        S = np.array([np.abs(np.fft.rfft(w, nf)) for w in A])
        S = S / S.max(1, keepdims=True)
        ax[1].fill_between(f, S.min(0), S.max(0), color='#4c72b0', alpha=0.3, lw=0)
        ax[1].plot(f, np.median(S, 0), color='#4c72b0', lw=1.4)
        ax[1].set_xlim(0, min(120, 0.5 / a.dt))
        ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('normalised amplitude')
        ax[1].set_title('amplitude spectra', loc='left')
        fig.tight_layout()
        fig.savefig(a.fig, dpi=160, bbox_inches='tight')
        print('\nfigure: %s' % a.fig)


if __name__ == '__main__':
    main()
