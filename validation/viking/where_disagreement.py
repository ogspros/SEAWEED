"""
where_disagreement.py — where does the residual disagreement with the recorded
signature actually live?

The whole-line extraction agrees with the Viking far-field signature at 0.827.
That single number says nothing about whether the main lobe is slightly wrong
everywhere, or nearly perfect with the error concentrated in the coda, or
correct in the passband and wrong at the edges.  The three would have quite
different consequences for designature, and only the second would justify the
L_w = 200 ms choice the paper reports as a negative result.

THREE DECOMPOSITIONS
--------------------
1. IN TIME, cumulative.  NCC computed on a growing window from t = 0.  If the
   curve is high and flat to ~60 ms and then falls, the main lobe is right and
   the coda is not.

2. IN TIME, local.  After matching the two by a single least-squares scale, the
   squared residual is accumulated along the window and normalised.  This says
   where the error is *made*, not where it has accumulated, and it is compared
   with the reference's own energy so that "the extraction has energy the truth
   does not" can be told apart from "the extraction is wrong where the truth has
   energy".

3. IN FREQUENCY.  Both signals are band-passed into sub-bands and the NCC is
   recomputed in each.  A dip in one band localises the disagreement
   spectrally; the amplitude ratio in the same band says whether the extraction
   has too much or too little energy there.

Everything is aggregated by family (A = 2.5 km blocks, D = 12.4 km combs,
C = 24.8 km combs, Z = --auto_select), so the question "does more span fix the
coda, or does it fix the passband?" gets an answer rather than a guess.

A caution built into the reading.  The reference is truncated to the same 200 ms
window as the extraction, and it carries 94 per cent of its energy in the first
100 ms; a residual concentrated late therefore means the extraction has energy
where the signature has essentially none, which is contamination, not a phase
error.

USAGE
-----
    python3 validation/viking/where_disagreement.py viking_blocks FarField.dat
"""
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ------------------------------------------------------------------ constants
DT_FF, DT_EX, UP = 1e-3, 4e-3, 4
VW, Z_REC, I_DELAY = 1448.0, 10.0, 27
NW, MAXLAG_MS = 50, 30
BAND = (8.0, 65.0)
SUB_BANDS = [(8, 18), (18, 28), (28, 38), (38, 48), (48, 65)]
WINDOWS_MS = [40, 60, 80, 100, 120, 140, 160, 200]
FAMILY_NAME = {'A': '2.5 km blocks', 'D': '12.4 km combs',
               'C': '24.8 km combs', 'Z': 'auto_select', 'B': 'contiguous, high fold'}
COL = {'A': '#c44e52', 'D': '#4c72b0', 'C': '#2f8a4f', 'Z': '#8172b3', 'B': '#dd8452'}
PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)_wavelet\.npy$')


# ------------------------------------------------------------------ reference
def bp(x, dt, f1=BAND[0], f2=BAND[1], order=4):
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
    g = np.zeros(k + 1); g[0] = 1.0; g[k] = -1.0
    return np.convolve(ff, g)


def reference(ff_g, lag_ms, f1=BAND[0], f2=BAND[1], n=NW):
    k = int(round(lag_ms))
    w = np.concatenate([np.zeros(k), ff_g]) if k >= 0 else ff_g[-k:]
    w = lp(np.pad(w, (0, 400)), DT_FF, 115.0)
    w4 = w[::UP][:n]
    if w4.size < n:
        w4 = np.pad(w4, (0, n - w4.size))
    return bp(w4, DT_EX, f1, f2)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best_lag(ex, ff_g):
    v = [(ncc(ex, reference(ff_g, L)), L) for L in range(-MAXLAG_MS, MAXLAG_MS + 1)]
    return max(v)


# ------------------------------------------------------------------ measures
def cumulative_ncc(ex, ref):
    """NCC on a growing window from t = 0."""
    out = []
    for wms in WINDOWS_MS:
        k = int(wms / (DT_EX * 1e3))
        out.append(ncc(ex[:k], ref[:k]))
    return np.array(out)


def residual_profile(ex, ref):
    """Squared residual along the window after one least-squares scale.

    Returns the cumulative fraction of the total residual energy, and the
    reference's own cumulative energy fraction for comparison.
    """
    s = float(ex @ ref / (ref @ ref + 1e-30))
    r = (ex - s * ref) ** 2
    cr = np.cumsum(r) / (r.sum() + 1e-30)
    ce = np.cumsum(ref ** 2) / ((ref ** 2).sum() + 1e-30)
    return cr, ce


def band_ncc(ex_raw, ff_g, lag):
    """NCC per sub-band, and the RELATIVE amplitude balance in that band.

    The extraction and the recorded signature are in different physical units,
    so a raw amplitude ratio only measures that scale.  Each band ratio is
    therefore divided by the broadband ratio: the result is 1 where the
    extraction has the same spectral balance as the signature, above 1 where it
    carries relatively too much energy, below where too little.
    """
    a0 = bp(ex_raw, DT_EX)
    b0 = reference(ff_g, lag)
    ref_scale = float(np.sqrt((a0 @ a0) / (b0 @ b0 + 1e-30)))
    out = []
    for (f1, f2) in SUB_BANDS:
        a = bp(ex_raw, DT_EX, f1, f2)
        b = reference(ff_g, lag, f1, f2)
        amp = float(np.sqrt((a @ a) / (b @ b + 1e-30))) / (ref_scale + 1e-30)
        out.append((ncc(a, b), amp))
    return np.array(out)


# ------------------------------------------------------------------ main
def main(d, ffp):
    ff_g = far_field(ffp)
    fams = {}
    for p in sorted(glob.glob(os.path.join(d, '*_wavelet.npy'))):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        raw = np.load(p).astype(float)
        if raw.size != NW:
            continue
        ex = bp(raw, DT_EX)
        v, lag = best_lag(ex, ff_g)
        ref = reference(ff_g, lag)
        cr, ce = residual_profile(ex, ref)
        # ABSOLUTE residual, comparable across groups: the per-group
        # normalisation of cres tells where the error sits within a group but
        # hides how large that error is.
        sc = float(ex @ ref / (ref @ ref + 1e-30))
        # Normalised by the SCALED reference, so the measure is invariant to the
        # units of the extraction and comparable between groups.
        res_abs = float((((ex - sc * ref) ** 2).sum())
                        / (sc ** 2 * (ref ** 2).sum() + 1e-30))
        # Family B mixes folds 200/500/1001 and Z mixes 100/200/500, i.e. spans
        # of 5, 12.5 and 25 km.  Averaging over a family therefore averages over
        # the very variable this analysis is about, so the key carries the span.
        sel = os.path.basename(p).replace('_wavelet.npy', '_sel.npz')
        fold, span = int(m['n']), float(m['L'])
        if os.path.exists(sel):
            z = np.load(sel, allow_pickle=True)
            fold = int(z['n_use'])
            span = round(0.025 * (int(z['sp_last']) - int(z['sp_first']) - 1), 1)
        kind = {'A': 'blind', 'C': 'blind', 'D': 'blind',
                'B': 'blind', 'Z': 'auto'}.get(m['fam'], m['fam'])
        key = (kind, fold, round(span, 1))
        rec = dict(fam=m['fam'], key=key, ncc=v, cum=cumulative_ncc(ex, ref),
                   cres=cr, cener=ce, band=band_ncc(raw, ff_g, lag),
                   res_abs=res_abs)
        fams.setdefault(key, []).append(rec)

    if not fams:
        print('no wavelets found in %s' % d); return
    order = sorted(fams, key=lambda k: (k[2], k[1], k[0]))
    LBL = {k: '%s %d sh, %.1f km' % (k[0], k[1], k[2]) for k in order}

    # ---------------------------------------------------------- 1. in time
    print('=' * 78)
    print('1. IN TIME, CUMULATIVE  --  NCC on a growing window from t = 0')
    print('=' * 78)
    print('  %-24s %5s   ' % ('family', 'n') + '  '.join('%5d' % w for w in WINDOWS_MS))
    for f in order:
        C = np.array([r['cum'] for r in fams[f]])
        print('  %-24s %5d   ' % (LBL[f], len(C))
              + '  '.join('%5.3f' % v for v in C.mean(0)))
    print('\n  A curve that is high to 60 ms and then falls means the main lobe is')
    print('  recovered and the disagreement is in the coda.  Compare the drop from')
    print('  60 ms to 200 ms across families: that difference is what span buys.')

    # ---------------------------------------------------------- 2. where the error is made
    print()
    print('=' * 78)
    print('2. IN TIME, LOCAL  --  where the residual energy is actually made')
    print('=' * 78)
    marks = [40, 60, 80, 100, 140, 200]
    print('  %-24s ' % 'family' + '  '.join('%6s' % ('<%dms' % w) for w in marks))
    for f in order:
        C = np.array([r['cres'] for r in fams[f]]).mean(0)
        print('  %-24s ' % LBL[f]
              + '  '.join('%6.2f' % C[int(w / 4) - 1] for w in marks))
    print()
    print('  absolute residual energy, normalised by the reference energy, and the')
    print('  window length at which the cumulative NCC is highest:')
    print('  %-24s %8s %8s  %14s'
          % ('group', 'misfit', 'to 60ms', 'best window'))
    for f in order:
        ra = np.mean([r['res_abs'] for r in fams[f]])
        early = np.mean([r['res_abs'] * r['cres'][14] for r in fams[f]])
        C = np.array([r['cum'] for r in fams[f]]).mean(0)
        k = int(np.argmax(C))
        print('  %-24s %8.2f %8.2f  %7d ms %.3f'
              % (LBL[f], ra, early, WINDOWS_MS[k], C[k]))

    E = np.array([r['cener'] for r in fams[order[0]]]).mean(0)
    print('  %-24s ' % '(reference energy)'
          + '  '.join('%6.2f' % E[int(w / 4) - 1] for w in marks))
    print('\n  Residual accumulating well after the reference energy has stopped means')
    print('  the extraction carries energy the signature does not: contamination,')
    print('  not a phase error on the wavelet itself.')

    # ---------------------------------------------------------- 3. in frequency
    print()
    print('=' * 78)
    print('3. IN FREQUENCY  --  NCC per sub-band, and amplitude ratio extraction/reference')
    print('=' * 78)
    hdr = '  %-24s ' % 'family' + '  '.join('%11s' % ('%d-%d Hz' % b) for b in SUB_BANDS)
    print(hdr)
    for f in order:
        B = np.array([r['band'] for r in fams[f]]).mean(0)
        print('  %-24s ' % LBL[f]
              + '  '.join('%5.2f /%5.2f' % (v, a) for v, a in B))
    print('\n  First number is the agreement in that band, second the relative amplitude')
    print('  balance (>1 too much energy there, <1 too little).  A band where the NCC')
    print('  collapses but the balance is near 1 is a phase problem; a band where both')
    print('  are low is a band the extraction simply does not carry.')

    # ---------------------------------------------------------- figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axs[0]
    for f in order:
        C = np.array([r['cum'] for r in fams[f]])
        ax.errorbar(WINDOWS_MS, C.mean(0), yerr=C.std(0, ddof=1) if len(C) > 1 else None,
                    marker='o', ms=4, lw=1.5, capsize=2, color=plt.cm.viridis(min(f[2] / 25.0, 1.0)),
                    label=LBL[f])
    ax.set_xlabel('length of the compared window (ms)')
    ax.set_ylabel('NCC vs recorded signature')
    ax.set_title('(a)  cumulative in time', loc='left')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axs[1]
    t = np.arange(NW) * DT_EX * 1e3
    for f in order:
        C = np.array([r['cres'] for r in fams[f]]).mean(0)
        ax.plot(t, C, lw=1.6, color=plt.cm.viridis(min(f[2] / 25.0, 1.0)), label=LBL[f])
    ax.plot(t, np.array([r['cener'] for r in fams[order[0]]]).mean(0), 'k--', lw=1.4,
            label='reference energy')
    ax.set_xlabel('time (ms)'); ax.set_ylabel('cumulative fraction')
    ax.set_title('(b)  where the residual is made', loc='left')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')

    ax = axs[2]
    x = np.arange(len(SUB_BANDS))
    w = 0.8 / max(len(order), 1)
    for i, f in enumerate(order):
        B = np.array([r['band'] for r in fams[f]]).mean(0)
        ax.bar(x + i * w - 0.4 + w / 2, B[:, 0], w, color=plt.cm.viridis(min(f[2] / 25.0, 1.0)),
               label=LBL[f])
    ax.set_xticks(x); ax.set_xticklabels(['%d-%d' % b for b in SUB_BANDS], fontsize=8)
    ax.set_xlabel('band (Hz)'); ax.set_ylabel('NCC in band')
    ax.set_title('(c)  in frequency', loc='left')
    ax.legend(fontsize=8); ax.set_ylim(0, 1)

    fig.tight_layout()
    out = os.path.join(d, 'where_disagreement.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('\nfigure: %s' % out)


if __name__ == '__main__':
    dd = sys.argv[1] if len(sys.argv) > 1 else 'viking_blocks'
    ff = sys.argv[2] if len(sys.argv) > 2 else 'FarField.dat'
    main(dd, ff)
