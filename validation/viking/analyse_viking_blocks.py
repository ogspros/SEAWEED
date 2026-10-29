"""
Analyse the Viking block experiment: every extracted wavelet in a directory is
compared with the RECORDED far-field signature, and the results are grouped by
fold and by line length so that the two can be separated.

Usage
-----
    python3 validation/viking/analyse_viking_blocks.py <dir_with_wavelets> [FarField.dat]

The script expects file names of the form

    <FAMILY>_n<FOLD>_L<LENGTH>km_<label>_wavelet.npy
    e.g.  A_n100_L2.5km_blk3_wavelet.npy

which is what run_viking_blocks.sh produces.  Anything that does not parse is
reported and skipped, so a partial directory is fine.

What it prints
--------------
  * mean +- sd of the NCC against ground truth for every (fold, length) cell
  * the paired-family test A (2.5 km) vs C (25 km) at constant fold 100:
    this is the line-length effect with fold held fixed
  * the fold curve at fixed contiguous acquisition, now with scatter
  * the scatter of the origin lag, i.e. the repeatability of stage 5
"""
import sys, os, re, glob
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ constants
DT_FF, DT_EX = 1e-3, 4e-3
UP = 4
VW = 1448.0            # measured on this line by the pipeline
Z_REC = 10.0           # documented streamer depth
I_DELAY = 27           # recording delay of FarField.dat, samples at 1 ms
NW = 50                # 200 ms at 4 ms
MAXLAG_MS = 30
BANDS = [(8., 65.), (20., 65.)]
BAND_MAIN = (8., 65.)

PAT = re.compile(r'^(?P<fam>[A-Z])_n(?P<n>\d+)_L(?P<L>[\d.]+)km_(?P<lab>.+?)_wavelet\.npy$')


# ------------------------------------------------------------------ reference
def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    sos = butter(order, [f1 / ny, min(f2 / ny, 0.98)], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def lp(x, dt, fc, order=8):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, min(fc / ny, 0.98), btype='low', output='sos'), x)


def build_far_field(path):
    ff = np.loadtxt(path)[I_DELAY:]
    k = int(round((2.0 * Z_REC / VW) / DT_FF))
    g = np.zeros(k + 1); g[0] = 1.0; g[k] = -1.0     # receiver ghost
    return np.convolve(ff, g)


def reference_on_grid(ff_g, lag_ms, f1, f2, n=NW):
    k = int(round(lag_ms / (DT_FF * 1e3)))
    w = np.concatenate([np.zeros(k), ff_g]) if k >= 0 else ff_g[-k:]
    w = lp(np.pad(w, (0, 400)), DT_FF, 115.0)
    w4 = w[::UP][:n]
    if w4.size < n:
        w4 = np.pad(w4, (0, n - w4.size))
    return bp(w4, DT_EX, f1, f2)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best_match(ex, ff_g, f1, f2):
    lags = np.arange(-MAXLAG_MS, MAXLAG_MS + 1, 1.0)
    v = np.array([ncc(ex, reference_on_grid(ff_g, L, f1, f2)) for L in lags])
    k = int(np.argmax(v))
    return v[k], float(lags[k])


# ------------------------------------------------------------------ main
def main(d, ff_path):
    ff_g = build_far_field(ff_path)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, '*_wavelet.npy'))):
        base = os.path.basename(p)
        m = PAT.match(base)
        if not m:
            print('  ! name not parsed, skipped: %s' % base)
            continue
        w = np.load(p).astype(float)
        if w.size != NW:
            print('  ! unexpected length %d, skipped: %s' % (w.size, base))
            continue
        rec = dict(fam=m['fam'], n=int(m['n']), L=float(m['L']), lab=m['lab'])
        # Provenance.  The file name states the intent; the _sel.npz written by
        # run_real states what was done.  Where they disagree the _sel.npz is
        # trusted and the discrepancy is reported: mixing runs of different
        # selections gives wrong numbers silently.
        selp = p.replace('_wavelet.npy', '_sel.npz')
        if os.path.exists(selp):
            z = np.load(selp, allow_pickle=True)
            n_true, step = int(z['n_use']), int(z['sp_step'])
            span_km = (int(z['sp_last']) - int(z['sp_first']) - 1) * 0.025
            if n_true != rec['n'] or abs(span_km - rec['L']) > 0.6:
                print('  ! %s: name says n=%d L=%.1fkm, run says n=%d L=%.1fkm '
                      '- using the run' % (base, rec['n'], rec['L'], n_true, span_km))
            rec['n'], rec['L'], rec['step'] = n_true, round(span_km, 1), step
            rec['coh'] = float(z['coherence'])
            rec['align_ms'] = float(z['align_residual_ms'])
            rec['origin_lead_ms'] = float(z['origin_minus_lead_ms'])
        else:
            rec['step'] = 1
            rec['coh'] = rec['align_ms'] = rec['origin_lead_ms'] = float('nan')
        for (f1, f2) in BANDS:
            v, lag = best_match(bp(w, DT_EX, f1, f2), ff_g, f1, f2)
            rec['ncc_%d_%d' % (f1, f2)] = v
            if (f1, f2) == BAND_MAIN:
                rec['lag'] = lag
        k = int(80 / 4)
        rec['coda'] = float((w[k:] ** 2).sum() / (w ** 2).sum())
        rows.append(rec)

    if not rows:
        print('no usable wavelet found in %s' % d); return
    key = 'ncc_%d_%d' % BAND_MAIN

    # ---------------- cells -------------------------------------------------
    print('\n%-6s %6s %8s %5s   %-16s  %-16s  %s'
          % ('family', 'fold', 'length', 'n', 'NCC 8-65 Hz', 'NCC 20-65 Hz', 'coda>80ms'))
    print('-' * 92)
    cells = {}
    for r in rows:
        cells.setdefault((r['fam'], r['n'], r['L']), []).append(r)
    for (fam, n, L), g in sorted(cells.items(), key=lambda x: (x[0][1], x[0][2])):
        a = np.array([x[key] for x in g])
        b = np.array([x['ncc_20_65'] for x in g])
        c = np.array([x['coda'] for x in g])
        f = '%.3f +- %.3f' if len(g) > 1 else '%.3f'
        s1 = (f % (a.mean(), a.std(ddof=1))) if len(g) > 1 else '%.3f' % a.mean()
        s2 = (f % (b.mean(), b.std(ddof=1))) if len(g) > 1 else '%.3f' % b.mean()
        print('%-6s %6d %7.1fkm %5d   %-16s  %-16s  %.2f'
              % (fam, n, L, len(g), s1, s2, c.mean()))

    # ---------------- the decisive comparison -------------------------------
    print('\n' + '=' * 76)
    print('LINE LENGTH AT CONSTANT FOLD  (the confound the nested runs could not remove)')
    print('=' * 76)
    for band in BANDS:
        k = 'ncc_%d_%d' % band
        groups = {}
        for r in rows:
            # families A, C, D only: they are blind selections at fixed fold.
            # Z is chosen by coherence and belongs to its own comparison below.
            if r['n'] in range(95, 106) and r['fam'] in ('A', 'C', 'D'):
                # Group by sp_step, the DESIGN variable, not by the measured
                # span: rounding the span to 0.1 km split the ten-run cells
                # into bins of two to four and the rank test then compared
                # n=2 against n=10.
                groups.setdefault(r['step'], []).append(r[k])
        if len(groups) < 2:
            print('  band %g-%g Hz: need at least two lengths at fold 100' % band); continue
        print('\n  band %g-%g Hz, fold fixed at 100:' % band)
        for st in sorted(groups):
            v = np.array(groups[st])
            print('    step %2d (%5.1f km) : %.3f +- %.3f   (n=%d)  [%s]'
                  % (st, 0.025 * 100 * st, v.mean(),
                     v.std(ddof=1) if v.size > 1 else 0.0, v.size,
                     ' '.join('%.3f' % x for x in v)))
        Ls = sorted(groups)
        a, b = np.array(groups[Ls[0]]), np.array(groups[Ls[-1]])
        if a.size > 1 and b.size > 1:
            u, p = mannwhitneyu(b, a, alternative='greater')
            cliff = float(((b[:, None] > a[None, :]).sum()
                           - (b[:, None] < a[None, :]).sum()) / (a.size * b.size))
            print('    %.1f km vs %.1f km : delta = %+.3f, Mann-Whitney one-sided '
                  'p = %.5f, Cliff d = %+.3f'
                  % (0.025 * 100 * Ls[-1], 0.025 * 100 * Ls[0],
                     b.mean() - a.mean(), p, cliff))
        # THE DECISIVE COMPARISON: same interval of line, different fold.
        print('\n  same interval, fold x5 / x10:')
        for lab, pre, fam2, n2 in [('shots 0-500', 'c0', 'B', 500),
                                   ('shots 500-1000', 'c1', 'B', 500),
                                   ('whole line', None, 'B', 1000)]:
            if pre is None:
                lo = [r[k] for r in rows if r['fam'] == 'C']
                hi = [r[k] for r in rows if r['fam'] == fam2 and r['n'] >= n2]
            else:
                lo = [r[k] for r in rows
                      if r['fam'] == 'D' and str(r['lab']).startswith(pre)]
                blk = 'blk0' if pre == 'c0' else 'blk1'
                hi = [r[k] for r in rows if r['fam'] == fam2
                      and r['n'] == n2 and str(r['lab']) == blk]
            if lo and hi:
                print('    %-16s fold 100 %.3f +- %.3f (n=%d)  vs  fold %d %.3f'
                      '   difference %+.3f'
                      % (lab, np.mean(lo), np.std(lo, ddof=1) if len(lo) > 1 else 0.,
                         len(lo), n2, np.mean(hi), np.mean(hi) - np.mean(lo)))

    # ---------------- fold curve -------------------------------------------
    print('\n' + '=' * 76)
    print('FOLD AT CONTIGUOUS ACQUISITION (family B + family A), band %g-%g Hz' % BAND_MAIN)
    print('=' * 76)
    byn = {}
    for r in rows:
        if r['fam'] in ('A', 'B'):
            byn.setdefault(r['n'], []).append(r[key])
    for n in sorted(byn):
        v = np.array(byn[n])
        print('  fold %5d : %.3f +- %.3f  (n=%d runs)'
              % (n, v.mean(), v.std(ddof=1) if v.size > 1 else 0.0, v.size))

    # ---------------- convergence check, usable without ground truth --------
    print('\n' + '=' * 76)
    print('CONVERGENCE CHECK: spread across disjoint subsets vs accuracy')
    print('=' * 76)
    print('  The spread is measurable on ANY dataset -- it needs no reference.')
    print('  If it tracks the accuracy column, it is the QC stack coherence is not.')
    print('  %-10s %6s   %-18s  %-18s' % ('span', 'n', 'spread across subsets',
                                          'agreement with truth'))
    sp_x, sp_y = [], []
    for st in sorted({r['step'] for r in rows if r['fam'] in ('A', 'C', 'D')}):
        v = np.array([r[key] for r in rows
                      if r['fam'] in ('A', 'C', 'D') and r['step'] == st])
        if v.size < 3:
            continue
        sd = float(v.std(ddof=1))
        print('  %-10s %6d   %-18s  %-18s'
              % ('%.1f km' % (0.025 * 100 * st), v.size,
                 'sd = %.3f  (range %.3f)' % (sd, float(np.ptp(v))),
                 '%.3f' % v.mean()))
        sp_x.append(sd); sp_y.append(v.mean())
    if len(sp_x) > 2:
        print('  spread vs accuracy across cells: Spearman rho = %+.2f'
              % spearmanr(sp_x, sp_y)[0])

    # ---------------- alignment: how much is jitter costing? ----------------
    print('\n' + '=' * 76)
    print('ALIGNMENT RESIDUAL AGAINST SPAN')
    print('=' * 76)
    print('  sigma is the stacking jitter; the wavelet is attenuated by')
    print('  exp(-2 pi^2 f^2 sigma^2), so the last column is what is left at 40 Hz.')
    for st in sorted({r['step'] for r in rows if r['fam'] in ('A', 'C', 'D')}):
        g = [r for r in rows if r['fam'] in ('A', 'C', 'D') and r['step'] == st]
        a = np.array([r['align_ms'] for r in g])
        att = float(np.exp(-2 * np.pi ** 2 * (40.0 * a.mean() * 1e-3) ** 2))
        print('  %-10s sigma = %.2f +- %.2f ms   ->  amplitude at 40 Hz x %.2f'
              % ('%.1f km' % (0.025 * 100 * st), a.mean(),
                 a.std(ddof=1) if a.size > 1 else 0.0, att))

    # ---------------- does --auto_select help? ------------------------------
    zs = [r for r in rows if r['fam'] == 'Z']
    if zs:
        print('\n' + '=' * 76)
        print('COHERENCE-BASED SELECTION (--auto_select) AGAINST BLIND SELECTION')
        print('=' * 76)
        print('  --auto_select picks the contiguous block whose windows agree most')
        print('  with EACH OTHER.  The question is whether that block also agrees')
        print('  best with the recorded signature.')
        for r in sorted(zs, key=lambda x: x['n']):
            peers = [q[key] for q in rows
                     if q['fam'] in ('A', 'B') and q['n'] == r['n']]
            line = '  fold %4d : auto-selected %.3f' % (r['n'], r[key])
            if peers:
                pk = np.array(peers)
                pct = 100.0 * (pk < r[key]).mean()
                line += ('   blind blocks %.3f +- %.3f (n=%d), '
                         'auto sits at the %.0fth percentile'
                         % (pk.mean(), pk.std(ddof=1) if pk.size > 1 else 0.0,
                            pk.size, pct))
            if np.isfinite(r['coh']):
                line += '   [coherence %.3f]' % r['coh']
            print(line)
        print('  A percentile at or below 50 means the criterion the pipeline')
        print('  optimises is not the one that matters, which is the point.')

    # ---------------- repeatability of the origin ---------------------------
    lags = np.array([r['lag'] for r in rows])
    print('\norigin lag vs ground truth over all runs : %.1f +- %.1f ms  '
          '(min %.0f, max %.0f)'
          % (lags.mean(), lags.std(ddof=1) if lags.size > 1 else 0.0,
             lags.min(), lags.max()))
    ol = np.array([r['origin_lead_ms'] for r in rows])
    if np.isfinite(ol).any():
        print("origin_minus_lead_ms reported by the pipeline : %.1f +- %.1f ms"
              % (np.nanmean(ol), np.nanstd(ol, ddof=1) if ol.size > 1 else 0.0))
        print('  -> if the two above track each other, the ~20 ms offset of the '
              'extracted\n     wavelet is the window LEAD and not a bias of the '
              'origin estimate.')

    # Coherence against accuracy: the point is that they need not agree.
    ok = [r for r in rows if np.isfinite(r['coh'])]
    if len(ok) > 3:
        c = np.array([r['coh'] for r in ok]); v = np.array([r[key] for r in ok])
        print('\nstack coherence vs agreement with ground truth: Pearson r = %+.2f '
              '(n=%d runs)' % (np.corrcoef(c, v)[0, 1], len(ok)))
        print('  A near-zero or negative value is the result, not a failure: it is '
              'the\n  claim that coherence measures internal consistency, not '
              'accuracy.')

    # ---------------- figure ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for L, mk, col in [(2.5, 'o', '#c44e52'), (12.5, 's', '#4c72b0'), (25.0, '^', '#55a868')]:
        st = {2.5: 1, 12.5: 5, 25.0: 10}[L]
        v = [r[key] for r in rows if r['fam'] in ('A', 'C', 'D')
             and r['step'] == st]
        if not v:
            continue
        x = np.full(len(v), L) * (1 + 0.02 * np.random.RandomState(0).randn(len(v)))
        ax.plot(x, v, mk, color=col, alpha=0.55, ms=6)
        ax.errorbar([L], [np.mean(v)], yerr=[np.std(v, ddof=1) if len(v) > 1 else 0],
                    fmt=mk, color=col, ms=11, capsize=5, lw=2, mec='k')
    ax.set_xscale('log'); ax.set_xlabel('line length spanned (km), fold fixed at 100')
    ax.set_ylabel('NCC vs recorded signature')
    ax.set_title('(a)  line length at constant fold', loc='left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    for n in sorted(byn):
        v = np.array(byn[n])
        ax.plot(np.full(v.size, n), v, 'o', color='0.6', alpha=0.6, ms=5)
        ax.errorbar([n], [v.mean()], yerr=[v.std(ddof=1) if v.size > 1 else 0],
                    fmt='o', color='#4c72b0', ms=10, capsize=5, lw=2, mec='k')
    ax.set_xscale('log'); ax.set_xlabel('fold (contiguous shots)')
    ax.set_ylabel('NCC vs recorded signature')
    ax.set_title('(b)  fold, contiguous acquisition', loc='left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(d, 'viking_blocks.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    print('\nfigure: %s' % out)

    np.save(os.path.join(d, 'viking_blocks_results.npy'), np.array(rows, dtype=object))


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    ffp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, 'FarField.dat')
    main(d, ffp)
