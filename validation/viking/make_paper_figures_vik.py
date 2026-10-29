"""
make_paper_figures_vik.py — the two Viking figures of the paper (Figs. 3 and 4).

Run in the directory that holds the runs:

    python3 validation/viking/make_paper_figures_vik.py viking_conv41 FarField.dat

Produces, in ./figures/ :

    fig_viking_wavelet.{pdf,png}   the extraction against the recorded
                                   far-field signature: waveform and spectrum,
                                   with the spread of the ten combs as a band
    fig_viking_control.{pdf,png}   what controls the result: span, not fold

Everything meant to be adjusted sits in the CONFIG block below — sizes in
centimetres, colours, fonts, bands, output formats. Nothing else needs editing
for cosmetic changes.

The reference is FarField.dat (source + source ghost, recorded 250 m below the
array, 1 ms) convolved with the receiver ghost for a 10 m streamer at the
measured Vw, anti-aliased, decimated to 4 ms and band-passed like the
extraction. R_wb and 1/r are frequency-independent scalars at normal incidence
and do not affect a correlation. No 2D half-integration: Viking is 3D.
"""
import itertools
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, resample, sosfiltfilt

# ============================== CONFIG =======================================
CM = 1 / 2.54
FIG1_SIZE = (17.0 * CM, 6.6 * CM)      # full page width, two panels
FIG2_SIZE = (17.0 * CM, 6.6 * CM)      # full page width, two panels
FORMATS = ('pdf', 'png')               # pdf for the journal, png to look at
DPI = 400
OUTDIR = 'figures'

FONT = dict(family='sans-serif', size=8)
LABEL_SIZE = 8
TITLE_SIZE = 8.5
TICK_SIZE = 7
LEGEND_SIZE = 6.8

C_REF = '#000000'                      # recorded signature
C_SPAN = {1: '#c44e52', 5: '#4c72b0', 10: '#2f8a4f'}   # 2.5 / 12.4 / 24.8 km
C_HIFOLD = '#dd8452'
C_BLOCK200 = '#8172b3'
BAND_ALPHA = 0.30

BAND = (8.0, 65.0)                     # comparison band
BAND_LABEL = '8-65 Hz'
SPAN_KM = {1: 2.5, 5: 12.4, 10: 24.8}

# acquisition, all measured on this line by the workflow itself
VW = 1448.0
Z_REC = 10.0
I_DELAY = 27                           # recording delay of FarField.dat (1 ms samples)
DT_FF, DT_EX, UP = 1e-3, 4e-3, 4
NW, MAXLAG_MS = 50, 30
# Shift allowed when comparing two extractions.  It must be the one used by
# wavelet_spread.py for the span table, or the figure and the table report two
# different numbers for the same pairs: with 5 samples here and 7 there, the
# 2.5 km blocks read 0.626 in the figure and 0.633 in the table.  wavelet_spread
# takes int(30 ms / dt), i.e. 7 samples (28 ms) at 4 ms.
PAIR_LAG = int(30e-3 / DT_EX)
SINC_FACTOR = 8        # display-only interpolation of the plotted waveforms
# =============================================================================

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


def reference(ff_g, lag_ms, n=NW):
    k = int(round(lag_ms))
    w = np.concatenate([np.zeros(k), ff_g]) if k >= 0 else ff_g[-k:]
    w = lp(np.pad(w, (0, 400)), DT_FF, 115.0)
    w4 = w[::UP][:n]
    if w4.size < n:
        w4 = np.pad(w4, (0, n - w4.size))
    return bp(w4, DT_EX)


def ncc_pair(a, b, max_lag=PAIR_LAG):
    """NCC between two extractions, over a small shift.

    The shift is PAIR_LAG, the same as the span table, and it is passed
    explicitly by every caller."""
    n = a.size
    best = -2.0
    for L in range(-max_lag, max_lag + 1):
        u, v = (a[L:], b[:n - L]) if L >= 0 else (a[:n + L], b[-L:])
        if u.size > 3:
            best = max(best, ncc(u, v))
    return best


def reference_free_spread(ws):
    """One minus the mean agreement among the extractions themselves.

    Nothing in this uses the recorded signature, which is the point: it is the
    quantity a user has on a survey where no signature survives."""
    if len(ws) < 3:
        return float('nan')
    return 1.0 - float(np.mean([ncc_pair(ws[i], ws[j], PAIR_LAG)
                                for i, j in itertools.combinations(range(len(ws)), 2)]))


def sinc_interp(x, factor=SINC_FACTOR):
    """Band-limited reconstruction on a finer grid.

    The wavelet is sampled at 4 ms and band-passed well below Nyquist, so the
    continuous signal it represents is recovered exactly by Fourier
    interpolation. Joining 50 samples with straight lines, which is what a
    plain plot does, makes it look angular for no reason other than the display.
    This is reconstruction, not smoothing: no information is added or removed.
    """
    return resample(x, x.size * factor)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-30))


def best_lag(ex, ff_g):
    v = [(ncc(ex, reference(ff_g, L)), L) for L in range(-MAXLAG_MS, MAXLAG_MS + 1)]
    return max(v)


# ------------------------------------------------------------------ loading
def load_runs(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, '*_wavelet.npy'))):
        m = PAT.match(os.path.basename(p))
        if not m:
            continue
        w = np.load(p).astype(float)
        if w.size != NW:
            continue
        r = dict(fam=m['fam'], n=int(m['n']), lab=m['lab'], path=p, w=w, step=1)
        sel = p.replace('_wavelet.npy', '_sel.npz')
        if os.path.exists(sel):
            z = np.load(sel, allow_pickle=True)
            r.update(n=int(z['n_use']), step=int(z['sp_step']),
                     coh=float(z['coherence']),
                     first=int(z['sp_first']), last=int(z['sp_last']))
        rows.append(r)
    return rows


def apply_style():
    plt.rcParams.update({
        'font.family': FONT['family'], 'font.size': FONT['size'],
        'axes.labelsize': LABEL_SIZE, 'axes.titlesize': TITLE_SIZE,
        'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable text in the PDF
    })


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(OUTDIR, '%s.%s' % (name, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('  wrote %s' % p)


# =================================================== FIGURE 1: ground truth
# The best blind block of Fig. 4a, overlaid on Fig. 3 for comparison.  It is
# found with the recorded signature, so it illustrates what a lucky segment
# gives; it is not a result the workflow could reach on an archive line.
BEST_BLOCK = ('B', 200, 'blk2')          # shots 400-599
C_BEST = '#2166ac'


def fshift(x, s):
    """Delay x by s samples (fractional) on a zero-padded support."""
    n = len(x)
    xp = np.pad(np.asarray(x, float), (n, n))
    X = np.fft.rfft(xp)
    fr = np.fft.rfftfreq(len(xp))
    return np.fft.irfft(X * np.exp(-2j * np.pi * fr * s), len(xp))[n:2 * n]


def figure_wavelet(rows, ff_g):
    combs = [r for r in rows if r['fam'] == 'C']
    if not combs:
        print('  no comb runs found, figure 1 skipped'); return
    W, lags, nccs = [], [], []
    for r in combs:
        e = bp(r['w'], DT_EX)
        v, L = best_lag(e, ff_g)
        W.append(e / np.abs(e).max()); lags.append(L); nccs.append(v)
    W = np.array(W)
    ref = reference(ff_g, float(np.median(lags)))
    ref = ref / np.abs(ref).max()
    t = np.arange(NW) * DT_EX * 1e3

    # Same band-pass and normalisation as the combs.  The reference is drawn at
    # the combs' median lag, so the block is moved by the difference between
    # its own best lag and that one; the sign of reference() is not assumed,
    # both directions are tried and the one that matches the reference is kept.
    wb = None
    best = [r for r in rows if (r['fam'], r['n'], r['lab']) == BEST_BLOCK]
    if best:
        eb = bp(best[0]['w'], DT_EX)
        vb, Lb = best_lag(eb, ff_g)
        eb = eb / np.abs(eb).max()
        d = (float(np.median(lags)) - Lb) * 1e-3 / DT_EX
        cands = [fshift(eb, s) for s in (d, -d)]
        cc = [float(c @ ref / (np.linalg.norm(c) * np.linalg.norm(ref) + 1e-30))
              for c in cands]
        wb = cands[int(np.argmax(cc))]
        print('  best block %s: NCC %.3f at lag %+.1f ms; overlay correlation %.3f'
              % ('_'.join(map(str, BEST_BLOCK)), vb, Lb, max(cc)))
    else:
        print('  best block %s not found, overlay skipped' % (BEST_BLOCK,))

    fig, ax = plt.subplots(1, 2, figsize=FIG1_SIZE)

    # (a) waveform
    ts = np.linspace(t[0], t[-1] + DT_EX * 1e3, NW * SINC_FACTOR, endpoint=False)
    ax[0].fill_between(ts, sinc_interp(W.min(0)), sinc_interp(W.max(0)),
                       color=C_SPAN[10], alpha=BAND_ALPHA, lw=0,
                       label='extracted, %d combs' % len(combs))
    ax[0].plot(ts, sinc_interp(np.median(W, 0)), color=C_SPAN[10], lw=1.1)
    # The spread of the ten disjoint combs is narrower than the line width,
    # which says that the extractions repeat and that they depart from the
    # reference together.  That belongs in the caption: on the panel it sat on
    # the wavelet and on the NCC annotation.
    ax[0].plot(ts, sinc_interp(ref), color=C_REF, lw=1.3,
               label='recorded far field\n(+ receiver ghost)')
    if wb is not None:
        ax[0].plot(ts, sinc_interp(wb), color=C_BEST, lw=1.1, ls='--',
                   label='best 200-shot block\n(shots 400-599)')
    ax[0].axhline(0, color='0.8', lw=0.5, zorder=0)
    ax[0].set_xlim(0, ts[-1]); ax[0].set_xlabel('time (ms)')
    ax[0].set_ylabel('normalised amplitude')
    ax[0].set_title('(a)', loc='left', fontweight='bold')
    ax[0].legend(loc='upper right')
    ax[0].text(0.97, 0.05, 'NCC = %.3f $\\pm$ %.3f' % (np.mean(nccs), np.std(nccs, ddof=1)),
               transform=ax[0].transAxes, ha='right', fontsize=LEGEND_SIZE)

    # (b) spectra
    nf = 1024
    f = np.fft.rfftfreq(nf, DT_EX)
    S = np.array([np.abs(np.fft.rfft(w, nf)) for w in W])
    S = S / S.max(1, keepdims=True)
    A = np.abs(np.fft.rfft(ref, nf)); A /= A.max()
    ax[1].fill_between(f, S.min(0), S.max(0), color=C_SPAN[10], alpha=BAND_ALPHA, lw=0)
    ax[1].plot(f, np.median(S, 0), color=C_SPAN[10], lw=1.1, label='extracted')
    if wb is not None:
        Sb = np.abs(np.fft.rfft(wb, nf)); Sb /= Sb.max()
        ax[1].plot(f, Sb, color=C_BEST, lw=1.1, ls='--', label='best 200-shot block')
    ax[1].plot(f, A, color=C_REF, lw=1.3,
               label='recorded far field\n(+ receiver ghost)')
    ax[1].set_xlim(0, 90); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('normalised amplitude')
    ax[1].set_title('(b)', loc='left', fontweight='bold')
    ax[1].legend(loc='upper right')

    fig.tight_layout(pad=0.4)
    save(fig, 'fig_viking_wavelet')
    plt.close(fig)


# =============================================== FIGURE 2: what controls it
# Panels (c) and (d): the near-trace section the blocks of (a) come from.
CUBE_PATH = os.environ.get('VIKING_CUBE', 'viking_pick/viking_cube.npy')
CUBE_DT = 0.004                  # s, sampling of the cube (1500 samples, 6 s)
T_SEC = (0.4, 0.8)               # s, window shown in (c) and analysed in (d)
NEAR_CH = None                   # channel index of the near trace; None = detect
BEST_SPAN = (399.5, 599.5)       # edges of the best 200-shot block (Fig. 3)
DIRECT_SEC = (0.10, 0.43)        # s, direct-arrival window on the near trace, for (d)
C_DIRECT = '#b35806'
SECTION_UP = 4                   # display-only interpolation factor in time
GAIN_POW = 0.0                   # display gain t^GAIN_POW; 0 = none (state it in the caption if used)


def pick_near(cube):
    """The near channel is the one whose strongest early event arrives first:
    at 262 m the seafloor comes at about 0.5 s, at 3.2 km at about 2.3 s."""
    i0, i1 = int(0.3 / CUBE_DT), min(int(3.0 / CUBE_DT), cube.shape[1])
    times = {}
    for ch in (0, cube.shape[2] - 1):
        a = np.median(np.abs(np.asarray(cube[:50, i0:i1, ch], float)), axis=0)
        times[ch] = (i0 + int(np.argmax(a))) * CUBE_DT
    ch = min(times, key=times.get)
    print('  near channel: index %d (strongest early event %.2f s; other end '
          '%.2f s)' % (ch, times[ch], max(times.values())))
    return ch


def near_section(path, tsec):
    if not os.path.exists(path):
        print('  cube %s not found, panels (c) and (d) skipped' % path)
        return None
    cube = np.load(path, mmap_mode='r')
    ch = NEAR_CH if NEAR_CH is not None else pick_near(cube)
    i0, i1 = int(round(tsec[0] / CUBE_DT)), int(round(tsec[1] / CUBE_DT))
    S = np.asarray(cube[:, i0:i1, ch], float)
    return S, np.arange(i0, i1) * CUBE_DT


def figure_control(rows, ff_g):
    """Two panels: where along the line the segment falls, and whether the
    spread across disjoint subsets tracks the accuracy.

    The span and fold panels of the earlier four-panel version are dropped:
    they repeated Table 6 exactly, and a figure that restates a table costs the
    reader a comparison without giving them anything.  These two panels carry
    what no table in the paper contains."""
    for r in rows:
        e = bp(r['w'], DT_EX)
        r['ncc'] = best_lag(e, ff_g)[0]

    blind = [r for r in rows if r['fam'] in ('A', 'C', 'D')]
    fig = plt.figure(figsize=(FIG2_SIZE[0], FIG2_SIZE[1] * 2.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axc = fig.add_subplot(gs[1, 0], sharex=axs[0])
    axd = fig.add_subplot(gs[1, 1])

    # (a) position along the line
    ax = axs[0]
    for r in rows:
        if r['fam'] == 'A' and 'first' in r:
            ax.plot([r['first'], r['last']], [r['ncc']] * 2, color=C_SPAN[1], lw=2.2,
                    solid_capstyle='butt')
        if r['fam'] == 'B' and r['n'] == 200 and 'first' in r:
            ax.plot([r['first'], r['last']], [r['ncc']] * 2, color=C_BLOCK200, lw=2.2,
                    alpha=0.8, solid_capstyle='butt')
    cv = np.array([r['ncc'] for r in rows if r['fam'] == 'C'])
    if cv.size:
        ax.axhspan(cv.min(), cv.max(), color=C_SPAN[10], alpha=BAND_ALPHA, zorder=0)
        ax.axhline(cv.mean(), color=C_SPAN[10], lw=1.1)
    ax.plot([], [], color=C_SPAN[1], lw=2, label='blind blocks, 100 shots')
    ax.plot([], [], color=C_BLOCK200, lw=2, label='blind blocks, 200 shots')
    ax.plot([], [], color=C_SPAN[10], lw=2, label='combs over the whole line')
    ax.set_xlabel('shot position along the line')
    ax.set_ylabel('NCC vs recorded signature')
    ax.set_title('(a)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='none')
    ax.grid(alpha=0.25, lw=0.4)

    # (b) reference-free spread against accuracy
    ax = axs[1]
    for st in (1, 5, 10):
        sel = [r for r in blind if r['step'] == st]
        v = np.array([r['ncc'] for r in sel])
        if v.size < 3:
            continue
        sp = reference_free_spread([bp(r['w'], DT_EX) for r in sel])
        ax.errorbar([sp], [v.mean()], yerr=[v.std(ddof=1)], fmt='o',
                    color=C_SPAN[st], ms=6, capsize=3, lw=1.2, mec='k', mew=0.5,
                    label='%.1f km span' % SPAN_KM[st])
    ax.set_ylabel('NCC vs recorded signature')
    ax.set_title('(b)', loc='left', fontweight='bold')
    # the points run from lower left (wide spread, poor accuracy) to upper
    # right, so the upper left corner is the one that stays empty
    ax.legend(loc='upper left'); ax.grid(alpha=0.25, lw=0.4)
    ax.invert_xaxis()
    # the tick labels already run 0.35 -> 0.00, so the axis says it is
    # inverted without needing a word for it
    ax.set_xlabel('disagreement among the extractions')

    fig.tight_layout(pad=0.5)
    sec = near_section(CUBE_PATH, T_SEC)
    if sec is not None:
        S, tt = sec
        # (c) near-trace section, same shot axis as (a)
        from scipy.signal import resample_poly
        Sd = resample_poly(S, SECTION_UP, 1, axis=1)
        td = tt[0] + np.arange(Sd.shape[1]) * CUBE_DT / SECTION_UP
        if GAIN_POW:
            Sd = Sd * (td / td[0]) ** GAIN_POW
        clip = np.percentile(np.abs(Sd), 98)
        axc.imshow(Sd.T, aspect='auto', cmap='RdBu_r', vmin=-clip, vmax=clip,
                   extent=[-0.5, S.shape[0] - 0.5,
                           td[-1] + CUBE_DT / SECTION_UP, td[0]],
                   interpolation='none')
        for x in BEST_SPAN:
            axc.axvline(x, color='k', lw=0.7, ls='--')
        axc.set_xlabel('shot position along the line')
        axc.set_ylabel('two-way time (s)')
        axc.set_title('(c)', loc='left', fontweight='bold')

        # (d) average amplitude spectrum of the same window
        nf = 1024
        fs = np.fft.rfftfreq(nf, CUBE_DT)
        taper = np.hanning(S.shape[1])
        A = np.abs(np.fft.rfft(S * taper, nf, axis=1)).mean(0)
        A /= A.max()
        axd.plot(fs, A, color='k', lw=1.1,
                 label='near trace')
        secd = near_section(CUBE_PATH, DIRECT_SEC)
        if secd is not None:
            Sdir = secd[0]
            Ad = np.abs(np.fft.rfft(Sdir * np.hanning(Sdir.shape[1]), nf, axis=1)).mean(0)
            Ad /= Ad.max()
            axd.plot(fs, Ad, color=C_DIRECT, lw=1.0, ls=':',
                     label='direct arrival')
        combs = [r for r in rows if r['fam'] == 'C']
        if combs:
            fe = np.fft.rfftfreq(nf, DT_EX)
            Sw = []
            for r in combs:
                e = bp(r['w'], DT_EX)
                Sw.append(np.abs(np.fft.rfft(e / np.abs(e).max(), nf)))
            Sw = np.array(Sw)
            Sw /= Sw.max(1, keepdims=True)
            axd.plot(fe, np.median(Sw, 0), color=C_SPAN[10], lw=1.1, ls='--',
                     label='extracted')
        axd.set_xlim(0, 90); axd.set_ylim(0, 1.05)
        axd.set_xlabel('frequency (Hz)'); axd.set_ylabel('normalised amplitude')
        axd.set_title('(d)', loc='left', fontweight='bold')
        axd.legend(loc='upper right', framealpha=0.9, edgecolor='none')
        axd.grid(alpha=0.25, lw=0.4)
        fig.tight_layout(pad=0.4)
    else:
        axc.set_visible(False); axd.set_visible(False)
    save(fig, 'fig_viking_control')
    plt.close(fig)


def main(d, ffp):
    apply_style()
    ff_g = far_field(ffp)
    rows = load_runs(d)
    print('loaded %d runs from %s' % (len(rows), d))
    figure_wavelet(rows, ff_g)
    figure_control(rows, ff_g)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'viking_blocks'
    ffp = sys.argv[2] if len(sys.argv) > 2 else 'FarField.dat'
    main(d, ffp)
