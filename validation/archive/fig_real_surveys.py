"""
fig_real_surveys.py — the three archive surveys, side by side.

One figure with three columns rather than three figures, because the point is
the comparison: three sources (an explosive on MS29, airguns on the other two),
three bands, three line lengths, one workflow.  Three separate figures would
hide exactly what the reader should be putting next to each other.

Rows:
  1  near-trace section with the tracked seafloor and the first multiple
  2  the extracted wavelet: the whole-line extraction, with the range spanned by
     the disjoint combs of the longest cell behind it
  3  the amplitude spectrum of the whole-line wavelet, the agreement between
     the combs band by band, and the ghost notches predicted by the towing
     depths.  The three belong together: the spectrum says where the energy is
     and therefore which bands the agreement should be read in, and the notches
     are a prediction made from acquisition metadata alone, which the observed
     spectrum either confirms or does not

Row 3 is the recommendation of the paper made visible: the extraction is done on
shots spread over the whole line, and whether that was enough is measured on the
survey itself rather than assumed.

Run from the directory holding span_experiment/:

    python3 validation/archive/fig_real_surveys.py span_experiment
"""
import os
import re
import sys
import glob
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================== CONFIG =======================================
CM = 1 / 2.54
FIGSIZE = (17.0 * CM, 15.0 * CM)
FORMATS = ('pdf', 'png')
DPI = 400
NAME = 'fig_real_surveys'
FONT, LABEL, TITLE, TICK, LEG = 8, 8, 8.5, 7, 6.6

# One entry per column.  `cell` is the glob of the longest-span cell, which
# supplies both the spread behind the wavelet and the band agreement.
VW_GHOST = 1500.0     # for the ghost notches; a few per cent does not move them

SURVEYS = [
    dict(key='ms29', title='MS-29 — flexotir, 150 g',
         dt=0.004, band=(8., 45.), edges=[8, 15, 22, 30, 38, 45],
         cell='ms29_n100_L039km_S0200_*_wavelet.npy', tmax_ms=3000, col='#c44e52',
         z_src=14.0, z_rec=25.0),
    dict(key='it065', title='IT90-65 — airgun array, 22.5 L',
         dt=0.002, band=(8., 65.), edges=[8, 18, 28, 38, 48, 65],
         cell='it065_n100_L014km_S0600_*_wavelet.npy', tmax_ms=2500, col='#4c72b0',
         z_src=4.0, z_rec=5.0),
    dict(key='mepa', title='SISOC-MEPA — GI guns, 11 L',
         dt=0.002, band=(15., 100.), edges=[15, 30, 45, 60, 75, 100],
         cell='mepa_n100_L007km_S0300_*_wavelet.npy', tmax_ms=2500, col='#2f8a4f',
         z_src=5.0, z_rec=4.25),
]
# =============================================================================


def bp(x, dt, f1, f2, order=4, axis=-1):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x, axis=axis)


def ncc_lag(a, b, maxlag):
    best = -2.0
    for L in range(-maxlag, maxlag + 1):
        x, y = (a[L:], b[:len(b) - L]) if L > 0 else (a[:len(a) + L] if L < 0 else a,
                                                      b[-L:] if L < 0 else b)
        n = min(len(x), len(y))
        if n < 8:
            continue
        x = x[:n] - x[:n].mean(); y = y[:n] - y[:n].mean()
        d = np.sqrt((x @ x) * (y @ y))
        if d > 0:
            best = max(best, float(x @ y / d))
    return best


def band_agreement(ws, dt, edges, maxlag):
    out = []
    for f1, f2 in zip(edges[:-1], edges[1:]):
        wb = [bp(w, dt, f1, f2) for w in ws]
        v = [ncc_lag(wb[i], wb[j], maxlag)
             for i, j in itertools.combinations(range(len(wb)), 2)]
        out.append((0.5 * (f1 + f2), float(np.mean(v)), float(np.std(v, ddof=1))))
    return np.array(out)


def shift(w, L):
    """Delay w by L samples (advance if negative), zero-filled: unlike
    np.roll, nothing wraps round from the other end."""
    s = np.zeros_like(w)
    if L > 0:
        s[L:] = w[:-L]
    elif L < 0:
        s[:L] = w[-L:]
    else:
        s[:] = w
    return s


def align(ws, maxlag, ref):
    """Align every wavelet to ref, the whole-line wavelet in the same band, so
    the spread band is not an artefact of each extraction carrying its own time
    origin and sits where the whole line does.  Display only: the band
    agreement below is computed on the unaligned wavelets.  Returns the aligned
    wavelets and the shift applied to each, in samples (positive = delayed)."""
    out, lags = [], []
    for w in ws:
        best, bl = -2.0, 0
        for L in range(-maxlag, maxlag + 1):
            s = shift(w, L)
            x = s - s.mean(); y = ref - ref.mean()
            d = np.sqrt((x @ x) * (y @ y))
            if d > 0 and float(x @ y / d) > best:
                best, bl = float(x @ y / d), L
        s = shift(w, bl)
        out.append(s / (np.abs(s).max() + 1e-30)); lags.append(bl)
    return np.array(out), lags


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': FONT,
        'axes.labelsize': LABEL, 'axes.titlesize': TITLE,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'legend.fontsize': LEG, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def main(d):
    style()
    fig, ax = plt.subplots(3, len(SURVEYS), figsize=FIGSIZE,
                           gridspec_kw=dict(height_ratios=[1.5, 1, 1]))

    for c, S in enumerate(SURVEYS):
        dt = S['dt']
        cube = os.path.join(d, '%s_cube.npy' % S['key'])
        track = os.path.join(d, '%s_track.npz' % S['key'])
        full = os.path.join(d, '%s_reference_full_line_wavelet.npy' % S['key'])
        cells = sorted(glob.glob(os.path.join(d, S['cell'])))
        for p in (cube, track, full):
            if not os.path.exists(p):
                raise SystemExit('missing: %s' % p)
        if len(cells) < 3:
            raise SystemExit('fewer than three cell wavelets for %s' % S['key'])

        # ---- row 1: near-trace section ------------------------------------
        g = np.load(cube, mmap_mode='r')
        t0 = np.load(track, allow_pickle=True)['t0']
        Sn = g.shape[0]
        dec = max(1, Sn // 1200)
        sec = np.asarray(g[::dec, :, 0]).T.astype(float)
        kmax = min(sec.shape[0], int(S['tmax_ms'] * 1e-3 / dt))
        sec = sec[:kmax]
        lim = np.percentile(np.abs(sec), 98)
        a = ax[0, c]
        a.imshow(sec, aspect='auto', cmap='gray_r', vmin=-lim, vmax=lim,
                 extent=[0, Sn, kmax * dt * 1e3, 0], interpolation='nearest')
        x = np.arange(Sn)
        a.plot(x, t0 * 1e3, color=S['col'], lw=0.8)
        a.plot(x, 2 * t0 * 1e3, color=S['col'], lw=0.6, ls=':')
        a.set_title(S['title'], fontsize=TITLE)
        a.set_xlabel('shot'); 
        if c == 0:
            a.set_ylabel('two-way time (ms)')

        # ---- row 2: wavelet ------------------------------------------------
        maxlag = int(30e-3 / dt)
        ws = [np.load(p).astype(float) for p in cells]
        wf = bp(np.load(full).astype(float), dt, *S['band'])
        wf = wf / (np.abs(wf).max() + 1e-30)
        A, lags = align([bp(w, dt, *S['band']) for w in ws], maxlag, wf)
        print('%s: combs shifted onto the whole line by %s ms'
              % (S['key'], ', '.join('%+.0f' % (L * dt * 1e3) for L in lags)))
        t = np.arange(len(wf)) * dt * 1e3
        a = ax[1, c]
        a.fill_between(t, A.min(0), A.max(0), color=S['col'], alpha=0.30, lw=0,
                       label='%d combs' % len(ws))
        a.plot(t, wf, color=S['col'], lw=1.2, label='whole line')
        a.axhline(0, color='0.8', lw=0.5)
        a.set_xlim(0, t[-1]); a.set_xlabel('time (ms)')
        if c == 0:
            a.set_ylabel('normalised amplitude')
        a.legend(loc='upper right')

        # ---- row 3: spectrum, agreement, and the predicted notches ---------
        B = band_agreement(ws, dt, S['edges'], maxlag)
        a = ax[2, c]
        nf = 2048
        f = np.fft.rfftfreq(nf, dt)
        A = np.abs(np.fft.rfft(wf, nf)); A /= A.max()
        sp = a.twinx()
        sp.fill_between(f, 0, A, color='0.85', lw=0, zorder=0)
        sp.set_ylim(0, 2.4)          # keeps the spectrum in the lower half
        sp.set_yticks([])
        if c == len(SURVEYS) - 1:
            sp.set_ylabel('amplitude spectrum', fontsize=LABEL, color='0.45')
        a.set_zorder(sp.get_zorder() + 1); a.patch.set_visible(False)

        for z, lab in ((S['z_src'], 'source'), (S['z_rec'], 'receiver')):
            fn = VW_GHOST / (2 * z)
            if S['edges'][0] <= fn <= S['edges'][-1]:
                a.axvline(fn, color='0.35', ls='--', lw=0.7)
                a.text(fn, 0.615, ' %s ghost\n %.0f m' % (lab, z), fontsize=LEG - 0.6,
                       color='0.35', rotation=90, va='bottom')

        a.errorbar(B[:, 0], B[:, 1], yerr=B[:, 2], marker='o', ms=4,
                   color=S['col'], capsize=2, lw=1.1, mec='k', mew=0.4)
        a.set_ylim(0.6, 1.02)
        a.set_xlim(S['edges'][0], S['edges'][-1])
        a.set_xlabel('frequency (Hz)')
        if c == 0:
            a.set_ylabel('agreement between combs')
        a.grid(alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4)
    outdir = 'figures'
    os.makedirs(outdir, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(outdir, '%s.%s' % (NAME, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('wrote %s' % p)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'span_experiment')
