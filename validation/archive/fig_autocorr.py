r"""
fig_autocorr.py — stage 1 of the workflow, on the three real surveys.

Top row: the autocorrelation of the near-offset trace over ONE group of
\SI{800}{\metre}, the length the workflow actually averages over, with the two
lobes marked.  The one at t0 is negative because the multiple has
reversed polarity at the free surface, the one at 2*t0 is positive.  That
signature is what distinguishes the water bottom from an arbitrary periodicity,
and the figure shows it holding across an explosive source and two airgun
configurations whose wavelets differ by a factor of four in duration.

The group matters.  Averaged over the whole line the figure shows almost
nothing: where the seafloor moves, by several hundred milliseconds on IT90-65,
the lobes of different shots fall at different lags and cancel, leaving noise
with a positive value at the marked position on MS-29, the opposite of what
the method claims.  Averaging the autocorrelation along a line with relief destroys
the very structure it is meant to show, which is the point the bottom row makes
and the reason the anchors on that survey come from short groups.

The group shown is chosen as the most internally coherent of the line and its
first shot is printed in the panel, so that the reader can see it is one place
along the line rather than a summary of all of it.

What the circles are, and are not.  They mark the extremum within 60 ms of the
seafloor time already tracked, so they show that the lobes fall where the
tracking says and with the signs the free surface requires.  They are not the
initial search: that is done by Stage 1, which proposes several candidates and
judges them on the moveout of water, and its pick is drawn as well so that the
two can be told apart.

Bottom row: the same autocorrelation computed in short groups along IT90-65 and
displayed as an image.  The seafloor moves through hundreds of milliseconds over
the line (the script prints the value from the tracking file), so no
single value initialises the search everywhere -- which is why the anchors come
from the local autocorrelation on that survey and from the semblance on MS29,
where the relief fits inside the semblance search window.  The bifurcation is
easier to justify in a picture than in a sentence.

Run from the directory holding span_experiment/:

    python3 validation/archive/fig_autocorr.py span_experiment
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# autocorr_all and bp are imported rather than reimplemented: the figure must
# show what the workflow computes, and a second copy of the code would be free
# to drift from it -- which is exactly how the synthetic figure came to disagree
# with the matrix it illustrated.
from seaweed.pick_seafloor import autocorr_all, bp
try:
    from seaweed import pick_consensus as PC
except Exception:                     # the figure still draws without it
    PC = None

# ============================== CONFIG =======================================
CM = 1 / 2.54
FIGSIZE = (17.0 * CM, 11.0 * CM)
FORMATS = ('pdf', 'png')
DPI = 400
NAME = 'fig_autocorr'
FONT, LABEL, TITLE, TICK, LEG = 8, 8, 8.5, 7, 6.6

SURVEYS = [
    dict(key='ms29', title='MS-29 — flexotir', dt=0.004, band=(8., 45.),
         n_ch=3, lag_max=3.0, col='#c44e52'),   # 2*t0 is near 2.4 s here
    dict(key='it065', title='IT90-65 — airgun array', dt=0.002, band=(8., 65.),
         n_ch=3, lag_max=2.2, col='#4c72b0'),
    dict(key='mepa', title='SISOC-MEPA — GI guns', dt=0.002, band=(15., 100.),
         n_ch=3, lag_max=2.2, col='#2f8a4f'),
]
LAG_MIN = 0.15           # the zero-lag peak is 1 by construction and carries no
                         # information; normalising with it in makes every other
                         # lobe invisible
LOCAL = 'it065'          # the survey shown in the bottom panel
GROUP_M = 800.0
DXSP = {'ms29': 200.0, 'it065': 25.0, 'mepa': 25.0}
# =============================================================================


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': FONT,
        'axes.labelsize': LABEL, 'axes.titlesize': TITLE,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'legend.fontsize': LEG, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def best_group(ac, grp, step):
    """The most internally coherent group of the line.

    Not the deepest lobe, which would favour whichever group happens to sit over
    the strongest reflector, but the group whose shots agree with each other:
    the mean autocorrelation of the group, divided by the mean of the absolute
    autocorrelations, is 1 where every shot has its lobes at the same lag and
    falls towards 0 where they scatter.
    """
    best, bi = -1.0, 0
    for i in range(0, ac.shape[0] - grp + 1, step):
        blk = ac[i:i + grp]
        num = np.abs(blk.mean(0)).sum()
        den = np.abs(blk).mean(0).sum() + 1e-30
        if num / den > best:
            best, bi = num / den, i
    return bi, best


def near_trace(cube, n_ch, dt, band):
    g = np.load(cube, mmap_mode='r')
    near = np.asarray(g[:, :, :n_ch]).mean(axis=2).astype(float)
    return bp(near, dt, *band, axis=1)


def main(d):
    style()
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.45,
                          wspace=0.28)

    for c, S in enumerate(SURVEYS):
        cube = os.path.join(d, '%s_cube.npy' % S['key'])
        track = os.path.join(d, '%s_track.npz' % S['key'])
        for p in (cube, track):
            if not os.path.exists(p):
                raise SystemExit('missing: %s' % p)
        dt = S['dt']
        near = near_trace(cube, S['n_ch'], dt, S['band'])
        ac = autocorr_all(near, dt, S['lag_max'])
        # ONE group, as the workflow computes it, not the whole line.
        grp = max(5, int(round(GROUP_M / DXSP[S['key']])))
        i0g, coh = best_group(ac, grp, max(2, grp // 2))
        m = ac[i0g:i0g + grp].mean(axis=0)
        k0 = int(LAG_MIN / dt)
        m = m / (np.abs(m[k0:]).max() + 1e-30)
        lag = np.arange(m.size) * dt * 1e3

        # The t0 of THIS GROUP, not the median of the line: on IT90-65 the two
        # differ by nearly 180 ms and the marker landed on the positive flank of
        # the lobe, making the figure contradict the method it illustrates.
        t0_arr = np.load(track, allow_pickle=True)['t0']
        t0 = float(np.median(t0_arr[i0g:i0g + grp])) * 1e3
        a = fig.add_subplot(gs[0, c])
        a.axhline(0, color='0.8', lw=0.5)
        a.plot(lag, m, color=S['col'], lw=1.0)
        # The lobe is picked as the extremum within +-60 ms of the tracked
        # value, and the departure is annotated rather than hidden: the tracking
        # is deliberately held above the seafloor so as not to clip the onset of
        # the event, so the lobe sits a few tens of milliseconds from it by
        # design.  Placing the marker at the tracked value exactly put it on the
        # flank, where the sign is meaningless.
        half = int(round(0.060 / dt))
        for k, (tt, want_min) in enumerate(((t0, True), (2 * t0, False))):
            j0 = int(round(tt * 1e-3 / dt))
            lo, hi = max(1, j0 - half), min(m.size - 1, j0 + half)
            if lo >= hi:
                continue
            j = lo + (int(np.argmin(m[lo:hi])) if want_min
                      else int(np.argmax(m[lo:hi])))
            a.plot([j * dt * 1e3], [m[j]], 'o', ms=5, mfc='none', mec='k', mew=0.9)
            # The label is nudged inside the axes: at 2 t0 on the last panel
            # the lobe sits near the right edge and a centred label ran off it.
            x_lab = j * dt * 1e3
            lo_x, hi_x = LAG_MIN * 1e3, S['lag_max'] * 1e3
            frac = (x_lab - lo_x) / max(hi_x - lo_x, 1e-9)
            ha = 'center' if 0.18 < frac < 0.82 else ('left' if frac <= 0.18
                                                      else 'right')
            a.annotate('%s at %s%s' % ('−' if want_min else '+',
                                       '$t_0$' if k == 0 else '$2t_0$',
                                       '  (%+.0f ms)' % ((j - j0) * dt * 1e3)
                                       if k == 0 else ''),
                       xy=(x_lab, m[j]),
                       xytext=(x_lab, m[j] + (0.30 if m[j] < 0 else 0.24)),
                       ha=ha, fontsize=LEG,
                       arrowprops=dict(arrowstyle='-', lw=0.5, color='0.4'))
        # The pick Stage 1 actually produces on this group, from the
        # candidates and the moveout test, not from the tracked value.  This is
        # what documents the initialisation; the circles above only check that
        # the lobes sit where the tracking says.
        if PC is not None:
            try:
                gcube = np.load(cube, mmap_mode='r')
                offs = np.asarray(np.load(track, allow_pickle=True)['offs'],
                                  float)
                vw = float(np.load(track, allow_pickle=True)['vw'])
                t0_s1, _rep = PC.pick_t0(
                    np.asarray(gcube[i0g:i0g + grp]).astype(float), dt, offs, vw)
                a.axvline(float(np.median(t0_s1)) * 1e3, color='0.25',
                          ls='--', lw=0.9)
                pass          # the dashed line is named in the caption
            except Exception as exc:
                print('  stage 1 pick not drawn on %s: %s'
                      % (S['key'], str(exc)[:60]))
        a.set_xlim(LAG_MIN * 1e3, S['lag_max'] * 1e3)
        a.set_ylim(-1.25, 1.25)
        a.set_title(S['title'])
        # The shot range of the group goes in the caption: on the panel it
        # collided with the trough at t0 and with the stage 1 marker.
        print('  %-6s group shown: shots %d-%d' % (S['key'], i0g, i0g + grp - 1))
        a.set_xlabel('lag (ms)')
        if c == 0:
            a.set_ylabel('autocorrelation')

    # ---------------- the local autocorrelation, as an image -----------------
    S = [x for x in SURVEYS if x['key'] == LOCAL][0]
    dt = S['dt']
    # The axis is set by the data, not by the constant used for the top row:
    # where the seafloor is deepest 2*t0 reached 2360 ms against an axis that
    # stopped at 2200, and the dotted line left the panel.
    t0 = np.load(os.path.join(d, '%s_track.npz' % LOCAL), allow_pickle=True)['t0']
    # The relief the caption quotes, read from the tracking it is drawn from.
    # It was once written by hand as 634 ms; the tracking gives 575.
    print('  %s relief: t0 %.0f-%.0f ms, %.0f ms over the line'
          % (LOCAL, np.nanmin(t0) * 1e3, np.nanmax(t0) * 1e3,
             (np.nanmax(t0) - np.nanmin(t0)) * 1e3))
    lag_bot = 2.0 * float(np.nanmax(t0)) * 1.06
    near = near_trace(os.path.join(d, '%s_cube.npy' % LOCAL), S['n_ch'], dt,
                      S['band'])
    ac = autocorr_all(near, dt, lag_bot)
    grp = max(5, int(round(GROUP_M / DXSP[LOCAL])))
    step = max(2, grp // 2)
    cen, img = [], []
    for i in range(0, ac.shape[0] - grp + 1, step):
        cen.append(i + grp / 2)
        img.append(ac[i:i + grp].mean(0))
    img = np.array(img).T
    k0 = int(LAG_MIN / dt)
    img = img[k0:]
    img = img / (np.abs(img).max(axis=0, keepdims=True) + 1e-30)

    a = fig.add_subplot(gs[1, :])
    lim = np.percentile(np.abs(img), 99)
    im = a.imshow(img, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                  origin='lower', interpolation='nearest',
                  extent=[0, ac.shape[0], LAG_MIN * 1e3, lag_bot * 1e3])
    # Without a scale the colour carries no sign, and the sign is the whole
    # point: the lobe at t0 must be negative and the one at 2 t0 positive.
    cb = fig.colorbar(im, ax=a, pad=0.012, fraction=0.035)
    cb.set_label('autocorrelation, normalised per group', fontsize=LEG)
    cb.ax.tick_params(labelsize=TICK - 0.5)
    x = np.arange(t0.size)
    a.plot(x, t0 * 1e3, 'k-', lw=1.0, label='$t_0$ tracked')
    a.plot(x, 2 * t0 * 1e3, 'k:', lw=0.8, label='$2t_0$')
    a.axhline(float(np.median(t0)) * 1e3, color='0.35', ls='--', lw=0.8,
              label='a single global value')
    a.set_ylim(LAG_MIN * 1e3, lag_bot * 1e3)
    a.set_xlim(0, ac.shape[0])
    a.set_xlabel('shot')
    a.set_ylabel('lag (ms)')
    a.set_title('%s — the same autocorrelation in groups of %.0f m along the line'
                % (S['title'].split(' —')[0], GROUP_M))
    # Below the curves, not on top of them: at upper right the legend sat on the
    # 2*t0 line exactly where it is highest.
    a.legend(loc='lower right', ncol=3, framealpha=0.85, frameon=True,
             borderpad=0.3)

    for ext in FORMATS:
        os.makedirs('figures', exist_ok=True)
        p = os.path.join('figures', '%s.%s' % (NAME, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('wrote %s' % p)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'span_experiment')
