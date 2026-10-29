"""
fig_span.py — the cross-survey span experiment, for the paper.

Three panels, because the result is a negative one and a negative result needs
to be shown from more than one angle to be believed:

  (a) agreement against span, four surveys.  Four shapes, no ordering.
  (b) agreement against frequency at the longest span each line allows.  Each
      survey is plotted over its own band, which is also why the bands differ:
      comparing a survey outside the band where its energy lies measures noise
      against noise.
  (c) agreement between neighbouring stretches against distant ones.  This is
      the mechanism: where the answer depends on WHERE along the line the shots
      were taken, more line helps; where it does not, nothing does.  The number
      above each pair is the Pearson correlation between the agreement of a
      pair of stretches and the distance between them, over all pairs of that
      survey: negative where distant stretches agree less.

All numbers are agreements among extractions made on DIFFERENT stretches at a
fold of 100 shots.  Interleaved subsets of one stretch are excluded: they agree
at 0.98-1.00 whatever the span and would flatten every curve.

Everything meant to be adjusted is in the CONFIG block.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================== CONFIG =======================================
CM = 1 / 2.54
FIGSIZE = (17.0 * CM, 6.0 * CM)
FORMATS = ('pdf', 'png')
DPI = 400
OUTDIR = 'figures'
NAME = 'fig_span'

COL = {'Viking Graben': '#000000', 'MS-29': '#c44e52',
       'IT90-65': '#4c72b0', 'SISOC-MEPA': '#2f8a4f'}
MRK = {'Viking Graben': 'D', 'MS-29': 's', 'IT90-65': 'o', 'SISOC-MEPA': '^'}

FONT = 8
LABEL, TITLE, TICK, LEG = 8, 8.5, 7, 6.6
# =============================================================================

# span (km), agreement, s.d., number of stretches
SPAN = {
    'Viking Graben': [(2.5, 0.633, 0.174, 10)],
    'SISOC-MEPA':          [(2.5, 0.862, 0.060, 8), (5.0, 0.911, 0.042, 8),
                      (7.5, 0.934, 0.025, 6)],
    'MS-29':          [(19.8, 0.891, 0.050, 8), (39.8, 0.891, 0.041, 4)],
    'IT90-65':       [(2.5, 0.915, 0.031, 8), (5.0, 0.929, 0.035, 8),
                      (10.0, 0.939, 0.023, 7), (15.0, 0.939, 0.026, 5)],
}

# band centre (Hz), agreement, at the LONGEST span each line allows
BANDS = {
    'Viking Graben': ([13, 23, 33, 43, 56.5],
                      [0.867, 0.665, 0.870, 0.689, 0.809], 2.5),
    'MS-29':          ([11.5, 18.5, 26, 34, 41.5],
                      [0.929, 0.961, 0.964, 0.964, 0.922], 39.8),
    'IT90-65':       ([13, 23, 33, 43, 56.5],
                      [0.986, 0.919, 0.971, 0.959, 0.928], 15.0),
    'SISOC-MEPA':          ([22.5, 37.5, 52.5, 67.5, 87.5],
                      [0.932, 0.930, 0.961, 0.853, 0.913], 7.5),
}

# adjacent half, distant half, Pearson, at the SHORTEST span of each line
SEP = {
    'Viking Graben': (0.758, 0.505, -0.83, 2.5),
    'SISOC-MEPA':          (0.876, 0.836, -0.31, 2.5),
    'IT90-65':       (0.908, 0.928, +0.23, 2.5),
    'MS-29':          (0.882, 0.905, +0.10, 19.8),
}

ORDER = ['Viking Graben', 'SISOC-MEPA', 'MS-29', 'IT90-65']


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': FONT,
        'axes.labelsize': LABEL, 'axes.titlesize': TITLE,
        'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'legend.fontsize': LEG, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'lines.linewidth': 1.1,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def main():
    style()
    fig, ax = plt.subplots(1, 3, figsize=FIGSIZE)

    # ---------------------------------------------------------- (a) vs span
    a = ax[0]
    for s in ORDER:
        d = np.array([(x, y, e) for x, y, e, _ in SPAN[s]])
        a.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2], marker=MRK[s], ms=4,
                   color=COL[s], capsize=2, lw=1.1, mec='k', mew=0.4, label=s)
    a.set_xscale('log')
    a.set_xticks([2.5, 5, 10, 20, 40])
    a.set_xticklabels(['2.5', '5', '10', '20', '40'])
    a.set_xlabel('line span covered (km)')
    a.set_ylabel('agreement between stretches')
    a.set_ylim(0.4, 1.0)
    a.set_title('(a)', loc='left', fontweight='bold')
    a.legend(loc='lower right')
    a.grid(alpha=0.25, lw=0.4)

    # ------------------------------------------------------- (b) vs frequency
    a = ax[1]
    for s in ORDER:
        f, v, sp = BANDS[s]
        a.plot(f, v, marker=MRK[s], ms=4, color=COL[s], mec='k', mew=0.4,
               label='%s (%.1f km)' % (s, sp))
    a.set_xlabel('band centre (Hz)')
    a.set_ylabel('agreement in band')
    a.set_ylim(0.4, 1.0)
    a.set_title('(b)', loc='left', fontweight='bold')
    a.legend(loc='lower right')
    a.grid(alpha=0.25, lw=0.4)

    # --------------------------------------------------- (c) position effect
    a = ax[2]
    x = np.arange(len(ORDER))
    w = 0.22
    # Points joined by a segment, not bars: a bar drawn from an axis that starts
    # at 0.4 invites the eye to compare areas that mean nothing, and the first
    # version did exactly that.
    for i, s in enumerate(ORDER):
        adj, dis, r, _ = SEP[s]
        a.plot([i - w, i + w], [adj, dis], '-', color=COL[s], lw=1.0, zorder=1)
        a.plot([i - w], [adj], 'o', mfc='none', mec=COL[s], mew=1.2, ms=6,
               zorder=2)
        a.plot([i + w], [dis], 'o', color=COL[s], ms=6, mec='k', mew=0.4,
               zorder=2)
        a.text(i, max(adj, dis) + 0.05, '$r = %+.2f$' % r, ha='center',
               fontsize=LEG, color='0.25')
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker='o', mfc='none', mec='0.3', mew=1.2,
                             ls='none', ms=6, label='neighbouring stretches'),
                      Line2D([], [], marker='o', color='0.3', ls='none', ms=6,
                             label='distant stretches')],
             loc='upper left', bbox_to_anchor=(-0.02, 1.03), ncol=1)
    a.set_xticks(x)
    short = {'Viking Graben': 'Viking', 'SISOC-MEPA': 'MEPA'}
    a.set_xticklabels([short.get(s, s) for s in ORDER], rotation=20, ha='right')
    a.set_xlim(-0.6, len(ORDER) - 0.4)
    a.set_ylim(0.4, 1.22)
    a.set_ylabel('agreement')
    a.set_title('(c)', loc='left', fontweight='bold')

    fig.tight_layout(pad=0.4)
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(OUTDIR, '%s.%s' % (NAME, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('wrote %s' % p)


if __name__ == '__main__':
    main()
