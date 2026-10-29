"""
fig_deghost.py -- Figure: the extracted MS-29 wavelet against the independently
processed section of the same line (Brancatelli et al. 2022).

Run in the directory that holds the spectra and the wavelet:

    python3 validation/archive/fig_deghost.py Spettro_NO_deghost.txt Spettro_deghost.txt \
            span_experiment/ms29_reference_full_line_wavelet.npy

Produces, in ./figures/ :

    fig_deghost.{pdf,png}   three panels: the spectra, the recovered operator
                            with its fit, and the depth sweep

Panels
  (a) amplitude spectra in dB over the comparison band: their stack before and
      after deghosting, our extracted wavelet, and our wavelet after the same
      operator has been applied
  (b) the ratio of the two published spectra -- which is the operator that was
      applied -- with the fit of the damped inverse ghost.  The damping is
      measured here, not chosen, which is the whole point
  (c) agreement between the deghosted wavelet and the deghosted section, in a
      window around the notch, against the depth of the operator

Everything meant to be adjusted sits in the CONFIG block: sizes in centimetres,
colours, fonts, band, output formats.  The style block matches
make_paper_figures_vik.py so the figures look like one another.

The numbers printed to stdout are the ones quoted in the text; check them
against it if either is edited.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ============================== CONFIG =======================================
CM = 1 / 2.54
FIG_SIZE = (17.0 * CM, 6.0 * CM)       # full page width, three panels
FORMATS = ('pdf', 'png')
DPI = 400
OUTDIR = 'figures'

FONT = dict(family='sans-serif', size=8)
LABEL_SIZE = 8
TITLE_SIZE = 8.5
TICK_SIZE = 7
LEGEND_SIZE = 6.5

C_STACK = '#000000'                    # their section, before deghosting
C_STACK_DG = '#7f7f7f'                 # their section, after
C_WAV = '#4c72b0'                      # our extracted wavelet
C_WAV_DG = '#c44e52'                   # ours, after their operator
C_FIT = '#c44e52'
C_MARK = '#2f8a4f'
BAND_ALPHA = 0.10

BAND = (8.0, 45.0)                     # comparison band for MS-29
VW_ECHOS = 1510.0                      # water velocity used in the 2022 deghost
VW_HERE = 1514.5                       # measured here from the direct arrival
Z_REPORT = 25.0                        # acquisition report
NOTCH_HALFWIDTH = 8.0                  # window for the depth sweep, Hz
DT = 4e-3                              # MS-29 sample interval
DB_FLOOR = -32.0                       # bottom of panel (a)
# =============================================================================


# ------------------------------------------------------------------ loading
def load_spectrum(path):
    """Echos amplitude-spectrum export: No., X [Hz], Y, Normalized Y.

    The Y column is the one to use.  Normalized Y is scaled to each file's own
    maximum, so ratios taken on it are wrong by the ratio of the two maxima.
    """
    f, y = [], []
    with open(path) as fh:
        for line in fh:
            p = line.split('\t')
            if len(p) >= 3 and p[0].strip().isdigit():
                f.append(float(p[1]))
                y.append(float(p[2]))
    if not f:
        sys.exit('no data rows in %s' % path)
    return np.asarray(f), np.asarray(y)


def spectrum_df(f):
    """Frequency step over the whole axis: the exported X column is rounded to
    three decimals, so f[1] - f[0] is not accurate enough to set the FFT length."""
    return (f[-1] - f[0]) / (f.size - 1)


def ghost(f, z, vw):
    return np.abs(2.0 * np.sin(2.0 * np.pi * f * z / vw))


def fit_operator(f, ratio, vw, lo, hi):
    m = (f >= lo) & (f <= hi)

    def resid(p):
        c, z, eps = p
        return np.log(c / np.sqrt(ghost(f[m], z, vw) ** 2 + eps ** 2)) \
            - np.log(ratio[m])

    r = least_squares(resid, [1.0, Z_REPORT, 0.3],
                      bounds=([1e-3, 5.0, 1e-3], [1e3, 60.0, 5.0]))
    c, z, eps = r.x
    dof = max(m.sum() - 3, 1)
    sd = np.sqrt(np.diag(np.linalg.inv(r.jac.T @ r.jac)
                         * np.sum(r.fun ** 2) / dof))
    model = c / np.sqrt(ghost(f[m], z, vw) ** 2 + eps ** 2)
    return dict(c=c, z=z, eps=eps, z_sd=sd[1], mask=m,
                r=float(np.corrcoef(model, ratio[m])[0, 1]),
                n=int(m.sum()))


def db(x, ref):
    return 20.0 * np.log10(np.maximum(x, 1e-30) / ref)


def apply_style():
    plt.rcParams.update({
        'font.family': FONT['family'], 'font.size': FONT['size'],
        'axes.labelsize': LABEL_SIZE, 'axes.titlesize': TITLE_SIZE,
        'xtick.labelsize': TICK_SIZE, 'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in FORMATS:
        p = os.path.join(OUTDIR, '%s.%s' % (name, ext))
        fig.savefig(p, dpi=DPI, bbox_inches='tight')
        print('  wrote %s' % p)


# ------------------------------------------------------------------ figure
def main(p_nodg, p_dg, p_wav):
    apply_style()
    lo, hi = BAND

    f, A = load_spectrum(p_nodg)
    f2, B = load_spectrum(p_dg)
    if not np.allclose(f, f2):
        sys.exit('the two spectra are on different frequency axes')
    df = spectrum_df(f)

    ratio = B / np.maximum(A, 1e-30)
    fit = fit_operator(f, ratio, VW_ECHOS, lo, hi)
    fit_here = fit_operator(f, ratio, VW_HERE, lo, hi)
    f_notch = VW_ECHOS / (2 * fit['z'])

    print('operator: z_r = %.2f +/- %.2f m at Vw = %.1f m/s (%.2f m at %.1f), '
          'notch %.2f Hz, eps = %.3f, r = %.4f on %d points'
          % (fit['z'], fit['z_sd'], VW_ECHOS, fit_here['z'], VW_HERE,
             f_notch, fit['eps'], fit['r'], fit['n']))
    print('total correction applied by the operator: %.1f dB'
          % (20 * np.log10(ratio[fit['mask']].max()
                           / ratio[fit['mask']].min())))

    # the wavelet, on the same frequency grid
    w = np.load(p_wav).astype(float)
    nfft = int(round(1.0 / (DT * df)))
    fw = np.fft.rfftfreq(nfft, DT)
    W = np.abs(np.fft.rfft(w, nfft))
    n = min(fw.size, f.size)
    fw, W, A, B, ratio = fw[:n], W[:n], A[:n], B[:n], ratio[:n]
    print('wavelet: %d samples, %.0f ms, zero-padded to %d for df = %.5f Hz'
          % (w.size, w.size * DT * 1e3, nfft, df))

    def operator(z):
        return fit['c'] / np.sqrt(ghost(fw, z, VW_ECHOS) ** 2 + fit['eps'] ** 2)

    Wdg = W * operator(fit['z'])

    m = (fw >= lo) & (fw <= hi)
    mn = (fw >= max(lo, f_notch - NOTCH_HALFWIDTH)) \
        & (fw <= min(hi, f_notch + NOTCH_HALFWIDTH))

    print('agreement over %g-%g Hz: wavelet vs undeghosted stack r = %.4f; '
          'deghosted vs deghosted r = %.4f'
          % (lo, hi,
             np.corrcoef(W[m] / W[m].max(), A[m] / A[m].max())[0, 1],
             np.corrcoef(Wdg[m] / Wdg[m].max(), B[m] / B[m].max())[0, 1]))

    fig, ax = plt.subplots(1, 3, figsize=FIG_SIZE)

    # -------------------------------------------------- (a) the four spectra
    for x, c, ls, lab in [(A, C_STACK, '-', 'section, no deghost'),
                          (B, C_STACK_DG, '-', 'section, deghosted'),
                          (W, C_WAV, '-', 'extracted wavelet'),
                          (Wdg, C_WAV_DG, '--', 'wavelet, deghosted')]:
        ax[0].plot(fw[m], db(x[m], x[m].max()), color=c, ls=ls, lw=1.1,
                   label=lab)
    ax[0].axvline(f_notch, color=C_MARK, ls=':', lw=0.9)
    ax[0].text(f_notch + 0.7, 1.2, '%.1f Hz' % f_notch, color=C_MARK,
               fontsize=LEGEND_SIZE, va='top')
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(DB_FLOOR, 3)
    ax[0].set_xlabel('frequency (Hz)')
    ax[0].set_ylabel('amplitude (dB)')
    ax[0].set_title('(a)', loc='left', fontweight='bold')
    ax[0].legend(loc='lower center')
    ax[0].grid(alpha=0.25, lw=0.4)

    # ------------------------------------------- (b) the recovered operator
    mm = fit['mask'][:n]
    ax[1].plot(fw[mm], ratio[mm], color=C_STACK, lw=1.1,
               label='measured ratio')
    ax[1].plot(fw[mm],
               fit['c'] / np.sqrt(ghost(fw[mm], fit['z'], VW_ECHOS) ** 2
                                  + fit['eps'] ** 2),
               color=C_FIT, ls='--', lw=1.1, label='damped inverse ghost')
    ax[1].axvline(f_notch, color=C_MARK, ls=':', lw=0.9)
    ax[1].set_xlim(lo, hi)
    # headroom above the peak so the legend never sits on the curve
    ax[1].set_ylim(None, ratio[mm].max() * 1.22)
    ax[1].set_xlabel('frequency (Hz)')
    ax[1].set_ylabel('gain of the applied operator')
    ax[1].set_title('(b)', loc='left', fontweight='bold')
    ax[1].legend(loc='upper left')
    ax[1].grid(alpha=0.25, lw=0.4)
    # the fitted parameters are quoted in the caption rather than boxed on the
    # panel: the curve occupies both lower corners and the peak the centre, so
    # any box collides either with it or with the legend

    # ----------------------------------------------------- (c) depth sweep
    zs = np.arange(20.0, 31.001, 0.05)
    rr = np.array([np.corrcoef((W * operator(z))[mn]
                               / (W * operator(z))[mn].max(),
                               B[mn] / B[mn].max())[0, 1] for z in zs])
    zbest = zs[rr.argmax()]
    flat = zs[rr > rr.max() - 0.01]
    print('depth sweep: maximum at %.2f m, flat within 0.01 over %.2f-%.2f m'
          % (zbest, flat.min(), flat.max()))
    print('  the top is %.1f m wide at that level, so the %.2f m between the '
          'sweep and\n  the acquisition report is agreement between two '
          'estimates and not an\n  uncertainty of %.2f m'
          % (flat.max() - flat.min(), abs(zbest - Z_REPORT),
             abs(zbest - Z_REPORT)))
    if zbest <= zs[0] + 0.1 or zbest >= zs[-1] - 0.1:
        print('  WARNING: the maximum sits at the edge of the swept range; '
              'widen it before believing the number')

    ax[2].axvspan(flat.min(), flat.max(), color=C_MARK, alpha=BAND_ALPHA, lw=0)
    ax[2].plot(zs, rr, color=C_WAV_DG, lw=1.2)
    ax[2].axvline(Z_REPORT, color=C_STACK, ls='-', lw=0.9)
    ax[2].axvline(fit['z'], color=C_STACK_DG, ls='--', lw=0.9)
    ax[2].plot([zbest], [rr.max()], 'o', color=C_MARK, ms=4, mec='k', mew=0.4)
    ax[2].set_xlim(zs[0], zs[-1])
    ax[2].set_xlabel('operator streamer depth (m)')
    ax[2].set_ylabel('agreement around the notch')
    ax[2].set_title('(c)', loc='left', fontweight='bold')
    ax[2].grid(alpha=0.25, lw=0.4)
    ax[2].set_ylim(None, rr.max() + 0.10 * (rr.max() - rr.min()))
    ax[2].text(0.97, 0.04,
               'report %.1f m\noperator %.2f m\nwavelet %.2f m\n'
               'top flat over %.1f m'
               % (Z_REPORT, fit['z'], zbest, flat.max() - flat.min()),
               transform=ax[2].transAxes, ha='right', va='bottom',
               fontsize=LEGEND_SIZE,
               bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))

    fig.tight_layout(pad=0.4)
    save(fig, 'fig_deghost')
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
