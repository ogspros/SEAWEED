r"""
test_avo.py — do the wavelets from different channels differ by more than the
ghosts predict?

THE QUESTION
------------
The wavelet extracted at offset x is

    P(w, th) = S(w) Gs(w, th) Gr(w, th) Rwb(th) exp(-i w tp) / rp^alpha

and three things move together with the incidence angle th: the two ghost
operators, whose delay shortens as 2 z cos(th) / c so their notches climb; the
water-bottom reflection coefficient, which varies with angle through Zoeppritz;
and the contamination, because the sub-bottom moveout differs from the primary's.
A raw comparison between channels mixes all three.

Two of them, however, are CALCULABLE.  The ghosts follow from the towing depths,
which are in the acquisition report, and the spreading from the geometry.  So
divide the wavelet at channel n by the wavelet at channel 0, correct for the
predicted ghost ratio, and what is left is Rwb(th_n)/Rwb(th_0) together with the
difference in contamination.

WHAT THE TEST DECIDES
---------------------
Whether it is worth pursuing.  If the measured ratio is what the ghosts alone
predict, there is no angular signal to extract and the idea is closed in half an
hour.  If it departs systematically -- monotonically with angle, and by more
than the scatter between adjacent channels -- then the shape of the extracted
wavelet carries an AVO response of the seafloor.

That would matter because Rwb is currently the weakest quantity the workflow
reports: it is estimated from an amplitude ratio, and on MS-29 it comes out at
1.13, which is impossible.  An Rwb read from the SHAPE rather than the amplitude
would be immune to the gain that breaks the present estimate.

The test does not decide whether such an estimate would be accurate.  It decides
only whether there is anything there to measure.

    python3 validation/viking/test_avo.py <prefix> --channels 0 20 40 60 --offsets 125 625 1125 1625

where <prefix>_ch<NN>_wavelet.npy were produced by run_real with --n_ch 1 and
--ch_start <NN>.
"""
import argparse
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bp(x, dt, f1, f2, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x)


def ghost(f, z, theta, c):
    """Free-surface ghost for a source or receiver at depth z, angle theta.

    The delay is the two-way vertical path, 2 z cos(theta) / c: at oblique
    incidence the ghost leg is shorter in time, so the notch climbs.  A rigid
    treatment would also carry the obliquity factor in the reflection at the
    surface, which for a free surface is -1 at every angle.
    """
    return 1.0 - np.exp(-1j * 2 * np.pi * f * 2 * z * np.cos(theta) / c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--channels", type=int, nargs="+", required=True)
    ap.add_argument("--offsets", type=float, nargs="+", required=True,
                    help="offset in metres of each channel listed")
    ap.add_argument("--depth", type=float, required=True, help="water depth, m")
    ap.add_argument("--vw", type=float, default=1500.)
    ap.add_argument("--z_src", type=float, required=True)
    ap.add_argument("--z_rec", type=float, required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--band", type=float, nargs=2, default=[8., 65.])
    a = ap.parse_args()

    if len(a.channels) != len(a.offsets):
        raise SystemExit("--channels and --offsets must have the same length")

    ws, th = [], []
    for ch, off in zip(a.channels, a.offsets):
        p = "%s_ch%02d_wavelet.npy" % (a.prefix, ch)
        if not os.path.exists(p):
            raise SystemExit("missing: %s" % p)
        ws.append(bp(np.load(p).astype(float), a.dt, *a.band))
        # incidence at the seafloor for a two-way path 2D and offset x
        th.append(np.arctan2(off / 2.0, a.depth))
    n = len(ws[0])
    nf = 4096
    f = np.fft.rfftfreq(nf, a.dt)
    m = (f >= a.band[0]) & (f <= a.band[1])

    print("channel   offset    angle    ghost notch (Hz)")
    for ch, off, t in zip(a.channels, a.offsets, th):
        print("  %4d  %8.0f  %6.1f deg   src %5.0f   rec %5.0f"
              % (ch, off, np.degrees(t),
                 a.vw / (2 * a.z_src * np.cos(t)),
                 a.vw / (2 * a.z_rec * np.cos(t))))

    A = [np.fft.rfft(w, nf) for w in ws]
    # A ratio is only meaningful where both spectra carry energy.  The ghost
    # itself suppresses the low end -- at 12 Hz with a 5 m cable it is down to
    # a fifth of its peak -- and there the ratio is numerical noise: on a
    # synthetic built from ghosts alone it read -25 dB where the prediction was
    # -8.  Frequencies below the threshold are reported but excluded from the
    # residual, and their number is printed, since a test that quietly drops
    # half its samples is worse than one that fails.
    amp = np.array([np.abs(x) for x in A])
    peak = amp[:, m].max(axis=1, keepdims=True)
    strong = np.all(amp > 0.10 * peak, axis=0)
    print("\nspectral ratio channel n / channel %d, in dB:" % a.channels[0])
    print("  the PREDICTED column is the ghost change alone; MEASURED is what")
    print("  the extracted wavelets do; the residual is what the ghosts do not")
    print("  explain, and is where an angular response of the seafloor would be.")
    hdr = "  %6s" % "Hz"
    for ch in a.channels[1:]:
        hdr += "%22s" % ("ch %d: meas / pred" % ch)
    print(hdr)
    resid = {ch: [] for ch in a.channels[1:]}
    for ft in (12, 18, 24, 30, 38, 46, 55, 62):
        if ft < a.band[0] or ft > a.band[1]:
            continue
        i = np.argmin(np.abs(f - ft))
        line = "  %6.0f" % ft
        for k, ch in enumerate(a.channels[1:], start=1):
            meas = 20 * np.log10(np.abs(A[k][i]) / (np.abs(A[0][i]) + 1e-30))
            gp = (ghost(f[i], a.z_src, th[k], a.vw)
                  * ghost(f[i], a.z_rec, th[k], a.vw))
            g0 = (ghost(f[i], a.z_src, th[0], a.vw)
                  * ghost(f[i], a.z_rec, th[0], a.vw))
            pred = 20 * np.log10(np.abs(gp) / (np.abs(g0) + 1e-30))
            line += "%11.1f /%9.1f" % (meas, pred)
            if strong[i]:
                resid[ch].append(meas - pred)
        print(line + ("" if strong[i] else "   (weak, excluded)"))

    lo = f[m][strong[m]].min() if strong[m].any() else float('nan')
    hi = f[m][strong[m]].max() if strong[m].any() else float('nan')
    print("\nresidual after removing the predicted ghost change,")
    print("over %.0f-%.0f Hz where every channel has at least a tenth of its "
          "peak:" % (lo, hi))
    for ch in a.channels[1:]:
        r = np.array(resid[ch])
        print("  channel %3d : mean %+6.2f dB, s.d. %5.2f, max |r| %5.2f"
              % (ch, r.mean(), r.std(), np.abs(r).max()))
    print("\n  A residual that is flat and near zero means the channels differ")
    print("  only by their ghosts, and there is nothing further to extract.")
    print("  A residual that grows monotonically with angle, by more than the")
    print("  scatter between adjacent channels, is an angular response and")
    print("  worth pursuing.  Anything oscillatory is contamination, which")
    print("  arrives at a different delay on each channel.")


if __name__ == "__main__":
    main()
