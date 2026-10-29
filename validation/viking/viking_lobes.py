#!/usr/bin/env python3
"""
viking_lobes.py -- why does the Viking extraction have a larger first positive
lobe, a smaller second one and a longer coda than the recorded signature?

Three explanations are on the table, and they predict different things.

  GHOST AT THE WRONG ANGLE.  The reference applies the receiver ghost at
  normal incidence, 2 z / V_w.  The near channel sits at 262 m, so over
  ~375 m of water the ray leaves the vertical by ~19 degrees and the ghost
  delay is 2 z cos(theta) / V_w, some 6 per cent shorter.  If this is the
  cause, (a) a reference built with the ghost at the true angle agrees better
  than the vertical one, and (b) the imbalance grows channel by channel as
  the angle opens.

  SOMETHING THE ANGLE DOES NOT CHANGE.  The seafloor is not a single
  interface, so the water-bottom reflection carries the response of the
  first few metres of sediment; and the streamer and the far-field
  measurement had different recording filters.  Either is a fixed filter on
  every channel.  If this is the cause, the imbalance is the SAME on every
  channel and the angle-corrected reference does not help.

  CONTAMINATION.  Sub-bottom energy inside the window.  It mostly lives in the
  coda, varies with the stretch of line, and should not produce a stable
  imbalance of the main lobes across ten disjoint combs.

The script scores each directory of comb wavelets against the vertical and the
angle-corrected reference, and measures the lobes on both.

    python3 validation/viking/viking_lobes.py FarField.dat viking_blocks viking_ch1 viking_ch2 \\
        viking_ch4 viking_ch8 --ch 0 1 2 4 8

--ch gives, for each directory, the channel index it was extracted on, which
fixes the offset and hence the angle.  The reference chain is imported from
viking_phase_test.py, so the two scripts cannot drift apart.
"""
import argparse
import glob
import os

import numpy as np
from scipy.signal import resample

from viking_phase_test import PAT, bp, reference, ncc

UP = 8                                  # display/measurement upsampling


def far_field_angle(path, delay, z_rec, vw, dt_ff, cos_t):
    """FarField.dat with the receiver ghost at incidence angle theta, applied
    as an exact fractional delay in the frequency domain (the 1 ms grid of the
    file cannot resolve a 6 per cent change in a 14 ms delay)."""
    ff = np.loadtxt(path)[delay:]
    n = 1 << int(np.ceil(np.log2(ff.size + 400)))
    F = np.fft.rfft(ff, n)
    f = np.fft.rfftfreq(n, dt_ff)
    tau = 2.0 * z_rec * cos_t / vw
    return np.fft.irfft(F * (1.0 - np.exp(-2j * np.pi * f * tau)), n)[:ff.size + 60]


def best(ex, ff_g, cfg):
    return max((ncc(ex, reference(ff_g, L, cfg)), L)
               for L in range(-cfg["maxlag_ms"], cfg["maxlag_ms"] + 1))


def lobes(x, t_trough_ms, dt_ms):
    """First positive lobe, main trough and second positive lobe, relative to
    the trough, measured on an upsampled copy.  The trough is searched within
    +-8 ms of the reference trough so that both signals are measured on the
    same event."""
    y = resample(x, x.size * UP)
    d = dt_ms / UP
    t = np.arange(y.size) * d
    win = (t >= t_trough_ms - 8) & (t <= t_trough_ms + 8)
    it = np.flatnonzero(win)[np.argmin(y[win])]
    T = -y[it]
    pre = (t >= t[it] - 16) & (t < t[it])
    post = (t > t[it]) & (t <= t[it] + 26)
    p1 = y[pre].max() / T if pre.any() else np.nan
    p2 = y[post].max() / T if post.any() else np.nan
    coda = (t > t[it] + 60)
    e = (y ** 2)
    return p1, p2, e[coda].sum() / e.sum(), t[it]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("farfield")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ch", type=int, nargs="+", required=True,
                    help="channel index of each directory, same order")
    ap.add_argument("--fam", default="C")
    ap.add_argument("--near", type=float, default=262.0)
    ap.add_argument("--dx", type=float, default=25.0)
    ap.add_argument("--D", type=float, default=375.0,
                    help="water depth for the incidence angle")
    ap.add_argument("--z_rec", type=float, default=10.0)
    ap.add_argument("--vw", type=float, default=1448.0)
    ap.add_argument("--delay", type=int, default=27)
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 65.0])
    ap.add_argument("--maxlag_ms", type=int, default=30)
    ap.add_argument("--plot", default="viking_lobes.png")
    a = ap.parse_args()
    if len(a.ch) != len(a.dirs):
        raise SystemExit("--ch needs one channel index per directory")

    dt, dt_ff = 4e-3, 1e-3
    cfg = dict(dt=dt, dt_ff=dt_ff, up=4, nw=50, lp_hz=115.0,
               lo=a.band[0], hi=a.band[1], maxlag_ms=a.maxlag_ms)
    ff_v = far_field_angle(a.farfield, a.delay, a.z_rec, a.vw, dt_ff, 1.0)

    rows, curves = [], []
    for d, c in zip(a.dirs, a.ch):
        x = a.near + c * a.dx
        th = np.arctan(x / (2 * a.D))
        ff_a = far_field_angle(a.farfield, a.delay, a.z_rec, a.vw, dt_ff,
                               np.cos(th))
        ws = []
        for p in sorted(glob.glob(os.path.join(d, "*_wavelet.npy"))):
            m = PAT.match(os.path.basename(p))
            if m and m.group("fam") == a.fam:
                w = np.load(p).astype(float)
                if w.size == cfg["nw"]:
                    ws.append(bp(w, dt, *a.band))
        if not ws:
            print("%-16s no %s wavelets" % (d, a.fam))
            continue
        sv = [best(w, ff_v, cfg) for w in ws]
        sa = [best(w, ff_a, cfg) for w in ws]
        L = int(np.median([s[1] for s in sv]))
        rv = reference(ff_v, L, cfg)
        ra = reference(ff_a, int(np.median([s[1] for s in sa])), cfg)
        # the reference trough fixes where the lobes are measured
        tt = np.argmin(resample(rv, rv.size * UP)) * dt * 1e3 / UP
        W = np.mean([w / np.linalg.norm(w) for w in ws], axis=0)
        le = lobes(W, tt, dt * 1e3)
        lv = lobes(rv, tt, dt * 1e3)
        la = lobes(ra, tt, dt * 1e3)
        rows.append((d, c, x, np.degrees(th), len(ws),
                     np.mean([s[0] for s in sv]), np.mean([s[0] for s in sa]),
                     le, lv, la))
        curves.append((d, c, W / np.abs(W).max(), rv / np.abs(rv).max()))

    print("\nagreement with the recorded signature, band %g-%g Hz"
          % tuple(a.band))
    print("%-16s %3s %6s %6s %3s %9s %9s" % ("dir", "ch", "x(m)", "theta", "n",
                                            "vertical", "at angle"))
    for r in rows:
        print("%-16s %3d %6.0f %6.1f %3d %9.4f %9.4f"
              % (r[0][:16], r[1], r[2], r[3], r[4], r[5], r[6]))

    print("\nlobes relative to the main trough (first positive, second "
          "positive) and coda energy beyond trough + 60 ms")
    print("%-16s %22s %22s %22s" % ("dir", "extracted", "reference vertical",
                                     "reference at angle"))
    for r in rows:
        f = lambda l: "%5.2f %5.2f %8.3f" % (l[0], l[1], l[2])
        print("%-16s %22s %22s %22s" % (r[0][:16], f(r[7]), f(r[8]), f(r[9])))

    print("\nreading it")
    print("  'at angle' clearly above 'vertical', and the extracted lobes "
          "drifting with the\n  channel -> the ghost angle.  Lobes the same "
          "on every channel and no gain\n  at angle -> a filter the angle does "
          "not change: seafloor response or\n  recording filters.")

    if a.plot and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 2.8),
                               squeeze=False)
        t = np.arange(cfg["nw"] * UP) * dt * 1e3 / UP
        for k, (d, c, W, rv) in enumerate(curves):
            ax[0, k].plot(t, resample(rv, rv.size * UP), "k", lw=1.3,
                          label="recorded (vertical ghost)")
            ax[0, k].plot(t, resample(W, W.size * UP), "C3", lw=1.1,
                          label="extracted, mean of combs")
            ax[0, k].axhline(0, color="0.8", lw=0.5)
            ax[0, k].set_title("%s  (channel %d)" % (os.path.basename(d), c),
                               fontsize=8)
            ax[0, k].set_xlabel("time (ms)")
        ax[0, 0].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=200)
        print("\nwrote %s" % a.plot)


if __name__ == "__main__":
    main()
