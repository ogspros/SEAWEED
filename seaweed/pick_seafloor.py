"""pick_seafloor -- pick the seafloor along a whole line from the local
autocorrelation, and write it as a tracking (--load_track) or as anchors
(--load_anchors) for run_real.

Why
---
A single t0 measured at the start of a line and used to centre the semblance
search everywhere fails where the relief is large: on IT90-65, 76 km with the
seafloor between about 500 and 1100 ms, 44 anchors fell against the edge of
the search window and the tracking locked onto another event where the window
no longer contained the seafloor.  Lowering the stretch limit does not help,
since the semblance then stops resolving the velocity.  The autocorrelation
needs no prior and no velocity, so it can be measured locally, in a sliding
window along the line.

On --lag
--------
The search range of the period is the one parameter this script does not set
itself.  Three automatic rules were tried and each traded one failure for
another: half the envelope maximum breaks on gained data (on MS-29 it returned
0.12-3.03 s); a multiple of the envelope median is crossed late on the same
data, cutting the shallow end of the line; proportional margins readmit a
sub-bottom impostor.  So --lag is read off the section once per survey and
checked on the QC figure.  The envelope estimate is printed as a suggestion,
with a warning when the supplied range is much wider.

Method
------
1. Local period.  In sliding groups of shots, autocorrelate the near trace and
   take the most negative lobe in the lag range whose double is positive.
   Primary and multiples are periodic with period t0, geology is not, and the
   free-surface bounce makes the lobe at t0 negative and the one at 2 t0
   positive.  This gives a coarse t0(shot) that follows the relief.
2. Per-shot refinement.  Within a narrow window around the coarse t0, take the
   peak of the envelope of the near trace, corrected to zero offset.
3. Continuity.  Picks departing from a moving median by more than a threshold
   are replaced by the median rather than interpolated.

The output is a QC figure and a .npz for run_real, so the seafloor can be
inspected once and every run of an experiment uses the same pick.

--load_track skips stage [2] of run_real, where the t0_shift is applied, so
the shift is applied here: pass the same --t0_shift as to run_real.

Usage
-----
    python -m seaweed.pick_seafloor --cube line_cube.npy --near 125 \\
        --vw 1453 --dxsp 25 --lag 0.45 1.30 --t0_shift -25 \\
        --out line_track.npz --save_anchors line_anchors.npz
"""
import argparse
import json
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from seaweed import code_version

try:
    from seaweed.seafloor_robust import semblance_bottom
    HAVE_SEMBLANCE = True
except Exception:                       # noqa: BLE001
    HAVE_SEMBLANCE = False


def bp(x, dt, f1, f2, axis=-1, order=4):
    ny = 0.5 / dt
    return sosfiltfilt(butter(order, [f1 / ny, min(f2 / ny, 0.98)],
                              btype='band', output='sos'), x, axis=axis)


def autocorr_all(near, dt, max_lag_s):
    """Normalised autocorrelation of every near trace, by FFT.

    Vectorised because the per-shot pass below needs one for every shot of a
    3000-shot line, and a Python loop over np.correlate would dominate the run.
    """
    S, T = near.shape
    nf = 1 << int(np.ceil(np.log2(2 * T)))
    X = np.fft.rfft(near - near.mean(1, keepdims=True), nf, axis=1)
    ac = np.fft.irfft(np.abs(X) ** 2, nf, axis=1)[:, :int(max_lag_s / dt) + 2]
    return ac / (ac[:, :1] + 1e-30)


def pick_lobe(ac_row, klo, khi, min_depth=0.0):
    """Lag of the seafloor period: the deepest negative lobe whose double is
    positive.

    Primary and multiples are periodic with period t0; the free-surface bounce
    makes the lobe at t0 negative and the one at 2 t0 positive.  A laterally
    persistent sub-bottom sequence also produces negative lobes, at the delays
    between its own reflectors, and on a synthetic with an invariant sub-bottom
    one of those beat the seafloor (352 ms picked against a true 815).  Those
    lobes have no positive lobe at twice their lag, so the check removes them.

    Returns (lag in samples, depth, at_edge) or None.
    """
    n = len(ac_row)
    seg = ac_row[klo:khi + 1]
    if seg.size < 5 or not np.isfinite(seg).all():
        return None
    loc = np.where((seg[1:-1] < seg[:-2]) & (seg[1:-1] < seg[2:]))[0] + 1
    if loc.size == 0:
        loc = np.array([int(np.argmin(seg))])
    best = None
    for j in loc:
        k = klo + int(j)
        depth = float(-ac_row[k])
        if depth < min_depth:
            continue
        k2 = 2 * k
        if k2 >= n - 1 or ac_row[k2] <= 0:      # the multiple must be positive
            continue
        if best is None or depth > best[1]:
            best = (k, depth, (j <= 1) or (j >= seg.size - 2))
    if best is None:
        return None
    k, depth, edge = best
    if 0 < k < n - 1:
        y0, y1, y2 = ac_row[k - 1], ac_row[k], ac_row[k + 1]
        den = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
    return k, depth, edge


def local_period(ac, dt, lag_lo, lag_hi, group, step, min_depth):
    """Coarse t0 along the line, from group-averaged autocorrelations.

    Averaging over a group before picking, rather than picking per shot and
    averaging after, keeps one bad shot from moving the answer; the price is
    that relief inside the group smears the lobe, which is why this pass is
    only the coarse one and is refined per shot afterwards.
    """
    S = ac.shape[0]
    klo, khi = int(lag_lo / dt), int(lag_hi / dt)
    cen, t0, dep = [], [], []
    for a in range(0, S, step):
        b = min(S, a + group)
        # a group needs at least half its nominal size and three shots: at wide
        # shot spacing 800 m can be as few as five shots
        if b - a < max(3, min(group, group // 2 + 1)):
            continue
        r = pick_lobe(ac[a:b].mean(0), klo, khi, min_depth)
        if r is None:
            continue
        k, d, edge = r
        if edge:
            continue
        cen.append(0.5 * (a + b - 1)); t0.append(k * dt); dep.append(d)
    return np.array(cen), np.array(t0), np.array(dep)


def semblance_refine(g, dt, offs, t0_coarse, n_anchor, half_ms, vw_guess,
                     stretch_max):
    """Semblance anchors, with the search window centred on the local period.

    The autocorrelation measures a periodicity, the semblance a moveout: they
    are independent discriminators of the seafloor, and only the semblance
    tells a reflector travelling at water velocity from one that merely
    repeats.  It also returns Vw, which the autocorrelation cannot.  Centring
    each search on the local period replaces the single depth prior that fails
    on lines with large relief, and makes each search cheaper.
    """
    S = g.shape[0]
    idx = np.unique(np.linspace(0, S - 1, max(4, n_anchor)).astype(int))
    ai, at, av, aw = [], [], [], []
    for si in idx:
        c = t0_coarse[si] * 1e3
        r = semblance_bottom(np.asarray(g[si]).T, dt, offs,
                             V_range=(vw_guess - 70, vw_guess + 70), nV=29,
                             t0_min_ms=max(50.0, c - half_ms),
                             t0_max_ms=c + half_ms,
                             stretch_max=stretch_max, return_width=False)
        if r is None:
            continue
        # a peak against either end of the window is the search running out of room
        if abs(r['t0'] * 1e3 - (c - half_ms)) < 2 * dt * 1e3 or \
           abs(r['t0'] * 1e3 - (c + half_ms)) < 2 * dt * 1e3:
            continue
        ai.append(si); at.append(r['t0']); av.append(r['V'])
        aw.append(r.get('V_halfwidth', np.nan))
    return (np.array(ai), np.array(at), np.array(av), np.array(aw))


def auto_lag(near, env, dt, near_off, vw, margin=0.06, k_med=6.0,
             max_t0=3.0):
    """Bracket the seafloor two-way time without using the autocorrelation.

    The autocorrelation cannot bound itself: where a laterally persistent
    sub-bottom sequence produces a deeper negative lobe than the seafloor, the
    majority of groups agree on the wrong answer and no amount of
    outlier-repair helps, because the outliers are the correct ones.  The bound
    has to come from a different property, and the obvious one is amplitude:
    the seafloor is the first strong event after the direct arrival.  Sub-bottom
    reflectors are later and weaker, and the multiple is later still.

    For each shot, take the first sample after the direct arrival at which the
    envelope exceeds half of its maximum, and bracket the distribution over the
    whole line.
    """
    S, T = env.shape
    t_dir = near_off / vw
    k0 = int((t_dir + margin) / dt)
    if k0 >= T - 10:
        return None
    # search only the first seconds after the mute: a marine seafloor is within
    # about 3 s two-way (2250 m) wherever this workflow applies, and the gained
    # tail of the record would otherwise set the maximum and the median
    k1 = min(T, k0 + int(max_t0 / dt))
    seg = env[:, k0:k1]
    # threshold on the median of the trace, not on its maximum: on data
    # delivered with gain (MS-29) the largest envelope sample sits in the
    # amplified tail, and half the maximum is crossed at once or not at all.  A
    # multiple of the median is a background level, and gain, which flattens
    # the envelope, keeps it stable.
    med = np.median(seg, axis=1, keepdims=True) + 1e-30
    thr = k_med * med
    above = seg > thr
    ok = above.any(axis=1)
    first = np.argmax(above, axis=1) + k0
    ok &= (first > k0) & (first < k1 - 5)
    if ok.sum() < 0.5 * S:
        return None
    # the crossing is the arrival at the near offset; it is converted to zero
    # offset rather than allowed for by a margin, which a sub-bottom event can
    # slip inside
    t = first[ok] * dt
    t = np.sqrt(np.maximum(t ** 2 - (near_off / vw) ** 2, (dt * 5) ** 2))
    lo, hi = np.percentile(t, [1, 99])
    lo, hi = max(0.05, lo - 0.06), hi + 0.12
    # a bracket wider than about a second is not a bracket, and everything
    # downstream would inherit it
    if hi - lo > 1.2 or hi > 0.8 * T * dt:
        return None
    return lo, hi


def moving_median(x, w):
    w = max(3, int(w) | 1)
    h = w // 2
    p = np.pad(x, h, mode='edge')
    return np.array([np.median(p[i:i + w]) for i in range(len(x))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cube', required=True, help='.npy written by --save_cube')
    ap.add_argument('--near', type=float, required=True, help='near offset, m')
    ap.add_argument('--vw', type=float, default=1500.0)
    ap.add_argument('--t0_shift', type=float, default=0.0,
                    help='ms, baked into the output because --load_track skips '
                         'the stage that would otherwise apply it')
    ap.add_argument('--band', type=float, nargs=2, default=[15.0, 80.0])
    ap.add_argument('--lag', nargs=2, required=True, metavar=('T0MIN', 'T0MAX'),
                    help='range of the seafloor two-way time in seconds, read '
                         'off the section: a seafloor at D metres is at about '
                         '2*D/1500 s.  Pass "auto auto" to use the estimate made '
                         'from the envelope, but read the note in the header '
                         'first: it is a suggestion, not a substitute')
    ap.add_argument('--group_m', type=float, default=800.0,
                    help='length of an autocorrelation group IN METRES. This is '
                         'the parameter that transfers between surveys: 31 shots '
                         'is 775 m at 25 m shot spacing and 6.2 km at 200 m, and '
                         'a group that spans a scarp averages across it. What '
                         'matters is relief per unit distance, not per shot: on a '
                         'line whose seafloor changes by a couple of metres per '
                         'kilometre a much longer group is safe and gives a far '
                         'more robust autocorrelation, so raise this where the '
                         'section is smooth')
    ap.add_argument('--dxsp', type=float, default=None,
                    help='shot-point spacing in metres, needed for --group_m')
    ap.add_argument('--group', type=int, default=None,
                    help='override --group_m with an explicit number of shots')
    ap.add_argument('--step', type=int, default=None, help='group stride in shots')
    ap.add_argument('--repair_ms', type=float, default=120.0,
                    help='half-width of the second group-level pass, which '
                         're-picks each group around the median of its '
                         'neighbours. This is what makes the result insensitive '
                         'to --lag')
    ap.add_argument('--min_depth', type=float, default=0.04,
                    help='minimum depth of the negative lobe for a group to be '
                         'accepted; rejects groups where no periodicity is found')
    ap.add_argument('--refine_lag_ms', type=float, default=100.0,
                    help='half-width of the per-shot lag search around the coarse '
                         'curve')
    ap.add_argument('--refine_ms', type=float, default=60.0,
                    help='half-width of the per-shot envelope search around the '
                         'local period')
    ap.add_argument('--median', type=int, default=21, help='continuity median width')
    ap.add_argument('--jump_ms', type=float, default=40.0,
                    help='departure from the moving median above which a pick is '
                         'replaced by the median')
    ap.add_argument('--n_ch', type=int, default=3,
                    help='nearest channels averaged to form the near trace')
    ap.add_argument('--semblance', choices=['off', 'check', 'correct'],
                    default='off',
                    help='"check" runs the velocity semblance at anchors and only '
                         'REPORTS how far it lies from the pick, changing nothing; '
                         '"correct" also ties the pick to it. Check is the useful '
                         'mode: the semblance measures moveout and the '
                         'autocorrelation measures periodicity, so agreement '
                         'between them is evidence from two independent '
                         'properties that both are on the seafloor. Needs '
                         '--dx')
    ap.add_argument('--dx', type=float, default=None,
                    help='channel spacing, needed with --semblance')
    ap.add_argument('--n_anchor', type=int, default=200)
    ap.add_argument('--semb_half_ms', type=float, default=150.0,
                    help='half-width of the semblance search around the local '
                         'period. This replaces the global D_init window')
    ap.add_argument('--stretch', type=float, default=0.45)
    ap.add_argument('--save_anchors', default=None,
                    help='write sparse anchors for run_real --load_anchors. '
                         'This is the preferred output: run_real then does its '
                         'own 2D pilot cross-correlation, window rejection and '
                         'drift smoothing, which is what removes the spikes that '
                         'a finished t0 curve still carries')
    ap.add_argument('--n_anchor_out', type=int, default=200)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fig', default=None)
    a = ap.parse_args()

    meta = np.load(a.cube + '.meta.npz', allow_pickle=True)
    dt = float(meta['dt'])
    g = np.load(a.cube, mmap_mode='r')
    S, T, C = g.shape
    print('%s: %d shots x %d channels, dt %.1f ms, record %.0f ms'
          % (os.path.basename(a.cube), S, C, dt * 1e3, T * dt * 1e3))

    near = np.asarray(g[:, :, :max(1, a.n_ch)]).mean(axis=2).astype(float)
    near = bp(near, dt, *a.band, axis=1)
    env = np.abs(hilbert(near, axis=1))

    rng = auto_lag(near, env, dt, a.near, a.vw)
    if rng is not None:
        print('    envelope suggests --lag %.2f %.2f (a suggestion only)'
              % rng)
    else:
        print('    the envelope could not bracket the seafloor '
              '(gained data, or a long record)')
    auto = any(str(v).lower() == 'auto' for v in a.lag)
    if auto:
        if rng is None:
            raise SystemExit('--lag auto asked for, but the envelope estimate '
                             'failed.  Read the range off the section instead.')
        a.lag = list(rng)
        print('    using it, as requested')
    else:
        a.lag = [float(v) for v in a.lag]
        print('    --lag %.2f %.2f supplied' % (a.lag[0], a.lag[1]))
    if rng is not None and (a.lag[0] < rng[0] - 0.2 or a.lag[1] > rng[1] + 0.4):
        print('    NOTE: the supplied range is much wider than the envelope '
              'suggests. A wide\n    range lets a laterally persistent '
              'sub-bottom sequence win with a negative\n    lobe at a shorter '
              'lag than the seafloor; narrow it if the pick looks wrong.')

    # --- 1. coarse period, group-averaged -----------------------------------
    if a.group is None:
        if a.dxsp is None:
            raise SystemExit('give --dxsp (shot spacing in metres) so that '
                             '--group_m can be converted to shots, or set '
                             '--group explicitly')
        a.group = max(5, int(round(a.group_m / a.dxsp)))
        a.step = a.step or max(2, a.group // 2)
        print('    group = %d shots (%.0f m at %.0f m shot spacing), stride %d'
              % (a.group, a.group_m, a.dxsp, a.step))
    a.step = a.step or max(2, a.group // 2)

    ac = autocorr_all(near, dt, 2 * a.lag[1] + 0.05)
    cen, t0g, dep = local_period(ac, dt, a.lag[0], a.lag[1], a.group, a.step,
                                 a.min_depth)
    if cen.size < 3:
        raise SystemExit('the autocorrelation found nothing usable: check --lag, '
                         '--band and --min_depth')
    ngrp = len(range(0, S, a.step))
    print('\n[1] coarse pass: %d of %d groups accepted (%d rejected at the edge '
          'of the\n    search range or with too shallow a lobe)'
          % (cen.size, ngrp, ngrp - cen.size))
    print('    t0 %.0f-%.0f ms; a single global value would have been %.0f ms'
          % (t0g.min() * 1e3, t0g.max() * 1e3, np.median(t0g) * 1e3))
    print('    relief along the line: %.0f ms = %.0f m of water'
          % ((t0g.max() - t0g.min()) * 1e3, (t0g.max() - t0g.min()) * a.vw / 2))
    lo_edge = np.mean(t0g < a.lag[0] + 0.08)
    if lo_edge > 0.3:
        print('    WARNING: %.0f%% of the groups sit within 80 ms of the lower end'
              % (100 * lo_edge))
        print('    of --lag.  A laterally persistent sub-bottom sequence produces')
        print('    negative lobes at the delays between its own reflectors, and')
        print('    those are at SHORTER lags than the seafloor; on a synthetic with')
        print('    an invariant sub-bottom they beat the true period in most groups.')
        print('    Narrow --lag to the range the section actually shows before')
        print('    trusting this, and check the multiple in the QC figure.')
    # Group-level repair.  A group that picked the wrong lobe is an outlier
    # against its neighbours, so it is re-picked with the search restricted to a
    # window around their median.  The --lag range then only has to be right
    # for most groups, not for every one.  Two passes: a third changes nothing
    # once the outliers are gone, and unbounded iteration of a search centred on
    # its own estimate is how this kind of scheme diverges.
    klo_all, khi_all = int(a.lag[0] / dt), int(a.lag[1] / dt)
    wrep = int(a.repair_ms * 1e-3 / dt)
    for it in range(2):
        ref = moving_median(t0g, 5)
        moved = 0
        for j, c in enumerate(cen):
            aa, bb = int(c - a.group / 2), int(c + a.group / 2)
            kk = int(round(ref[j] / dt))
            r = pick_lobe(ac[max(0, aa):min(S, bb)].mean(0),
                          max(klo_all, kk - wrep), min(khi_all, kk + wrep),
                          a.min_depth * 0.5)
            if r is None:
                continue
            if abs(r[0] * dt - t0g[j]) > 0.5 * a.repair_ms * 1e-3:
                moved += 1
            t0g[j] = r[0] * dt
        if it == 0:
            print('    group repair: %d of %d groups re-picked around their '
                  'neighbours' % (moved, cen.size))

    t0g = moving_median(t0g, 5)
    t0_coarse = np.interp(np.arange(S), cen, t0g)

    # --- 2. per-shot period, searched only near the coarse curve ------------
    # the search is restricted so that the lobe cannot hop to the multiple or
    # to a sub-bottom periodicity; the coarse curve is smooth, so this is a
    # bounded refinement
    win = int(a.refine_lag_ms * 1e-3 / dt)
    t0_shot = t0_coarse.copy()
    nrej = 0
    for s_ in range(S):
        c = int(round(t0_coarse[s_] / dt))
        r = pick_lobe(ac[s_], max(1, c - win), min(ac.shape[1] - 2, c + win),
                      a.min_depth * 0.5)
        if r is None or r[2]:
            nrej += 1
            continue
        t0_shot[s_] = r[0] * dt
    print('\n[2] per-shot pass within +-%.0f ms of the coarse curve: %d of %d '
          'shots kept\n    the coarse value (%.1f%%)'
          % (a.refine_lag_ms, nrej, S, 100.0 * nrej / S))
    pick = t0_shot

    # --- 2b. envelope refinement -------------------------------------------
    dtt = np.sqrt(pick ** 2 + (a.near / a.vw) ** 2) - pick
    half = int(a.refine_ms * 1e-3 / dt)
    for s_ in range(S):
        c = int(round((pick[s_] + dtt[s_]) / dt))
        lo, hi = max(0, c - half), min(T, c + half + 1)
        if hi - lo >= 5:
            k = lo + int(np.argmax(env[s_, lo:hi]))
            pick[s_] = k * dt - dtt[s_]

    # --- 2c. semblance anchors, centred on the local period ----------------
    vw = a.vw
    if a.semblance != 'off':
        if not HAVE_SEMBLANCE:
            raise SystemExit('seaweed.seafloor_robust could not be imported; '
                             'drop --semblance or check the installation')
        offs = (a.near + a.dx * np.arange(C)).astype(float)
        ai, at, av, aw = semblance_refine(g, dt, offs, pick, a.n_anchor,
                                          a.semb_half_ms, a.vw, a.stretch)
        if ai.size < 4:
            raise SystemExit('the semblance accepted fewer than four anchors; '
                             'widen --semb_half_ms or check --near/--dx')
        vw_semb = float(np.median(av))
        dev = (np.interp(ai, np.arange(S), pick) - at) * 1e3
        print('\n[2c] semblance %s on %d of %d anchors (window +-%.0f ms around '
              'the local period)'
              % (a.semblance.upper(), ai.size, max(4, a.n_anchor), a.semb_half_ms))
        print('     Vw %.1f m/s from the semblance (peak half-width %.0f m/s); '
              '%.1f supplied' % (vw_semb, np.nanmedian(aw), a.vw))
        print('     autocorrelation minus semblance: %+.1f +- %.1f ms '
              '(median %+.1f)'
              % (dev.mean(), dev.std(ddof=1) if dev.size > 1 else 0.0,
                 np.median(dev)))
        agree = float(np.mean(np.abs(dev - np.median(dev)) < 30))
        print('     %.0f%% of anchors within 30 ms of the common offset'
              % (100 * agree))
        if abs(np.median(dev)) > 40 or agree < 0.7:
            print('     THE TWO DISAGREE.  They measure different properties --')
            print('     a periodicity against a moveout -- so a constant offset of')
            print('     a few tens of ms is expected (onset against envelope peak).')
            print('     A large median, or scattered anchors, means one of them is')
            print('     on the wrong event, and the pick should not be used until')
            print('     the QC figure says which.')
        else:
            print('     Consistent: two independent properties of the wavefield')
            print('     place the seafloor at the same time.')
        if abs(vw_semb - a.vw) / max(a.vw, 1) > 0.04:
            print('     NOTE: the semblance velocity departs from the supplied one')
            print('     by more than 4 per cent, which is 4 per cent on the water')
            print('     depth. The semblance averages the whole column, the direct')
            print('     arrival the surface layer: in a stratified column they')
            print('     legitimately differ.')
        if a.semblance == 'correct':
            drift = moving_median(np.interp(np.arange(S), ai, at) - pick, a.median)
            pick = pick + drift
            vw = vw_semb
            print('     pick tied to the semblance anchors, Vw taken from them')

    # --- 3. continuity -----------------------------------------------------
    for _ in range(2):
        med = moving_median(pick, a.median)
        bad = np.abs(pick - med) > a.jump_ms * 1e-3
        pick[bad] = med[bad]
    n_bad = int(bad.sum())
    resid = (pick - moving_median(pick, a.median)) * 1e3
    print('\n[2] per-shot refinement and continuity:')
    print('    %d of %d picks replaced by the moving median (%.1f%%)'
          % (n_bad, S, 100.0 * n_bad / S))
    print('    residual about the moving median: %.1f ms r.m.s.'
          % np.sqrt(np.mean(resid ** 2)))
    print('    departure from the coarse curve: %.1f +- %.1f ms'
          % (np.mean((pick - t0_coarse) * 1e3), np.std((pick - t0_coarse) * 1e3)))

    t0 = pick + a.t0_shift * 1e-3
    D = t0 * vw / 2
    print('\n[3] seafloor: t0 %.0f-%.0f ms -> D %.0f-%.0f m (shift %+.0f ms applied)'
          % (t0.min() * 1e3, t0.max() * 1e3, D.min(), D.max(), a.t0_shift))

    off = meta['off_hdr']
    plain = (int, float, str, bool, list, type(None))
    params = dict(script='pick_seafloor', code=code_version(),
                  **{k: v for k, v in vars(a).items() if isinstance(v, plain)})
    np.savez(a.out, t0=t0, vw=vw, D_init=float(np.median(D)), dt=dt,
             offs=(off if off.size else np.array([])), S=S,
             segy=str(meta['segy']) if 'segy' in meta else '',
             shot0=meta['shot0'] if 'shot0' in meta else -1,
             nshot=meta['nshot'] if 'nshot' in meta else -1,
             t0_shift=a.t0_shift, picked_by='pick_seafloor.py',
             params=json.dumps(params))
    print('    written to %s  (use it with --load_track)' % a.out)

    if a.save_anchors:
        # anchors are taken from the smoothed curve, not the per-shot pick: the
        # spikes are what the cross-correlation tracking replaces, and they must not
        # constrain it
        sm = moving_median(t0, max(a.median, 11))
        idx = np.unique(np.linspace(0, S - 1, max(8, a.n_anchor_out)).astype(int))
        np.savez(a.save_anchors, i=idx, t=sm[idx], V=np.full(idx.size, vw),
                 params=json.dumps(params))
        print('    %d anchors written to %s (every %d shots)'
              % (idx.size, a.save_anchors, max(1, S // idx.size)))
        print('    prefer this to --load_track: run_real then keeps its own')
        print('    cross-correlation tracking and drift smoothing')

    # --- QC figure ---------------------------------------------------------
    figp = a.fig or (os.path.splitext(a.out)[0] + '_pick.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    dec = max(1, S // 2000)
    sec = near[::dec].T
    lim = np.percentile(np.abs(sec), 98)
    tmax = min(T, int(1.8 * (t0.max() + 0.4) / dt))
    ax[0].imshow(sec[:tmax], aspect='auto', cmap='gray_r', vmin=-lim, vmax=lim,
                 extent=[0, S, tmax * dt * 1e3, 0])
    ax[0].plot(np.arange(S), t0 * 1e3, 'r', lw=1.2, label='pick (zero offset)')
    ax[0].plot(np.arange(S), 2 * t0 * 1e3, 'y:', lw=1.0, label='first multiple')
    ax[0].set_ylabel('ms'); ax[0].legend(loc='upper right', fontsize=8)
    ax[0].set_title('seafloor pick from the local autocorrelation', loc='left')
    ax[1].plot(np.arange(S), t0_coarse * 1e3, '0.6', lw=2,
               label='local autocorrelation')
    ax[1].plot(np.arange(S), t0 * 1e3, 'r', lw=1, label='refined + continuity')
    ax[1].set_xlabel('shot'); ax[1].set_ylabel('t0 (ms)')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figp, dpi=150, bbox_inches='tight')
    print('    QC figure: %s' % figp)
    print('\nLook at the figure before using the file.  The red line must follow')
    print('the first strong reflector everywhere, and the yellow line must sit on')
    print('the multiple; if the yellow line misses, the pick is on the wrong event.')


if __name__ == '__main__':
    main()
