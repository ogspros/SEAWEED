#!/usr/bin/env python3
"""anchor_qc -- the stage-1 anchors checked against the raw onset of the near trace.

A diagnostic: it flags, it corrects nothing, and nothing in the workflow reads
what it writes.  The picker scores candidates by semblance along the water
hyperbola, which does not separate cycles: on 13 channels, moving t0 by one
period changes the differential moveout by a few ms, and on Viking the anchors
fell one cycle early or late in five groups of eleven, which the tracking then
turned into a step of about 20 ms.  The onset of the near trace needs no
tracking and no picker, so where it can be trusted it says whether every anchor
sits on the same cycle.

Onset: band-passed near trace, envelope, walking back from the envelope peak
until it drops below max(0.08 peak, 4 x RMS of the first 60 ms of the segment),
as find_origin does but without its guard.  The segment starts at the end of
the direct arrival (direct_end; 150 ms after it where the end cannot be
measured).  The onset is trusted block by block, only where the answer
does not depend on how it is measured: every variant of threshold, segment
start and detector must move it uniformly (within 2 samples of its median
shift on 80 per cent of the shots) and the onset must be smooth along the line.
Where it is not trusted the anchor is reported as not assessable, not given a
number: on MS-29 the envelope peak falls on events hundreds of ms below the
seafloor over most of the line.

Each anchor is compared, as an arrival time at the near offset, with the onset
of the shots around it; the median of those differences is the convention of
the line, and an anchor more than T_dom/2 from it is flagged as off its cycle.
T_dom = 4 x the first zero of the autocorrelation of the onset-aligned stack.

    python -m seaweed.anchor_qc line_cube.npy line_track.npz line_anchors.npz \\
        line_anchor_qc.npz

line_track.npz is read only for the sampling interval, the offsets and the
water velocity, as in anchors_from_consensus.
"""
import argparse
import json
import os

import numpy as np

from seaweed import code_version
from seaweed.run_real import BAND
from seaweed.wavelet_extractor_v2 import arrival_time, bandpass, envelope

# Variants that must leave the onset where it is, up to a uniform shift.  A
# threshold below the noise is not one of them: the walk back then runs into
# the noise and measures nothing (MEPA, where the noise is 0.2-0.3 of the peak),
# so the low-fraction variant keeps the noise floor.
VARIANTS = {"2x noise": dict(frac=0.0, k=2.0),
            "8x noise": dict(frac=0.0, k=8.0),
            "0.04 peak, noise floor kept": dict(frac=0.04, k=4.0),
            "0.16 peak": dict(frac=0.16, k=0.0),
            "segment 100 ms earlier": dict(lo_ms=-100.0),
            "segment 150 ms later": dict(lo_ms=150.0),
            "first crossing of 8x noise": dict(first=8.0)}
COHERENT = 0.8           # share of shots that must agree, per variant and block
# end of the direct arrival: the envelope of the stack of the near traces back
# under the find_origin threshold, plus its guard
DIRECT_HALF_MS, DIRECT_NOISE_MIN_MS, DIRECT_GUARD_MS, DIRECT_MAX_S = 50.0, 20.0, 8.0, 1.0


def direct_end(traces, dt, x0, vw, frac_peak=0.08, floor_k=4.0):
    """End of the direct arrival on the near trace, in s, and what decided it.

    traces: band-passed near traces, one per shot.  With a fixed geometry the
    direct arrives at x0 / vw on every shot, so the stack is their mean.  The
    peak is the envelope maximum within x0 / vw +- 50 ms; the threshold is the
    one of find_origin, max(0.08 peak, 4 x RMS of the noise before x0 / vw -
    50 ms when at least 20 ms of it are recorded); the end is the first sample
    after the peak back under it, plus find_origin's 8-ms guard.  NaN when no
    peak or no return under the threshold within 1 s."""
    st = np.mean(np.asarray(traces, float), axis=0)
    env = envelope(st)
    td = x0 / vw
    info = dict(t_direct_ms=td * 1e3)
    a = max(0, int(round((td - DIRECT_HALF_MS * 1e-3) / dt)))
    b = min(env.size, int(round((td + DIRECT_HALF_MS * 1e-3) / dt)) + 1)
    if b - a < 3:
        return np.nan, dict(info, note="no direct window")
    ipk = a + int(np.argmax(env[a:b]))
    thr = frac_peak * env[ipk]
    n_noise = int(round((td - DIRECT_HALF_MS * 1e-3) / dt))
    noise = n_noise * dt >= DIRECT_NOISE_MIN_MS * 1e-3
    if noise:
        thr = max(thr, floor_k * float(np.sqrt(np.mean(st[:n_noise] ** 2))))
    i, lim = ipk, min(env.size - 1, ipk + int(round(DIRECT_MAX_S / dt)))
    while i < lim and env[i] > thr:
        i += 1
    info.update(peak_ms=ipk * dt * 1e3, noise_floor=bool(noise))
    if env[i] > thr:
        return np.nan, dict(info, note="the envelope does not return under the threshold")
    t_end = i * dt + DIRECT_GUARD_MS * 1e-3
    return t_end, dict(info, end_ms=t_end * 1e3, duration_ms=(t_end - td) * 1e3)


def raw_onset(x, dt, t_lo, t_hi, frac=0.08, k=4.0, noise_ms=60.0, first=None,
              lo_ms=0.0):
    """Onset of the strongest event in [t_lo, t_hi] s, in seconds.

    Returns (onset, envelope peak, noise over peak); NaN for a dead trace.
    first: start from the first sample above first x the noise RMS, and take
    the largest envelope value within 60 ms after it as the peak.
    """
    a = max(0, int(round((t_lo + lo_ms * 1e-3) / dt)))
    b = min(len(x), int(round(t_hi / dt)))
    nn = max(3, int(round(noise_ms * 1e-3 / dt)))
    seg = x[a:b]
    if b - a <= nn + 3 or np.std(seg) < 1e-12:
        return np.nan, np.nan, np.nan
    env = envelope(seg)
    noise = float(np.sqrt(np.mean(seg[:nn] ** 2)))
    ipk = int(np.argmax(env))
    if first is not None:
        up = np.flatnonzero(env[nn:] > first * noise)
        if up.size:
            j = nn + int(up[0])
            ipk = j + int(np.argmax(env[j:j + int(round(0.060 / dt))]))
    thr = max(frac * env[ipk], k * noise)
    i = ipk
    while i > 0 and env[i] > thr:
        i -= 1
    return (a + i) * dt, (a + ipk) * dt, float(env[:nn].max() / env[ipk])


def trusted(onset, variants, dt, block):
    """Per shot: whether its block passes, and the per-block table."""
    S = onset.size
    loc = np.array([np.nanmedian(onset[max(0, i - 5):i + 6]) for i in range(S)])
    ok = np.zeros(S, bool)
    rows = []
    # a short tail joins the block before it rather than being judged alone
    starts = list(range(0, S, block))
    if len(starts) > 1 and S - starts[-1] < block // 2:
        starts.pop()
    for b0, b1 in zip(starts, starts[1:] + [S]):
        s = slice(b0, b1)
        live = ~np.isnan(onset[s])
        if live.sum() < 5:
            rows.append(dict(first=b0, last=s.stop - 1, worst=np.nan,
                             which="dead traces", smooth=np.nan, trusted=False))
            continue
        worst, which = 1.0, ""
        for name, v in variants.items():
            d = (v[s] - onset[s])[live & ~np.isnan(v[s])]
            share = (float(np.mean(np.abs(d - np.median(d)) <= 2 * dt + 1e-9))
                     if d.size else 0.0)
            if share < worst:
                worst, which = share, name
        smooth = float(np.mean(np.abs(onset[s] - loc[s])[live] <= 3 * dt + 1e-9))
        good = worst >= COHERENT and smooth >= COHERENT
        ok[s] = good & ~np.isnan(onset[s])
        rows.append(dict(first=b0, last=s.stop - 1, worst=worst, which=which,
                         smooth=smooth, trusted=bool(good)))
    return ok, rows


def t_dom(traces, onset, ok, dt, win_ms=200.0):
    """4 x the first zero of the autocorrelation of the onset-aligned stack."""
    n = int(round(win_ms * 1e-3 / dt))
    segs = [tr[int(round(o / dt)):int(round(o / dt)) + n]
            for tr, o, g in zip(traces, onset, ok) if g]
    segs = [s for s in segs if s.size == n]
    if len(segs) < 3:
        return np.nan
    st = np.mean(segs, axis=0)
    st = st - st.mean()
    ac = np.correlate(st, st, "full")[n - 1:]
    i = int(np.argmax(ac <= 0))
    if i < 1:
        return np.nan
    return 4e3 * dt * (i - 1 + ac[i - 1] / (ac[i - 1] - ac[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("track_in", help="only for dt, offsets and vw")
    ap.add_argument("anchors")
    ap.add_argument("qc_out")
    ap.add_argument("--block", type=int, default=100,
                    help="shots per block in which the onset is trusted or not")
    ap.add_argument("--local", type=int, default=10,
                    help="half-width, in shots, of the onset an anchor is "
                         "compared with; local rather than the whole group, "
                         "where the seafloor has relief")
    ap.add_argument("--t_hi", type=float, default=None,
                    help="end of the onset search, in s; default the end of the "
                         "picker's search, min(1.5, record / 2.2)")
    ap.add_argument("--tdom_ms", type=float, default=None,
                    help="T_dom in ms; default measured on the onset-aligned stack")
    a = ap.parse_args()

    g = np.load(a.cube, mmap_mode="r")
    z = np.load(a.track_in, allow_pickle=True)
    dt = float(z["dt"]); offs = np.asarray(z["offs"], float); vw = float(z["vw"])
    S, T = g.shape[0], g.shape[1]
    x0 = float(offs[0])
    t_hi = a.t_hi if a.t_hi else arrival_time(min(1.50, T * dt / 2.2), x0, vw, 0)

    traces = np.array([bandpass(np.asarray(g[s][:, 0], float), dt, BAND)
                       for s in range(S)])
    t_end, d_info = direct_end(traces, dt, x0, vw)
    t_lo = t_end if np.isfinite(t_end) else x0 / vw + 0.150
    print("direct arrival at %.0f ms, %s" % (x0 / vw * 1e3, "ends at %.0f ms" % (t_end * 1e3) if np.isfinite(t_end)
                                            else "end NOT MEASURABLE (%s): search from 150 ms after it" % d_info.get("note")))
    base = np.array([raw_onset(tr, dt, t_lo, t_hi) for tr in traces])
    onset = base[:, 0]
    variants = {name: np.array([raw_onset(tr, dt, t_lo, t_hi, **kw)[0]
                                for tr in traces])
                for name, kw in VARIANTS.items()}
    ok, rows = trusted(onset, variants, dt, a.block)
    td = a.tdom_ms if a.tdom_ms else t_dom(traces, onset, ok, dt)

    za = np.load(a.anchors, allow_pickle=True)
    ai = np.asarray(za["i"], int); at = np.asarray(za["t"], float)
    diff = np.full(ai.size, np.nan)
    for j, (i, t) in enumerate(zip(ai, at)):
        s = slice(max(0, i - a.local), min(S, i + a.local + 1))
        if ok[s].mean() >= 0.5:
            diff[j] = (arrival_time(float(t), x0, vw, 0)
                       - np.median(onset[s][ok[s]])) * 1e3
    assessed = ~np.isnan(diff)
    conv = float(np.median(diff[assessed])) if assessed.sum() >= 3 else np.nan
    status = np.where(~assessed | np.isnan(conv), "not assessable",
                      np.where(np.abs(diff - conv) > td / 2, "off cycle", "ok"))

    print("onset of the near trace (%.0f m): search %.0f-%.0f ms, trusted on "
          "%.0f%% of the shots" % (x0, t_lo * 1e3, t_hi * 1e3, 100 * ok.mean()))
    for r in rows:
        if not r["trusted"]:
            print("  shots %4d-%4d not trusted (worst variant %.2f, %s; smooth %.2f)"
                  % (r["first"], r["last"], r["worst"], r["which"], r["smooth"]))
    print("T_dom %.1f ms; convention of the line (anchor - onset, median) %s"
          % (td, "%+.0f ms" % conv if np.isfinite(conv)
             else "undefined (fewer than three anchors assessable)"))
    for i, t, d, s in zip(ai, at, diff, status):
        print("  anchor %5d  t0 %.3f s  %s  %s" % (
            i, t, "%+5.0f ms" % d if np.isfinite(d) else "    -   ", s))
    print("%d anchors: %d ok, %d off cycle, %d not assessable"
          % (ai.size, np.sum(status == "ok"), np.sum(status == "off cycle"),
             np.sum(status == "not assessable")))

    params = dict(script="anchor_qc", code=code_version(),
                  cube=os.path.basename(a.cube),
                  track_in=os.path.basename(a.track_in),
                  anchors=os.path.basename(a.anchors), block=a.block,
                  local=a.local, t_lo=t_lo, t_hi=t_hi, t_dom_ms=td, direct_end=d_info,
                  convention_ms=conv, coherent=COHERENT,
                  variants=list(VARIANTS))
    np.savez(a.qc_out, onset=onset, onset_trusted=ok, anchor_i=ai,
             anchor_t=at, anchor_minus_onset_ms=diff, anchor_status=status,
             params=json.dumps(params))
    print("wrote %s" % a.qc_out)


if __name__ == "__main__":
    main()
