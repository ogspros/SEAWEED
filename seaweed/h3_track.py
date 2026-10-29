#!/usr/bin/env python3
"""h3_track -- the onset tracking: the seafloor from dense onset anchors, with
the stage-1 tracking as a prior only.

The stage-1 tracking can be right to within a few ms over long stretches and
wrong by tens of ms over others, and those slow errors are what the multiple
windows, and R_wb, cannot absorb.  The raw onset of the near traces, where it
can be measured, follows manual picks to an IQR of a few ms.  The onset
tracking therefore takes the absolute position from onsets, picked around the
stage-1 curve, and keeps the cross-correlation only to carry it between them.
Every threshold below is a physical quantity of the line: T_dom, the shot
spacing, the 500-m tracking window.

  1. prior     the stage-1 tracking, corrected block by block (100 shots) where
               an earlier event with its own water-layer multiple shows that it
               sits on a later event
  2. anchors   one every round(250 m / shot spacing) shots: the onset within
               +-2.5 T_dom of the prior.  Variant h2: channel 0.  Variant h2m:
               the median of the moveout-corrected onsets of channel indices
               1-3, kept only when the three agree within 2 samples.  An onset
               counts when its envelope peak exceeds 4 x the noise RMS and the
               walk back stops inside the window
  3. fallback  where at least 3 consecutive anchor positions are unreliable,
               the stage-1 anchors of that stretch are used, moved to the onset
               convention by the line median of (onset anchor - prior); the
               stretch is flagged, with the causes of its failed positions.
               Where the onset tracking is not applicable to the whole unit
               (T_dom not measurable on the provisional onsets, or fewer than 2
               anchors), the stage-1 tracking it was given is the output,
               unchanged, and the unit is flagged with the reason; the run does
               not fail
  4. tracking  the cross-correlation of run_real, drift-corrected on these
               anchors with a moving median over +-n_group/2 shots
  5. jitter    on every shot the h2m onset replaces the tracked time where it is
               reliable and within T_dom/2 of it; no further smoothing
  6. convention the median lateness of the onset rule, calibrated on 24
               synthetic calibration records as the picker's 9 ms was, is
               subtracted once from the output: from every shot and from the
               saved anchors

The operating domain is flagged, never acted on: the separation between the
direct arrival and the seafloor at the near offset, shot by shot on the output,
against max(S_STAR x T_dom, end of the direct).  s* = 5 T_dom comes from
synthetic records, where the direct lasts about 1.4 T_dom; on real lines the
direct, with a slow component near 11 Hz, can last 9-11 T_dom, and the end of
the direct is then the active bound.  The separation is the smallest of three,
shot by shot: the output tracking, the stage-1 anchors interpolated along the
line, and the first energy after the end of the direct (anchor_qc.raw_onset,
first crossing of 8 x the noise); a tracking locked on a later event would
otherwise look deeper, and in the domain.  The flag reports which of the three
is active.  All three are measured after the end of the direct, so a seafloor
inside a long direct is invisible to them; a fourth, the water-layer
reverberation period from the autocorrelation of the near traces
(estimate_t0_autocorr, defaults: no tracking, no direct rejection), puts a shot
outside when it places the seafloor before the end of the direct.  Its search
starts at 1.2 wavelet lengths (240 ms two-way), so it cannot report a shallower
seafloor.

A cycle check (a block whose anchors jump by k T_dom from its neighbours) is
reported, and moves the stage-1 anchors of a fallback stretch in that block.

The output has the layout of run_real --save_track, so the extraction runs on
it with --load_track.

    python -m seaweed.h3_track cube.npy stage1_track.npz stage1_anchors.npz \\
        h3_track.npz --dxsp 25 [--variant h2m] [--n_group N] [--n_ch_track 3]
"""
import argparse
import json
import os
import sys

import numpy as np

from seaweed import code_version
from seaweed.anchor_qc import direct_end, raw_onset
from seaweed.anchor_qc import t_dom as onset_tdom
from seaweed.run_real import BAND
from seaweed.wavelet_extractor_v2 import (arrival_time, bandpass, envelope, estimate_t0_autocorr,
                                          track_seafloor_xcorr)

FRAC, K_NOISE, NOISE_MS = 0.08, 4.0, 60.0     # the onset rule of find_origin
M_TDOM = 2.5                                  # onset search, +- M_TDOM x T_dom
AGREE_SAMP = 2                                # h2m: the three channels within 2 samples
H2M_CH = (1, 2, 3)
ANCHOR_M, WINDOW_M = 250.0, 500.0
BLOCK = 100
MIN_STRETCH = 3                               # unreliable anchor positions in a row
PROVISIONAL_MS = 80.0                         # onset search before T_dom is known
MULT_SEARCH_MS = 80.0                         # matched-filter search, as run_real
DIRECT_GAP_MS = 150.0                         # where the end of the direct cannot be measured (anchor_qc)
# median of (onset anchor - true t0) on 24 synthetic calibration records, one
# per main-grid cell and used in no evaluation; the threshold rule is early, by
# about 0.4 T_dom, against the true seafloor
CONVENTION_MS = {"h2": -11.0, "h2m": -13.0}
# operating domain: separation from the direct >= S_STAR x T_dom; a line with
# more than OUT_SHARE of its shots below is outside, partially outside when a
# 100-shot block has its median below (isolated shots are only counted)
S_STAR, OUT_SHARE = 5.0, 0.5


# why an onset is not reliable, as counted in the fallback flags
WHY = ("outside the data", "below 4x the noise", "walk back outside the window", "h2m channels disagree")


def onset_why(x, dt, t_c, half):
    """onset() with the cause when it is not reliable: (onset s, reliable, cause
    or "")."""
    n0 = int(round(NOISE_MS * 1e-3 / dt))
    a = int(round((t_c - half) / dt)); b = int(round((t_c + half) / dt))
    if a - n0 < 0 or b > len(x) or b - a < 3:
        return np.nan, False, WHY[0]
    seg = np.asarray(x[a - n0:b], float)
    if np.std(seg) < 1e-12:
        return np.nan, False, WHY[0]
    env = envelope(seg)
    noise = float(np.sqrt(np.mean(seg[:n0] ** 2)))
    ipk = n0 + int(np.argmax(env[n0:]))
    thr = max(FRAC * env[ipk], K_NOISE * noise)
    i = ipk
    while i > n0 and env[i] > thr:
        i -= 1
    above, inside = bool(env[ipk] > K_NOISE * noise), bool(i > n0)
    return (a - n0 + i) * dt, above and inside, ("" if above and inside else WHY[1] if not above else WHY[2])


def onset(x, dt, t_c, half):
    """Onset of the strongest event within t_c +- half (s), noise from the 60 ms
    before the window.  Returns (onset s, reliable): reliable when the envelope
    peak exceeds K_NOISE x the noise RMS and the walk back ends inside the window."""
    return onset_why(x, dt, t_c, half)[:2]


def vertical(t, x, vw):
    return np.sqrt(max(t * t - (x / vw) ** 2, 1e-12))


def shot_onset_why(gb, s, dt, offs, vw, t0p, half, variant):
    """shot_onset() with the cause when it is not reliable."""
    chans = (0,) if variant == "h2" else H2M_CH
    tv = []
    for c in chans:
        if c >= gb.shape[2]:
            return np.nan, False, WHY[0]
        o, ok, why = onset_why(gb[s, :, c], dt, arrival_time(t0p, offs[c], vw), half)
        if not ok:
            return np.nan, False, why
        tv.append(vertical(o, offs[c], vw))
    if np.ptp(tv) > AGREE_SAMP * dt + 1e-9:
        return np.nan, False, WHY[3]
    return float(np.median(tv)), True, ""


def shot_onset(gb, s, dt, offs, vw, t0p, half, variant):
    """Vertical-time onset of shot s around the prior t0p, and whether it is reliable."""
    return shot_onset_why(gb, s, dt, offs, vw, t0p, half, variant)[:2]


SEP_SOURCES = ("tracking", "stage-1 anchors", "first energy after the direct")


def first_energy(traces, dt, t_end, t_hi):
    """Per shot, the first energy after the end of the direct on the near trace
    (band-passed): anchor_qc.raw_onset from t_end, first crossing of 8 x the
    noise RMS of its first 60 ms; NaN where it cannot be measured."""
    if not np.isfinite(t_end):
        return np.full(len(traces), np.nan)
    return np.array([raw_onset(np.asarray(x, float), dt, t_end, t_hi, first=8.0)[0] for x in traces])


def water_autocorr_t0(graw, dt, offs):
    """t0 per shot (s) from the autocorrelation of the near-trace reverberation:
    estimate_t0_autocorr with its defaults, on every shot; NaN where it fails."""
    S = graw.shape[0]
    try:
        t, _ = estimate_t0_autocorr(np.asarray(graw, float), dt, offs)
    except Exception:
        t = None
    if t is None:
        return np.full(S, np.nan)
    t = np.asarray(t, float)
    return t if t.size == S else np.full(S, np.nan)


def validity_domain(t0, x0, vw, T, direct_ms=float("nan"), source="h3_track", anchors=None, first_onset=None,
                    autocorr_t0=None):
    """The validity flag of a tracking: share of shots, and 100-shot blocks by
    their median, whose direct-seafloor separation at the near offset is below
    max(S_STAR x T_dom, the duration of the direct after x0 / vw, direct_ms).
    The separation of each shot is the smallest of those available: the
    tracking t0, the stage-1 anchors (i, t) interpolated along the line, the
    first-energy onsets (arrival times, s).  T in s; NaN -> outside the domain,
    T_dom not measurable.  An unmeasured direct end leaves S_STAR x T_dom
    alone; the active bound and the active separation are reported.  A shot
    whose water-layer autocorrelation t0 (autocorr_t0) puts the seafloor before
    the end of the direct is below in any case; decided_by says which test made
    the flag."""
    t0 = np.asarray(t0, float)
    cand = [arrival_time(t0, x0, vw) - x0 / vw]
    if anchors is not None and len(anchors[0]):
        ai, at = np.asarray(anchors[0], int), np.asarray(anchors[1], float)
        o = np.argsort(ai)
        cand.append(arrival_time(np.interp(np.arange(t0.size), ai[o], at[o]), x0, vw) - x0 / vw)
    else:
        cand.append(np.full(t0.size, np.nan))
    cand.append(np.asarray(first_onset, float) - x0 / vw if first_onset is not None else np.full(t0.size, np.nan))
    C = np.vstack(cand)
    sep = np.nanmin(C, axis=0)
    which = np.nanargmin(C, axis=0)
    share = {SEP_SOURCES[k]: float(np.mean(which == k)) for k in range(3)}
    d = dict(s_star_tdom=S_STAR, t_dom_ms=float(T * 1e3), t_dom_source=source, direct_ms=float(direct_ms),
             separation_median_ms=float(np.median(sep) * 1e3), separation_min_ms=float(np.min(sep) * 1e3),
             separation_active=max(share, key=share.get), separation_active_share=share,
             separation_by_source_median_ms={SEP_SOURCES[k]: (float(np.nanmedian(C[k]) * 1e3) if np.isfinite(C[k]).any()
                                                              else float("nan")) for k in range(3)})
    if not np.isfinite(T):
        return dict(d, flag="outside the domain, T_dom not measurable", threshold_ms=float("nan"),
                    active="-", fraction_below=float("nan"), blocks_below=[])
    by_direct = bool(np.isfinite(direct_ms) and direct_ms * 1e-3 > S_STAR * T)
    thr = direct_ms * 1e-3 if by_direct else S_STAR * T
    d["active"] = "end of the direct" if by_direct else "%g T_dom" % S_STAR
    below3 = sep < thr
    below4 = np.zeros(t0.size, bool)
    if autocorr_t0 is not None and np.isfinite(direct_ms):
        sep4 = arrival_time(np.asarray(autocorr_t0, float), x0, vw) - x0 / vw
        below4 = np.isfinite(sep4) & (sep4 < direct_ms * 1e-3)
        d["separation_autocorr_median_ms"] = (float(np.nanmedian(sep4) * 1e3) if np.isfinite(sep4).any()
                                              else float("nan"))
    below = below3 | below4; frac = float(below.mean())

    def _blocks(use4):
        return [(int(b0), int(min(t0.size, b0 + BLOCK) - 1)) for b0 in range(0, t0.size, BLOCK)
                if np.median(sep[b0:b0 + BLOCK]) < thr or (use4 and below4[b0:b0 + BLOCK].mean() > 0.5)]

    def _flag(f, blk):
        # "partially" from the 100-shot blocks only; isolated shots are counted
        return "outside the domain" if f > OUT_SHARE else "partially outside the domain" if blk else "in the domain"
    blocks = _blocks(True)
    flag = _flag(frac, blocks)
    in_blk = np.zeros(t0.size, bool)
    for b0, b1 in blocks:
        in_blk[b0:b1 + 1] = True
    d["isolated_shots_below"] = int(np.sum(below & ~in_blk))
    d["decided_by"] = ("-" if flag == "in the domain" else "water-layer autocorrelation"
                       if flag != _flag(float(below3.mean()), _blocks(False)) else "separations")
    d["fraction_below_autocorr_only"] = float(np.mean(below4 & ~below3))
    return dict(d, flag=flag, threshold_ms=float(thr * 1e3), fraction_below=frac, blocks_below=blocks)


def domain_line(dom):
    if not np.isfinite(dom["threshold_ms"]):
        return "operating domain: %s" % dom["flag"].upper()
    return ("operating domain: %s (separation from the direct %.0f ms median, %.0f ms min, from the %s, against %.0f ms "
            "= %s [%g T_dom = %.0f ms, end of the direct %s]; %.1f%% of the shots below, %d blocks of %d)"
            % (dom["flag"].upper() if dom["fraction_below"] else dom["flag"], dom["separation_median_ms"],
               dom["separation_min_ms"], dom["separation_active"], dom["threshold_ms"], dom["active"],
               dom["s_star_tdom"], dom["s_star_tdom"] * dom["t_dom_ms"],
               "%.0f ms" % dom["direct_ms"] if np.isfinite(dom["direct_ms"]) else "not measured",
               100 * dom["fraction_below"], len(dom["blocks_below"]), BLOCK)
            + ("; decided by the %s" % dom["decided_by"] if dom.get("decided_by", "-") != "-" else "")
            + ("; %d isolated shots below, not counted for the state" % dom["isolated_shots_below"]
               if dom.get("isolated_shots_below") else ""))


def save_unit_fallback(a, meta, t0p, vw, dt, offs, S, reason, extra, d_info, f_on, t_ac):
    """Onset tracking not applicable to the whole unit: the stage-1 tracking
    given as prior is the output, unchanged (no event correction, no convention
    constant: stage 1 carries its own), with the stage-1 anchors, flagged with
    the reason."""
    za = np.load(a.stage1_anchors, allow_pickle=True)
    si = np.asarray(za["i"], int); st_ = np.asarray(za["t"], float)
    k = (si >= 0) & (si < S)
    print(f"  ONSET TRACKING NOT APPLICABLE ON THIS UNIT ({reason}): stage-1 tracking used")
    dom = validity_domain(t0p, offs[0], vw, extra["t_dom_ms"] * 1e-3, d_info.get("duration_ms", float("nan")),
                          anchors=(si[k], st_[k]), first_onset=f_on, autocorr_t0=t_ac)
    dom["first_energy_s"] = [None if not np.isfinite(x) else float(x) for x in f_on]
    dom["autocorr_t0_s"] = [None if not np.isfinite(x) else float(x) for x in t_ac]
    print("  " + domain_line(dom))
    params = dict(script="h3_track", code=code_version(), variant=a.variant, applicable=False, fallback="unit",
                  fallback_reason=reason, convention_ms=CONVENTION_MS[a.variant], convention_applied=False,
                  n_ch_track=a.n_ch_track, prior=os.path.basename(a.prior_track),
                  stage1_anchors=os.path.basename(a.stage1_anchors), unreliable_fraction=1.0,
                  unreliable_stretches=[(0, int(S - 1))], unreliable_stretch_causes=[{reason: 1}],
                  onset_used_fraction=0.0, domain=dom, direct_end=d_info, **extra)
    np.savez(a.out, t0=t0p, vw=vw, D_init=0.0, dt=dt, offs=offs, S=S, segy=str(meta["segy"]), shot0=-1,
             nshot=int(meta["nshot"]), t0_shift=0.0, params=json.dumps(params, default=float),
             anchors_i=si[k], anchors_t=st_[k], anchors_kind=np.array(["stage1"] * int(k.sum())),
             unreliable=np.ones(S, bool), onset_used=np.zeros(S, bool), prior_corrected=t0p)
    print(f"  saved {a.out}")


def aligned_stack(gb, shots, dt, t_arr, lo, hi, c=0):
    """Mean of channel-c traces of shots, each cut on [t_arr[s] + lo, t_arr[s] + hi]."""
    n = int(round((hi - lo) / dt)); out, k = np.zeros(n), 0
    for s in shots:
        a = int(round((t_arr[s] + lo) / dt))
        if a < 0 or a + n > gb.shape[1]:
            continue
        out += gb[s, a:a + n, c]; k += 1
    return out / max(k, 1), k


def matched(st, tpl, centre, R, sign):
    """Best score sign x normalised correlation of tpl within centre +- R samples."""
    tpl = tpl - tpl.mean(); tpl /= (np.linalg.norm(tpl) + 1e-12)
    best, lag = -np.inf, 0
    for k in range(max(0, centre - R), min(len(st) - len(tpl), centre + R) + 1):
        seg = st[k:k + len(tpl)] - st[k:k + len(tpl)].mean()
        n = np.linalg.norm(seg)
        if n < 1e-12:
            continue
        v = sign * float(seg @ tpl) / n
        if v > best:
            best, lag = v, k - centre
    return best, lag


def event_check(gb, b_shots, dt, offs, vw, t0p, T, t_direct_end):
    """Block check against a stage-1 curve on a later event.  Returns a dict with
    the flags and, when the correction is proven, the vertical shift to apply.
    The earlier event is searched after t_direct_end (s, near trace)."""
    x0 = offs[0]
    arr = np.array([arrival_time(t, x0, vw) for t in t0p])
    lo = max(-0.600, float(np.max(t_direct_end - arr[b_shots])))
    hi = 0.300
    if lo > -2 * T:
        return dict(candidate=False, multiple_flag=False, corrected=0.0, note="no room above the prior")
    st, k = aligned_stack(gb, b_shots, dt, arr, lo, hi)
    if k < 5:
        return dict(candidate=False, multiple_flag=False, corrected=0.0, note="too few traces")
    i_zero = int(round(-lo / dt))
    tz = i_zero * dt
    o_p, ok_p = onset(st, dt, tz, M_TDOM * T)
    res = dict(candidate=False, multiple_flag=False, corrected=0.0, note="")
    # the prior's own multiple: lag of the matched filter from 2 t0
    L = int(round(0.200 / dt)); R = int(round(MULT_SEARCH_MS * 1e-3 / dt))
    t0m = np.median(t0p[b_shots])
    # the template is the primary stack from the onset o_rel; the multiple stack
    # is cut on the multiple predicted from t0p + dv, and the search is centred
    # where the prior's lead (o_p - tz) puts the multiple onset, twice as far
    lead = (o_p - tz) if ok_p else 0.0
    def mult_score(dv, o_rel):
        i0 = int(round(o_rel / dt))
        tpl = st[i0:i0 + L]
        if len(tpl) < L:
            return -np.inf, 0
        arr_m = np.array([arrival_time(t + dv, x0, vw, 1) for t in t0p])
        sm, km = aligned_stack(gb, b_shots, dt, arr_m, -0.100, 0.300)
        c = int(round(0.100 / dt)) + int(round(2 * lead / dt))
        return matched(sm, tpl, c, R, -1.0)
    if ok_p:
        sc_p, lag_p = mult_score(0.0, o_p)
        dev = abs(lag_p) * dt / 2.            # implied error of the vertical time, s
        res["multiple_flag"] = bool(abs(lag_p) >= R - 1 or dev > max(0.02 * t0m, T))
        res["prior_multiple_score"] = float(sc_p)
    else:
        sc_p = -np.inf
    # an earlier event between the direct and the reach of the onset search:
    # within M_TDOM x T_dom of the prior the anchors' own search already finds the
    # onset, and two cycles of the same wavelet have multiples too alike to choose
    # between (MEPA, development iteration 1: candidates 22-36 ms early scored
    # 0.69-0.81 against 0.67-0.81)
    top = (o_p if ok_p else tz) - M_TDOM * T
    half = 0.5 * (top - NOISE_MS * 1e-3)
    if half > 2 * dt:
        o_c, ok_c = onset(st, dt, NOISE_MS * 1e-3 + half, half)
        if ok_c:
            res["candidate"] = True
            dv = (o_c - (o_p if ok_p else tz)) * float(np.median(arr[b_shots] / t0p[b_shots]))
            sc_c, _ = mult_score(dv, o_c)
            res["candidate_shift_ms"] = float(dv * 1e3)
            res["candidate_multiple_score"] = float(sc_c)
            if sc_c > 0 and sc_c >= sc_p:
                res["corrected"] = float(dv)
    return res


def drift_smooth(ai, d, w):
    """run_real --drift_window_shots: per anchor, the median over the anchors
    within +-w shots, at least 3, odd, ties to the lower index."""
    if len(d) < 3:
        return d
    n_max = len(d) - (1 - len(d) % 2)
    out = []
    for i in range(len(d)):
        dist = np.abs(ai - ai[i])
        m = int(np.sum(dist <= w)); m = min(max(3, m + (1 - m % 2)), n_max)
        out.append(np.median(d[np.lexsort((ai, dist))[:m]]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cube"); ap.add_argument("prior_track"); ap.add_argument("stage1_anchors"); ap.add_argument("out")
    ap.add_argument("--dxsp", type=float, required=True, help="shot spacing, m")
    ap.add_argument("--variant", choices=("h2", "h2m"), default="h2m")
    ap.add_argument("--n_group", type=int, default=None, help="default max(5, round(500 m / dxsp))")
    ap.add_argument("--n_ch_track", type=int, default=3)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)       # the log shows the stages as they end

    meta = np.load(a.cube + ".meta.npz", allow_pickle=True)
    g = np.load(a.cube, mmap_mode="r")
    P = np.load(a.prior_track, allow_pickle=True)
    t0p = np.asarray(P["t0"], float); S = min(g.shape[0], t0p.size); t0p = t0p[:S].copy()
    vw = float(P["vw"]); offs = np.asarray(P["offs"], float); dt = float(meta["dt"])
    nch = max(4, a.n_ch_track)
    graw = np.ascontiguousarray(g[:S, :, :nch]).astype(np.float32)
    gb = bandpass(graw.astype(float), dt, BAND, axis=1)
    ng = a.n_group or max(5, int(round(WINDOW_M / a.dxsp)))
    na = max(1, int(round(ANCHOR_M / a.dxsp)))
    x0 = offs[0]

    # T_dom of the line, from the provisional onsets of channel 0 (anchor_qc's definition)
    prov = np.array([onset(gb[s, :, 0], dt, arrival_time(t0p[s], x0, vw), PROVISIONAL_MS * 1e-3) for s in range(S)],
                    dtype=object)
    on0 = np.array([p[0] for p in prov], float); ok0 = np.array([p[1] for p in prov], bool)
    T = onset_tdom(gb[:, :, 0], np.nan_to_num(on0), ok0, dt) * 1e-3
    # the end of the direct: the event check searches after it, the domain uses it
    t_dend, d_info = direct_end(gb[:, :, 0], dt, x0, vw)
    f_on = first_energy(gb[:, :, 0], dt, t_dend, arrival_time(min(1.50, gb.shape[1] * dt / 2.2), x0, vw))
    t_ac = water_autocorr_t0(graw[:, :, :1], dt, offs[:1])
    if not np.isfinite(T):
        return save_unit_fallback(a, meta, t0p, vw, dt, offs, S, "t_dom_not_measurable",
                                  dict(t_dom_ms=float("nan"), n_group=ng, anchor_every=na), d_info, f_on, t_ac)
    t0_stage1 = t0p.copy()
    print(f"h3_track: {S} shots, dt {dt*1e3:.0f} ms, T_dom {T*1e3:.1f} ms, n_group {ng}, anchor every {na} shots, "
          f"variant {a.variant}")

    # 1. prior, corrected for a later event block by block; the earlier event is
    # searched after the end of the direct (anchor_qc.direct_end)
    if not np.isfinite(t_dend):
        t_dend = x0 / vw + DIRECT_GAP_MS * 1e-3
    print(f"  direct arrival ends at {t_dend*1e3:.0f} ms" + ("" if "end_ms" in d_info else
          f" (end NOT MEASURABLE, {d_info.get('note')}: {DIRECT_GAP_MS:.0f} ms after it)"))
    blocks = []
    for b0 in range(0, S, BLOCK):
        bs = np.arange(b0, min(S, b0 + BLOCK))
        r = event_check(gb, bs, dt, offs, vw, t0p, T, t_dend)
        r.update(first=int(b0), last=int(bs[-1]))
        if r["corrected"]:
            t0p[bs] = t0p[bs] + r["corrected"]
        blocks.append(r)
    n_corr = sum(1 for r in blocks if r["corrected"])
    print(f"  event check: {sum(r['candidate'] for r in blocks)} blocks with an earlier event, "
          f"{sum(r['multiple_flag'] for r in blocks)} with an inconsistent multiple, {n_corr} corrected")

    # 2. dense onset anchors
    slots = np.arange(0, S, na)
    an = np.array([shot_onset_why(gb, s, dt, offs, vw, t0p[s], M_TDOM * T, a.variant) for s in slots], dtype=object)
    at = np.array([x[0] for x in an], float); aok = np.array([x[1] for x in an], bool)
    why = [x[2] for x in an]

    # cycle check, per block, on the anchors against the prior
    r_b = []
    for r in blocks:
        m = aok & (slots >= r["first"]) & (slots <= r["last"])
        r_b.append(float(np.median(at[m] - t0p[slots[m]])) if m.sum() >= 3 else np.nan)
    r_b = np.array(r_b)
    for j, r in enumerate(blocks):
        nb = [r_b[i] for i in range(max(0, j - 2), min(len(blocks), j + 3)) if i != j and np.isfinite(r_b[i])]
        r["cycle_k"] = 0
        if np.isfinite(r_b[j]) and len(nb) >= 2:
            trend = float(np.median(nb)); jump = r_b[j] - trend; k = int(round(jump / T))
            if k and abs(jump - k * T) <= T / 4 and np.max(np.abs(np.array(nb) - trend)) <= T / 4:
                r["cycle_k"] = k

    # 3. fallback stretches: stage-1 anchors moved to the onset convention
    za = np.load(a.stage1_anchors, allow_pickle=True)
    si = np.asarray(za["i"], int); st_ = np.asarray(za["t"], float)
    conv = float(np.median(at[aok] - t0p[slots[aok]])) if aok.sum() else 0.0
    bad_stretch = np.zeros(S, bool); stretches = []; causes = []
    run0 = None
    for j in range(len(slots) + 1):
        bad = j < len(slots) and not aok[j]
        if bad and run0 is None:
            run0 = j
        if not bad and run0 is not None:
            if j - run0 >= MIN_STRETCH:
                s0 = slots[run0]; s1 = slots[j] - 1 if j < len(slots) else S - 1
                bad_stretch[s0:s1 + 1] = True; stretches.append((int(s0), int(s1)))
                causes.append({w: why[run0:j].count(w) for w in WHY if why[run0:j].count(w)})
            run0 = None
    ai = list(slots[aok]); atv = list(at[aok]); kind = ["onset"] * len(ai)
    for i, t in zip(si, st_):
        if i < S and bad_stretch[i]:
            j = min(i // BLOCK, len(blocks) - 1)
            ai.append(int(i)); atv.append(float(t + conv + blocks[j]["cycle_k"] * T)); kind.append("stage1")
    order = np.argsort(ai)
    ai = np.asarray(ai, int)[order]; atv = np.asarray(atv, float)[order]; kind = [kind[k] for k in order]
    if ai.size < 2:
        return save_unit_fallback(a, meta, t0_stage1, vw, dt, offs, S, "fewer_than_2_onset_anchors",
                                  dict(t_dom_ms=T * 1e3, n_group=ng, anchor_every=na, blocks=blocks,
                                       anchors_onset=int(aok.sum()), anchor_positions=int(slots.size)), d_info, f_on,
                                  t_ac)
    print(f"  anchors: {int(aok.sum())} onset of {slots.size} positions, {sum(k == 'stage1' for k in kind)} stage-1 in "
          f"{len(stretches)} unreliable stretches ({100*bad_stretch.mean():.1f}% of the line); "
          f"convention of the stage-1 anchors {conv*1e3:+.1f} ms; "
          f"{sum(1 for r in blocks if r['cycle_k'])} blocks with a cycle jump")

    # 4. tracking between the anchors
    t0x = track_seafloor_xcorr(graw, dt, offs[:nch], t0_init_s=float(atv[0]), vw=vw,
                               n_group=ng, n_ch=a.n_ch_track)
    d = atv - t0x[ai]
    d_sm = drift_smooth(ai, d, ng // 2)
    t0 = t0x + np.interp(np.arange(S), ai, d_sm)

    # 5. jitter: the h2m onset where it is reliable and within T_dom/2
    used = np.zeros(S, bool)
    for s in range(S):
        tv, ok = shot_onset(gb, s, dt, offs, vw, t0[s], M_TDOM * T, "h2m")
        if ok and abs(tv - t0[s]) <= T / 2:
            t0[s] = tv; used[s] = True
    conv_out = CONVENTION_MS[a.variant] * 1e-3
    t0 = np.maximum(t0 - conv_out, 1e-3)
    atv = atv - conv_out
    print(f"  tracking: t0 {t0.min()*1e3:.0f}-{t0.max()*1e3:.0f} ms, scatter {np.std(t0)*1e3:.1f} ms; "
          f"onset used on {100*used.mean():.1f}% of the shots; drift spread {np.ptp(d_sm)*1e3:.0f} ms")

    dom = validity_domain(t0, x0, vw, T, d_info.get("duration_ms", float("nan")), anchors=(si, st_), first_onset=f_on,
                          autocorr_t0=t_ac)
    dom["first_energy_s"] = [None if not np.isfinite(x) else float(x) for x in f_on]
    dom["autocorr_t0_s"] = [None if not np.isfinite(x) else float(x) for x in t_ac]
    print("  " + domain_line(dom))
    if stretches:
        print(f"  onset-tracking fallback on {len(stretches)} stretches, {100*bad_stretch.mean():.1f}% of the line")
    params = dict(script="h3_track", code=code_version(), variant=a.variant, t_dom_ms=T * 1e3, n_group=ng,
                  anchor_every=na, n_ch_track=a.n_ch_track, convention_ms=CONVENTION_MS[a.variant], prior=os.path.basename(a.prior_track),
                  stage1_anchors=os.path.basename(a.stage1_anchors), convention_stage1_ms=conv * 1e3,
                  unreliable_fraction=float(bad_stretch.mean()), unreliable_stretches=stretches,
                  applicable=True, fallback=("stretches" if stretches else "none"), fallback_reason="",
                  unreliable_stretch_causes=causes, convention_applied=True, domain=dom,
                  onset_used_fraction=float(used.mean()), blocks=blocks,
                  anchors_onset=int(aok.sum()), anchor_positions=int(slots.size), direct_end=d_info)
    np.savez(a.out, t0=t0, vw=vw, D_init=0.0, dt=dt, offs=offs, S=S, segy=str(meta["segy"]), shot0=-1,
             nshot=int(meta["nshot"]), t0_shift=0.0, params=json.dumps(params, default=float),
             anchors_i=ai, anchors_t=atv, anchors_kind=np.array(kind), unreliable=bad_stretch, onset_used=used,
             prior_corrected=t0p)
    print(f"  saved {a.out}")


if __name__ == "__main__":
    main()
