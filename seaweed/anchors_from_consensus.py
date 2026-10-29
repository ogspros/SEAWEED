#!/usr/bin/env python3
"""anchors_from_consensus -- stage 1: seafloor anchors from the consensus picker.

The line is divided into groups of shots.  In each group the consensus picker
proposes candidate seafloor times, from the autocorrelation and from the
semblance, and scores them; the best is kept as an anchor only if its
semblance is high enough and it agrees with the running median of its
neighbours.  A few certain anchors are what the tracking needs: they say which
event to follow, and the cross-correlation of run_real follows it from shot to
shot.  Used as a tracker instead, deciding group by group, the picker made the
Viking seafloor oscillate between 320 and 390 m where the section is nearly
flat.

The direct arrival is not a candidate: a candidate that arrives at the near
offset before the end of the direct (anchor_qc.direct_end: the envelope of the
stacked near traces back under the find_origin threshold, plus 8 ms) is dropped
before the best is chosen.  On Viking with groups of 50 shots, four
consecutive groups won on the direct (207-211 ms) and agreed with each other,
so the neighbour check kept them.

With fewer than four anchors the program goes on:
  reduced       2 or 3 anchors kept: they are written as they are; with only 2
                groups the neighbour check does not apply (the median of two
                values is their mean, equidistant from both), only the
                semblance threshold
  single group  fewer than 2: the whole line is picked as one group (same
                shots, candidates and direct rejection) and that time is
                written at the first and last shot.  The drift is then
                CONSTANT along the whole line: the slope of the seafloor is
                ignored, and on a dipping real line the window of h3_track may
                not contain the seafloor.  The whole line is flagged
Both are written in params (stage1_fallback) and in the log.

    python -m seaweed.anchors_from_consensus line_cube.npy line_track.npz \\
        line_anchors.npz

line_cube.npy and line_track.npz come from a first pass of run_real
(--save_cube, --save_track); the track is read only for the sampling
interval, the offsets and the water velocity.  Then:

    python -m seaweed.run_real ... --load_anchors line_anchors.npz
"""
import argparse
import json
import os

import numpy as np

from seaweed import code_version
from seaweed import pick_consensus as PC
from seaweed.anchor_qc import direct_end
from seaweed.run_real import BAND
from seaweed.wavelet_extractor_v2 import arrival_time, bandpass


def pick_group(g, sel, dt, offs, vw, x0, t_end, label, n_drop):
    """The best candidate of the shots sel (AGC, autocorrelation and semblance
    candidates, those before the end of the direct dropped, scored): (t0 in the
    convention of the tracking, score, semblance) or None."""
    gg = PC.agc(np.asarray(g[sel]).astype(float), dt)
    hi = min(1.50, (gg.shape[1] * dt) / 2.2)
    try:
        cands = sorted(set(np.round(
            PC.autocorr_candidates(gg, dt, 0.10, hi)
            + PC.semblance_candidates(gg, dt, offs, vw, 0.10, hi), 4)))
        if np.isfinite(t_end):
            near = [c for c in cands if arrival_time(c, x0, vw) < t_end]
            cands = [c for c in cands if c not in near]
            n_drop.append(len(near))
            if near:
                print("  %s: %d candidates on the direct dropped (%s ms)"
                      % (label, len(near), ", ".join("%.0f" % (c * 1e3) for c in near)))
        scored = [PC.score_candidate(gg, dt, offs, t, vw) for t in cands]
    except Exception as e:
        print("  %s failed: %s" % (label, str(e)[:50]))
        return None
    if not scored:
        return None
    j = int(np.argmax([s for s, _ in scored]))
    # the convention offset of the picker is subtracted here too, so that the
    # anchors are in the convention the tracking expects
    return cands[j] - PC.CONVENTION_MS * 1e-3, scored[j][0], scored[j][1]["semblance"]


def decide(t, sem, tol_ms, min_semblance):
    """Which groups become anchors and what fallback that implies: (median of
    the neighbours, off-neighbour flags, kept flags, "none" | "reduced" |
    "single group").  With two groups the neighbour check does not apply."""
    t = np.asarray(t, float); sem = np.asarray(sem, float)
    med = PC._running_median(t, 5) if t.size else t
    off = np.zeros(t.size, bool) if t.size == 2 else np.abs(t - med) * 1e3 > tol_ms
    keep = ~off & (sem >= min_semblance)
    fallback = "none" if (t.size >= 4 and keep.sum() >= 4) else "reduced" if keep.sum() >= 2 else "single group"
    return med, off, keep, fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("track_in", help="only for dt, offsets and vw")
    ap.add_argument("anchors_out")
    ap.add_argument("--group", type=int, default=100,
                    help="shots per group; wide groups give fewer and surer "
                         "anchors, which is what the tracker wants")
    ap.add_argument("--tol_ms", type=float, default=60.0,
                    help="how far a group may sit from its neighbours' median "
                         "and still be kept")
    ap.add_argument("--min_semblance", type=float, default=0.35)
    ap.add_argument("--max_shots", type=int, default=48,
                    help="shots used to score the candidates of a group, "
                         "spread over it.  The moveout is as clear on twenty "
                         "shots as on two hundred, and the cost grows with the "
                         "number; on IT90-65 the whole group took 80 minutes "
                         "for the same score.  0 uses every shot")
    ap.add_argument("--keep_direct", action="store_true",
                    help="keep the candidates that arrive before the end of the "
                         "direct, as in the first version of the workflow")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()

    g = np.load(a.cube, mmap_mode="r")
    z = np.load(a.track_in, allow_pickle=True)
    dt = float(z["dt"]); offs = np.asarray(z["offs"], float); vw = float(z["vw"])
    S = g.shape[0]
    x0 = float(offs[0])
    t_end, d_info = np.nan, {}
    if not a.keep_direct:
        t_end, d_info = direct_end([bandpass(np.asarray(g[s][:, 0], float), dt, BAND) for s in range(S)], dt, x0, vw)
        print("direct arrival at %.0f ms on the near trace (%.0f m); %s" % (
            1e3 * x0 / vw, x0, "it ends at %.0f ms (%.0f ms after it): earlier candidates dropped"
            % (t_end * 1e3, d_info["duration_ms"]) if np.isfinite(t_end)
            else "its END IS NOT MEASURABLE (%s): no candidate dropped" % d_info.get("note", "")))
    n_drop = []

    rows = []
    for a0 in range(0, S, a.group):
        b = min(S, a0 + a.group)
        sel = np.arange(a0, b)
        if a.max_shots and sel.size > a.max_shots:
            sel = sel[np.linspace(0, sel.size - 1, a.max_shots).round().astype(int)]
        p = pick_group(g, sel, dt, offs, vw, x0, t_end, "group %4d-%4d" % (a0, b), n_drop)
        if p is not None:
            rows.append(dict(centre=0.5 * (a0 + b - 1), t0=p[0], score=p[1], sem=p[2]))

    t = np.array([r["t0"] for r in rows])
    sem = np.array([r["sem"] for r in rows])
    med, off, keep, fallback = decide(t, sem, a.tol_ms, a.min_semblance)
    if t.size:
        print("%d groups, %d kept as anchors (%d rejected: %d off their "
              "neighbours, %d weak)"
              % (len(rows), keep.sum(), (~keep).sum(), int(np.sum(off)), int(np.sum(sem < a.min_semblance))))

    single = None
    if fallback == "none":
        ai = np.array([int(round(r["centre"])) for r, k in zip(rows, keep) if k])
        at = t[keep]
    elif fallback == "reduced":
        ai = np.array([int(round(r["centre"])) for r, k in zip(rows, keep) if k])
        at = t[keep]
        why = ("only %d groups produced a candidate" % len(rows)) if len(rows) < 4 else \
              ("%d of %d groups kept" % (keep.sum(), len(rows)))
        print("FEWER THAN FOUR ANCHORS (%d; %s): continuing with them" % (keep.sum(), why))
    else:
        sel = np.arange(S)
        if a.max_shots and sel.size > a.max_shots:
            sel = sel[np.linspace(0, sel.size - 1, a.max_shots).round().astype(int)]
        p = pick_group(g, sel, dt, offs, vw, x0, t_end, "single group 0-%d" % S, [])
        if p is None:
            raise SystemExit("no seafloor candidate on the whole line")
        single = dict(t0=float(p[0]), score=float(p[1]), semblance=float(p[2]), weak=bool(p[2] < a.min_semblance),
                      reason="%d groups with a candidate, %d kept" % (len(rows), keep.sum()), flagged=[0, int(S - 1)])
        ai = np.array([0, S - 1]); at = np.array([p[0], p[0]])
        print("STAGE 1 FELL BACK TO A SINGLE GROUP (%s): t0 %.3f s, semblance %.2f%s, written at shots 0 and %d. "
              "The drift is CONSTANT along the whole line: the seafloor slope is ignored, and on a dipping line "
              "the window of h3_track may not contain the seafloor. The whole line is flagged"
              % (single["reason"], p[0], p[2], " (WEAK)" if single["weak"] else "", S - 1))
    # every group is kept, rejected ones included, with the parameters that
    # decided: the file says by itself how its anchors were chosen
    params = dict(script="anchors_from_consensus", code=code_version(),
                  cube=os.path.basename(a.cube),
                  track_in=os.path.basename(a.track_in), group=a.group,
                  tol_ms=a.tol_ms, min_semblance=a.min_semblance,
                  max_shots=a.max_shots, running_median=5,
                  convention_ms=PC.CONVENTION_MS, vw=vw,
                  keep_direct=a.keep_direct, direct_end=d_info, candidates_dropped=n_drop,
                  stage1_fallback=fallback, n_kept=int(keep.sum()), single_group=single)
    np.savez(a.anchors_out, i=ai, t=at, V=np.full(ai.size, vw),
             params=json.dumps(params),
             group_centre=np.array([r["centre"] for r in rows]),
             group_t0=t, group_median=med,
             group_score=np.array([r["score"] for r in rows]),
             group_semblance=np.array([r["sem"] for r in rows]),
             group_kept=keep)
    print("anchors at shots:", ", ".join(str(i) for i in ai))
    print("times           :", ", ".join("%.3f" % x for x in at))
    print("wrote %s" % a.anchors_out)

    if a.compare:
        q = np.load(a.compare, allow_pickle=True)
        qi = np.asarray(q["i"], int); qt = np.asarray(q["t"], float)
        ref = np.interp(ai, qi, qt)
        d = (at - ref) * 1e3
        print("against %s: median %+.0f ms, p10 %+.0f, p90 %+.0f, max |.| %.0f"
              % (a.compare, np.median(d), np.percentile(d, 10),
                 np.percentile(d, 90), np.abs(d).max()))


if __name__ == "__main__":
    main()
