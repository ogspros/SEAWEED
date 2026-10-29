#!/usr/bin/env python3
"""Unit tests of the onset-tracking pieces on small analytic gathers (no Devito, no data).

  1. onset(): the threshold walk back finds the onset of a clean arrival within one
     sample, and calls a window with no arrival unreliable
  2. shot_onset(): h2m recovers the vertical time from three moveout-corrected
     channels, and refuses when one channel is displaced by a cycle
  3. drift_smooth(): the median over +-w shots, at least 3 and odd, as
     run_real --drift_window_shots
  4. extract_wavelet(align='two_stage') on a line whose windows carry a smooth
     static: the same wavelet as the single alignment; and on a line where a
     quarter of the windows are one cycle late, the two-stage alignment keeps
     the stack on one cycle where the single one is pulled
  5. run_real.bistability() flags a stack whose second half is one cycle late
  6. run_real.apply_recipe(): --recipe final sets --align two_stage,
     --bistab_check, --t0_shift 0 and n_group from the 500-m rule whatever the
     tracking, an explicit flag wins, --recipe paper keeps the old defaults
  7. h3_track on a unit where the onset tracking is not applicable (noise
     only): no error, the
     stage-1 tracking comes out unchanged, flagged with the reason
  8. the operating-domain flag (max(5 T_dom, end of the direct) from the
     direct, with the active bound; the separation the smallest of tracking,
     stage-1 anchors and first energy; the water-layer autocorrelation before
     the end of the direct puts a shot outside) and the --group rule
     of seafloor_pipeline
  9. anchors_from_consensus.decide: no fallback, reduced (2 groups without the
     neighbour check, 3 groups with it), single group

    python tests/test_h3.py          # exit 0 when every test passes
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seaweed import h3_track as H                                       # noqa: E402
from seaweed.run_real import BAND, apply_recipe, bistability            # noqa: E402
from seaweed.wavelet_extractor_v2 import arrival_time, bandpass, extract_wavelet  # noqa: E402

DT = 0.002
FAILS = []


def check(name, cond, detail=""):
    print("%-64s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILS.append(name)


def causal_wavelet(f=30.0, dt=DT):
    """A minimum-phase-like wavelet starting at sample 0: decaying sinusoid."""
    t = np.arange(int(0.15 / dt)) * dt
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.03)


def trace(T, onset_s, w, dt=DT, amp=1.0, noise=0.0, rng=None, band=None):
    x = np.zeros(T)
    i = int(round(onset_s / dt))
    x[i:i + len(w)] += amp * w[:max(0, min(len(w), T - i))]
    if noise:
        x += noise * rng.standard_normal(T)
    # h3_track measures onsets on band-passed traces (BAND of run_real)
    return bandpass(x, dt, BAND) if band else x


def main():
    rng = np.random.default_rng(1)
    w = causal_wavelet()
    T = 1500
    # 1. onset: the threshold rule is early on an abrupt onset (the envelope of a
    # causal wavelet rises before it), by a constant: that is the convention
    # constant, reported apart.  What the onset tracking needs is that it does
    # not depend on the noise
    on = []
    # (the walk back meets the 8 % threshold on the slow precursor of the
    # zero-phase band-pass, so 1 % noise moves it by a few ms: IQR, not range)
    for k in range(20):
        o, ok = H.onset(trace(T, 1.000, w, noise=0.01, rng=rng, band=True), DT, 1.010, 0.060)
        on.append(o if ok else np.nan)
    on = np.array(on); bias = float(np.nanmedian(on) - 1.000)
    iqr = float(np.subtract(*np.percentile(on, [75, 25])))
    check("1a onset reproducible: IQR over 20 noise draws <= 2 samples",
          np.isfinite(on).all() and iqr <= 2 * DT + 1e-9,
          "bias %+.0f ms, IQR %.0f ms, range %.0f ms" % (bias * 1e3, iqr * 1e3, np.ptp(on) * 1e3))
    o, ok = H.onset(trace(T, 2.5, w, noise=0.01, rng=rng, band=True), DT, 1.010, 0.060)
    check("1b no arrival in the window -> unreliable", not ok)

    # 2. shot_onset, h2m: the same constant on every channel, so the vertical
    # time is t0 plus the vertical bias on (nearly) every draw
    vw = 1500.0; offs = np.array([100., 125., 150., 175.]); t0 = 0.900
    good = 0
    for k in range(10):
        g = np.zeros((1, T, 4))
        for c in range(4):
            g[0, :, c] = trace(T, arrival_time(t0, offs[c], vw), w, noise=0.005, rng=rng, band=True)
        tv, ok = H.shot_onset(g, 0, DT, offs, vw, 0.905, 0.07, "h2m")
        good += bool(ok and abs(tv - (t0 + bias * arrival_time(t0, offs[2], vw) / t0)) <= 2 * DT + 1e-9)
    check("2a h2m reliable and at t0 + bias (2 samples) on >= 8 of 10 draws", good >= 8, "%d/10" % good)
    g2 = g.copy(); g2[0, :, 2] = trace(T, arrival_time(t0, offs[2], vw) + 0.033, w, noise=0.005, rng=rng, band=True)
    tv, ok = H.shot_onset(g2, 0, DT, offs, vw, 0.905, 0.07, "h2m")
    check("2b one channel a cycle late -> h2m refuses", not ok)

    # 3. drift_smooth
    ai = np.array([0, 10, 20, 30, 40, 50]); d = np.array([0., 0., 9., 0., 0., 0.])
    sm = H.drift_smooth(ai, d, 5)
    check("3a an isolated anchor is absorbed (median of 3)", np.allclose(sm, 0.0), str(sm))
    d2 = np.array([0., 0., 5., 5., 5., 5.])
    sm2 = H.drift_smooth(ai, d2, 5)
    check("3b a step longer than the window is followed", np.allclose(sm2[3:], 5.0) and sm2[0] == 0.0, str(sm2))

    # 4. two-stage alignment
    S, C = 120, 1
    t0s = 0.6 + 0.004 * np.sin(np.arange(S) / 15.0)
    gl = np.zeros((S, 1000, C))
    for s in range(S):
        gl[s, :, 0] = trace(1000, arrival_time(t0s[s], 100., vw), causal_wavelet(25.0), noise=0.02, rng=rng)
    t_track = np.full(S, 0.6)
    w1, q1 = extract_wavelet(gl, DT, t_track, np.array([100.]), vw=vw, band=(6., 90.))
    w2, q2 = extract_wavelet(gl, DT, t_track, np.array([100.]), vw=vw, band=(6., 90.), align="two_stage",
                             align_group=20)
    ncc = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)))
    check("4a smooth static: two-stage = single (NCC >= 0.99)", ncc >= 0.99, "NCC %.4f, T_dom %.0f ms"
          % (ncc, q2["align_tdom_ms"]))
    # no true statics, a ringing wavelet and strong noise: single windows can lock
    # one cycle off within +-40 ms; the two-stage alignment cannot leave +-T_dom/2
    # of its group.  (It also cannot follow a TRUE static over T_dom/2 inside a
    # group of 500 m: that is the price, declared in the paper.)
    def ringing(f=25.0, dt=DT):
        t = np.arange(int(0.2 / dt)) * dt
        return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.08)
    gj = np.zeros((S, 1000, C))
    for s in range(S):
        gj[s, :, 0] = trace(1000, arrival_time(0.6, 100., vw), ringing(), noise=0.6, rng=rng)
    _, qa = extract_wavelet(gj, DT, t_track, np.array([100.]), vw=vw, band=(6., 90.))
    _, qb = extract_wavelet(gj, DT, t_track, np.array([100.]), vw=vw, band=(6., 90.), align="two_stage",
                            align_group=20)
    half = 0.5 * qb["align_tdom_ms"]
    ja = int(np.sum(np.abs(qa["lags_ms"] - np.median(qa["lags_ms"])) > half))
    jb = int(np.sum(np.abs(qb["lags_ms"] - np.median(qb["lags_ms"])) > half))
    check("4b no statics, noisy ringing: two-stage has no window beyond T_dom/2", jb == 0 and jb <= ja,
          "windows beyond T_dom/2: single %d, two-stage %d of %d" % (ja, jb, S))

    # 5. bistability: second half of the line one cycle late, as one block
    gk = np.zeros((200, 1000, 1))
    for s in range(200):
        gk[s, :, 0] = trace(1000, arrival_time(0.6, 100., vw) + (0.040 if s >= 100 else 0.0),
                            causal_wavelet(25.0), noise=0.01, rng=rng)
    _, qk = extract_wavelet(gk, DT, np.full(200, 0.6), np.array([100.]), vw=vw, band=(6., 90.))
    bk = bistability(qk, DT)
    _, qc = extract_wavelet(gl, DT, t_track, np.array([100.]), vw=vw, band=(6., 90.))
    bc = bistability(qc, DT)
    check("5a half the line a cycle late -> bistable", bk["bistable"], str(bk["offsets_samp"]))
    check("5b smooth static only -> not bistable", not bc["bistable"], str(bc["offsets_samp"]))

    # 6. the recipes of run_real
    def ns(**k):
        d = dict(recipe="paper", align=None, bistab_check=None, t0_shift=None, n_group=None, dxsp=None); d.update(k)
        return argparse.Namespace(**d)
    r = ns(); apply_recipe(r)
    check("6a paper: single, no check, 0 ms, n_group 20",
          (r.align, r.bistab_check, r.t0_shift, r.n_group) == ("single", False, 0.0, 20))
    r = ns(recipe="final", dxsp=25.0); apply_recipe(r)
    check("6b final, 25 m: two_stage, check, 0 ms, n_group 20",
          (r.align, r.bistab_check, r.t0_shift, r.n_group) == ("two_stage", True, 0.0, 20))
    r = ns(recipe="final", dxsp=200.0); apply_recipe(r)
    check("6c final, 200 m: n_group 5", r.n_group == 5)
    r = ns(recipe="final", dxsp=25.0, align="single", n_group=9); apply_recipe(r)
    check("6d final: an explicit flag wins", (r.align, r.bistab_check, r.n_group) == ("single", True, 9))
    try:
        apply_recipe(ns(recipe="final")); ok6 = False
    except SystemExit:
        ok6 = True
    check("6e final without --dxsp and --n_group -> refused", ok6)

    # 7. h3_track where the onset tracking is not applicable to the unit: noise only
    with tempfile.TemporaryDirectory() as tmp:
        S7, T7 = 30, 1500
        cube = os.path.join(tmp, "cube.npy")
        np.save(cube, (0.01 * rng.standard_normal((S7, T7, 4))).astype(np.float32))
        np.savez(cube + ".meta.npz", dt=DT, off_hdr=offs.astype(np.float32), shot0=-1, nshot=S7, segy="noise")
        prior = 0.9 + 0.001 * np.arange(S7)
        np.savez(os.path.join(tmp, "s1.npz"), t0=prior, vw=vw, dt=DT, offs=offs)
        np.savez(os.path.join(tmp, "a1.npz"), i=np.arange(0, S7, 5), t=prior[::5], V=np.full(6, vw))
        rc = subprocess.run([sys.executable, "-m", "seaweed.h3_track", cube, os.path.join(tmp, "s1.npz"),
                             os.path.join(tmp, "a1.npz"), os.path.join(tmp, "h3.npz"), "--dxsp", "25",
                             "--variant", "h2"], capture_output=True, text=True,
                            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")).returncode
        ok7 = rc == 0 and os.path.exists(os.path.join(tmp, "h3.npz"))
        z = np.load(os.path.join(tmp, "h3.npz"), allow_pickle=True) if ok7 else None
        p7 = json.loads(str(z["params"])) if ok7 else {}
        check("7a not applicable -> exit 0 and a tracking file", ok7, "exit %d" % rc)
        check("7b the stage-1 tracking, unchanged", ok7 and np.array_equal(z["t0"], prior))
        check("7c flagged: applicable False, fallback unit, reason",
              ok7 and p7["applicable"] is False and p7["fallback"] == "unit"
              and p7["fallback_reason"] in ("t_dom_not_measurable", "fewer_than_2_onset_anchors")
              and bool(np.all(z["unreliable"])) and set(z["anchors_kind"]) == {"stage1"}, p7.get("fallback_reason", ""))

    # 8. operating domain and --group rule
    from seaweed.seafloor_pipeline import group_rule
    x0 = 262.0; td = 0.025
    deep = H.validity_domain(np.full(300, 0.48), x0, vw, td)
    shallow = H.validity_domain(np.full(300, 0.11), x0, vw, td)
    mixed = H.validity_domain(np.r_[np.full(240, 0.48), np.full(60, 0.11)], x0, vw, td)
    check("8a 480 ms at 262 m: in the domain", deep["flag"] == "in the domain", "%.0f ms" % deep["separation_median_ms"])
    check("8b 110 ms at 262 m (sep < 125 ms): outside", shallow["flag"] == "outside the domain",
          "%.0f ms" % shallow["separation_median_ms"])
    check("8c a fifth of the line shallow: partially, its block flagged",
          mixed["flag"] == "partially outside the domain" and mixed["blocks_below"] == [(200, 299)], str(mixed["blocks_below"]))
    check("8d T_dom not measurable: outside, flagged", "not measurable" in H.validity_domain(np.full(9, .5), x0, vw, np.nan)["flag"])
    long_d = H.validity_domain(np.full(300, 0.33), x0, vw, td, 243.0)
    short_d = H.validity_domain(np.full(300, 0.33), x0, vw, td, 43.0)
    check("8f direct lasting 243 ms > 5 T_dom: the end of the direct is the active bound",
          long_d["active"] == "end of the direct" and long_d["threshold_ms"] == 243.0
          and long_d["flag"] == "outside the domain", "sep %.0f ms" % long_d["separation_median_ms"])
    check("8g direct lasting 43 ms: 5 T_dom is the active bound", short_d["active"] == "5 T_dom"
          and short_d["flag"] == "in the domain")
    late = H.validity_domain(np.full(300, 0.48), x0, vw, td, 43.0, anchors=([0, 299], [0.11, 0.11]),
                             first_onset=np.full(300, np.nan))
    check("8h tracking late, stage-1 anchors shallow: the anchors are the active separation, outside",
          late["separation_active"] == "stage-1 anchors" and late["flag"] == "outside the domain", late["separation_active"])
    fe = H.validity_domain(np.full(300, 0.48), x0, vw, td, 43.0, anchors=([0, 299], [0.48, 0.48]),
                           first_onset=np.full(300, arrival_time(0.11, x0, vw)))
    check("8i first energy early: it is the active separation, outside", fe["separation_active"]
          == "first energy after the direct" and fe["flag"] == "outside the domain")

    ac = H.validity_domain(np.full(300, 0.48), x0, vw, td, 243.0, anchors=([0, 299], [0.48, 0.48]),
                           first_onset=np.full(300, np.nan), autocorr_t0=np.full(300, 0.20))
    check("8j water-layer autocorrelation before the end of the direct: outside, decided by it",
          ac["flag"] == "outside the domain" and ac["decided_by"] == "water-layer autocorrelation", ac["decided_by"])
    ok_ac = H.validity_domain(np.full(300, 0.48), x0, vw, td, 243.0, anchors=([0, 299], [0.48, 0.48]),
                              first_onset=np.full(300, np.nan), autocorr_t0=np.full(300, 0.47))
    check("8k autocorrelation agreeing with the tracking: in the domain", ok_ac["flag"] == "in the domain")

    iso = np.full(300, 0.48); iso[[17, 150]] = 0.11
    one = H.validity_domain(iso, x0, vw, td, 43.0)
    check("8l two isolated shallow shots: in the domain, counted", one["flag"] == "in the domain"
          and one["isolated_shots_below"] == 2, str(one["isolated_shots_below"]))

    # 9. stage-1 anchors with fewer than four groups (anchors_from_consensus.decide)
    from seaweed.anchors_from_consensus import decide
    _, _, k, fb = decide([0.50, 0.51, 0.49, 0.50, 0.52], [0.9] * 5, 120.0, 0.35)
    check("9a five groups that agree: no fallback", fb == "none" and k.all())
    _, off, k, fb = decide([0.50, 1.10], [0.9, 0.9], 120.0, 0.35)
    check("9b two groups 600 ms apart: no neighbour check, both kept, reduced", fb == "reduced" and k.all() and not off.any())
    _, _, k, fb = decide([0.50, 1.10], [0.9, 0.2], 120.0, 0.35)
    check("9c two groups, one weak: single group", fb == "single group" and k.sum() == 1)
    _, off, k, fb = decide([0.50, 0.51, 1.20], [0.9] * 3, 120.0, 0.35)
    check("9d three groups, one off its neighbours: 2 kept, reduced", fb == "reduced" and k.sum() == 2 and off[2])
    _, _, k, fb = decide([0.50, 1.30, 0.51, 1.40, 0.49], [0.9] * 5, 120.0, 0.35)
    check("9e five groups, two off their neighbours: 3 kept, reduced", fb == "reduced" and k.sum() == 3, str(k))
    _, _, k, fb = decide([0.5], [0.9], 120.0, 0.35)
    check("9f one group: single group", fb == "single group")
    check("8e --group rule: Viking 100, IT90-65 100, MEPA 100, MS-29 12, short line N/4",
          [group_rule(25, 1012), group_rule(25, 3051), group_rule(25, 1932), group_rule(200, 800), group_rule(25, 60)]
          == [100, 100, 100, 12, 15])

    print("\n%s" % ("PASS: every onset-tracking test" if not FAILS else "FAIL: %s" % ", ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
