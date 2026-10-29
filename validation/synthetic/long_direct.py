#!/usr/bin/env python3
"""The "long direct" class: a robustness check of the operating domain, not a
validation.  The slow component of the real direct arrival,
measured on Viking, is added to synthetic records near the edge of the
operating domain, and the complete method is run with and without it.

  1. Viking (cube, first channels, every shot): the stack of the near traces
     aligned on x / vw, low-passed below 20 Hz, is the template; its envelope
     peak against that of the BAND-passed direct gives the amplitude ratio; the
     same ratio on the first 24 channels, fitted as (x / x0)^-p, gives the
     decay with offset
  2. records of picker_records.py with true separation between 5 T_dom
     and 300 ms (the edge: where the end of a 240-ms direct would become the
     active bound), and every 19th record above 300 ms as controls
  3. on each, the template is added at x_c / vw on every channel c, scaled by
     ratio x the synthetic direct's envelope peak x (x_c / x0)^-p
  4. seafloor_pipeline on the original and on the modified cube

    DATA=DIR python validation/synthetic/long_direct.py run RECORDS OUT [--jobs 8] [--only id,id]
    python validation/synthetic/long_direct.py report OUT

DATA holds viking_pick/viking_cube.npy and viking_conv41/viking_track_conv.npz
(the Viking cube and the first-pass tracking of validation/real/README.md).
--only restricts step 2 to the records named, after the selection.
"""
import collections
import glob
import json
import os
import subprocess
import sys

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [ROOT]
DATA = os.environ.get("DATA", ".")
from seaweed.run_real import BAND                                       # noqa: E402
from seaweed.wavelet_extractor_v2 import arrival_time, bandpass, envelope, estimate_vw_direct  # noqa: E402

# the template ends before the Viking seafloor (313 ms after the direct at its
# shallowest), tapered over its last 40 ms; a first version ran to 450 ms and
# carried the low-passed Viking seafloor with it
F_SLOW, N_CH, PRE, POST, TAPER = 20.0, 24, 0.050, 0.290, 0.040


def viking_template():
    g = np.load(os.path.join(DATA, "viking_pick", "viking_cube.npy"), mmap_mode="r")
    z = np.load(os.path.join(DATA, "viking_conv41", "viking_track_conv.npz"), allow_pickle=True)
    dt = float(z["dt"]); offs = np.asarray(z["offs"], float); vw = float(z["vw"])
    t0 = float(np.min(np.asarray(z["t0"], float)))               # shallowest seafloor of the line
    sos = butter(4, F_SLOW, "low", fs=1 / dt, output="sos")
    amps, ratios, tpl, info = [], [], None, {}
    for c in range(N_CH):
        x = offs[c]; td = x / vw
        sep = arrival_time(t0, x, vw) - td                          # seafloor after the direct on this channel
        post = min(POST, sep - 0.020)                               # measured before the seafloor only
        a = int(round((td - PRE) / dt)); b = int(round((td + POST) / dt))
        st = np.mean(np.asarray(g[:, a:b, c], float), axis=0)
        slow = sosfiltfilt(sos, st); full = bandpass(st, dt, BAND)
        n = int(round((PRE + post) / dt)); nt = int(round(TAPER / dt))
        w = np.r_[np.ones(n - nt), 0.5 * (1 + np.cos(np.linspace(0, np.pi, nt))), np.zeros(slow.size - n)]
        slow = slow * w
        amps.append(envelope(slow).max()); ratios.append(envelope(slow).max() / envelope(full[:n]).max())
        if c == 0:
            tpl = slow / envelope(slow).max()
            e = envelope(slow); half = float(np.sum(e > 0.5 * e.max()) * dt * 1e3)
            A = np.abs(np.fft.rfft(slow, 4096)); f = np.fft.rfftfreq(4096, dt); fpk = float(f[np.argmax(A)])
            info.update(post_ch0_ms=post * 1e3)
    p = -float(np.polyfit(np.log(offs[:N_CH] / offs[0]), np.log(np.array(amps) / amps[0]), 1)[0])
    info.update(dt=dt, x0=float(offs[0]), vw=vw, ratio=float(ratios[0]), decay_p=p, half_ms=half, f_peak=fpk,
                ratios=[float(r) for r in ratios])
    return tpl, info


def add_long_direct(g, dt, offs, vw, tpl, info):
    tp = resample_poly(tpl, int(round(info["dt"] / dt)), 1) if abs(info["dt"] - dt) > 1e-9 else tpl
    S, T, C = g.shape
    st = np.mean(np.asarray(g[:, :, 0], float), axis=0)
    td0 = offs[0] / vw
    a = max(0, int(round((td0 - 0.05) / dt))); b = int(round((td0 + 0.05) / dt))
    pk = envelope(bandpass(st, dt, BAND))[a:b].max()
    out = np.array(g, np.float32)
    n0 = int(round(PRE / dt))
    for c in range(C):
        amp = info["ratio"] * pk * (offs[c] / offs[0]) ** (-info["decay_p"])
        i0 = int(round(offs[c] / vw / dt)) - n0
        lo, hi = max(0, i0), min(T, i0 + tp.size)
        if hi > lo:
            out[:, lo:hi, c] += (amp * tp[lo - i0:hi - i0]).astype(np.float32)
    return out, float(pk)


def pipeline(cube, out, nshot, vw, dt, offs):
    fp = out + "_first_pass.npz"
    np.savez(fp, t0=np.zeros(nshot), vw=vw, dt=dt, offs=offs)
    rc = subprocess.run([sys.executable, "-m", "seaweed.seafloor_pipeline", out, "--segy", "synthetic", "--nshot",
                         str(nshot), "--dxsp", "25", "--load_cube", cube, "--first_pass", fp, "--n_ch_track", "3"],
                        capture_output=True, text=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1"))
    open(out + ".log", "w").write(rc.stdout + rc.stderr)
    return rc.returncode


def one(args):
    cube, out, tpl, info, role = args
    uid = os.path.basename(cube)[:-len("_cube.npy")]
    meta = np.load(cube + ".meta.npz", allow_pickle=True); tr = np.load(cube.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
    dt = float(meta["dt"]); offs = np.asarray(meta["off_hdr"], float); g = np.load(cube, mmap_mode="r")
    vw, _x, _n = estimate_vw_direct(np.asarray(g), dt, offs, D_guess=float(np.median(tr["D"])))
    gl, pk = add_long_direct(g, dt, offs, float(tr["vw_true"]), tpl, info)
    d = os.path.join(out, uid); os.makedirs(d)
    lc = os.path.join(d, "long_cube.npy"); np.save(lc, gl)
    np.savez(lc + ".meta.npz", **{k: meta[k] for k in meta.files})
    t0t = np.asarray(tr["t0_true"], float); T = float(tr["tdom_true_ms"])
    sep = float(np.median(arrival_time(t0t, offs[0], float(tr["vw_true"])) - offs[0] / float(tr["vw_true"]))) * 1e3
    res = dict(uid=uid, role=role, T_ms=T, sep_ms=sep, sep_T=sep / T, direct_peak=pk)
    for lab, c in (("orig", cube), ("long", lc)):
        rc = pipeline(c, os.path.join(d, lab), g.shape[0], vw, dt, offs)
        if rc:
            res[lab + "_error"] = rc; continue
        p = os.path.join(d, lab)
        a = np.load(os.path.join(p, "stage1_anchors.npz"), allow_pickle=True)
        pa = json.loads(str(a["params"]))
        z = np.load(os.path.join(p, "track.npz"), allow_pickle=True)
        s = json.load(open(os.path.join(p, "summary.json")))
        e1 = float(np.median(np.asarray(a["t"]) - t0t[np.asarray(a["i"], int)])) * 1e3
        ec = float(np.median(np.asarray(z["t0"], float) - t0t)) * 1e3
        res[lab] = dict(direct_end_ms=pa["direct_end"].get("duration_ms"), dropped=int(sum(pa["candidates_dropped"])),
                        s1_err_ms=e1, complete_err_ms=ec, s1_cycle=bool(abs(e1) < T / 2), complete_cycle=bool(abs(ec) < T / 2),
                        h3_applicable=s["h3_applicable"], reason=s["h3_fallback_reason"], domain=s["domain"]["flag"],
                        active=s["domain"].get("active"), threshold_ms=s["domain"]["threshold_ms"])
    os.remove(lc)
    return res


def run(records, out, jobs, only=None):
    from multiprocessing import Pool
    os.makedirs(out)
    tpl, info = viking_template()
    json.dump(info, open(os.path.join(out, "viking_slow_direct.json"), "w"), indent=1)
    np.save(os.path.join(out, "viking_slow_direct_template.npy"), tpl)
    cubes = [c for c in sorted(glob.glob(os.path.join(records, "*_cube.npy")))
             if str(np.load(c.replace("_cube.npy", "_truth.npz"), allow_pickle=True)["kind"]) in ("grid", "sec")]
    seps = []
    for c in cubes:
        tr = np.load(c.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
        x0 = float(np.asarray(np.load(c + ".meta.npz", allow_pickle=True)["off_hdr"])[0]); v = float(tr["vw_true"])
        s = float(np.median(arrival_time(np.asarray(tr["t0_true"], float), x0, v) - x0 / v)) * 1e3
        seps.append((c, s, float(tr["tdom_true_ms"])))
    edge = [(c, "edge") for c, s, T in seps if 5 * T <= s < 300.0]
    ctrl = [(c, "control") for k, (c, s, T) in enumerate([x for x in seps if x[1] >= 300.0]) if k % 19 == 0]
    print("template: %s; %d edge records, %d controls" % (json.dumps(info), len(edge), len(ctrl)), flush=True)
    if only:
        edge = [x for x in edge if os.path.basename(x[0])[:-len("_cube.npy")] in only]
        ctrl = [x for x in ctrl if os.path.basename(x[0])[:-len("_cube.npy")] in only]
    with Pool(jobs) as pool, open(os.path.join(out, "records.jsonl"), "w") as fh:
        for r in pool.imap_unordered(one, [(c, out, tpl, info, role) for c, role in edge + ctrl]):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()


def report(d):
    R = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
    info = json.load(open(os.path.join(d, "viking_slow_direct.json")))
    print("# Long-direct class (robustness check, not validation): `%s`, %d records\n" % (d, len(R)))
    print("Viking slow direct (< 20 Hz, before the seafloor, tapered): peak %.1f Hz, %.0f ms above half its peak, %.3f of the "
          "band-passed direct on the near channel, decay (x/x0)^-%.2f\n" % (info["f_peak"], info["half_ms"], info["ratio"], info["decay_p"]))
    print("| role | n | pipeline failures orig / long | end of the direct, median ms orig / long | active bound = end of the direct orig / long "
          "| candidates dropped orig / long | stage 1 on the cycle orig / long | complete on the cycle orig / long | complete within -100/+50 orig / long "
          "| onset tracking not applicable orig / long | domain flags (long) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for role in ("edge", "control"):
        L = [r for r in R if r["role"] == role]
        g = lambda lab, k: [r[lab][k] for r in L if lab in r]
        med = lambda lab: np.median([x for x in g(lab, "direct_end_ms") if x is not None])
        print("| %s | %d | %d / %d | %.0f / %.0f | %d / %d | %d / %d | %d / %d | %d / %d | %d / %d | %d / %d | %s |" % (
            role, len(L), sum(1 for r in L if "orig" not in r), sum(1 for r in L if "long" not in r), med("orig"), med("long"),
            sum(a == "end of the direct" for a in g("orig", "active")), sum(a == "end of the direct" for a in g("long", "active")),
            sum(g("orig", "dropped")), sum(g("long", "dropped")), sum(g("orig", "s1_cycle")), sum(g("long", "s1_cycle")),
            sum(g("orig", "complete_cycle")), sum(g("long", "complete_cycle")),
            sum(-100 <= e <= 50 for e in g("orig", "complete_err_ms")), sum(-100 <= e <= 50 for e in g("long", "complete_err_ms")),
            sum(not a for a in g("orig", "h3_applicable")), sum(not a for a in g("long", "h3_applicable")),
            dict(collections.Counter("%s (%s)" % (a, b) for a, b in zip(g("long", "domain"), g("long", "active"))))))
    E = [r for r in R if r["role"] == "edge"]
    print("\nedge records: separation %.0f-%.0f ms (%.1f-%.1f T_dom)\n" % (min(r["sep_ms"] for r in E), max(r["sep_ms"] for r in E),
          min(r["sep_T"] for r in E), max(r["sep_T"] for r in E)))
    print("| record | role | separation (ms) | end of the direct orig -> long (ms) | dropped long | complete error orig -> long (ms) | domain long (active) |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(R, key=lambda r: (r["role"], r["sep_ms"])):
        o = r.get("orig", {}); l = r.get("long", {})
        f = lambda x: "%.0f" % x if x is not None else "-"
        print("| %s | %s | %.0f | %s -> %s | %s | %s -> %s | %s (%s) |" % (
            r["uid"], r["role"], r["sep_ms"], f(o.get("direct_end_ms")), f(l.get("direct_end_ms")), l.get("dropped", "-"),
            "%+.0f" % o["complete_err_ms"] if o else "FAIL", "%+.0f" % l["complete_err_ms"] if l else "FAIL", l.get("domain"), l.get("active")))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
        only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
        run(sys.argv[2], sys.argv[3], jobs, only)
    else:
        report(sys.argv[2])
