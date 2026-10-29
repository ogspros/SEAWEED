#!/usr/bin/env python3
"""The bench of 72 synthetic lines (Sect. 3.2): tracking along a line with
explicit bathymetry, stage 1 against the onset tracking.

Lines: 60 shots every 125 m, 4 seafloor depths under the first shot (120, 250,
450, 650 m) x 3 slopes (0, 0.46, 1.34 degrees: about 0, 12 and 35 ms over 10
shots) x 2 classes (R0: first reflector 150 m below the seafloor; R1: 20 m
below, about one period) x 3 seeds; 3 m RMS roughness, 60 Hz, 3000-m streamer.

  generate  fire every line in groups of 10 shots, one job per group
            (OUT/bench_groups), then assemble one cube per line (OUT/bench:
            <id>_cube.npy, .meta.npz, _truth.npz, the layout of picker_records.py).
            About 8 hours on 48 processes
  run       per line: stage-1 anchors (anchors_from_consensus --group 10
            --tol_ms 120 --max_shots 48), the stage-1 tracking (run_real, 500-m
            window, three channels) and the onset tracking (h3_track h2); the
            first pass gives only the sampling, offsets and the water velocity
            of the direct arrival
  report    a line passes when >= 90 % of its shots are within T_dom/2 of the
            true seafloor (arrival time at the near trace) and the RMS of the
            medians of 20-shot blocks, median removed, is <= T_dom/4 (true
            T_dom); the complete method is the onset tracking where applicable,
            stage 1 where not

    python validation/synthetic/bench_lines.py generate OUT [--jobs 48]
    python validation/synthetic/bench_lines.py run BENCH OUT [--jobs 12]
    python validation/synthetic/bench_lines.py report BENCH OUT
"""
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("DEVITO_LOGGING", "ERROR")
os.environ.setdefault("DEVITO_ARCH", "gcc")
os.environ.pop("DEVITO_PLATFORM", None)
os.environ.setdefault("DEVITO_LANGUAGE", "C")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]
import numpy as np                                                      # noqa: E402

CONFIG = os.path.join(HERE, "matrix_v2.yaml")
B_DEPTHS = (120.0, 250.0, 450.0, 650.0)
B_SLOPES = (0.0, 0.46, 1.34)
B_REFL = (("R0", 150.0), ("R1", 20.0))       # first_layer_m
B_SEED0, B_REPS = 920000, 3
ROUGH_M = 3.0
N_LINE, STRIDE, GROUP = 60, 5, 10
DX = 125.0                                    # shot spacing, m (every 5th position at 25 m)


def bench_lines():
    out = []
    for i_r, (rname, fl) in enumerate(B_REFL):
        for i_d, D in enumerate(B_DEPTHS):
            for i_s, sl in enumerate(B_SLOPES):
                for rep in range(B_REPS):
                    seed = B_SEED0 + 1000 * i_r + 100 * i_d + 10 * i_s + rep
                    out.append(dict(run_id="C_%s_D%d_s%s_r%d" % (rname, D, ("%.2f" % sl).replace(".", "p"), rep),
                                    axis="C", seed=seed, refl=rname, slope_deg=sl,
                                    params=dict(geom="g3000", D_m=D, fmax_hz=60.0, first_layer_m=fl, slope_deg=sl,
                                                roughness_m=ROUGH_M, p_fault=0.0, p_folds=0.0, p_lens=0.0)))
    return out


def model_and_shots(run, cfg, shot_idx, n_shots):
    """The shots shot_idx of a bench line, with the bathymetry fixed by slope_deg
    and roughness_m, ZMAX from the deepest seafloor of the line and the record
    length from it."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    import generate_line2_hf as G
    from velocity_generator import generate_velocity_model
    from generate_parallel_hf import make_source_fn
    from examples.seismic import Model
    p = run["params"]; m = cfg["meta"]
    geom = G.GEOMS[p["geom"]]; offs = G.geom_offsets(geom)
    XMAX = G.domain_for_shots_geom(n_shots + 4, geom)
    xs_all = G.shot_positions_geom(XMAX, geom)
    # the seafloor is D_m under the first shot of the line and deepens by
    # tan(slope) per metre along it: the generator's slope starts at x = 0
    tn_ = np.tan(np.radians(p["slope_deg"]))
    x_first = xs_all[0] - offs[0] / 2.
    x_last = xs_all[n_shots - 1] - offs[0] / 2.
    d_gen = p["D_m"] - tn_ * x_first
    dmax = p["D_m"] + tn_ * (x_last - x_first) + 4 * ROUGH_M
    rng = np.random.default_rng(run["seed"])
    vp, rho, kwb, meta = generate_velocity_model(
        rng, XMAX=XMAX, ZMAX=dmax + 1600.0, DH=G.DH,
        D_water_range=(d_gen, d_gen),
        p_fault=p["p_fault"], p_folds=p["p_folds"], p_lens=p["p_lens"],
        slope_deg=p["slope_deg"], roughness_m=p["roughness_m"],
        first_layer_m=p["first_layer_m"])
    model = Model(vp=(vp / 1000.), b=1.0 / rho, origin=(0., 0.), shape=vp.shape,
                  spacing=(G.DH, G.DH), space_order=8, nbl=G.NB, bcs="damp", fs=True)
    shot_idx = np.asarray(shot_idx)
    src_fn, w_bare = make_source_fn(10_000 + run["seed"], fmax=p["fmax_hz"])
    dt_out = m["dt_ms"] / 1000.
    tn = float(np.clip(2.6 * (2 * dmax / m["v_water"]) * 1000. + 800., 1500., 4500.))
    NT = int(round(tn / 1000. / dt_out))
    g = np.zeros((len(shot_idx), NT, len(offs)), np.float32)
    for j, i in enumerate(shot_idx):
        d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offs)
        fr = Fraction(float(dtd) / 1000. / dt_out).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
        d = d[:NT] if d.shape[0] > NT else np.pad(d, ((0, NT - d.shape[0]), (0, 0)))
        g[j] = d
    idx = np.clip(((xs_all[shot_idx] - offs[0] / 2.) / G.DH).astype(int), 0, vp.shape[0] - 1)
    j0 = int(np.median(idx)); kw = int(kwb[j0])
    col = vp[j0, kw:]; dv = np.flatnonzero(np.abs(np.diff(col)) > 20.0)
    refl = None
    if dv.size:
        k = int(dv[0]) + 1
        tw = 2.0 * np.sum(G.DH / col[:k])
        z1, z2 = col[k - 1] * rho[j0, kw + k - 1], col[k] * rho[j0, kw + k]
        refl = dict(delay_ms=float(tw * 1e3), rc=float((z2 - z1) / (z2 + z1)))
    return g, dt_out, offs, w_bare, kwb[idx] * G.DH, dict(tn_ms=tn, dmax_m=float(dmax), xmax_m=float(XMAX), refl_info=refl)


def first_zero_tdom(st, dt):
    st = st - st.mean()
    ac = np.correlate(st, st, "full")[st.size - 1:]
    i = int(np.argmax(ac <= 0))
    if i < 1:
        return np.nan
    return 4.0 * dt * (i - 1 + ac[i - 1] / (ac[i - 1] - ac[i]))


def job_group(args):
    run, q, cfg, out = args
    t = time.time()
    try:
        shots = (np.arange(N_LINE) * STRIDE)[q * GROUP:(q + 1) * GROUP]
        g, dt, offs, w_bare, D, info = model_and_shots(run, cfg, shots, 1 + (N_LINE - 1) * STRIDE)
    except Exception as e:
        return dict(uid=run["run_id"], unit="g%d" % q, error=str(e)[:300])
    np.savez(os.path.join(out, "bench_groups", "%s_g%d.npz" % (run["run_id"], q)), g=g, dt=dt, offs=offs,
             w_bare=w_bare, D=D, info=json.dumps(info))
    return dict(uid=run["run_id"], unit="g%d" % q, seconds=time.time() - t)


def generate(out, jobs, only=None):
    import yaml
    from multiprocessing import Pool
    os.environ["OMP_NUM_THREADS"] = "1"
    cfg = yaml.safe_load(open(CONFIG))
    os.makedirs(os.path.join(out, "bench_groups"), exist_ok=True)
    J = [(r, q, cfg, out) for r in sorted(bench_lines(), key=lambda r: (-r["params"]["D_m"], -r["slope_deg"]))
         for q in range(N_LINE // GROUP)
         if (only is None or "%s_g%d" % (r["run_id"], q) in only)
         and not os.path.exists(os.path.join(out, "bench_groups", "%s_g%d.npz" % (r["run_id"], q)))]
    with Pool(jobs, maxtasksperchild=1) as pool, open(os.path.join(out, "jobs.jsonl"), "a") as fh:
        for r in pool.imap_unordered(job_group, J):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("%s %s %s" % (r["uid"], r["unit"], r.get("error", "%.0f s" % r.get("seconds", 0))), flush=True)
    if only is None:
        assemble(out, cfg)


def assemble(out, cfg):
    from airgun_generator import DT as AG_DT
    b = os.path.join(out, "bench"); os.makedirs(b, exist_ok=True)
    vwt = float(cfg["meta"]["v_water"])
    for run in bench_lines():
        fs = [os.path.join(out, "bench_groups", "%s_g%d.npz" % (run["run_id"], q)) for q in range(N_LINE // GROUP)]
        if not all(os.path.exists(f) for f in fs):
            print("incomplete", run["run_id"]); continue
        zs = [np.load(f, allow_pickle=True) for f in fs]
        g = np.concatenate([z["g"] for z in zs]); D = np.concatenate([z["D"] for z in zs])
        uid = run["run_id"]
        np.save(os.path.join(b, uid + "_cube.npy"), np.ascontiguousarray(g, np.float32))
        np.savez(os.path.join(b, uid + "_cube.npy.meta.npz"), dt=float(zs[0]["dt"]),
                 off_hdr=np.asarray(zs[0]["offs"], np.float32), shot0=-1, nshot=g.shape[0], segy=uid)
        np.savez(os.path.join(b, uid + "_truth.npz"), D=D, vw_true=vwt, t0_true=2 * D / vwt,
                 tdom_true_ms=1e3 * first_zero_tdom(np.asarray(zs[0]["w_bare"], float), AG_DT), w_bare=zs[0]["w_bare"],
                 kind=run["refl"], slope_deg=run["slope_deg"], D0=run["params"]["D_m"], info=str(zs[0]["info"]),
                 params=json.dumps(run["params"]), shot_spacing_m=DX)
        print("assembled", uid, g.shape)


def sh(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                              env=dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")).returncode


def one(args):
    cube, out = args
    from seaweed.wavelet_extractor_v2 import estimate_vw_direct
    PY = sys.executable
    uid = os.path.basename(cube)[:-len("_cube.npy")]
    o = os.path.join(out, uid); os.makedirs(o)
    g = np.load(cube, mmap_mode="r"); meta = np.load(cube + ".meta.npz", allow_pickle=True)
    tr = np.load(cube.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
    dt = float(meta["dt"]); offs = np.asarray(meta["off_hdr"], float); S = g.shape[0]
    ng = max(5, int(round(500.0 / DX)))
    vw, _x, _n = estimate_vw_direct(np.asarray(g), dt, offs, D_guess=float(np.median(tr["D"])))
    res = dict(uid=uid, vw_est=float(vw))
    tin = os.path.join(o, "first_pass.npz"); np.savez(tin, t0=np.zeros(S), vw=vw, dt=dt, offs=offs)
    s1a, s1t = os.path.join(o, "s1_anchors.npz"), os.path.join(o, "s1_track.npz")
    if sh([PY, "-m", "seaweed.anchors_from_consensus", cube, tin, s1a, "--group", "10", "--tol_ms", "120",
           "--max_shots", "48"], os.path.join(o, "s1_anchors.log")) or not os.path.exists(s1a):
        res["error"] = "stage-1 anchors"; return res
    sh([PY, "-m", "seaweed.run_real", "--segy", uid, "--nshot", str(S), "--load_cube", cube, "--load_anchors", s1a,
        "--save_track", s1t, "--n_group", str(ng), "--n_ch_track", "3", "--t0_shift", "0", "--out", os.path.join(o, "s1")],
       os.path.join(o, "s1_track.log"))
    if not os.path.exists(s1t):
        res["error"] = "stage-1 tracking"; return res
    sh([PY, "-m", "seaweed.h3_track", cube, s1t, s1a, os.path.join(o, "h2_track.npz"), "--dxsp", str(DX),
        "--variant", "h2", "--n_ch_track", "3", "--n_group", str(ng)], os.path.join(o, "h2_track.log"))
    return res


def run(bench, out, jobs):
    from multiprocessing import Pool
    os.makedirs(out)
    with Pool(jobs) as pool, open(os.path.join(out, "lines.jsonl"), "w") as fh:
        for r in pool.imap_unordered(one, [(c, out) for c in sorted(glob.glob(os.path.join(bench, "*_cube.npy")))]):
            fh.write(json.dumps(r) + "\n"); fh.flush()


def line_ok(dd, T):
    dd = np.asarray(dd, float)
    inside = float(np.mean(np.abs(dd) <= T / 2))
    med = np.median(dd)
    bm = np.array([np.median(dd[b:b + 20]) - med for b in range(0, dd.size, 20) if dd[b:b + 20].size >= 10])
    rms = float(np.sqrt(np.mean(bm ** 2))) if bm.size else np.nan
    return inside >= 0.90 and rms <= T / 4


def report(bench, out):
    from seaweed.wavelet_extractor_v2 import arrival_time
    res = {"stage 1": {}, "onset tracking": {}, "complete": {}}
    for c in sorted(glob.glob(os.path.join(bench, "*_cube.npy"))):
        uid = os.path.basename(c)[:-len("_cube.npy")]
        tr = np.load(c.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
        x0 = float(np.asarray(np.load(c + ".meta.npz", allow_pickle=True)["off_hdr"])[0])
        t0t = np.asarray(tr["t0_true"], float); vwt = float(tr["vw_true"]); T = float(tr["tdom_true_ms"]) * 1e-3
        cls = str(tr["kind"])
        d = {}
        for k, f in (("stage 1", "s1_track.npz"), ("onset tracking", "h2_track.npz")):
            p = os.path.join(out, uid, f)
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                d[k] = line_ok(arrival_time(np.asarray(z["t0"], float), x0, float(z["vw"])) - arrival_time(t0t, x0, vwt), T)
        d["complete"] = d.get("onset tracking", d.get("stage 1", False))
        for k in res:
            res[k].setdefault(cls, []).append(bool(d.get(k, False)))
    print("# Bench of 72 lines (`%s`)\n" % out)
    print("| tracking | R0 lines passing | R1 lines passing | total |\n|---|---|---|---|")
    for k, v in res.items():
        print("| %s | %d / %d | %d / %d | %d |" % (k, sum(v.get("R0", [])), len(v.get("R0", [])), sum(v.get("R1", [])),
                                                len(v.get("R1", [])), sum(v.get("R0", [])) + sum(v.get("R1", []))))


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else None
    if sys.argv[1] == "generate":
        only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
        generate(sys.argv[2], jobs or 48, only)
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], jobs or 12)
    else:
        report(sys.argv[2], sys.argv[3])
