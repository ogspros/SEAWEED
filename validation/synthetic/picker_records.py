#!/usr/bin/env python3
"""The synthetic records of the picker tests, generated once from their seeds
and kept as cubes, so that every chain runs on the same data.

  grid   the 56 main-grid records of the picker test (eval_picker.py)
  sec    the 160 secondary-axis records (eval_sec)
  calib  24 main-grid records, one per cell, disjoint from the 56: the
         calibration of the convention constant of the onset tracking

The run ids are in picker_run_ids.json; models and shots are those of the v2
matrix (matrix_v2.yaml and its manifest), regenerated as make_fig_synthetic.py
does.  Per record: <id>_cube.npy (shots, samples, channels; float32),
<id>_cube.npy.meta.npz (the layout of run_real --save_cube) and <id>_truth.npz
(depth under every shot, true water velocity and t0, true T_dom, the bare
source wavelet, the model parameters).  About 15 minutes per record on one
core.

    python validation/synthetic/picker_records.py --data DIR --out OUT [--kinds grid,sec] [--only id,id] [--jobs 48]

DIR holds wavelet_matrix_v2/manifest.jsonl.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("DEVITO_LOGGING", "ERROR")
os.environ.setdefault("DEVITO_ARCH", "gcc")
os.environ.pop("DEVITO_PLATFORM", None)
os.environ.setdefault("DEVITO_LANGUAGE", "C")
os.environ["OMP_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [ROOT, HERE]
import numpy as np                                                      # noqa: E402

CONFIG = os.path.join(HERE, "matrix_v2.yaml")
IDS = os.path.join(HERE, "picker_run_ids.json")


def model_and_shots(run, cfg):
    """make_fig_synthetic.regenerate, returning the depth under every shot."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    import generate_line2_hf as G
    from velocity_generator import generate_velocity_model
    from generate_parallel_hf import make_source_fn
    from examples.seismic import Model
    p = run["params"]; der = run["derived"]; m = cfg["meta"]
    geom = G.GEOMS[p["geom"]]; offs = G.geom_offsets(geom)
    n_shots = m["n_shots"]
    XMAX = G.domain_for_shots_geom(n_shots + 4, geom)
    rng = np.random.default_rng(run["seed"])
    vp, rho, kwb, meta = generate_velocity_model(
        rng, XMAX=XMAX, ZMAX=der["ZMAX_m"], DH=G.DH,
        D_water_range=(p["D_m"], p["D_m"]),
        p_fault=p["p_fault"], p_folds=p["p_folds"], p_lens=p["p_lens"],
        slope_deg=p["slope_deg"], roughness_m=p["roughness_m"],
        first_layer_m=p["first_layer_m"])
    model = Model(vp=(vp / 1000.), b=1.0 / rho, origin=(0., 0.), shape=vp.shape,
                  spacing=(G.DH, G.DH), space_order=8, nbl=G.NB, bcs="damp", fs=True)
    xs_all = G.shot_positions_geom(XMAX, geom)
    n_shots = min(n_shots, len(xs_all))
    shot_idx = np.arange(n_shots)
    src_fn, w_bare = make_source_fn(10_000 + run["seed"], fmax=p["fmax_hz"])
    dt_out = m["dt_ms"] / 1000.
    tn = float(np.clip(2.6 * (2 * p["D_m"] / m["v_water"]) * 1000. + 800.,
                       1500., 4500.))
    NT = int(round(tn / 1000. / dt_out))
    g = np.zeros((len(shot_idx), NT, len(offs)), np.float32)
    zs = p.get("z_src_m", None); zr = p.get("z_rec_m", None)
    for j, i in enumerate(shot_idx):
        try:
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offs,
                             z_src=zs, z_rec=zr)
        except TypeError:
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offs)
        fr = Fraction(float(dtd) / 1000. / dt_out).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
        d = d[:NT] if d.shape[0] > NT else np.pad(d, ((0, NT - d.shape[0]), (0, 0)))
        g[j] = d
    idx = np.clip(((xs_all[shot_idx] - offs[0] / 2.) / G.DH).astype(int),
                  0, vp.shape[0] - 1)
    return g, dt_out, offs, w_bare, kwb[idx] * G.DH


def t_dom_true(w_bare):
    """Four times the first zero of the autocorrelation of the bare source."""
    from airgun_generator import DT as AG_DT
    w = np.asarray(w_bare, float); w = w - w.mean()
    ac = np.correlate(w, w, "full")[w.size - 1:]
    i = int(np.argmax(ac <= 0))
    return 4e3 * AG_DT * (i - 1 + ac[i - 1] / (ac[i - 1] - ac[i]))


def job(args):
    run, kind, cfg, out = args
    t = time.time()
    try:
        g, dt, offs, w_bare, D = model_and_shots(run, cfg)
    except Exception as e:
        return dict(uid=run["run_id"], kind=kind, error=str(e)[:300])
    vwt = float(cfg["meta"]["v_water"]); uid = run["run_id"]
    np.save(os.path.join(out, uid + "_cube.npy"), np.ascontiguousarray(g, np.float32))
    np.savez(os.path.join(out, uid + "_cube.npy.meta.npz"), dt=dt, off_hdr=np.asarray(offs, np.float32),
             shot0=-1, nshot=g.shape[0], segy=uid)
    np.savez(os.path.join(out, uid + "_truth.npz"), D=D, vw_true=vwt, t0_true=2 * D / vwt,
             tdom_true_ms=t_dom_true(w_bare), w_bare=w_bare, kind=kind, params=json.dumps(run["params"]))
    return dict(uid=uid, kind=kind, shots=int(g.shape[0]), seconds=time.time() - t)


def main():
    import yaml
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory holding wavelet_matrix_v2/manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="grid,sec")
    ap.add_argument("--only", default=None, help="comma-separated run ids")
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(a.out + " exists")
    os.makedirs(a.out)
    cfg = yaml.safe_load(open(CONFIG))
    man = {json.loads(l)["run_id"]: json.loads(l) for l in open(os.path.join(a.data, "wavelet_matrix_v2", "manifest.jsonl"))}
    ids = json.load(open(IDS))
    only = set(a.only.split(",")) if a.only else None
    jobs = [(man[r], k, cfg, a.out) for k in a.kinds.split(",") for r in ids[k] if only is None or r in only]
    with Pool(a.jobs, maxtasksperchild=1) as pool, open(os.path.join(a.out, "jobs.jsonl"), "w") as fh:
        for r in pool.imap_unordered(job, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print("%s %s %s" % (r["uid"], r["kind"], r.get("error", "%.0f s" % r.get("seconds", 0))), flush=True)


if __name__ == "__main__":
    main()
