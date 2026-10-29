"""Runner of the v7 matrix: the synthetics of the paper with the final recipe,
or, with --recipe paper, as published.

Every model is shot once (run_matrix_v5.generate_model) and then goes through
the passes of its matrix, each the unchanged process() of run_matrix_v5 (true
seafloor) or run_matrix_v6 (automatic chain) fed with those gathers:

  v2       v5_bias (30 dB, t0 bias -100..+100 ms), v5_noise (0-30 dB), v6 (chain)
  shallow  shallow_v5 (30 dB), shallow_chain (30 dB, final recipe only)
  z        z_v5 (30 dB)

--recipe paper  the passes as published: extract_wavelet with the single
                alignment, the chain of pick_consensus.pick_line (groups of 10,
                120 ms) interpolated between group centres
--recipe final  extract_wavelet with the two-stage alignment (align_group =
                n_group of the 500-m rule) and the bistability check of
                run_real on every primary; the chain is seaweed.seafloor_pipeline
                on the noise-free gathers, as v6 picked them (stage 1 with the
                end-of-direct rejection, --group from the rule, h3_track h2 with
                its fallback), whose tracking replaces pick_line; fallback,
                reason and operating domain are recorded per record

The scoring (origin bias, overlap, NCC against the synthetic truth) is that of
run_matrix_v5, unchanged.  The configurations and manifests are those of the
published runs, read from --data (read only).

    python validation/synthetic/run_matrix_v7.py --matrix v2 --recipe final \\
        --data DIR --out OUT --shard 0 --n_shards 48 [--only id,id]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, ROOT]
import numpy as np                                                      # noqa: E402
import yaml                                                             # noqa: E402

import run_matrix_v5 as R5                                              # noqa: E402  (sets the Devito env)
import run_matrix_v6 as R6                                              # noqa: E402
from seaweed import pick_consensus as PC                               # noqa: E402
from seaweed.run_real import bistability, n_group_rule                 # noqa: E402

def matrices(d):
    """The passes of each matrix, with the configurations and manifests of the
    published runs under d (--data)."""
    return {
        "v2": dict(manifest=d + "/wavelet_matrix_v2/manifest.jsonl",
                   passes=[("v5_bias", d + "/matrix_v5_bias.yaml", R5), ("v5_noise", d + "/matrix_v5_noise.yaml", R5),
                           ("v6", d + "/matrix_v6.yaml", R6)]),
        "shallow": dict(manifest=d + "/wavelet_matrix_shallow/manifest.jsonl",
                        passes=[("shallow_v5", d + "/matrix_v5_shallow.yaml", R5),
                                ("shallow_chain", d + "/matrix_v6.yaml", R6, "final")]),
        "z": dict(manifest=d + "/wavelet_matrix_z/manifest.jsonl",
                  passes=[("z_v5", d + "/matrix_v5_z.yaml", R5)]),
    }


DXSP = R5.G.SHOT_DX                 # shot spacing of the synthetic lines, 25 m
STATE = {}                          # what the patched functions leave for the records


def final_extract(orig, ng):
    """extract_wavelet with the final recipe: two-stage alignment; the
    bistability check of run_real on each 200-ms primary."""
    def f(gathers, dt, t0, offsets, *a, **k):
        k.setdefault("align", "two_stage"); k.setdefault("align_group", ng)
        w, q = orig(gathers, dt, t0, offsets, *a, **k)
        # the primary of each record is the first successful order-0 call on its
        # gathers and window (the comparison extraction that follows reuses both)
        key = (id(gathers), id(t0))
        if w is not None and k.get("order", 0) == 0 and k.get("template") is None \
                and k.get("w_len_ms") == STATE.get("w_len_ms") and key not in STATE.setdefault("seen", set()):
            STATE["seen"].add(key)
            b = bistability(q, dt)
            STATE.setdefault("bistab", []).append(dict(bistable=bool(b["bistable"]), bistab_blocks=int(b["n_blocks"]),
                                                       bistab_out=int(b["n_out"])))
        return w, q
    return f


def final_pick_line(g, dt, offs, vw, group=None, tol_ms=None, **_):
    """The tracking of seafloor_pipeline on these gathers, in place of pick_line."""
    tmp = tempfile.mkdtemp(prefix="v7_", dir=STATE["tmp"])
    try:
        cube = os.path.join(tmp, "cube.npy")
        np.save(cube, np.ascontiguousarray(g, np.float32))
        np.savez(cube + ".meta.npz", dt=dt, off_hdr=np.asarray(offs, np.float32), shot0=-1, nshot=g.shape[0],
                 segy="synthetic")
        fp = os.path.join(tmp, "first_pass.npz")
        np.savez(fp, t0=np.zeros(g.shape[0]), vw=vw, dt=dt, offs=np.asarray(offs, float))
        out = os.path.join(tmp, "p")
        rc = subprocess.run([sys.executable, "-m", "seaweed.seafloor_pipeline", out, "--segy", "synthetic",
                             "--nshot", str(g.shape[0]), "--dxsp", str(DXSP), "--load_cube", cube, "--first_pass", fp,
                             "--n_ch_track", "3"], capture_output=True, text=True, cwd=ROOT,
                            env=dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1")).returncode
        if rc:
            raise RuntimeError("seafloor_pipeline exit %d" % rc)
        z = np.load(os.path.join(out, "track.npz"), allow_pickle=True)
        s = json.load(open(os.path.join(out, "summary.json")))
        STATE["chain"] = dict(h3_applicable=s["h3_applicable"], h3_fallback=s["h3_fallback"],
                              h3_fallback_reason=s["h3_fallback_reason"],
                              h3_unreliable_fraction=s["h3_unreliable_fraction"], stage1_group=s["group"],
                              domain_flag=s["domain"]["flag"], domain_active=s["domain"].get("active"),
                              domain_threshold_ms=s["domain"]["threshold_ms"],
                              direct_ms=s["domain"].get("direct_ms"), t_dom_ms=s["domain"]["t_dom_ms"])
        return np.asarray(z["t0"], float), {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_passes(run, mat, recipe):
    cfgs = [(p[0], yaml.safe_load(open(p[1])), p[2], (p[3] if len(p) > 3 else None)) for p in mat["passes"]]
    md = R5.generate_model(run, cfgs[0][1])
    out = []
    for name, cfg, mod, only in cfgs:
        if only and only != recipe:
            continue
        saved = (mod.generate_model, mod.extract_wavelet, PC.pick_line)
        mod.generate_model = lambda r, c, _md=md: _md
        STATE.clear(); STATE.update(tmp=STATE_TMP, w_len_ms=cfg["meta"]["w_len_ms"])
        try:
            if recipe == "final":
                mod.extract_wavelet = final_extract(saved[1], n_group_rule(DXSP))
                PC.pick_line = final_pick_line
            t = time.time()
            recs = mod.process(run, cfg)
        finally:
            mod.generate_model, mod.extract_wavelet, PC.pick_line = saved
        bl = iter(STATE.get("bistab", []))
        for r in recs:
            r.update(pass_name=name, recipe=recipe, pass_seconds=time.time() - t)
            if recipe == "final" and r.get("ok"):
                r.update(next(bl, {}))
            if "chain" in STATE:
                r.update(STATE["chain"])
        out += recs
    return out


def main():
    global STATE_TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=("v2", "shallow", "z"), required=True)
    ap.add_argument("--recipe", choices=("paper", "final"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True,
                    help="directory holding the configurations (matrix_*.yaml) and the manifests "
                         "(wavelet_matrix_*/manifest.jsonl) of the published runs, read only")
    ap.add_argument("--only", default=None, help="comma-separated run_ids")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()
    mat = matrices(os.path.abspath(a.data))[a.matrix]
    runs = [json.loads(l) for l in open(mat["manifest"])]
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        runs = [r for r in runs if r["run_id"] in want]
        if want - {r["run_id"] for r in runs}:
            raise SystemExit("unknown run_id: %s" % ", ".join(sorted(want - {r["run_id"] for r in runs})))
    runs = runs[a.shard::a.n_shards]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    STATE_TMP = str(out / "tmp"); os.makedirs(STATE_TMP, exist_ok=True)
    rp = out / f"results_{a.matrix}_shard{a.shard:03d}.jsonl"
    done = {json.loads(l)["run_id"] for l in open(rp)} if rp.exists() else set()
    runs = [r for r in runs if r["run_id"] not in done]
    print(f"[{a.matrix} {a.recipe} shard {a.shard}] {len(runs)} runs ({len(done)} already done)", flush=True)
    wp_path = out / f"results_{a.matrix}_shard{a.shard:03d}_wavelets.npz"
    waves = {}
    if wp_path.exists():
        with np.load(wp_path) as z:
            waves = {k: z[k] for k in z.files}
    t0 = time.time()
    with open(rp, "a") as f:
        for i, r in enumerate(runs, 1):
            for rec in run_passes(r, mat, a.recipe):
                w_arr = rec.pop("_wavelet", None); r_arr = rec.pop("_reference", None)
                if w_arr is not None:
                    key = f"{rec['pass_name']}|{rec['run_id']}|{rec.get('snr_db')}|{rec.get('t0_bias_ms')}"
                    waves[key] = w_arr; waves[key + "|ref"] = r_arr
                f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            np.savez_compressed(wp_path, **waves)
            el = time.time() - t0
            print(f"[shard {a.shard}] {i}/{len(runs)} {el/i:.0f}s/run ETA {(len(runs)-i)*el/i/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
