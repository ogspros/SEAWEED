"""Expands matrix_v2.yaml into the list of runs and writes the manifest.

Grid of water depth x first-reflector depth, ZMAX scaled with D, structural
elements off in the grid, usable maximum offset computed per cell.

    python validation/synthetic/build_matrix_v2.py --config validation/synthetic/matrix_v2.yaml [--dry-run]
"""
import os
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import yaml

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

# near offset of each geometry (from generate_line2_hf.GEOMS)
NEAR = {"g1500": 50.0, "g3000": 125.0, "g4500": 150.0}
V_SED_NOMINAL = 1900.0     # only to convert first_layer_m into an indicative two-way time


def make_seed(base, cell_id, rep):
    h = hashlib.sha256(f"{base}|{cell_id}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


def d_min(L_w, x, v):
    """Least depth that can be treated: below it the direct arrival enters the window."""
    return 0.5 * v * L_w * np.sqrt(1.0 + 2.0 * x / (v * L_w))


def sep_margin(D, x, L_w, v):
    """Dimensionless separation margin m = (t_pri - t_dir)/L_w.

    It collapses depth, offset and wavelet length onto one axis, which is the
    right regressor for the applicability threshold; D_min is m >= 1.
    """
    return (np.sqrt(x ** 2 + 4.0 * D ** 2) - x) / v / L_w


def x_max_usable(D, L_w, v):
    """Inverse: the offset beyond which the direct arrival contaminates the primary.

    Ignoring it on one model (D = 314 m, 1625 m of offset) made the depth 12%
    wrong and R_wb wrong by a factor of 2.7.
    """
    return max((4.0 * D ** 2 / (v * L_w) - v * L_w) / 2.0, 0.0)


def build_cells(cfg):
    ref = dict(cfg["reference"])
    cells, seen = [], set()

    def add(cid, axis, over):
        p = dict(ref); p.update(over)
        key = tuple(sorted(p.items()))
        if key in seen:
            return
        seen.add(key)
        cells.append({"cell_id": cid, "axis": axis, "params": p})

    for D in cfg["grid"]["D_m"]:
        for fl in cfg["grid"]["first_layer_m"]:
            add(f"grid_D{D:.0f}_L{fl:.0f}", "grid_D_contam",
                {"D_m": D, "first_layer_m": fl})

    for axis, vals in (cfg.get("secondary_axes") or {}).items():
        for v in vals:
            tag = str(v).replace(".", "p")
            add(f"sec_{axis}_{tag}", f"secondary_{axis}", {axis: v})

    # Paired cells: several parameters changed together, where they are not
    # physically independent.  z_src and z_rec enter the ghost as the product
    # G_s*G_r, and varying one at a time would give configurations of no real
    # acquisition (a source at 15 m with a streamer at 6).
    for name, over in (cfg.get("paired_cells") or {}).items():
        add(f"pair_{name}", f"paired_{name.split('_')[0]}", dict(over))
    return cells


def build_runs(cfg, cells):
    m = cfg["meta"]; v = m["v_water"]; L_w = m["w_len_ms"] / 1000.0
    runs = []
    for c in cells:
        p = c["params"]; near = NEAR.get(p["geom"], 125.0)
        xmax = x_max_usable(p["D_m"], L_w, v)
        n_ch_max = 1 + int(max(xmax - near, 0.0) / 25.0)  # typical spacing 25 m
        for rep in range(m["n_replicas"]):
            runs.append({
                "run_id": f"{c['cell_id']}_r{rep:03d}",
                "cell_id": c["cell_id"], "axis": c["axis"], "replica": rep,
                "seed": make_seed(m["seed_base"], c["cell_id"], rep),
                "params": p,
                "derived": {
                    "ZMAX_m": p["D_m"] + p["zmax_margin_m"],
                    "near_off_m": near,
                    "D_min_m": float(d_min(L_w, near, v)),
                    "x_max_usable_m": float(xmax),
                    "n_ch_max": int(n_ch_max),
                    "sep_margin": float(sep_margin(p["D_m"], near, L_w, v)),
                    # n_ch asked by the cell, truncated to what is allowed
                    "n_ch_used": int(min(p.get("n_ch", 1), max(n_ch_max, 1))),
                    "n_ch_truncated": bool(p.get("n_ch", 1) > n_ch_max),
                    "first_refl_twt_ms": float(2 * p["first_layer_m"]
                                               / V_SED_NOMINAL * 1e3),
                    # falsifiable prediction: below D_min the direct arrival contaminates the
                    # primary on the near trace
                    "expected_pass": bool(p["D_m"] >= d_min(L_w, near, v)),
                },
            })
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cells = build_cells(cfg)
    runs = build_runs(cfg, cells)
    nph = len(cfg["posthoc"]["snr_db"])

    print(f"cells {len(cells)} | runs {len(runs)} | records expected ~{len(runs)*nph}")
    nf = sum(1 for r in runs if not r["derived"]["expected_pass"])
    print(f"predicted failures (D < D_min): {nf} runs ({100*nf/len(runs):.0f}%)")
    print("\nusable x_max by depth (L_w 200 ms, near 125 m):")
    for D in cfg["grid"]["D_m"]:
        xm = x_max_usable(D, cfg["meta"]["w_len_ms"] / 1000., cfg["meta"]["v_water"])
        nch = 1 + int(max(xm - 125., 0.) / 25.)
        print(f"  D={D:6.0f} m -> x_max {xm:7.0f} m -> n_ch max {nch:3d}")

    if args.dry_run:
        return
    out = Path(cfg["paths"]["out_root"]); out.mkdir(parents=True, exist_ok=True)
    with open(out / cfg["paths"]["manifest"], "w") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    yaml.safe_dump(cfg, open(out / "matrix_config_frozen.yaml", "w"), sort_keys=False)
    print(f"\nmanifest -> {out / cfg['paths']['manifest']}")


if __name__ == "__main__":
    main()
