# =============================================================================
#  The true R_wb of a synthetic line, rebuilt from its seed.
#
#  The velocity model is not saved with the line, but the seed is, and
#  generate_velocity_model is deterministic: vp and rho are regenerated (no
#  Devito, a few seconds) and the impedance contrast at the seafloor is read.
#  This is where the range of R_wb quoted for the synthetics comes from.
#
#      python validation/synthetic/true_rwb.py --root dataset_geomvar --line test/line_00235.npz
# =============================================================================
import os, argparse, numpy as np
from velocity_generator import generate_velocity_model, VP_WATER, RHO_WATER
import generate_line2_hf as G

Z_WATER = VP_WATER * RHO_WATER


def true_rwb(d, n_shots_gen=20):
    """Returns (R_wb per column, vp below the seafloor per column, kwb)."""
    seed = int(d["seed"])
    geom = G.GEOMS[str(d["geom"])] if "geom" in d.files else G.GEOMS["g3000"]
    # XMAX from the length of kwb_depth when available (exact), otherwise
    # rebuilt as in make_line: domain_for_shots_geom(n_shots+4, geom)
    if "kwb_depth" in d.files and np.asarray(d["kwb_depth"]).size > 1:
        XMAX = (np.asarray(d["kwb_depth"]).size - 1) * G.DH
    else:
        XMAX = G.domain_for_shots_geom(n_shots_gen + 4, geom)
    # the same sequence as make_line: rng = default_rng(seed) passed to build_model
    rng = np.random.default_rng(seed)
    vp, rho, kwb, meta = generate_velocity_model(rng, XMAX=XMAX, ZMAX=2800.0,
                                                 DH=G.DH)
    nx = vp.shape[0]
    vp_sf = np.array([vp[j, kwb[j]] for j in range(nx)])
    rho_sf = np.array([rho[j, kwb[j]] for j in range(nx)])
    Z2 = vp_sf * rho_sf
    R = (Z2 - Z_WATER) / (Z2 + Z_WATER)
    return R, vp_sf, kwb, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_geomvar")
    ap.add_argument("--line", required=True)
    ap.add_argument("--n_shots_gen", type=int, default=20)
    args = ap.parse_args()

    d = np.load(os.path.join(args.root, args.line), allow_pickle=True)
    R, vp_sf, kwb, meta = true_rwb(d, args.n_shots_gen)

    # consistency check: the regenerated bathymetry must match the saved one
    ok = "n/a"
    if "kwb_depth" in d.files and np.asarray(d["kwb_depth"]).size > 1:
        kd = np.asarray(d["kwb_depth"], float)
        err = np.abs(kd - kwb * G.DH).max() if len(kd) == len(kwb) else np.inf
        ok = f"{err:.1f} m" if np.isfinite(err) else "different lengths"

    print(f"line {args.line} | seed {int(d['seed'])} | geom "
          f"{str(d['geom']) if 'geom' in d.files else '?'}")
    print(f"  regenerated against saved bathymetry: max diff {ok}")
    print(f"  (if it is not ~0, the reconstruction is wrong and R_wb does not hold)")
    print(f"\n  mean D_water: {np.mean(kwb)*G.DH:.0f} m   "
          f"(npz: {float(d['D_water']):.0f} m)")
    print(f"  vp just below the seafloor: {vp_sf.min():.0f}-{vp_sf.max():.0f} m/s "
          f"(median {np.median(vp_sf):.0f})")
    print(f"\n  >>> true R_wb: median {np.median(R):.3f}  "
          f"[min {R.min():.3f}, max {R.max():.3f}] <<<")
    print(f"      range assumed before this check: 0.34-0.47")

    # R_wb below the reflection points of the near trace
    if "offsets" in d.files:
        offs = np.asarray(d["offsets"], float)
        geom = G.GEOMS[str(d["geom"])]
        XMAX = (len(kwb) - 1) * G.DH
        xs = G.shot_positions_geom(XMAX, geom)[:d["gathers"].shape[0]]
        idx = np.clip(((xs - offs[0] / 2.) / G.DH).astype(int), 0, len(R) - 1)
        print(f"\n  R_wb below the reflection points of the near trace: "
              f"median {np.median(R[idx]):.3f} "
              f"[{R[idx].min():.3f}, {R[idx].max():.3f}]")
        print(f"      (the value to compare with the estimate)")


if __name__ == "__main__":
    main()
