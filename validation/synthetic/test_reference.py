# =============================================================================
#  WHICH REFERENCE DOES THE SIMULATION ACTUALLY PRODUCE?
#
#  Everything the paper reports about the synthetics is measured against a
#  reference built ANALYTICALLY as (bare source wavelet) x (ghost operator).
#  Devito, by contrast, generates the ghost physically at the free surface, with
#  the source two cells below it, and propagates in TWO dimensions.  Three
#  things could therefore make the analytic reference wrong:
#
#    (a) the free-surface reflection coefficient may not be exactly -1 in the
#        discrete implementation, and the source at z = 6 m sits only 2 cells
#        from the boundary, where interpolation stencils straddle it;
#    (b) the analytic ghost assumes vertical incidence, delay 2z/V, whereas the
#        true delay is 2z cos(theta)/V -- a 1.2 per cent difference at the near
#        trace, so this one should be negligible;
#    (c) in 2D the far field is NOT the source time function: the Green function
#        goes as omega^{-1/2} e^{i pi/4}, i.e. a half integration.
#
#  This script removes the extraction entirely.  It cuts the RAW near-trace
#  window at the arrival time computed from the TRUE bathymetry, stacks nothing,
#  aligns nothing, and compares it against four candidate references.  Whatever
#  wins is what the reference should be.
#
#  USAGE:
#    python validation/synthetic/test_reference.py --config validation/synthetic/matrix_v2.yaml --n 4
# =============================================================================
import os, json, argparse
import numpy as np
import yaml
from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt

os.environ.setdefault('DEVITO_LOGGING', 'ERROR')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import generate_line2_hf as G
from velocity_generator import generate_velocity_model
from generate_parallel_hf import make_source_fn, apply_ghost
from airgun_generator import DT as AG_DT
from examples.seismic import Model

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

BAND = (6., 90.)
Z_DEV = 6.0


def nrm(x):
    x = np.asarray(x, float) - np.mean(x)
    return x / (np.linalg.norm(x) + 1e-12)


def best_ncc(a, b, max_lag=60):
    """Maximum correlation over lag, on full overlap of the shorter series."""
    a = nrm(a); n = min(len(a), len(b))
    best, bl = -2.0, 0
    for L in range(-max_lag, max_lag + 1):
        i0 = max(0, L); i1 = min(len(a), len(b) + L)
        if i1 - i0 < int(0.8 * n):
            continue
        x = nrm(a[i0:i1]); y = nrm(b[i0 - L:i0 - L + (i1 - i0)])
        v = float(x @ y)
        if v > best:
            best, bl = v, L
    return best, bl


def omega_filter(x, dt, p=0.5, phase_deg=45.):
    """Apply omega^{-p} e^{i phase}.  Theory gives p = 1/2 and phase = +45 deg
    for the 2D far field; both are left free so they can be MEASURED rather than
    assumed.  If the empirical optimum sits at p = 1/2 the asymptotic Green
    function explains the data; if it sits elsewhere something else contributes.
    """
    n = len(x)
    X = np.fft.rfft(x, 2 * n)
    f = np.fft.rfftfreq(2 * n, dt)
    om = 2 * np.pi * np.maximum(f, 1e-9)
    H = np.exp(1j * np.radians(phase_deg)) * om ** (-p)
    H[0] = 0.
    return np.fft.irfft(X * H, 2 * n)[:n]


def half_integrate(x, dt):
    return omega_filter(x, dt, 0.5, 45.)


def bp(x, dt):
    nyq = 0.5 / dt
    b, a = butter(4, [BAND[0] / nyq, min(BAND[1] / nyq, .95)], 'band')
    return filtfilt(b, a, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--wl_ms", type=float, default=200.)
    ap.add_argument("--standalone", action="store_true",
                    help="build the models here instead of reading the manifest, "
                         "so the ceiling is measured on exactly the configuration "
                         "used by test_nshots.py (D, first reflector, folds, lens, "
                         "dip and roughness)")
    ap.add_argument("--D", type=float, default=400.)
    ap.add_argument("--first_layer", type=float, default=150.)
    ap.add_argument("--flat", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m = cfg["meta"]
    if args.standalone:
        kwg = (dict(p_fault=0., p_folds=0., p_lens=0., slope_deg=0.,
                    roughness_m=0.) if args.flat else
               dict(p_fault=0., p_folds=1., p_lens=1., slope_deg=1.5,
                    roughness_m=8.))
        cand = [dict(run_id=f"standalone_r{i:03d}", seed=90_000 + i,
                     params=dict(D_m=args.D, first_layer_m=args.first_layer,
                                 fmax_hz=60., geom="g3000", **kwg),
                     derived=dict(ZMAX_m=args.D + 1600.))
                for i in range(args.n)]
        desc = ("laterally invariant" if args.flat
                else "folds + lens + 1.5 deg dip + 8 m roughness")
        print(f"{args.n} models built here: D = {args.D:.0f} m, first reflector "
              f"{args.first_layer:.0f} m, {desc}\n")
    else:
        root = cfg["paths"]["out_root"]
        runs = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
        # deep water, flat seafloor, no structure: the cleanest possible test
        cand = [r for r in runs if r["params"]["D_m"] >= 350
                and r["params"]["slope_deg"] == 0 and r["params"]["p_fault"] == 0
                and r["axis"] == "grid_D_contam"][:args.n]
        print(f"{len(cand)} models from the manifest, deep water, flat seafloor\n")

    hdr = f"{'model':>22} {'w_bare':>8} {'+ghost':>8} {'+2D':>8} {'+ghost+2D':>10}"
    print(hdr); print("-" * len(hdr))
    acc = {k: [] for k in ("bare", "ghost", "twod", "both")}
    scan = None
    for run in cand:
        p = run["params"]; der = run["derived"]
        geom = G.GEOMS[p["geom"]]; offs = G.geom_offsets(geom)
        XMAX = G.domain_for_shots_geom(m["n_shots"] + 4, geom)
        rng = np.random.default_rng(run["seed"])
        vp, rho, kwb, meta = generate_velocity_model(
            rng, XMAX=XMAX, ZMAX=der["ZMAX_m"], DH=G.DH,
            D_water_range=(p["D_m"], p["D_m"]),
            p_fault=p.get("p_fault", 0.), p_folds=p.get("p_folds", 0.),
            p_lens=p.get("p_lens", 0.), slope_deg=p.get("slope_deg", 0.),
            roughness_m=p.get("roughness_m", 0.),
            first_layer_m=p["first_layer_m"])
        model = Model(vp=(vp / 1000.), b=1.0 / rho, origin=(0., 0.),
                      shape=vp.shape, spacing=(G.DH, G.DH), space_order=8,
                      nbl=G.NB, bcs="damp", fs=True)
        xs = G.shot_positions_geom(XMAX, geom)
        src_fn, w_bare = make_source_fn(10_000 + run["seed"], fmax=p["fmax_hz"])
        dt_out = m["dt_ms"] / 1000.
        tn = float(np.clip(2.6 * (2 * p["D_m"] / m["v_water"]) * 1000. + 800.,
                           1500., 4500.))
        # ONE shot only: no stack, no alignment, nothing
        d, dtd = G.shoot(model, xs[0], src_fn, tn=tn, offsets=offs)
        fr = Fraction(float(dtd) / 1000. / dt_out).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0)
        # TRUE arrival time from the TRUE bathymetry under the reflection point
        j = int(np.clip((xs[0] - offs[0] / 2.) / G.DH, 0, vp.shape[0] - 1))
        D_true = kwb[j] * G.DH
        t_arr = np.hypot(2 * D_true, offs[0]) / m["v_water"]
        wl = int(round(args.wl_ms * 1e-3 / dt_out))
        k0 = int(round(t_arr / dt_out)) - int(round(0.020 / dt_out))
        if k0 < 0 or k0 + 2 * wl > d.shape[0]:
            print(f"  {run['run_id']}: window outside the record"); continue
        raw = bp(d[k0:k0 + 2 * wl, 0].astype(float), dt_out)

        w = np.asarray(w_bare, float)
        if abs(dt_out - AG_DT) > 1e-9:
            w = resample_poly(w, int(round(len(w) * AG_DT / dt_out)), len(w))
        refs = {
            "bare":  bp(w, dt_out)[:wl],
            "ghost": bp(apply_ghost(w.astype(np.float32), dt_out, Z_DEV, Z_DEV)
                        .astype(float), dt_out)[:wl],
            "twod":  bp(half_integrate(w, dt_out), dt_out)[:wl],
            "both":  bp(half_integrate(
                        apply_ghost(w.astype(np.float32), dt_out, Z_DEV, Z_DEV)
                        .astype(float), dt_out), dt_out)[:wl],
        }
        vals = {k: best_ncc(raw, v)[0] for k, v in refs.items()}
        for k in acc:
            acc[k].append(vals[k])
        print(f"{run['run_id']:>22} {vals['bare']:8.3f} {vals['ghost']:8.3f} "
              f"{vals['twod']:8.3f} {vals['both']:10.3f}")
        scan = (raw, w, dt_out)

    print("-" * len(hdr))
    print(f"{'mean':>22} " + " ".join(f"{np.mean(acc[k]):8.3f}" for k in
                                      ("bare", "ghost", "twod")) +
          f" {np.mean(acc['both']):10.3f}")

    # --- is p = 1/2 the right exponent, and +45 deg the right phase? ---------
    if scan:
        print("\nExponent scan on the last model (ghost applied, then omega^-p):")
        raw_l, w_l, dt_l = scan
        wg_l = apply_ghost(w_l.astype(np.float32), dt_l, Z_DEV, Z_DEV).astype(float)
        print(f"{'p':>6} " + " ".join(f"{ph:6.0f}d" for ph in (0, 22.5, 45, 67.5, 90)))
        bestv, bestpp = -2, None
        for p in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
            row = []
            for ph in (0, 22.5, 45, 67.5, 90):
                rr = bp(omega_filter(wg_l, dt_l, p, ph), dt_l)[:len(raw_l) // 2]
                v = best_ncc(raw_l, rr)[0]
                row.append(f"{v:7.3f}")
                if v > bestv:
                    bestv, bestpp = v, (p, ph)
            print(f"{p:6.2f} " + " ".join(row))
        print(f"  best: p = {bestpp[0]:.2f}, phase = {bestpp[1]:.0f} deg "
              f"-> NCC {bestv:.3f}   (theory: p = 0.50, phase = 45 deg)")
    print()
    print("The winning column is the reference the simulation actually produces.")
    print("If 'ghost' beats 'bare', the analytic ghost is right in kind.")
    print("If '+2D' columns win, the half integration must be included.")
    print("If ALL are low (< 0.8), the analytic reference is wrong for a reason")
    print("we have not identified, and the synthetic numbers understate the")
    print("method rather than measuring it.")


if __name__ == "__main__":
    main()
