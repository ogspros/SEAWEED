# =============================================================================
#  Does the geology decorrelate as the line gets longer?
#
#  Distance covered and number of shots are separated, which a naive test
#  (more shots = longer line) confuses:
#     A) 20 shots, 25 m apart   ->  475 m covered
#     B) 60 shots, 25 m apart   -> 1475 m covered   (more shots, same scale)
#     C) 60 shots, 125 m apart  -> 7375 m covered   (same shots, much longer)
#  B against A isolates the sqrt(N) gain, C against B the decorrelation of
#  the geology.  The folds of the generator have wavelengths of 2000-8000 m,
#  so over 475 m the geology stacks coherently.
#
#      python validation/synthetic/test_line_length.py --n_models 3 --out longline_test
#      python validation/synthetic/test_line_length.py --analyze longline_test     (re-analysis only)
# =============================================================================
import os, time, argparse, numpy as np
os.environ.setdefault('DEVITO_LOGGING', 'ERROR')
os.environ.setdefault('DEVITO_ARCH', 'gcc')
os.environ.pop('DEVITO_PLATFORM', None)
os.environ.setdefault('DEVITO_LANGUAGE', 'C')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt

import generate_line2_hf as G
from generate_parallel_hf import make_source_fn, apply_ghost
from airgun_generator import DT as AG_DT
from seaweed.wavelet_extractor_v2 import track_seafloor_xcorr, extract_wavelet
from seaweed.seafloor_robust import semblance_bottom

BAND = (6., 90.); Z_DEV = 6.0; VW = 1500.
STRIDE_FAR = 5  # stride 5 x 25 m = 125 m
N_SHORT, N_LONG = 20, 60


def nrm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def ncc_shift(a, b, max_lag=40, min_overlap=60):
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    def _c(x, y):
        x = x - x.mean(); y = y - y.mean()
        dd = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / dd) if dd > 1e-12 else 0.0
    best = -2.
    for L in range(-max_lag, max_lag + 1):
        a0 = max(0, L); a1 = min(na, nb + L); ov = a1 - a0
        if ov < min_overlap:
            continue
        best = max(best, _c(a[a0:a1], b[a0 - L:a0 - L + ov]))
    return best


def reference(w_bare, dt, wl):
    w = np.asarray(w_bare, np.float64)
    if abs(dt - AG_DT) > 1e-9:
        w = resample_poly(w, int(round(len(w) * AG_DT / dt)), len(w))
    wg = apply_ghost(w.astype(np.float32), dt, Z_DEV, Z_DEV).astype(np.float64)
    nyq = 0.5 / dt
    b, a = butter(4, [BAND[0] / nyq, min(BAND[1] / nyq, .95)], 'band')
    wg = filtfilt(b, a, wg)
    wg = wg[:wl] if len(wg) >= wl else np.pad(wg, (0, wl - len(wg)))
    return nrm(wg - wg.mean())


# --------------------------------------------------------------------------- #
def generate(seed, geom_name, out_dir, tn=4500.0, dt_out=0.002, fmax=45.0):
    """Generate one model and fire only the shots the three sets need (the union of their positions)."""
    geom = G.GEOMS[geom_name]; offsets = G.geom_offsets(geom)
    n_needed = 1 + (N_LONG - 1) * STRIDE_FAR  # coverage of set C
    XMAX = G.domain_for_shots_geom(n_needed + 4, geom)
    rng = np.random.default_rng(seed)
    model, vp, rho, kwb, meta = G.build_model(XMAX, rng=rng)
    xs_all = G.shot_positions_geom(XMAX, geom)
    if len(xs_all) < n_needed:
        raise RuntimeError(f"dominio troppo corto: {len(xs_all)} < {n_needed}")

    idx_B = np.arange(N_LONG)  # stride 1
    idx_C = np.arange(N_LONG) * STRIDE_FAR  # stride STRIDE_FAR
    idx_all = np.unique(np.concatenate([idx_B, idx_C]))
    src_fn, w_bare = make_source_fn(10_000 + seed, fmax=fmax)

    NT = int(round(tn / 1000. / dt_out))
    G_out = np.zeros((len(idx_all), NT, len(offsets)), np.float32)
    t0 = time.time()
    for k, i in enumerate(idx_all):
        d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offsets)
        fr = Fraction(float(dtd) / 1000. / float(dt_out)).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
        d = d[:NT] if d.shape[0] > NT else np.pad(d, ((0, NT - d.shape[0]), (0, 0)))
        G_out[k] = d
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"    shot {k+1}/{len(idx_all)}  {el/(k+1):.1f}s/shot", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    f = os.path.join(out_dir, f"long_{seed:06d}.npz")
    np.savez(f, gathers=G_out, dt=dt_out, w_bare=w_bare,
             idx_all=idx_all, idx_B=idx_B, idx_C=idx_C,
             xs=xs_all[idx_all], offsets=offsets.astype(np.float32),
             geom=geom_name, D_water=meta['D_water'],
             kwb_depth=meta.get('kwb_depth'), fmax=fmax, seed=seed)
    print(f"  saved {f}  ({time.time()-t0:.0f}s, {len(idx_all)} shots)")
    return f


def analyze(f, wl_ms=200.):
    d = np.load(f, allow_pickle=True)
    g = d["gathers"]; dt = float(d["dt"]); D = float(d["D_water"])
    offs = np.asarray(d["offsets"], float); xs = np.asarray(d["xs"], float)
    idx_all = list(np.asarray(d["idx_all"]))
    wl = int(round(wl_ms * 1e-3 / dt))
    w_ref = reference(d["w_bare"], dt, wl)

    pos = {v: k for k, v in enumerate(idx_all)}
    sets = {
        "A  20 shot passo  25 m": [pos[i] for i in np.arange(N_SHORT)],
        "B  60 shot passo  25 m": [pos[i] for i in np.asarray(d["idx_B"])],
        "C  60 shot passo 125 m": [pos[i] for i in np.asarray(d["idx_C"])],
    }
    out = {}
    for lab, rows in sets.items():
        sub = g[rows]
        span = xs[rows].max() - xs[rows].min()
        r0 = semblance_bottom(sub[0].T, dt, offs,
                              t0_min_ms=max(150, 2 * D / VW * 1e3 - 200),
                              t0_max_ms=2 * D / VW * 1e3 + 200)
        t0i = r0['t0'] if r0 else 2 * D / VW
        # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
        # were computed; it is written out so that the default of 20 does not change them.
        t0 = track_seafloor_xcorr(sub, dt, offs, t0_init_s=t0i, n_group=5)
        w, qc = extract_wavelet(sub, dt, t0, offs, w_len_ms=wl_ms, n_ch=1, order=0,
                                band=BAND)
        v = np.nan if w is None else ncc_shift(nrm(np.asarray(w, float)), w_ref)
        out[lab] = (v, span, len(rows), None if w is None else qc["stack_coherence"])
        print(f"  {lab} | span {span/1000:5.2f} km | NCC {v:.3f} | "
              f"coh {out[lab][3]:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_models", type=int, default=3)
    ap.add_argument("--seed_base", type=int, default=900000)
    ap.add_argument("--geom", default="g3000")
    ap.add_argument("--fmax", type=float, default=45.0,
                    help="a low band: the regime where the window collects the most "
                         "geology, so the most sensitive case")
    ap.add_argument("--out", default="longline_test")
    ap.add_argument("--analyze", default=None, help="re-analyse a directory only")
    args = ap.parse_args()

    if args.analyze:
        import glob
        files = sorted(glob.glob(os.path.join(args.analyze, "long_*.npz")))
        allr = {}
        for f in files:
            print(f"\n{os.path.basename(f)}")
            for k, v in analyze(f).items():
                allr.setdefault(k, []).append(v[0])
        print("\n=== MEAN OVER THE MODELS ===")
        for k, v in allr.items():
            a = np.array(v, float)
            print(f"  {k}: NCC {np.nanmean(a):.3f} +- {np.nanstd(a):.3f} (n={len(a)})")
        if len(allr) == 3:
            ks = list(allr)
            A, B, C = (np.nanmean(np.array(allr[k], float)) for k in ks)
            print(f"\n  B - A = {B-A:+.3f}  (effect of the number of shots, sqrt(N))")
            print(f"  C - B = {C-B:+.3f}  (effect of the LENGTH of the line)")
            print("\n  A large C-B means the geology decorrelates and long lines are needed.")
            print("  C-B ~ 0 means the length does not matter.")
        return

    for k in range(args.n_models):
        seed = args.seed_base + k
        print(f"\n[{k+1}/{args.n_models}] model seed {seed}", flush=True)
        generate(seed, args.geom, args.out, fmax=args.fmax)
    print(f"\nthen: python {os.path.basename(__file__)} --analyze {args.out}")


if __name__ == "__main__":
    main()
