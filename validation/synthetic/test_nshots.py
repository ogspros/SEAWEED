# =============================================================================
#  DOES THE FOLD HELP?  Accuracy against the number of shots stacked.
#
#  The expectation is a sqrt(N) improvement saturating at some point.  The
#  reference includes the 2D far-field factor, and the comparison slides with
#  full overlap.
#
#  The design separates two effects that are otherwise confounded:
#    - MORE SHOTS at the same spacing: isolates the sqrt(N) gain on random noise
#    - the SAME shots spread over a longer line: isolates the decorrelation of
#      the geology, which is what would make the contamination average out
#  If neither helps, the limit is local coherent contamination -- the first
#  sub-bottom reflector sits at essentially the same delay under every shot, so
#  it stacks as coherently as the wavelet does and no amount of fold removes it.
#
#  USAGE:
#    python validation/synthetic/test_nshots.py --config validation/synthetic/matrix_v2.yaml --models 3 --snr 30
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
from seaweed.wavelet_extractor_v2 import (track_seafloor_xcorr, extract_wavelet,
                                  estimate_vw_direct)
from seaweed.seafloor_robust import semblance_bottom
from examples.seismic import Model

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

BAND = (6., 90.)
Z_DEV = 6.0


def nrm(x):
    x = np.asarray(x, float) - np.mean(x)
    return x / (np.linalg.norm(x) + 1e-12)


def two_d_factor(x, dt, p=0.5, phase_deg=45.):
    n = len(x)
    X = np.fft.rfft(x, 2 * n)
    f = np.fft.rfftfreq(2 * n, dt)
    H = np.exp(1j * np.radians(phase_deg)) * (2 * np.pi * np.maximum(f, 1e-9)) ** (-p)
    H[0] = 0.
    return np.fft.irfft(X * H, 2 * n)[:n]


def reference(w_bare, dt, wl):
    w = np.asarray(w_bare, np.float64)
    if abs(dt - AG_DT) > 1e-9:
        w = resample_poly(w, int(round(len(w) * AG_DT / dt)), len(w))
    wg = apply_ghost(w.astype(np.float32), dt, Z_DEV, Z_DEV).astype(np.float64)
    wg = two_d_factor(wg, dt)
    nyq = 0.5 / dt
    b, a = butter(4, [BAND[0] / nyq, min(BAND[1] / nyq, .95)], 'band')
    wg = filtfilt(b, a, wg)
    return nrm(wg[:wl] if len(wg) >= wl else np.pad(wg, (0, wl - len(wg))))


def slide(wlong, ref, o_max):
    wl = len(ref); y = ref - ref.mean(); ny = np.linalg.norm(y)
    hi = min(len(wlong) - wl, int(o_max))
    best = -2.0
    for o in range(0, max(hi, 0) + 1):
        seg = wlong[o:o + wl]
        x = seg - seg.mean(); nx = np.linalg.norm(x)
        if nx * ny > 1e-12:
            best = max(best, float(x @ y / (nx * ny)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--out", default="nshots")
    ap.add_argument("--snr", type=float, default=30.)
    ap.add_argument("--D", type=float, default=400.)
    ap.add_argument("--first_layer", type=float, default=150.)
    ap.add_argument("--flat", action="store_true",
                    help="laterally invariant model. Useful as a CONTROL: with "
                         "no lateral variability every shot sees the same 1D "
                         "medium, all gathers are identical, and neither more "
                         "shots nor a longer line can change anything. A first "
                         "attempt used this by accident and returned the same "
                         "value to three decimals for every configuration")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m = cfg["meta"]
    geom = G.GEOMS["g3000"]; offs = G.geom_offsets(geom)
    dt = m["dt_ms"] / 1000.
    wl = int(round(m["w_len_ms"] * 1e-3 / dt))
    vw = m["v_water"]

    # configurations: (label, n_shots, shot decimation)
    #   decimation 1 -> consecutive shots, short line
    #   decimation 5 -> same count spread over five times the distance
    confs = [("10 shots, dense", 10, 1), ("20 shots, dense", 20, 1),
             ("40 shots, dense", 40, 1), ("80 shots, dense", 80, 1),
             ("20 shots, spread x5", 20, 5)]
    n_max = max(n * d for _, n, d in confs)

    # Shard BY MODEL: each model is an independent forward-modelling job, and
    # the five configurations reuse the same 80 shots, so there is nothing to
    # share between models.  Comparing configurations that may differ by a few
    # hundredths also needs more than three replicas to mean anything.
    mine = [i for i in range(args.models) if i % args.n_shards == args.shard]
    if not mine:
        print(f"shard {args.shard}: nothing to do"); return
    print(f"D = {args.D:.0f} m, first reflector {args.first_layer:.0f} m, "
          f"models {mine} of {args.models}, {n_max} shots per model, "
          f"SNR {args.snr:.0f} dB nominal")
    model_desc = ("LATERALLY INVARIANT (control)" if args.flat
                  else "folds + lens + 1.5 deg dip + 8 m roughness")
    print(f"model: {model_desc}\n")
    if args.snr >= 20:
        print("  NOTE: at this nominal SNR the effective level in the window "
              "after\n  stacking is above 30 dB, so noise cannot limit "
              "anything and the\n  sqrt(N) gain has nothing to act on. Run "
              "also at --snr -10.\n")
    res = {lab: [] for lab, _, _ in confs}
    coh_l = {lab: [] for lab, _, _ in confs}
    scat_l = {lab: [] for lab, _, _ in confs}
    t0sd_l = {lab: [] for lab, _, _ in confs}
    for im in mine:
        seed = 90_000 + im
        XMAX = G.domain_for_shots_geom(n_max + 4, geom)
        rng = np.random.default_rng(seed)
        # LATERAL VARIABILITY IS REQUIRED for this test to mean anything.
        # Folds, a lens and a gently dipping seafloor make successive shots see
        # different geology, which is the only way spreading the line can
        # decorrelate the contamination.  Without them the experiment cannot
        # answer its own question.
        kw = (dict(p_fault=0., p_folds=0., p_lens=0., slope_deg=0.,
                   roughness_m=0.) if args.flat else
              dict(p_fault=0., p_folds=1., p_lens=1., slope_deg=1.5,
                   roughness_m=8.))
        vp, rho, kwb, meta = generate_velocity_model(
            rng, XMAX=XMAX, ZMAX=args.D + 1600., DH=G.DH,
            D_water_range=(args.D, args.D),
            first_layer_m=args.first_layer, **kw)
        model = Model(vp=(vp / 1000.), b=1.0 / rho, origin=(0., 0.),
                      shape=vp.shape, spacing=(G.DH, G.DH), space_order=8,
                      nbl=G.NB, bcs="damp", fs=True)
        xs = G.shot_positions_geom(XMAX, geom)
        src_fn, w_bare = make_source_fn(10_000 + seed, fmax=60.)
        w_ref = reference(w_bare, dt, wl)
        tn = float(np.clip(2.6 * (2 * args.D / vw) * 1000. + 800., 1500., 4500.))
        NT = int(round(tn / 1000. / dt))
        n_gen = min(n_max, len(xs))
        g = np.zeros((n_gen, NT, len(offs)), np.float32)
        for i in range(n_gen):
            d, dtd = G.shoot(model, xs[i], src_fn, tn=tn, offsets=offs)
            fr = Fraction(float(dtd) / 1000. / dt).limit_denominator(2000)
            d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
            g[i] = (d[:NT] if d.shape[0] > NT else
                    np.pad(d, ((0, NT - d.shape[0]), (0, 0))))
        # noise, calibrated as in the matrix
        ns = float(np.sqrt(np.mean(g ** 2)) / (10 ** (args.snr / 20.)))
        g = g + np.random.default_rng(seed).normal(0, ns, g.shape).astype(np.float32)

        print(f"  model {im + 1}/{args.models}")
        for lab, n_sh, dec in confs:
            idx = np.arange(0, min(n_sh * dec, n_gen), dec)[:n_sh]
            if len(idx) < n_sh:
                res[lab].append(np.nan); continue
            gs = g[idx]
            r0 = semblance_bottom(gs[0].T, dt, offs,
                                  t0_min_ms=max(120, 2 * args.D / vw * 1e3 - 200),
                                  t0_max_ms=2 * args.D / vw * 1e3 + 200)
            # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
            # were computed; it is written out so that the default of 20 does not change them.
            t0 = track_seafloor_xcorr(gs, dt, offs,
                                      t0_init_s=(r0['t0'] if r0 else 2 * args.D / vw),
                                      vw=vw, n_group=5)
            PAD = 120.
            wp_l, qp = extract_wavelet(gs, dt, t0, offs,
                                       w_len_ms=m["w_len_ms"] + 2 * PAD,
                                       lead_ms=80. + PAD, n_ch=1, order=0,
                                       band=BAND, vw=vw)
            if wp_l is None:
                res[lab].append(np.nan); continue
            sep = (np.hypot(2 * args.D, offs[0]) - offs[0]) / vw
            v = slide(nrm(np.asarray(wp_l, float)), w_ref,
                      max(4, int(round(0.5 * sep / dt))))
            res[lab].append(v)
            # DIAGNOSTICS.  A sqrt(N) gain requires the windows to be aligned:
            # if the tracking drifts as it propagates along the line, more shots
            # means more misalignment, and windows summed out of phase cancel
            # the very signal the fold was meant to reinforce.  Coherence and
            # alignment scatter say directly whether that is happening.
            coh_l[lab].append(float(qp["stack_coherence"]))
            scat_l[lab].append(float(qp.get("align_scatter_ms", np.nan)))
            t0sd_l[lab].append(float(np.std(t0) * 1e3))
            print(f"    {lab:22s} NCC {v:.3f} | coh {qp['stack_coherence']:.3f} "
                  f"| align scatter {qp.get('align_scatter_ms', np.nan):5.1f} ms "
                  f"| t0 sd {np.std(t0)*1e3:5.1f} ms  (span "
                  f"{(xs[idx[-1]] - xs[idx[0]]) / 1000.:.2f} km)")

    tag = f"{args.out}_snr{int(args.snr)}"
    fn = f"{tag}_shard{args.shard:03d}.json"
    json.dump({"ncc": {k: list(map(float, v)) for k, v in res.items()},
               "coh": {k: list(map(float, v)) for k, v in coh_l.items()},
               "scatter_ms": {k: list(map(float, v)) for k, v in scat_l.items()},
               "t0_sd_ms": {k: list(map(float, v)) for k, v in t0sd_l.items()}},
              open(fn, "w"))
    print(f"\nwritten: {fn}")

    # merge whatever shards are already on disk, so a partial run is still
    # readable and the last shard to finish prints the complete table
    import glob
    allr = {lab: [] for lab, _, _ in confs}
    for f in sorted(glob.glob(f"{tag}_shard*.json")):
        d = json.load(open(f))
        dn = d.get("ncc", d)          # tolerate the older flat format
        for k in allr:
            allr[k].extend(dn.get(k, []))
    n_done = len(allr["20 shots, dense"])
    print(f"\n{n_done} models pooled from {len(glob.glob(f'{tag}_shard*.json'))} shards")
    print(f"{'configuration':>22} {'mean':>7} {'sd':>7} {'sem':>7}   vs 20 dense")
    base = np.nanmean(allr["20 shots, dense"])
    for lab, _, _ in confs:
        a = np.array(allr[lab], float)
        n = np.sum(np.isfinite(a))
        sem = np.nanstd(a) / max(np.sqrt(n), 1)
        print(f"{lab:>22} {np.nanmean(a):7.3f} {np.nanstd(a):7.3f} {sem:7.3f} "
              f"{np.nanmean(a) - base:+9.3f}")
    # pool the diagnostics too: they say WHY, not just whether
    import glob as _g
    diag = {k: {lab: [] for lab, _, _ in confs} for k in ("coh", "scatter_ms", "t0_sd_ms")}
    for f in sorted(_g.glob(f"{tag}_shard*.json")):
        d = json.load(open(f))
        for k in diag:
            if k in d:
                for lab in diag[k]:
                    diag[k][lab].extend(d[k].get(lab, []))
    if any(diag["coh"][lab] for lab, _, _ in confs):
        print(f"\n{'configuration':>22} {'coherence':>10} {'scatter ms':>11} "
              f"{'t0 sd ms':>9}")
        for lab, _, _ in confs:
            print(f"{lab:>22} {np.nanmean(diag['coh'][lab]):10.3f} "
                  f"{np.nanmean(diag['scatter_ms'][lab]):11.2f} "
                  f"{np.nanmean(diag['t0_sd_ms'][lab]):9.1f}")
        print("  If coherence falls or the alignment scatter grows with shot")
        print("  count, the windows are being summed out of phase and the fold")
        print("  is working against itself.")

    print("\n  Compare each difference against the standard error: with a few")
    print("  models the scatter between geological realisations easily exceeds")
    print("  the effect being looked for.")
    print("\n  A sqrt(N) gain would show as a steady rise with shot count.")
    print("  If the spread configuration matches the dense one at equal count,")
    print("  decorrelating the geology does not help either, and the limit is")
    print("  the local contamination that stacks as coherently as the wavelet.")


if __name__ == "__main__":
    main()
