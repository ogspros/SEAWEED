"""Runner of the v2 matrix: generates the models, extracts, writes results.jsonl.

  - generation: generate_velocity_model, a Devito Model and G.shoot, as in
    make_line but with every parameter of the cell explicit (G.build_model
    does not pass them on);
  - extraction: wavelet_extractor_v2 (physical anchoring, origin decided on
    the stack, matched filter with imposed polarity for the multiple).

The models generated here are the ones the v5 and v6 matrices re-extract.

    python validation/synthetic/run_matrix_v2.py --config validation/synthetic/matrix_v2.yaml --shard 0 --n_shards 30
"""
import os, json, time, argparse, traceback
os.environ.setdefault('DEVITO_LOGGING', 'ERROR')
os.environ.setdefault('DEVITO_ARCH', 'gcc')
os.environ.pop('DEVITO_PLATFORM', None)
os.environ.setdefault('DEVITO_LANGUAGE', 'C')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import resample_poly, butter, filtfilt
from examples.seismic import Model

import generate_line2_hf as G
from velocity_generator import generate_velocity_model, VP_WATER, RHO_WATER
from generate_parallel_hf import make_source_fn, apply_ghost
from airgun_generator import DT as AG_DT
from seaweed.wavelet_extractor_v2 import (track_seafloor_xcorr, extract_wavelet,
                                  estimate_vw_direct, statistical_spectrum)
from seaweed.seafloor_robust import semblance_bottom

# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

BAND = (6., 90.)
Z_DEV = 6.0  # Z_SRC = Z_REC in shoot(): the ghost the data actually contain
Z_WATER = VP_WATER * RHO_WATER


def nrm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def ncc_shift(a, b, max_lag=40, min_overlap=60):
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    def _c(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / d) if d > 1e-12 else 0.0
    best, blag = -2., 0
    for L in range(-max_lag, max_lag + 1):
        a0 = max(0, L); a1 = min(na, nb + L); ov = a1 - a0
        if ov < min_overlap:
            continue
        v = _c(a[a0:a1], b[a0 - L:a0 - L + ov])
        if v > best:
            best, blag = v, L
    return best, blag


def two_d_factor(x, dt, p=0.5, phase_deg=45.):
    """Apply omega^{-p} e^{i phase}: the 2D far-field factor.

    In three dimensions the far field reproduces the source time function; in
    two dimensions it does not.  The 2D Green function is G ~ H_0^{(2)}(omega r/c),
    asymptotically proportional to omega^{-1/2} e^{i pi/4}, so a 2D simulation
    records a HALF INTEGRATION of the injected signature.  A reference built as
    (bare wavelet) x (ghost) does not describe the data, and comparing the two
    penalises high frequencies by sqrt(f).

    Measured on raw single-shot windows, with no extraction involved:
        bare wavelet 0.430 | + ghost 0.816 | + 2D 0.538 | + ghost + 2D 0.881.
    Both corrections are needed; the second is worth about 0.07.

    p is FIXED at the theoretical 1/2, not fitted.  An empirical scan prefers
    p ~ 1, but a larger exponent strips more high-frequency energy from the
    reference and thereby absorbs every other high-frequency loss of the
    simulation -- grid dispersion at 5.6 points per wavelength, source and
    receiver interpolation, the damping boundaries.  Fitting it would tune the
    reference until it matched the data, which stops measuring the method.
    """
    n = len(x)
    X = np.fft.rfft(x, 2 * n)
    f = np.fft.rfftfreq(2 * n, dt)
    H = np.exp(1j * np.radians(phase_deg)) * (2 * np.pi * np.maximum(f, 1e-9)) ** (-p)
    H[0] = 0.
    return np.fft.irfft(X * H, 2 * n)[:n]


def spec_corr(a, b, dt, band=BAND, nfft=1024):
    """Correlation between two amplitude spectra over the useful band."""
    A = np.abs(np.fft.rfft(np.asarray(a, float), nfft))
    B = np.abs(np.fft.rfft(np.asarray(b, float), nfft))
    f = np.fft.rfftfreq(nfft, dt)
    m = (f >= band[0]) & (f <= min(band[1], 0.5 / dt * 0.95))
    if m.sum() < 5:
        return float("nan")
    x = A[m] - A[m].mean(); y = B[m] - B[m].mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else float("nan")


def ncc_slide(wlong, ref, o_lo=0, o_hi=None, o_max=None):
    """Slide the reference through a LONGER extraction: full overlap always.

    With equal lengths only wl - |lag| samples can be compared, and the lag
    reaches the search limit in a few per cent of cases, so the correlation is
    then computed on 60 per cent of the wavelet and is not comparable with the
    rest.  Extracting with a margin removes the problem.

    o_max BOUNDS the slide, and must be set from the physics.  Unbounded
    sliding measures the wrong thing in shallow water: with the direct arrival
    and the primary only 16 ms apart, the comparison simply finds whichever of
    the two resembles the reference better, and the question "which event did
    the workflow extract" can no longer be answered.  Limiting the slide to half
    the direct/primary separation makes it impossible to swap the two events
    while still absorbing the bias of the origin criterion.
    """
    wl = len(ref)
    y = ref - ref.mean(); ny = np.linalg.norm(y)
    top = len(wlong) - wl
    if o_hi is None:
        o_hi = top if o_max is None else min(top, int(o_max))
    lo = int(max(0, o_lo)); hi = int(min(top, o_hi))
    best, bo = -2.0, lo
    for o in range(lo, hi + 1):
        seg = wlong[o:o + wl]
        x = seg - seg.mean(); nx = np.linalg.norm(x)
        if nx * ny < 1e-12:
            continue
        v = float(x @ y / (nx * ny))
        if v > best:
            best, bo = v, o
    return best, bo


def reference(w_bare, dt, wl, z_src=Z_DEV, z_rec=Z_DEV, two_d=True):
    """Reference wavelet: w_bare with the ghost Devito actually produced, at the
    depths of the cell.

    wl is in samples; a value in milliseconds (a float) is caught here rather
    than failing in a slice further on.
    """
    if not isinstance(wl, (int, np.integer)):
        raise TypeError(f"reference(): wl must be in samples (int), "
                        f"got {wl!r}")
    w = np.asarray(w_bare, np.float64)
    if abs(dt - AG_DT) > 1e-9:
        w = resample_poly(w, int(round(len(w) * AG_DT / dt)), len(w))
    wg = apply_ghost(w.astype(np.float32), dt, z_src, z_rec).astype(np.float64)
    if two_d:
        wg = two_d_factor(wg, dt)
    nyq = 0.5 / dt
    b, a = butter(4, [BAND[0] / nyq, min(BAND[1] / nyq, .95)], 'band')
    wg = filtfilt(b, a, wg)
    wg = wg[:wl] if len(wg) >= wl else np.pad(wg, (0, wl - len(wg)))
    return nrm(wg - wg.mean())


# --------------------------------------------------------------------------- #
def generate_model(run, cfg):
    """Generate the line with the parameters of the cell; returns a dict."""
    p = run["params"]; der = run["derived"]; m = cfg["meta"]
    geom = G.GEOMS[p["geom"]]; offsets = G.geom_offsets(geom)
    n_shots = m["n_shots"]
    XMAX = G.domain_for_shots_geom(n_shots + 4, geom)
    rng = np.random.default_rng(run["seed"])

    vp, rho, kwb, meta = generate_velocity_model(
        rng, XMAX=XMAX, ZMAX=der["ZMAX_m"], DH=G.DH,
        D_water_range=(p["D_m"], p["D_m"]),  # imposed water depth
        p_fault=p["p_fault"], p_folds=p["p_folds"], p_lens=p["p_lens"],
        slope_deg=p["slope_deg"], roughness_m=p["roughness_m"],
        first_layer_m=p["first_layer_m"])

    model = Model(vp=(vp / 1000.), b=1.0 / rho, origin=(0., 0.), shape=vp.shape,
                  spacing=(G.DH, G.DH), space_order=8, nbl=G.NB, bcs="damp", fs=True)
    xs_all = G.shot_positions_geom(XMAX, geom)
    n_shots = min(n_shots, len(xs_all))
    src_fn, w_bare = make_source_fn(10_000 + run["seed"], fmax=p["fmax_hz"])

    dt_out = m["dt_ms"] / 1000.
    # tn adapted to the depth: the record must reach past the first multiple
    tn = float(np.clip(2.6 * (2 * p["D_m"] / m["v_water"]) * 1000. + 800., 1500., 4500.))
    NT = int(round(tn / 1000. / dt_out))
    g = np.zeros((n_shots, NT, len(offsets)), np.float32)
    zs = p.get("z_src_m", None); zr = p.get("z_rec_m", None)
    for i in range(n_shots):
        try:
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offsets,
                             z_src=zs, z_rec=zr)
        except TypeError:
            # shoot() without z_src/z_rec would leave the depths at 6/6 m: better to
            # stop than to generate data inconsistent with the reference
            if zs is not None and (zs != Z_DEV or zr != Z_DEV):
                raise RuntimeError("shoot() non accetta z_src/z_rec: applicare "
                                   "patch_shoot_z.py prima di usare l'asse z")
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offsets)
        fr = Fraction(float(dtd) / 1000. / dt_out).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
        d = d[:NT] if d.shape[0] > NT else np.pad(d, ((0, NT - d.shape[0]), (0, 0)))
        g[i] = d

    # true R_wb below the reflection points of the near trace (a covariate)
    x_mid = xs_all[:n_shots] - offsets[0] / 2.
    idx = np.clip((x_mid / G.DH).astype(int), 0, vp.shape[0] - 1)
    Z2 = np.array([vp[j, kwb[j]] * rho[j, kwb[j]] for j in idx])
    R_true = (Z2 - Z_WATER) / (Z2 + Z_WATER)

    return dict(gathers=g, dt=dt_out, offsets=offsets.astype(np.float32),
                w_bare=w_bare, D_true=(kwb[idx] * G.DH).astype(float),
                R_wb_true=float(np.median(R_true)), n_layers=int(meta["n_layers"]),
                has_fault=("fault" in meta), has_lens=bool(meta.get("lens", False)),
                tn_ms=tn)


def process(run, cfg):
    p = run["params"]; der = run["derived"]; m = cfg["meta"]
    md = generate_model(run, cfg)
    g = md["gathers"]; dt = md["dt"]; offs = md["offsets"]
    D_nom = float(np.median(md["D_true"]))
    wl = int(round(m["w_len_ms"] * 1e-3 / dt))
    w_ref = reference(md["w_bare"], dt, wl,      # wl in SAMPLES, not ms
                      z_src=p.get("z_src_m", Z_DEV), z_rec=p.get("z_rec_m", Z_DEV),
                      two_d=True)
    w_ref_1d = reference(md["w_bare"], dt, wl,   # continuity with earlier runs
                         z_src=p.get("z_src_m", Z_DEV),
                         z_rec=p.get("z_rec_m", Z_DEV), two_d=False)
    n_ch = der["n_ch_used"]

    # D_guess sizes the window on the direct/primary separation.  It is not
    # circular: it is the nominal depth of the cell, and on real data it would
    # be a rough seafloor pick.  It is needed because in shallow water a fixed
    # window locks onto the primary (1566 m/s measured for water at 1500, D=100).
    vw, _x0, n_pick = estimate_vw_direct(g, dt, offs, D_guess=D_nom)
    r0 = semblance_bottom(g[0].T, dt, offs,
                          t0_min_ms=max(120, 2 * D_nom / vw * 1e3 - 200),
                          t0_max_ms=2 * D_nom / vw * 1e3 + 200)
    t0i = r0['t0'] if r0 else 2 * D_nom / vw
    # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
    # were computed; it is written out so that the default of 20 does not change them.
    t0 = track_seafloor_xcorr(g, dt, offs, t0_init_s=t0i, vw=vw, n_group=5)

    recs = []
    # The bias on t0 is a parameter of the extraction, not of the generation:
    # it is applied to the same gathers, to inject a systematic misplacement of
    # the window, which stack coherence cannot see because every shot is wrong
    # by the same amount.
    biases = cfg["posthoc"].get("t0_bias_ms", [0.0])
    for snr in cfg["posthoc"]["snr_db"]:
        rng = np.random.default_rng(run["seed"] + int(snr))
        gg = g
        snr_eff = float('inf')
        if snr < 60:
            ns = np.sqrt(np.mean(g ** 2)) / (10 ** (snr / 20.))
            gg = g + ns * rng.normal(0, 1, g.shape).astype(np.float32)
            # nominal SNR (calibrated on the whole gather) against the effective SNR
            # in the primary window: they differ by about 4 dB because the gather is
            # mostly empty, plus about 13 dB of stack gain over 20 shots.  It is
            # recorded so that levels are comparable between cells with different
            # record lengths or depths.
            t_a = np.sqrt((2 * D_nom / vw) ** 2 + (offs[0] / vw) ** 2)
            k0 = int(round(t_a / dt)) - int(round(0.080 / dt))
            W = int(round((0.080 + m["w_len_ms"] * 1e-3 + 0.060) / dt))
            if 0 <= k0 and k0 + W <= g.shape[1]:
                rms_win = float(np.sqrt(np.mean(g[:, k0:k0 + W, 0] ** 2)))
                snr_eff = float(20 * np.log10(rms_win / (ns + 1e-30))
                                + 10 * np.log10(max(g.shape[0], 1)))
        rec_base = dict(run_id=run["run_id"], cell_id=run["cell_id"], axis=run["axis"],
                   replica=run["replica"], seed=run["seed"], snr_db=snr,
                   snr_eff_db=snr_eff,
                   **{k: p[k] for k in ("geom", "D_m", "fmax_hz", "first_layer_m",
                                        "slope_deg", "roughness_m", "p_fault",
                                        "p_folds", "p_lens")},
                   z_src_m=p.get("z_src_m", Z_DEV), z_rec_m=p.get("z_rec_m", Z_DEV),
                   **{k: der[k] for k in ("D_min_m", "x_max_usable_m", "n_ch_used",
                                          "n_ch_truncated", "first_refl_twt_ms",
                                          "expected_pass", "sep_margin")},
                   D_true=D_nom, R_wb_true=md["R_wb_true"],
                   n_layers=md["n_layers"], has_fault=md["has_fault"],
                   has_lens=md["has_lens"], tn_ms=md["tn_ms"],
                   vw_est=float(vw), vw_npick=int(n_pick),
                   t0_track_ms=float(np.mean(t0) * 1e3),
                   t0_true_ms=float(2 * D_nom / m["v_water"] * 1e3))
        for bias in biases:
          t0b = t0 + bias * 1e-3
          rec = dict(rec_base, t0_bias_ms=float(bias))
          try:
              # Extract with a margin, so that the comparison always has full overlap;
              # the sliding comparison discards the surplus.  The margin adapts to the
              # record: a fixed 120 ms pushes the window before t = 0 in shallow water,
              # where every window would be rejected (264 of 3200 records, all at
              # D = 100-180 m, the hardest cells).
              t_arr_min = float(np.min(np.sqrt(t0b ** 2 + (offs[0] / vw) ** 2)))
              lead_max = max(20., 0.90 * t_arr_min * 1e3)
              PAD = float(np.clip(lead_max - 80., 0., 120.))
              wp_l, qp = extract_wavelet(gg, dt, t0b, offs,
                                         w_len_ms=m["w_len_ms"] + 2 * PAD,
                                         lead_ms=min(80. + PAD, lead_max),
                                         n_ch=n_ch, order=0, band=BAND)
              if wp_l is None:
                  rec.update(ok=False, error=qp.get("error")); recs.append(rec); continue
              wlong = nrm(np.asarray(wp_l, float))
              # The slide is bounded twice.  Half the direct/primary separation keeps it
              # from swapping those two events in shallow water; in deep water that bound
              # is hundreds of ms and the reference could latch onto a later sub-bottom
              # event (matches at +306 to +442 ms were seen).  The margin added to the
              # extraction is the tighter bound: sliding beyond it means the origin was
              # not where the workflow put it, which is a different failure.
              sep_s = (np.hypot(2 * D_nom, offs[0]) - offs[0]) / vw
              npad_s = int(round(PAD * 1e-3 / dt))
              o_lim = max(2, min(npad_s, int(round(0.5 * sep_s / dt))))
              o_lo = max(0, npad_s - o_lim)
              o_hi = min(len(wlong) - wl, npad_s + o_lim)
              v_p, off_p = ncc_slide(wlong, w_ref, o_lo=o_lo, o_hi=o_hi)
              v_p_1d, _ = ncc_slide(wlong, w_ref_1d, o_lo=o_lo, o_hi=o_hi)
              lag_p = off_p - int(round(PAD * 1e-3 / dt))
              rec["pad_ms"] = PAD
              rec["slide_lo_ms"] = float(o_lo * dt * 1e3)
              rec["slide_hi_ms"] = float(o_hi * dt * 1e3)
              # physical position of the compared segment relative to the predicted
              # primary: near zero means the primary was locked, near minus the
              # separation means the direct arrival was
              pos_ms = (qp["origin_minus_lead_ms"] + off_p * dt * 1e3)
              wp = wlong[off_p:off_p + wl]
              # counted here, not derived later: frac_padded is what the extractor filled
              # in its own output, n_invented_kept how much of it entered the segment the
              # score was computed on; they differ because the slide usually steps clear
              # of the filled tail
              rec["frac_padded"] = float(qp.get("frac_padded", 0.0))
              _nz = np.nonzero(wp)[0]
              rec["n_invented_kept"] = int(len(wp) - _nz[-1] - 1) if len(_nz) \
                  else int(len(wp))
              # the aligned segment and its reference are kept, so that a figure never
              # has to rebuild the experiment to draw it
              rec["_wavelet"] = np.asarray(wp, np.float32)
              rec["_reference"] = np.asarray(w_ref, np.float32)
              # t0 corrected by the measured primary time, then the multiple
              t0c = np.sqrt(np.maximum(
                  (np.sqrt(t0b ** 2 + (offs[0] / vw) ** 2)
                   + qp["origin_minus_lead_ms"] * 1e-3) ** 2 - (offs[0] / vw) ** 2, 1e-12))
              wm, qm = extract_wavelet(gg, dt, t0c, offs, w_len_ms=m["w_len_ms"],
                                       n_ch=n_ch, order=1, band=BAND, template=wp,
                                       template_search_ms=80., template_expected_sign=-1)
              # --- independent statistical check, calibrated against the truth ---
              # On real data the extraction is compared with a statistical estimate of the
              # amplitude spectrum, because there is no truth.  Here the truth is known,
              # so the check itself is calibrated: how well the statistical estimate
              # recovers the true spectrum, and how well its agreement with the
              # extraction predicts the true accuracy.  The models have 6 to 14 layers,
              # so this also measures what the violated whiteness assumption costs.
              try:
                  f_st, A_st, n_st = statistical_spectrum(
                      gg, dt, t0b, offs, n_ch=min(8, gg.shape[2]),
                      win_ms=700., below_ms=250., band=BAND)
              except Exception:
                  f_st, A_st, n_st = None, None, 0
              if A_st is not None:
                  nf = 1024
                  fr_ = np.fft.rfftfreq(nf, dt)
                  A_i = np.interp(fr_, f_st, A_st)
                  bm = (fr_ >= BAND[0]) & (fr_ <= min(BAND[1], 0.5 / dt * 0.95))
                  def _sc(x):
                      X = np.abs(np.fft.rfft(np.asarray(x, float), nf))
                      u = X[bm] - X[bm].mean(); v2 = A_i[bm] - A_i[bm].mean()
                      d = np.linalg.norm(u) * np.linalg.norm(v2)
                      return float(u @ v2 / d) if d > 1e-12 else float("nan")
                  rec.update(spec_extr_vs_stat=_sc(wp),
                             spec_true_vs_stat=_sc(w_ref),
                             spec_extr_vs_true=spec_corr(wp, w_ref, dt),
                             n_stat_windows=int(n_st))
              rec.update(ok=True, ncc_prim_true=v_p,
                         ncc_prim_true_1d=v_p_1d, lag_prim=lag_p,
                         match_pos_ms=float(pos_ms),
                         sep_direct_primary_ms=float(sep_s * 1e3),
                         coh_prim=qp["stack_coherence"], n_win_prim=qp["n_windows"],
                         origin_minus_lead_ms=qp["origin_minus_lead_ms"],
                         align_res_ms=qp["align_residual_ms_mean"],
                         amp_prim=float(np.sqrt(np.mean(
                             np.asarray(qp["w_raw"], float)[off_p:off_p + wl] ** 2))))
              if wm is not None:
                  v_m, lag_m = ncc_shift(nrm(-np.asarray(wm, float)), w_ref)
                  tpl = nrm(np.asarray(wp, float) - np.mean(wp))
                  pm = float((qm["w_raw"] - qm["w_raw"].mean()) @ tpl)
                  # qp["w_raw"] is the long extraction, with its margin; the projection
                  # uses the trimmed segment that was compared
                  wpr = np.asarray(qp["w_raw"], float)[off_p:off_p + wl]
                  pp = float((wpr - wpr.mean()) @ tpl)
                  r1 = float(np.hypot(2 * D_nom, offs[0]))
                  r2 = float(np.hypot(4 * D_nom, offs[0]))
                  rec.update(ncc_mult_true=v_m, coh_mult=qm["stack_coherence"],
                             n_win_mult=qm["n_windows"],
                             # blind indicators, available on real data
                             validator=qm["template_score"],
                             validator_sign=qm["template_sign"],
                             dt_matched_ms=(qm["origin_minus_lead_ms"]
                                            - qp["origin_minus_lead_ms"]),
                             R_wb_est=abs(pm / (pp + 1e-30)) * (r2 / r1) ** 0.5)
              else:
                  rec.update(mult_error=qm.get("error"))
          except Exception:
            rec.update(ok=False, error=traceback.format_exc(limit=3))
          recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--only", default=None,
                    help="comma-separated run_ids; keep only these.  Use --out "
                         "as well, so the matrix is not touched")
    ap.add_argument("--out", default=None,
                    help="override paths.out_root from the config.  Regenerating "
                         "a few records into the matrix directory would mix them "
                         "with runs made under different code")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src_root = Path(cfg["paths"]["out_root"])
    runs = [json.loads(l) for l in open(src_root / cfg["paths"]["manifest"])]
    out = Path(args.out) if args.out else src_root
    out.mkdir(parents=True, exist_ok=True)
    if args.out:
        print(f"manifest read from {src_root}, results written to {out}")
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        runs = [r for r in runs if r["run_id"] in want]
        missing = want - {r["run_id"] for r in runs}
        if missing:
            raise SystemExit("unknown run_id: %s" % ", ".join(sorted(missing)))
        print("--only: %d run(s) kept" % len(runs))
    runs = runs[args.shard::args.n_shards]
    if args.limit:
        runs = runs[:args.limit]
    rp = out / f"results_shard{args.shard:03d}.jsonl"
    done = set()
    if rp.exists():
        done = {json.loads(l)["run_id"] for l in open(rp)}
    runs = [r for r in runs if r["run_id"] not in done]
    print(f"[shard {args.shard}] {len(runs)} runs ({len(done)} already done)", flush=True)

    # wavelets: a companion archive keyed like the records, rewritten whole
    # after each run (npz has no append; a few thousand short arrays take
    # milliseconds)
    wp_path = out / f"results_shard{args.shard:03d}_wavelets.npz"
    waves = {}
    if wp_path.exists():
        with np.load(wp_path) as z:
            waves = {k: z[k] for k in z.files}

    t0 = time.time()
    with open(rp, "a") as f:
        for i, r in enumerate(runs, 1):
            for rec in process(r, cfg):
                w_arr = rec.pop("_wavelet", None)
                r_arr = rec.pop("_reference", None)
                if w_arr is not None:
                    key = (f"{rec['run_id']}|{rec.get('snr_db')}|"
                           f"{rec.get('t0_bias_ms')}")
                    waves[key] = w_arr
                    waves[key + "|ref"] = r_arr
                f.write(json.dumps(rec) + "\n")
            f.flush()
            np.savez_compressed(wp_path, **waves)
            if i % 5 == 0:
                el = time.time() - t0
                print(f"[shard {args.shard}] {i}/{len(runs)} {el/i:.0f}s/run "
                      f"ETA {(len(runs)-i)*el/i/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
