# run_real -- the SEAWEED workflow on a SEG-Y line: from the gathers to the
# effective wavelet and its checks.
#
#   [0]  t0 from the autocorrelation, and the amplitude trend of the record
#   [1]  water velocity from the direct arrival
#   [2]  seafloor tracking: anchors (semblance, or a file from stage 1) and
#        cross-correlation between them; [2b] amplitude trend between primary
#        and first multiple
#   [3]  applicability: separation of the direct arrival from the primary
#   [4]  extraction: alignment and weighted stack of the near-channel windows
#   [5]  the first and second multiples: depth from the lag, two R_wb
#   [6]  comparison with a reference wavelet, when one is given (--ref)
#
# There is no ground truth on a real line.  The checks test consistency: the
# multiples test the amplitude assumptions, the statistical spectrum tests the
# amplitude spectrum, and a reference made by the same stacking approach
# measures stability, not accuracy.
#
# Spreading is 3D on field data (--alpha 1); the synthetics of the paper are 2D
# (0.5), and the wrong exponent changes R_wb by about 40%.  The water velocity
# is estimated, not assumed: 1451 m/s was measured in the cold water of the
# Ross Sea.
#
#     python -m seaweed.run_real --segy line.sgy --near 125 --dx 25 \\
#         --nshot 1000 --out line
import os, argparse, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly

from seaweed.wavelet_extractor_v2 import (track_seafloor_xcorr, extract_wavelet,
                                  estimate_vw_direct, arrival_time, envelope,
                                  estimate_t0_autocorr,
                                  estimate_vw_direct_semblance,
                                  statistical_spectrum, find_origin)
from seaweed.seafloor_robust import semblance_bottom
from seaweed import code_version

BAND = (6., 90.)


def nrm(x):
    return x / (np.linalg.norm(x) + 1e-9)


def ncc_shift(a, b, max_lag=60, min_overlap=None):
    # the minimum overlap scales with the wavelet: at dt = 4 ms a 200 ms
    # wavelet has only 50 samples, and a fixed overlap of 60 was never reached
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    if min_overlap is None:
        min_overlap = max(20, int(0.6 * min(na, nb)))
    def _c(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / d) if d > 1e-12 else 0.0
    best, blag, bov = -2., 0, 0
    for L in range(-max_lag, max_lag + 1):
        a0 = max(0, L); a1 = min(na, nb + L); ov = a1 - a0
        if ov < min_overlap:
            continue
        v = _c(a[a0:a1], b[a0 - L:a0 - L + ov])
        if v > best:
            best, blag, bov = v, L, ov
    return best, blag, bov


def load_segy(path, shot0=None, nshot=100, nch=None):
    """Load consecutive shots. Returns (S,T,C), dt, offsets from the headers."""
    import segyio as sio
    with sio.open(path, "r", ignore_geometry=True) as f:
        ffid = f.attributes(sio.TraceField.FieldRecord)[:]
        offs = f.attributes(sio.TraceField.offset)[:]
        dt = sio.tools.dt(f) / 1e6
        shots = np.unique(ffid)
        s0 = shot0 if shot0 is not None else int(shots[0])
        use = shots[np.searchsorted(shots, s0):][:nshot]
        data, off_ref = [], None
        for sid in use:
            idx = np.where(ffid == sid)[0]
            o = np.abs(offs[idx]); order = np.argsort(o)
            idx = idx[order]; o = o[order]
            if nch:
                idx, o = idx[:nch], o[:nch]
            tr = np.stack([f.trace[i] for i in idx])     # (C, T)
            if off_ref is None:
                off_ref = o.astype(np.float32)
            if tr.shape[0] != len(off_ref):
                continue                                  # incomplete shot
            data.append(tr.T)                             # (T, C)
    if not data:
        raise RuntimeError("no complete shot loaded")
    T = min(d.shape[0] for d in data)
    g = np.stack([np.nan_to_num(d[:T]) for d in data]).astype(np.float32)
    return g, dt, off_ref


def bistability(qp, dt, block=50, min_win=10, tol=2):
    """Origins of the partial stacks of `block` shots against the full stack's,
    from the aligned windows and weights of the primary extraction.  Bistable
    when over 10 per cent of the blocks are
    beyond `tol` samples, or the origins split into two clusters more than
    T_dom/2 apart; T_dom is that of the two-stage alignment or, with the single
    one, 4 x the first zero of the autocorrelation of the stack."""
    up = 4
    al = qp["aligned_segments"]; wts = np.clip(qp["coh_per_window"], 0, None)
    shot = np.array([s for s, _ in qp["windows_used"]]); noise = qp["noise_segments"]

    def origin(m):
        st = resample_poly((wts[m, None] * al[m]).sum(0) / (wts[m].sum() + 1e-12), 1, up)
        st = st - st.mean()
        return find_origin(st, dt, noise_seg=noise[m].ravel())[0]

    i_full = origin(np.ones(len(shot), bool))
    td = qp.get("align_tdom_ms", float("nan"))
    if not np.isfinite(td):
        st = qp["stack_upsampled"] - qp["stack_upsampled"].mean()
        ac = np.correlate(st, st, "full")[len(st) - 1:]
        iz = int(np.argmax(ac <= 0)); td = 4e3 * iz * dt / up
    d = []
    for b0 in range(0, int(shot.max()) + 1, block):
        m = (shot >= b0) & (shot < b0 + block)
        if m.sum() >= min_win:
            d.append(origin(m) - i_full)
    d = np.sort(np.array(d, float))
    n_out = int(np.sum(np.abs(d) > tol)); gap = float(np.max(np.diff(d)) * dt * 1e3) if d.size > 1 else 0.
    return dict(n_blocks=int(d.size), n_out=n_out, gap_ms=gap, tdom_ms=float(td),
                offsets_samp=d, bistable=bool(n_out > 0.1 * max(d.size, 1) or gap > 0.5 * td))


# The final recipe: what it sets when the flag is not given.  The paper recipe
# keeps the defaults of the first version of the workflow.
RECIPES = {"paper": dict(align="single", bistab_check=False, t0_shift=0.0, n_group=20),
           "final": dict(align="two_stage", bistab_check=True, t0_shift=0.0, n_group=None)}


def n_group_rule(dxsp):
    """Shots in 500 m, at least 5: the tracking window of the final recipe."""
    return max(5, int(round(500.0 / dxsp)))


def apply_recipe(args):
    """Fill the flags left unset (None) from --recipe; an explicit flag wins.
    Returns the names of the flags the recipe set."""
    rec = RECIPES[args.recipe]
    if rec["n_group"] is None and args.n_group is None:
        if args.dxsp is None:
            raise SystemExit("--recipe final needs --dxsp (shot spacing, m) for n_group = max(5, round(500 m / dxsp))")
        rec = dict(rec, n_group=n_group_rule(args.dxsp))
    set_ = []
    for k, v in rec.items():
        if getattr(args, k) is None:
            setattr(args, k, v); set_.append(k)
    return set_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segy", required=True)
    ap.add_argument("--shot0", type=int, default=None)
    ap.add_argument("--nshot", type=int, default=100)
    ap.add_argument("--nch_load", type=int, default=None)
    ap.add_argument("--near", type=float, default=None,
                    help="near offset in m; header value used if omitted")
    ap.add_argument("--dx", type=float, default=None, help="channel spacing in m")
    ap.add_argument("--D_init", type=float, default=None,
                    help="approximate seafloor depth. If omitted it is estimated "
                         "from the autocorrelation (recommended: a D_init wrong "
                         "by 200 m dropped the MS29 correlation from 0.94 to 0.50)")
    ap.add_argument("--wl_ms", type=float, default=200.)
    ap.add_argument("--n_ch", type=int, default=1)
    ap.add_argument("--ch_start", type=int, default=0,
                    help="first channel used for the extraction, counting from "
                         "zero.  With --n_ch 1 this selects a single channel, "
                         "which is how the wavelet is measured at a chosen "
                         "incidence angle rather than at normal incidence")
    ap.add_argument("--weights", default="coherence",
                    choices=["coherence", "uniform", "snr"],
                    help="how the aligned windows are weighted in the stack. "
                         "'coherence' is the published default; 'uniform' keeps "
                         "the diversity between windows, which is what averages "
                         "out contamination that varies along the line; 'snr' "
                         "judges each window on its own amplitude against its "
                         "pre-arrival noise")
    ap.add_argument("--align", choices=("single", "two_stage"), default=None,
                    help="window alignment: 'single' (each window within "
                         "+-40 ms of the pilot, the default) or 'two_stage' ("
                         "groups of --n_group shots shifted in bulk within +-40 ms, "
                         "then each window within +-T_dom/2 of its group; the "
                         "default of --recipe final)")
    ap.add_argument("--bistab_check", action="store_true", default=None,
                    help="after the primary extraction, the origins of partial "
                         "stacks of 50 shots against the full stack's: flagged "
                         "bistable when over 10%% of the blocks are beyond 2 "
                         "samples or the origins split into two clusters more "
                         "than T_dom/2 apart.  It flags and corrects nothing "
                         "(on by default with --recipe final)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="spreading exponent: 1.0 = 3D (real data), 0.5 = 2D")
    ap.add_argument("--ref", default=None, help="reference wavelet .npy")
    ap.add_argument("--ref_dt", type=float, default=None, help="sampling interval of the reference (s)")
    ap.add_argument("--t0_shift", type=float, default=None,
                    help="shift of the tracked seafloor in ms, NEGATIVE to move "
                         "earlier. Cross-correlation locks onto the body of the "
                         "waveform, not its onset: -10/-25 ms keeps both primary "
                         "and multiple fully inside the window")
    ap.add_argument("--n_group", type=int, default=None,
                    help="adjacent shots combined in the 2D tracking window "
                         "(robustness to noisy channels and shots); 20 with "
                         "--recipe paper, max(5, round(500 m / --dxsp)) with "
                         "--recipe final")
    ap.add_argument("--recipe", choices=tuple(RECIPES), default="paper",
                    help="defaults of --align, --bistab_check, --t0_shift and "
                         "--n_group: 'paper' (single, off, 0, 20: the published "
                         "runs) or 'final' (two_stage, on, 0, the 500-m rule), "
                         "the extraction of seaweed.seafloor_pipeline for cells "
                         "and combs on its tracking.  An explicit flag wins")
    ap.add_argument("--dxsp", type=float, default=None,
                    help="shot spacing in m, for the n_group of --recipe final")
    ap.add_argument("--n_ch_track", type=int, default=3,
                    help="near channels used in tracking")
    ap.add_argument("--sp_first", type=int, default=None,
                    help="first shot index used for EXTRACTION (tracking always "
                         "uses the whole line). Default: all shots")
    ap.add_argument("--sp_last", type=int, default=None,
                    help="last shot index used for extraction (exclusive)")
    ap.add_argument("--auto_select", type=int, default=None,
                    help="automatically pick the best CONTIGUOUS run of this "
                         "many shots, by stack coherence. The workflow is: track "
                         "and extract over the whole line, look at where the "
                         "windows agree, then re-extract on the best portion")
    ap.add_argument("--stretch_semb", type=float, default=0.45,
                    help="NMO stretch limit for the reflection semblance. This "
                         "sets the usable offset from the geometry itself, "
                         "x <= 2 D sqrt((1+s)^2-1), so it adapts to water depth "
                         "instead of being a fixed distance: 420 m over a 200 m "
                         "seafloor, 1430 m over a 683 m one")
    ap.add_argument("--x_max_semb", type=float, default=0.,
                    help="hard offset limit in metres for the semblance, "
                         "overriding the stretch criterion. 0 disables it")
    ap.add_argument("--n_ch_direct", type=int, default=40,
                    help="near channels used for the direct-arrival velocity. "
                         "The direct wave decays as 1/sqrt(r) and the far traces "
                         "are noisy; note also that it measures the SURFACE "
                         "layer, not the whole water column")
    ap.add_argument("--wiggle_gain", type=float, default=2.0,
                    help="display gain for the aligned-window wiggle panel; "
                         "affects the figure only")
    ap.add_argument("--drift_smooth", type=int, default=5,
                    help="width, in anchors and odd, of the moving median applied "
                         "to the drift correction; with fewer anchors it narrows "
                         "to the largest odd number available. 1 disables "
                         "smoothing (exact interpolation through every anchor, "
                         "which turns anchor noise into fast undulations of the "
                         "tracked seafloor)")
    ap.add_argument("--drift_window_shots", type=int, default=0,
                    help="TRIAL, not a default: take the drift median over the "
                         "anchors within this many shots of each anchor instead "
                         "of over --drift_smooth anchors; widened to the nearest "
                         "anchors until there are at least three, and always an "
                         "odd number.  0 (default) keeps --drift_smooth")
    ap.add_argument("--n_anchor", type=int, default=20,
                    help="semblance anchors along the line")
    ap.add_argument("--use_semblance_vw", action="store_true",
                    help="use the semblance Vw instead of the direct-arrival one")
    ap.add_argument("--sp_step", type=int, default=1,
                    help="use every Nth shot of the selected range for the "
                         "EXTRACTION. 1 = contiguous (default). With sp_step>1 "
                         "the same fold is spread over N times the line, which "
                         "separates fold from line length: the contamination "
                         "decorrelates with distance, the stack does not care "
                         "how far apart the windows are")
    ap.add_argument("--save_track", default=None,
                    help="write the tracked seafloor (t0, Vw, D_init) to this "
                         ".npz after tracking the whole line, with the "
                         "parameters and anchors that produced it (key params)")
    ap.add_argument("--load_track", default=None,
                    help="reuse a cached tracking instead of recomputing it. "
                         "Guarantees that every run of an experiment sees the "
                         "SAME seafloor, so that only the shot selection varies")
    ap.add_argument("--save_cube", default=None,
                    help="write the loaded gather to this .npy (plus a .meta.npz "
                         "for dt and the header offsets) so that later runs on "
                         "the same line can memory-map it instead of re-reading "
                         "the SEG-Y")
    ap.add_argument("--load_cube", default=None,
                    help="memory-map a cube written by --save_cube. The array "
                         "handed on is identical, so results must not change")
    ap.add_argument("--load_anchors", default=None,
                    help="take the seafloor anchors from this .npz (keys i, t "
                         "and optionally V) instead of running the semblance. "
                         "Everything else in the tracking stage is unchanged, so "
                         "the 2D pilot cross-correlation, the drift smoothing and "
                         "the t0_shift still apply")
    ap.add_argument("--out", default="real_qc")
    args = ap.parse_args()
    by_recipe = apply_recipe(args)
    print(f"recipe {args.recipe}: align {args.align}, bistab_check {args.bistab_check}, "
          f"t0_shift {args.t0_shift:g} ms, n_group {args.n_group}"
          + (f" (set by the recipe: {', '.join(by_recipe)})" if args.recipe != "paper" else ""))
    if args.drift_smooth < 1 or args.drift_smooth % 2 == 0:
        ap.error("--drift_smooth must be odd and positive: an even width "
                 "has no centre")

    # Cached tracking: stages [0] to [2] are skipped when it is given; the
    # shot selection and the extraction are unaffected.
    TRK = None
    if args.load_track:
        TRK = np.load(args.load_track, allow_pickle=True)
        print(f"\n*** tracking loaded from {args.load_track}: steps [0], [1] "
              f"and [2] are SKIPPED ***")

    if args.load_cube:
        meta = np.load(args.load_cube + ".meta.npz", allow_pickle=True)
        g = np.load(args.load_cube, mmap_mode="r")
        dt = float(meta["dt"])
        off_hdr = meta["off_hdr"]
        if off_hdr.ndim == 0 or off_hdr.size == 0:
            off_hdr = None
        print(f"\ngather memory-mapped from {args.load_cube}: {g.shape}, "
              f"{g.dtype}, {g.nbytes/1e9:.2f} GB on disk")
        # The shot selection is an index into this cube, so a cube built with a
        # different --shot0/--nshot would silently shift every selection.
        for k, v in (("shot0", args.shot0), ("nshot", args.nshot)):
            saved = meta[k].item() if k in meta else None
            if saved is not None and v is not None and int(saved) != int(v):
                raise SystemExit(
                    f"cube was built with --{k} {saved} but --{k} {v} was given: "
                    f"the shot indices would not mean the same thing")
    else:
        g, dt, off_hdr = load_segy(args.segy, args.shot0, args.nshot, args.nch_load)
    S, T, C = g.shape
    if args.save_cube:
        np.save(args.save_cube, np.ascontiguousarray(g))
        np.savez(args.save_cube + ".meta.npz", dt=dt,
                 off_hdr=(off_hdr if off_hdr is not None else np.array([])),
                 shot0=(-1 if args.shot0 is None else args.shot0),
                 nshot=(-1 if args.nshot is None else args.nshot),
                 segy=os.path.basename(args.segy))
        print(f"gather cached to {args.save_cube} "
              f"({np.asarray(g).nbytes/1e9:.2f} GB); later runs can use "
              f"--load_cube {args.save_cube}")
    offs = off_hdr
    if args.near is not None and args.dx is not None:
        offs = (args.near + args.dx * np.arange(C)).astype(np.float32)
        print(f"offsets SET from the command line: {offs[0]:.0f}-{offs[-1]:.0f} m")
        if off_hdr is not None and len(off_hdr) == C:
            d = np.abs(offs - off_hdr).max()
            print(f"  max departure from headers: {d:.0f} m"
                  + ("   <-- CHECK" if d > args.dx else ""))
    print(f"\n{os.path.basename(args.segy)}: {S} shot x {C} channels | dt {dt*1e3:.1f} ms "
          f"| record length {T*dt*1e3:.0f} ms")
    print(f"offset {offs[0]:.0f}-{offs[-1]:.0f} m | initial D "
          + (f"{args.D_init:.0f} m (supplied)" if args.D_init is not None
             else "to be estimated from the autocorrelation"))

    wl = int(round(args.wl_ms * 1e-3 / dt))
    nyq = 0.5 / dt
    if BAND[1] / nyq > 0.95:
        print(f"  NOTE: upper band edge {BAND[1]:.0f} Hz close to Nyquist "
              f"({nyq:.0f} Hz): the band-pass will be truncated")

    # --- 0. t0 from the autocorrelation: no dependence on D_init ---
    # The primary and its multiples are periodic with period t0, the sub-bottom
    # geology is not.  This measures a time, so it needs no prior and no Vw, and
    # polarity adds a constraint: the lobe at t0 is negative, the one at 2 t0
    # positive.
    t0_ac, qc_ac = (None, None) if TRK is not None else \
        estimate_t0_autocorr(g, dt, offs, n_shots=min(S, 60))
    if TRK is not None:
        args.D_init = float(TRK["D_init"])
        print(f"[0] D_init from the cached tracking: {args.D_init:.0f} m")
    if t0_ac is not None:
        print(f"[0] t0 from autocorrelation: {np.median(t0_ac)*1e3:.0f} ms "
              f"({qc_ac['n_groups']} groups, {qc_ac['n_rejected']} rejected, "
              f"spread {qc_ac['t0_spread_ms']:.0f} ms)")
        print(f"    lobe at t0 {qc_ac['lobe1']:+.3f} (expected NEGATIVE), "
              f"at 2*t0 {qc_ac['lobe2']:+.3f} (expected positive)"
              + ("" if qc_ac['lobe1'] < 0 else "   <-- ANOMALOUS POLARITY"))
    if args.D_init is None:
        if t0_ac is None:
            raise SystemExit("autocorrelation failed: supply --D_init")
        args.D_init = float(np.median(t0_ac)) * 1500. / 2.
        print(f"    -> D_init inferred: {args.D_init:.0f} m (with a provisional Vw of 1500)")
    elif t0_ac is not None:
        d_ac = float(np.median(t0_ac)) * 1500. / 2.
        if abs(d_ac - args.D_init) > 0.25 * args.D_init:
            print(f"    WARNING: D_init supplied as {args.D_init:.0f} m against "
                  f"{d_ac:.0f} m from the autocorrelation (departure "
                  f"{100*(args.D_init-d_ac)/d_ac:+.0f}%)")

    # --- 0b. amplitude trend of the whole record ---
    # Archive SEG-Y files are often delivered with spherical-divergence
    # compensation or AGC applied.  That leaves the waveform unchanged but breaks
    # the amplitude ratios between events, and so any R_wb estimate.  Raw data
    # decay with time; a flat or rising trend suggests a gain.
    rms_t = np.sqrt(np.mean(g[:, :, 0] ** 2, axis=0))
    tt_ = np.arange(len(rms_t)) * dt
    m_ = (tt_ > 0.3) & (tt_ < min(4.0, tt_[-1]))
    # Over the whole record, however, the trend is not a test: late noise
    # flattens it on raw data (SISOC-MEPA), and a record that starts before
    # the first arrival makes it rise (MS-29).  It is reported here for
    # reference, and the verdict is taken in [2b], between the primary and the
    # first multiple, once the seafloor is known.
    gain_exp = float('nan')        # set in [2b]
    if m_.sum() > 50:
        gain_whole = float(np.polyfit(np.log(tt_[m_]),
                                      np.log(rms_t[m_] + 1e-30), 1)[0])
        print("\n[0b] amplitude trend over the whole record: RMS ~ t^%+.2f"
              % gain_whole)
        print("     not a test of applied gain: late noise flattens it and a "
              "record starting\n     before the first arrival raises it.  "
              "The test is [2b].")

    # --- 1. Vw from the direct arrival ---
    if TRK is not None:
        vw = float(TRK["vw"])
        print(f"\n[1] Vw from the cached tracking: {vw:.1f} m/s")
    else:
        vw, x0_off, npk = estimate_vw_direct(g, dt, offs, D_guess=args.D_init,
                                        n_ch_max=args.n_ch_direct)
        vw_sem, sem_pk, n_ch_sem = estimate_vw_direct_semblance(
            g, dt, offs, D_guess=args.D_init, n_ch_max=args.n_ch_direct)
        print(f"\n[1] Vw from the direct arrival: {vw:.1f} m/s ({npk} picks)  "
              f"[assumed 1500 -> {100*(vw-1500)/1500:+.1f}%]")
        print(f"    fit intercept: {x0_off:+.1f} m. Lumps together the error in the "
              f"nominal near offset\n    and the picking bias (x0 = dx - V*dt): it "
              f"does not separate them, but if |x0|\n    exceeds the channel spacing "
              f"the near offset from the field report needs checking.")

    if TRK is not None:
        t0 = np.asarray(TRK["t0"], float)
        if t0.size != S:
            raise SystemExit(
                f"cached tracking has {t0.size} shots but {S} were loaded: the "
                f"cache must be built with the same --shot0/--nshot")
        print(f"\n[2] seafloor from the cached tracking ({t0.size} shots)")
        trk_params = (json.loads(str(TRK["params"])) if "params" in TRK
                      else dict(note="cached tracking without parameters"))
        if trk_params.get("script") == "h3_track" and "applicable" in trk_params:
            if not trk_params["applicable"]:
                print(f"    ONSET TRACKING NOT APPLICABLE ON THIS UNIT ({trk_params['fallback_reason']}): "
                      f"stage-1 tracking used")
            elif trk_params["fallback"] == "stretches":
                print(f"    onset-tracking fallback on {len(trk_params['unreliable_stretches'])} stretches, "
                      f"{100*trk_params['unreliable_fraction']:.1f}% of the line")
    else:
        # --- 2. seafloor tracking, anchored ---
        # Cross-correlation keeps the seafloor continuous from shot to shot but
        # does not fix its absolute position: on the first shot there is no pilot,
        # and on data with strong sub-bottom reflections the error propagates along
        # the line (on IT90-65, 85 m).  Anchors fix it at intervals, and the
        # cross-correlation follows the seafloor between them.
        anchors_i, anchors_t, Vsem = [], [], []
        step = max(1, S // max(args.n_anchor, 1))
        t_lo = max(0.100, 2 * args.D_init / vw - 0.300)
        t_hi = 2 * args.D_init / vw + 0.300
        raw_i, raw_t, raw_V, sem_full = [], [], [], []
        # what the tracking was made with, written by --save_track
        trk_params = dict(script="run_real", code=code_version(),
                          n_group=args.n_group, n_ch_track=args.n_ch_track,
                          drift_smooth=args.drift_smooth,
                          drift_window_shots=args.drift_window_shots,
                          D_init=args.D_init)
        if args.load_anchors:
            za = np.load(args.load_anchors, allow_pickle=True)
            ai = np.asarray(za["i"], int)
            at = np.asarray(za["t"], float)
            inside = (ai >= 0) & (ai < S)
            if inside.sum() < 2:
                raise SystemExit(f"{args.load_anchors} has fewer than two "
                                 f"anchors inside the {S} shots loaded")
            if inside.sum() < 4:   # stage 1 reduced or single group (anchors_from_consensus)
                print(f"    FEWER THAN FOUR ANCHORS ({int(inside.sum())}): drift correction "
                      f"from {int(inside.sum())} anchors")
            ai, at = ai[inside], at[inside]
            av = (np.asarray(za["V"], float)[inside] if "V" in za
                  else np.full(ai.size, vw))
            # taken as given: whatever produced them has applied continuity already,
            # and the global-median validation below would be wrong on a line with
            # relief
            anchors_i = list(ai); anchors_t = list(at); Vsem = list(av)
            trk_params["anchors"] = dict(
                source="file", file=os.path.basename(args.load_anchors),
                params=(json.loads(str(za["params"])) if "params" in za
                        else None))
            sem_full = [dict(t0=t, V=v) for t, v in zip(at, av)]
            print(f"[2] anchors read from {args.load_anchors}: "
                  f"{len(anchors_i)} anchors, t0 "
                  f"{min(anchors_t)*1e3:.0f}-{max(anchors_t)*1e3:.0f} ms")
            print("    the semblance is skipped; the cross-correlation "
                  "tracking, the drift")
            print("    smoothing and the t0_shift are unchanged")
            if len(anchors_i) > 2:
                gap = int(np.median(np.diff(sorted(anchors_i))))
                print(f"    median anchor spacing {gap} shots" +
                      ("" if gap <= 30 else "  (sparse: consider more)"))
        else:
            trk_params["anchors"] = dict(
                source="semblance", n_anchor=args.n_anchor,
                stretch_semb=args.stretch_semb, x_max_semb=args.x_max_semb)
            for si in range(0, S, step):
                rr = semblance_bottom(g[si].T, dt, offs,
                                      t0_min_ms=t_lo * 1e3, t0_max_ms=t_hi * 1e3,
                                      stretch_max=args.stretch_semb,
                                      x_max=(args.x_max_semb if args.x_max_semb > 0 else None))
                if rr:
                    raw_i.append(si); raw_t.append(rr['t0']); raw_V.append(rr['V'])
                    sem_full.append(rr)
        # anchor validation: an anchor against the edge of the search window is
        # the window running out, not a measurement (on MS-29, with a D_init far
        # from the truth, anchors spread over 536-1108 ms on a flat seafloor), and a
        # single bad anchor drags the whole tracking
        if raw_t and not args.load_anchors:
            rt = np.asarray(raw_t)
            at_edge = (rt < t_lo + 0.020) | (rt > t_hi - 0.020)
            med = np.median(rt[~at_edge]) if (~at_edge).sum() else np.median(rt)
            mad = np.median(np.abs(rt - med)) + 1e-6
            keep = (~at_edge) & (np.abs(rt - med) < max(4 * mad, 0.060))
            n_edge = int(at_edge.sum()); n_out = int((~at_edge & ~keep).sum())
            if n_edge or n_out:
                print(f"[2] anchors rejected: {n_edge} at the edge of the search window"
                      + (" (D_init probably wrong)" if n_edge else "")
                      + f", {n_out} anomalous")
            anchors_i = [raw_i[k] for k in range(len(rt)) if keep[k]]
            anchors_t = [raw_t[k] for k in range(len(rt)) if keep[k]]
            Vsem = [raw_V[k] for k in range(len(rt)) if keep[k]]
        if not anchors_i:
            print("[2] semblance failed everywhere: using D_init")
            trk_params["anchors"] = dict(source="D_init")
            t0i = 2 * args.D_init / vw
            t0 = track_seafloor_xcorr(g, dt, offs, t0_init_s=t0i, vw=vw,
                                      n_group=args.n_group, n_ch=args.n_ch_track)
            t0 = np.maximum(t0 + args.t0_shift * 1e-3, 1e-3)
        else:
            Vs = float(np.median(Vsem))
            print(f"[2] semblance over {len(anchors_i)} anchors: t0 "
                  f"{min(anchors_t)*1e3:.0f}-{max(anchors_t)*1e3:.0f} ms | median V {Vs:.0f} m/s")
            # anchors read from a file carry no semblance peak, so there is no
            # resolution to report
            hw = [x['V_halfwidth'] for x in sem_full if 'V_halfwidth' in x]
            hw_med = (float(np.nanmedian(hw)) if np.isfinite(hw).any()
                      else float('nan'))
            xs = [x['x_max_used'] for x in sem_full if 'x_max_used' in x]
            if np.isfinite(hw_med) and xs:
                print(f"    velocity resolution: peak half-width {hw_med:.0f} m/s, "
                      f"offsets to {float(np.median(xs)):.0f} m "
                      f"(stretch limit {args.stretch_semb:.2f})")
            else:
                print("    velocity resolution: skipped (no semblance on anchors "
                      "read from a file)")
            if np.isfinite(hw_med) and hw_med > 60.:
                print("    NOTE: broad peak, so this velocity is uncertain by tens "
                      "of m/s, and a")
                print("    3 per cent error here is 3 per cent on the water depth. "
                      "It measures the")
                print("    WHOLE COLUMN, whereas the direct arrival measures the "
                      "SURFACE layer:")
                print("    in a stratified column the two legitimately differ.")
            if args.use_semblance_vw and not args.load_anchors:
                # the semblance finds Vw from coherence along the hyperbola over many
                # traces; the direct-arrival picking depends on a per-trace threshold and
                # can overestimate (1574 against about 1451 m/s on IT90-65)
                print(f"    Vw: {vw:.0f} (direct) -> {Vs:.0f} (semblance)")
                vw = Vs
            t0 = track_seafloor_xcorr(g, dt, offs, t0_init_s=anchors_t[0], vw=vw,
                                      n_group=args.n_group, n_ch=args.n_ch_track)
            # Drift correction: the tracked-minus-anchor difference, interpolated
            # between anchors.  np.interp holds the end values outside the anchor range,
            # so a missing first anchor leaves the start of the line with the drift of
            # the first surviving one; denser anchors (--n_anchor) constrain the
            # tracking locally.
            d_anch = np.array(anchors_t) - t0[np.array(anchors_i)]
            # The drift is smoothed with a local moving median, not interpolated
            # exactly through every anchor: each anchor carries an independent error of
            # a few ms, and exact interpolation turns them into undulations of the
            # seafloor with the period of the anchor spacing.  A low-order polynomial
            # is worse (a cubic over 31 anchors left 20.6 ms of residual) and bends at
            # the ends.
            # The width is a number of anchors, odd so that the window is
            # centred, and does not depend on how many anchors there are: it
            # narrows only when there are fewer anchors than the width.
            k = min(args.drift_smooth, len(d_anch) - (1 - len(d_anch) % 2))
            if args.drift_window_shots > 0 and len(d_anch) >= 3:
                ai_ = np.array(anchors_i)
                n_max = len(d_anch) - (1 - len(d_anch) % 2)
                d_sm, widths = [], []
                for i in range(len(d_anch)):
                    dist = np.abs(ai_ - ai_[i])
                    m = int(np.sum(dist <= args.drift_window_shots))
                    m = min(max(3, m + (1 - m % 2)), n_max)
                    near = np.lexsort((ai_, dist))[:m]
                    d_sm.append(np.median(d_anch[near])); widths.append(m)
                d_sm = np.array(d_sm)
                k = f"+-{args.drift_window_shots} shots ({min(widths)}-{max(widths)} anchors)"
            elif k > 1 and len(d_anch) >= 3:
                d_sm = np.array([np.median(d_anch[max(0, i - k // 2):i + k // 2 + 1])
                                 for i in range(len(d_anch))])
            else:
                d_sm = d_anch
            drift = np.interp(np.arange(S), anchors_i, d_sm)
            print(f"    drift: {len(anchors_i)} anchors, moving median over {k}, "
                  f"anchor scatter {np.std(d_anch - d_sm)*1e3:.1f} ms")
            t0 = t0 + drift
            if anchors_i[0] > 0:
                print(f"    NOTE: first valid anchor at shot {anchors_i[0]}; shots "
                      f"0-{anchors_i[0]-1} are only extrapolated. Raise --n_anchor "
                      f"or check the search window if the start looks mis-locked.")
            # The shift is applied after the drift correction: the drift is (anchor -
            # tracked), so a shift applied before it would cancel exactly.  It is
            # applied once: up to commit 72e6fec it was applied a second time just
            # above, so a --t0_shift of -25 moved the seafloor by -50 ms.
            t0 = np.maximum(t0 + args.t0_shift * 1e-3, 1e-3)
            if args.t0_shift:
                print(f"    shift applied after drift correction: "
                      f"{args.t0_shift:+.0f} ms")
            if len(anchors_i) < 3:
                print(f"    WARNING: only {len(anchors_i)} valid anchors, "
                      f"the drift correction is unreliable")
            print(f"    drift relative to anchors: mean {np.mean(drift)*1e3:+.0f} ms, "
                  f"spread {np.ptp(drift)*1e3:.0f} ms")
            trk_params.update(
                anchors_i=[int(x) for x in anchors_i],
                anchors_t=[float(x) for x in anchors_t],
                drift_at_anchors_ms=[float(x) for x in d_anch * 1e3],
                anchor_scatter_ms=float(np.std(d_anch - d_sm) * 1e3),
                drift_spread_ms=float(np.ptp(drift) * 1e3))
    D_tr = t0 * vw / 2.
    print(f"    tracked seafloor: t0 {t0.min()*1e3:.0f}-{t0.max()*1e3:.0f} ms "
          f"-> D {D_tr.min():.0f}-{D_tr.max():.0f} m (scatter {np.std(t0)*1e3:.1f} ms)")
    if args.save_track:
        np.savez(args.save_track, t0=t0, vw=vw, D_init=args.D_init, dt=dt,
                 offs=offs, S=S, segy=os.path.basename(args.segy),
                 shot0=(-1 if args.shot0 is None else args.shot0),
                 nshot=args.nshot, t0_shift=args.t0_shift,
                 params=json.dumps(trk_params))
        print(f"    tracking saved to {args.save_track} "
              f"(reuse it with --load_track to keep the seafloor identical "
              f"across runs)")

    # --- 2b. amplitude trend between primary and first multiple ---
    # Between the primary and the first multiple the near trace holds signal
    # and nothing else, and raw data decay there.  The window keeps one wavelet
    # length from both events.  On the side of the multiple this margin
    # matters most: the multiple is predicted at 2 t0, so it moves twice as far
    # as a tracking error, and a narrower margin lets it into the window, where
    # it reads as a gain.  The exponent is fitted per shot on 20 ms RMS bins and
    # the median over the shots is reported.  It qualifies the amplitude ratios
    # of [5]; the waveform is normalised and does not depend on it.
    x_near = float(offs[0])
    t_beg = np.sqrt(t0 ** 2 + (x_near / vw) ** 2) + args.wl_ms * 1e-3
    t_end = np.sqrt((2 * t0) ** 2 + (x_near / vw) ** 2) - args.wl_ms * 1e-3
    nbin = max(1, int(round(0.020 / dt)))
    slopes = []
    for s_ in range(S):
        i0_ = int(np.ceil(t_beg[s_] / dt)); i1_ = int(np.floor(t_end[s_] / dt))
        nb_ = (min(i1_, T) - i0_) // nbin
        if nb_ < 4:
            continue
        seg_ = np.asarray(g[s_, i0_:i0_ + nb_ * nbin, 0], float)
        rms_ = np.sqrt(np.mean(seg_.reshape(nb_, nbin) ** 2, axis=1))
        if np.any(rms_ <= 0):
            continue
        tc_ = (i0_ + (np.arange(nb_) + 0.5) * nbin) * dt
        slopes.append(float(np.polyfit(np.log(tc_), np.log(rms_), 1)[0]))
    if len(slopes) >= 10:
        gain_exp = float(np.median(slopes))
        raw_like = gain_exp < -0.4
        print("[2b] amplitude trend between primary and first multiple: "
              "RMS ~ t^%+.2f (%d shots)" % (gain_exp, len(slopes)))
        if raw_like:
            print("     decaying, as raw data do")
        else:
            print("     flat or rising: a gain may have been applied.  Amplitude "
                  "ratios between\n     events, and so R_wb in [5], should be "
                  "read with caution; the waveform\n     is unaffected, since it "
                  "is normalised.")
    else:
        print("[2b] amplitude trend between primary and first multiple: not "
              "measured (less than\n     80 ms between them, one wavelet length "
              "away from each, on %d shots of %d)" % (S - len(slopes), S))

    # --- 3. applicability ---
    Lw = args.wl_ms * 1e-3
    D_med = float(np.median(D_tr))
    x_max = max((4 * D_med ** 2 / (vw * Lw) - vw * Lw) / 2., 0.)
    D_min = 0.5 * vw * Lw * np.sqrt(1 + 2 * offs[0] / (vw * Lw))
    print(f"[3] applicability: D {D_med:.0f} m vs D_min {D_min:.0f} m -> "
          f"{'OK' if D_med >= D_min else 'OUT OF SPEC'}")
    print(f"    maximum usable offset {x_max:.0f} m -> "
          f"{max(1, 1 + int((x_max - offs[0]) / max(offs[1]-offs[0], 1))) if C > 1 else 1} channels")

    # --- 4. primary extraction ---
    # Tracking runs over the whole line and only the extraction is restricted
    # to the selected shots: the seafloor is followed continuously, while the
    # wavelet is stacked only where the windows agree.  On a real line quality
    # varies, and averaging over the poor portions costs more than the sqrt(N)
    # it gains.
    s_first = 0 if args.sp_first is None else max(0, args.sp_first)
    s_last = S if args.sp_last is None else min(S, args.sp_last)

    if args.auto_select:
        w_scan, q_scan = extract_wavelet(g, dt, t0, offs, w_len_ms=args.wl_ms, weights=args.weights, ch_start=args.ch_start,
                             align=args.align, align_group=args.n_group,
                                         n_ch=args.n_ch, order=0, band=BAND, vw=vw)
        if w_scan is not None:
            # per-shot coherence, from the per-window values
            coh_shot = np.full(S, np.nan)
            acc = {}
            for (ss, cc), cv in zip(q_scan["windows_used"], q_scan["coh_per_window"]):
                acc.setdefault(ss, []).append(cv)
            for ss, vals in acc.items():
                coh_shot[ss] = np.mean(vals)
            n_sel = int(np.clip(args.auto_select, 5, S))
            valid = np.nan_to_num(coh_shot, nan=-1.0)
            csum = np.concatenate([[0.], np.cumsum(valid)])
            means = (csum[n_sel:] - csum[:-n_sel]) / n_sel
            i_best = int(np.argmax(means))
            s_first, s_last = i_best, i_best + n_sel
            print(f"\n[3b] auto-selected shots {s_first}-{s_last-1}: mean "
                  f"coherence {means[i_best]:.3f} against {np.nanmean(coh_shot):.3f} "
                  f"over the whole line")
        else:
            print("\n[3b] scan for auto-selection failed, using the whole line")

    # a slice, not fancy indexing: with sp_step == 1 it is a view and costs
    # nothing
    sp_step = max(1, args.sp_step)
    sl = slice(s_first, s_last, sp_step)
    t0_ex = np.ascontiguousarray(t0[sl])
    # from a memory-mapped cube the slice is materialised once, instead of
    # reading from disk at every access
    g_ex = (np.ascontiguousarray(g[sl])
            if (sp_step > 1 or args.load_cube) else g[sl])
    n_use = len(t0_ex)
    if n_use < 5:
        print(f"\nToo few shots selected ({n_use})"); return
    if (s_first, s_last) != (0, S) or sp_step > 1:
        span = (s_last - s_first - 1) * 1  # shots spanned, not shots used
        print(f"    extraction on {n_use} shots: {s_first}-{s_last-1} "
              f"step {sp_step} ({n_use}/{S} of the line, spanning "
              f"{span+1} shot positions)")
    # the multiple validator is local: it uses the depth of the shots
    # stacked, not the median of the whole line
    D_med = float(np.median(D_tr[sl]))

    wp, qp = extract_wavelet(g_ex, dt, t0_ex, offs, w_len_ms=args.wl_ms, weights=args.weights, ch_start=args.ch_start,
                             align=args.align, align_group=args.n_group,
                             n_ch=args.n_ch, order=0, band=BAND, vw=vw,
                             shot_start=0, n_shots=n_use)
    if wp is None:
        print(f"\nEXTRACTION FAILED: {qp.get('error')}"); return
    wp = nrm(np.asarray(wp, float))
    print(f"\n[4] PRIMARY: {qp['n_windows']} windows | coherence {qp['stack_coherence']:.3f} "
          f"| origin-lead {qp['origin_minus_lead_ms']:+.0f} ms "
          f"| align. residual {qp['align_residual_ms_mean']:.1f} ms")

    bistab = None
    if args.bistab_check:
        bistab = bistability(qp, dt)
        print(f"    partial stacks of 50 shots: {bistab['n_out']}/{bistab['n_blocks']} beyond 2 samples, "
              f"largest gap between origins {bistab['gap_ms']:.0f} ms (T_dom {bistab['tdom_ms']:.0f} ms)"
              + ("   <-- BISTABLE: inspect before use" if bistab["bistable"] else ""))

    # --- 5. multiple and validator ---
    t0c = np.sqrt(np.maximum(
        (np.sqrt(t0_ex ** 2 + (offs[0] / vw) ** 2)
         + qp["origin_minus_lead_ms"] * 1e-3) ** 2 - (offs[0] / vw) ** 2, 1e-12))
    wm, qm = extract_wavelet(g_ex, dt, t0c, offs, w_len_ms=args.wl_ms, weights=args.weights, ch_start=args.ch_start,
                             align=args.align, align_group=args.n_group,
                             n_ch=args.n_ch,
                             order=1, band=BAND, vw=vw, template=wp,
                             template_search_ms=80., template_expected_sign=-1,
                             shot_start=0, n_shots=n_use)
    multiple_reason = ""
    if wm is None:
        print(f"[5] MULTIPLE failed: {qm.get('error')}  "
              f"(record too short? multiple at "
              f"{2*np.median(t0c)*1e3:.0f} ms of {T*dt*1e3:.0f} ms)")
        multiple_reason = str(qm.get("error") or "not extracted")
        print(f"    MULTIPLE NOT AVAILABLE ({multiple_reason}): primary wavelet saved; "
              f"multiple-based diagnostics (depth from the lag, R_wb, second multiple, "
              f"matched-filter validator) not available")
        wm = None
    else:
        wmn = nrm(-np.asarray(wm, float))
        # Water depth from the lag between the two origins measured on the
        # stacks, not from the tracking.  Per shot the primary origin sits at
        # arrival(t0c, 0) - Lp and the multiple's at arrival(t0c, 1) + ol_m - Lm,
        # Lp and Lm the mean alignment shifts of the two stacks; both origins
        # are "onset - guard", so the guard cancels.  The vertical time tau
        # solves sqrt((2 tau)^2 + a^2) - sqrt(tau^2 + a^2) = dT, a = x/vw, in
        # closed form.  A constant shift of the tracking moves neither t0c nor
        # the multiple window, so it leaves D unchanged.  Up to e90f74a the depth
        # was D_med plus the difference of the two origin-leads, and moved with
        # the tracking by about shift/(2 t0).
        a_x = offs[0] / vw
        dT = (arrival_time(t0c, offs[0], vw, 1) - arrival_time(t0c, offs[0], vw, 0)
              + (qm["origin_minus_lead_ms"] - qm["lag_mean_ms"] + qp["lag_mean_ms"]) * 1e-3)
        dT = np.maximum(dT, 1e-6)
        tau = np.sqrt(dT * (5 * dT + 2 * np.sqrt(4 * dT ** 2 + 9 * a_x ** 2)) / 9.)
        D_meas = float(np.median(vw * tau / 2.))
        r1 = float(np.hypot(2 * D_med, offs[0])); r2 = float(np.hypot(4 * D_med, offs[0]))
        # amplitudes by projection onto the primary wavelet, not by the RMS of the
        # window: at multiple times deep sub-bottom reflections are laterally
        # coherent, survive the stack and inflate the RMS even when the shape is
        # right
        tpl = nrm(wp - wp.mean())
        pp = float((qp["w_raw"] - qp["w_raw"].mean()) @ tpl)
        pm = float((qm["w_raw"] - qm["w_raw"].mean()) @ tpl)
        R = abs(pm / (pp + 1e-30)) * (r2 / r1) ** args.alpha
        print(f"[5] MULTIPLE: {qm['n_windows']} windows | coherence {qm['stack_coherence']:.3f}")
        print(f"    >>> VALIDATOR {qm['template_score']:.3f}  sign {qm['template_sign']:+d}"
              + ("" if qm['template_sign'] < 0 else "   <-- EXPECTED -1"))
        print(f"    >>> D from the primary-multiple lag: {D_meas:.0f} m "
              f"(tracked {D_med:.0f} m, departure {100*(D_meas-D_med)/D_med:+.1f}%)")
        print(f"        multiple origin-lead {qm['origin_minus_lead_ms']:+.0f} ms, mean alignment "
              f"shift primary {qp['lag_mean_ms']:+.1f} / multiple {qm['lag_mean_ms']:+.1f} ms"
              + ("   <-- AT THE SEARCH BOUNDARY: a bound, not a depth"
                 if abs(qm['origin_minus_lead_ms']) >= 80. - dt * 1e3 else ""))
        print(f"    >>> R_wb {R:.3f} (spreading {'3D' if args.alpha==1 else '2D'} "
              f"x{(r2/r1)**args.alpha:.2f}) -- qualitative indication only")

        # --- second multiple: internal consistency check on R_wb ---
        # The second-to-first multiple ratio should give the same R_wb as the
        # first-multiple-to-primary ratio.  Agreement supports the amplitude
        # assumptions; disagreement says one of them fails without saying which.
        # The second multiple has sign (-1)^2 = +1, the same as the primary.
        t_m2 = float(arrival_time(float(np.median(t0c)), offs[0], vw, 2))
        need = t_m2 + (args.wl_ms + 140.) * 1e-3      # lead + w_len + tail
        if need > T * dt:
            print(f"    2nd multiple at {t_m2*1e3:.0f} ms would need "
                  f"{need*1e3:.0f} ms of record, only {T*dt*1e3:.0f} ms "
                  f"available:\n    R_wb consistency check SKIPPED.")
        else:
            w2, q2 = extract_wavelet(g_ex, dt, t0c, offs, w_len_ms=args.wl_ms, weights=args.weights, ch_start=args.ch_start,
                             align=args.align, align_group=args.n_group,
                                     n_ch=args.n_ch, order=2, band=BAND, vw=vw,
                                     template=wp, template_search_ms=80.,
                                     template_expected_sign=+1,
                                     shot_start=0, n_shots=n_use)
            if w2 is None:
                print(f"    2nd multiple: extraction failed ({q2.get('error')})")
            else:
                p2 = float((q2["w_raw"] - q2["w_raw"].mean()) @ tpl)
                r3 = float(np.hypot(6 * D_med, offs[0]))
                R2 = abs(p2 / (pm + 1e-30)) * (r3 / r2) ** args.alpha
                sg2 = q2["template_sign"]
                print(f"    2nd multiple: {q2['n_windows']} windows | coherence "
                      f"{q2['stack_coherence']:.3f} | score "
                      f"{q2['template_score']:.3f} sign {sg2:+d}"
                      + ("" if sg2 > 0 else "   <-- EXPECTED +1"))
                print(f"    >>> R_wb from 2nd/1st multiple: {R2:.3f}  "
                      f"against {R:.3f} from 1st/primary")
                d_rel = 200. * abs(R2 - R) / (R2 + R + 1e-12)
                print(f"        relative discrepancy {d_rel:.0f}%"
                      + ("  -> consistent: the amplitude ratio is usable"
                         if d_rel < 25 else "  -> INCONSISTENT"))
                if d_rel >= 25:
                    if np.isfinite(gain_exp) and gain_exp > -0.4:
                        print("        Cause is most likely the gain already "
                              "applied to the data (see [2b]).\n        No single "
                              "spreading exponent can reconcile the two "
                              "estimates when\n        the gain is not a simple "
                              "function of time, as with AGC.")
                    else:
                        # not a gain, by [2b]; nothing here says which of the amplitude
                        # assumptions fails, so no cause is named
                        print("        The amplitude trend does not point to a gain "
                              "(see [2b]), and\n        this check does not say which "
                              "assumption fails; treat R_wb as an\n        order of "
                              "magnitude only.")

    # --- 6. comparison against a previously published estimate ---
    wr = None
    if args.ref and os.path.exists(args.ref):
        wr = np.asarray(np.load(args.ref), float)
        if args.ref_dt and abs(args.ref_dt - dt) > 1e-9:
            from scipy.signal import resample_poly
            from fractions import Fraction
            fr = Fraction(args.ref_dt / dt).limit_denominator(100)
            wr = resample_poly(wr, fr.numerator, fr.denominator)
            print(f"\n[6] reference resampled from {args.ref_dt*1e3:.0f} a {dt*1e3:.0f} ms")
        b, a = butter(4, [BAND[0]/nyq, min(BAND[1]/nyq, .95)], 'band')
        wr = filtfilt(b, a, wr)
        wr = nrm((wr[:wl] if len(wr) >= wl else np.pad(wr, (0, wl-len(wr)))))
        v, lag, ov = ncc_shift(wp, wr)
        print(f"[6] NCC(extracted, {os.path.basename(args.ref)}) = {v:.3f} "
              f"(lag {lag:+d}, overlap {ov}/{wl})")
        print("    NB: the reference is an ESTIMATE, not ground truth: a low value")
        print("        does not say which of the two is wrong.")

    # --- figure ---
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # every panel is lettered and every axis says what it shows
    for a_, lab in zip(ax.ravel(), "abcdef"):
        a_.set_title("(%s)" % lab, loc="left", fontweight="bold", fontsize=9)
    sec = g[:, :, 0].T
    cl = np.percentile(np.abs(sec), 97) + 1e-12
    tt = np.arange(T) * dt * 1e3
    ax[0, 0].imshow(sec, aspect='auto', cmap='gray_r', vmin=-cl, vmax=cl,
                    extent=[0, S, tt[-1], 0])
    ax[0, 0].plot(np.arange(S) + .5, t0 * 1e3, 'r.-', ms=2, lw=.8, label='seafloor')
    ax[0, 0].plot(np.arange(S) + .5, 2 * t0 * 1e3, 'y:', lw=1, label='multiple')
    ax[0, 0].set_ylim(min(4 * np.median(t0) * 1e3, tt[-1]), 0)
    # an onset tracking (h3_track) carries its anchors and its fallback stretches:
    # onset anchors where the onset was reliable, stage-1 anchors where it was not
    status = []
    if TRK is not None and "anchors_kind" in TRK:
        ai_, at_ = np.asarray(TRK["anchors_i"], int), np.asarray(TRK["anchors_t"], float) * 1e3
        kd = np.asarray(TRK["anchors_kind"]).astype(str)
        on = kd == "onset"
        if on.any():
            ax[0, 0].plot(ai_[on] + .5, at_[on], 'o', mfc='none', mec='c', ms=4, mew=.9,
                          label='onset anchors (reliable onset)')
        if (~on).any():
            ax[0, 0].plot(ai_[~on] + .5, at_[~on], 'x', color='m', ms=5, mew=1.2,
                          label='stage-1 anchors (onset unreliable)')
        bad = np.asarray(TRK["unreliable"], bool) if "unreliable" in TRK else np.zeros(S, bool)
        edges = np.flatnonzero(np.diff(np.r_[0, bad[:S].astype(int), 0]))
        for k, (b0, b1) in enumerate(zip(edges[::2], edges[1::2])):
            ax[0, 0].axvspan(b0, b1, color='m', alpha=.12, lw=0,
                             label='stage-1 fallback stretch' if k == 0 else None)
        dom = trk_params.get("domain", {})
        if dom.get("flag"):
            status.append('operating domain: %s' % dom["flag"])
    if bistab is not None:
        status.append('BISTABLE: %d/%d partial stacks off by > 2 samples' % (bistab['n_out'], bistab['n_blocks'])
                      if bistab['bistable'] else 'partial stacks: %d/%d off by > 2 samples'
                      % (bistab['n_out'], bistab['n_blocks']))
    # the status goes above the title, one item per line, clear of the panel letter
    title_a = ''.join(x + '\n' for x in status) + 'near trace + tracked seafloor'
    ax[0, 0].legend(fontsize=7); ax[0, 0].set_xlabel('shot'); ax[0, 0].set_ylabel('ms')
    ax[0, 0].set_title(title_a, fontsize=9, loc='center')

    # the tracked depth looks jagged because the vertical scale is expanded:
    # a sub-sample lag at dt = 2 ms is quantised at 1.5 m.  A running mean over
    # the picks separates the trend from the quantisation without hiding either.
    ax[0, 1].plot(D_tr, '.', color='0.7', ms=2.5, label='per shot')
    if len(D_tr) >= 9:
        k = max(3, (len(D_tr) // 40) * 2 + 1)
        ker = np.ones(k) / k
        D_sm = np.convolve(np.pad(D_tr, (k // 2, k // 2), mode='edge'), ker, 'valid')
        ax[0, 1].plot(D_sm, 'r-', lw=1.4,
                      label=f'running mean over {k} shots')
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_xlabel('shot'); ax[0, 1].set_ylabel('D (m)')
    ax[0, 1].set_title(f'tracked depth (Vw {vw:.0f} m/s)', fontsize=9,
                       loc='center')

    al = qp.get('aligned_segments')
    if al is not None:
        # the windows are aligned after upsampling, so the axis is drawn in ms from
        # the window length the extractor reports
        dt_al = dt * qp['win_samp'] / al.shape[1]
        ta = np.arange(al.shape[1]) * dt_al * 1e3
        sub = al[:min(len(al), 40)]
        sc = np.percentile(np.abs(sub), 98) + 1e-12
        for i, tr in enumerate(sub):
            y = i + tr / sc * .45 * args.wiggle_gain
            ax[0, 2].plot(ta, y, 'k', lw=.5)
            ax[0, 2].fill_between(ta, i, y, where=(y > i),
                                  color='k', alpha=.55, lw=0)
        ax[0, 2].invert_yaxis()
    ax[0, 2].set_xlabel('time within the window (ms)')
    ax[0, 2].set_ylabel('window, in the order extracted')
    ax[0, 2].set_title('aligned windows (primary), first %d'
                       % min(len(al) if al is not None else 0, 40),
                       fontsize=9, loc='center')

    pc = qp.get('coh_per_window')
    if pc is not None:
        ax[1, 0].plot(pc, 'b.-', ms=3, lw=.6)
        # a reading line, not a threshold: no window is rejected on coherence.
        # Every window enters the stack, weighted by this value under --weights
        # coherence.
        ax[1, 0].axhline(.7, color='g', ls=':', lw=0.9,
                         label='0.7, for reading only: no window is rejected')
        ax[1, 0].set_ylim(-.1, 1.05)
        ax[1, 0].legend(fontsize=6.5, loc='lower left')
    ax[1, 0].set_xlabel('window'); ax[1, 0].set_ylabel('coherence against the pilot')
    ax[1, 0].set_title('coherence per window (weight in the stack: %s)'
                       % args.weights, fontsize=9, loc='center')

    # drawn with a band-limited interpolation for display: at dt = 4 ms a
    # 200 ms wavelet has only 50 samples, which are kept visible
    t = np.arange(wl) * dt * 1e3
    ti = np.linspace(t[0], t[-1], 8 * wl)
    def _sm(y):
        from scipy.signal import resample
        return resample(np.asarray(y, float), 8 * wl)
    ax[1, 1].plot(ti, _sm(wp), 'b', lw=1.8, label='extracted (primary)')
    ax[1, 1].plot(t, wp, 'b.', ms=3, alpha=.45)
    if wm is not None:
        ax[1, 1].plot(ti, _sm(wmn), 'g', lw=1.2,
                      label="multiple (polarity reversed)")
    ax[1, 1].legend(fontsize=7); ax[1, 1].set_xlabel('time from the window start (ms)')
    ax[1, 1].set_ylabel('amplitude (normalised)')
    ax[1, 1].set_title('wavelets', fontsize=9, loc='center')

    # Amplitude spectra, with the independent statistical estimate, which
    # shares none of the assumptions of this workflow.  A reference made by the
    # same stacking approach is not shown: agreement with it would measure
    # reproducibility, not accuracy.
    nf = 4096; fr = np.fft.rfftfreq(nf, dt)
    band_m = (fr >= BAND[0]) & (fr <= min(BAND[1], nyq * 0.95))
    def _nz(y):
        A = np.abs(np.fft.rfft(y, nf)); return A / (A[band_m].max() + 1e-12)
    ax[1, 2].set_xlabel('frequency (Hz)')
    ax[1, 2].set_ylabel('amplitude (normalised to the in-band peak)')
    ax[1, 2].plot(fr[band_m], _nz(wp)[band_m], 'b', lw=1.6, label='primary')
    if wm is not None:
        ax[1, 2].plot(fr[band_m], _nz(wmn)[band_m], 'g', lw=1.1, label='multiple')
    f_st, A_st, n_st = statistical_spectrum(g, dt, t0, offs)
    if A_st is not None:
        A_i = np.interp(fr, f_st, A_st); A_i = A_i / (A_i[band_m].max() + 1e-12)
        rr = float(np.corrcoef(_nz(wp)[band_m], A_i[band_m])[0, 1])
        ax[1, 2].plot(fr[band_m], A_i[band_m], 'k--', lw=1.6,
                      label='independent statistical\n(r=%+.2f over %.0f-%.0f Hz)'
                            % (rr, BAND[0], min(BAND[1], nyq * 0.95)))
        print(f"\n[7] Independent check: amplitude spectrum against the "
              f"statistical estimate\n    ({n_st} windows below the seafloor, "
              f"white-reflectivity assumption): r = {rr:+.3f}")
        print(f"    This shares no assumptions with the workflow: no seafloor, "
              f"no tracking,\n    no alignment. It constrains the MODULUS only "
              f"-- phase is not recoverable\n    without assuming it, and an "
              f"airgun signature is mixed-phase.")
        # the statistical estimate does not stack, so it keeps the full
        # incoherent noise while the extracted wavelet attenuates it by sqrt(N)
        hi_m = (fr >= 20.) & (fr <= min(BAND[1], nyq * 0.95))
        r_hi = float(np.corrcoef(_nz(wp)[hi_m], A_i[hi_m])[0, 1])
        print(f"    Restricted to 20-{min(BAND[1], nyq*0.95):.0f} Hz, where the "
              f"pre-arrival noise no longer\n    dominates: r = {r_hi:+.3f}")
    ax[1, 2].set_xlim(0, min(120, nyq)); ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_xlabel('Hz'); ax[1, 2].set_title('amplitude spectra', fontsize=9)
    plt.tight_layout(); plt.savefig(args.out + ".png", dpi=135)
    print(f"\nfigure: {args.out}.png")
    np.save(args.out + "_wavelet.npy", wp.astype(np.float32))
    # the multiple carries the same source and ghosts, with its sub-bottom
    # contamination at a different delay.  Saved as extracted, without the
    # polarity flip, so that the measured sign stays readable in the file.
    if wm is not None:             # not extracted: reported in the log and in _sel.npz
        np.save(args.out + "_wavelet_multiple.npy", wm.astype(np.float32))
    # provenance: which shots produced the wavelet, and the checks of the stack
    np.savez(args.out + "_sel.npz",
             sp_first=s_first, sp_last=s_last, sp_step=sp_step, n_use=n_use,
             coherence=qp["stack_coherence"],
             origin_minus_lead_ms=qp["origin_minus_lead_ms"],
             align_residual_ms=qp["align_residual_ms_mean"],
             # the scatter of the applied shifts (see wavelet_extractor_v2: it is not
             # the residual misalignment, nor a stacking low-pass)
             align_scatter_ms=qp.get("align_scatter_ms", float("nan")),
             # per-window coherences, which are the stack weights themselves
             coh_per_window=np.asarray(qp.get("coh_per_window", []),
                                       dtype=np.float32),
             weights=str(qp.get("weights", "coherence")),
             D_med=D_med, vw=vw, dt=dt,
             # water depth from the primary-multiple lag (step [5]), NaN when the
             # multiple was not extracted
             D_mult=(D_meas if wm is not None else float("nan")),
             align=args.align,
             bistable=(bistab["bistable"] if bistab else False),
             bistab_offsets_samp=(bistab["offsets_samp"] if bistab else np.zeros(0)),
             cached_tracking=(args.load_track or ""),
             recipe=args.recipe,
             multiple_available=wm is not None, multiple_reason=multiple_reason,
             # onset tracking: applicability and fallback (absent keys: not an h3_track file)
             **({f"h3_{k}": trk_params[k] for k in ("applicable", "fallback", "fallback_reason",
                                                    "unreliable_fraction")}
                if args.load_track and trk_params.get("script") == "h3_track" and "applicable" in trk_params
                else {}))
    print(f"wavelet: {args.out}_wavelet.npy")


if __name__ == "__main__":
    main()
