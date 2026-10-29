# =============================================================================
#  PAPER FIGURE: extracted wavelet against ground truth, on synthetic models
#
#  WHY THIS FIGURE.  The tables report correlations, but a correlation of 0.79
#  means nothing to a reader until they can see what it looks like.  This figure
#  shows the best, the median and the worst case of the matrix, so every number
#  elsewhere in the paper becomes interpretable: whether 0.79 is a wavelet fit
#  for deconvolution or one that is barely recognisable.
#
#  It re-runs the extraction on the three selected models rather than storing
#  waveforms during the matrix run, so it always reflects the current code.
#
#  USAGE:
#    python validation/synthetic/make_fig_synthetic.py --config validation/synthetic/matrix_v2.yaml --out fig_synthetic
# =============================================================================
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    return x / (np.linalg.norm(x) + 1e-9)


def ncc_shift(a, b, max_lag=40, min_overlap=60):
    a = np.asarray(a, float); b = np.asarray(b, float); na, nb = len(a), len(b)
    def _c(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / d) if d > 1e-12 else 0.0
    best, blag = -2., 0
    for L in range(-max_lag, max_lag + 1):
        a0 = max(0, L); a1 = min(na, nb + L)
        if a1 - a0 < min_overlap:
            continue
        v = _c(a[a0:a1], b[a0 - L:a0 - L + (a1 - a0)])
        if v > best:
            best, blag = v, L
    return best, blag


def regenerate(run, cfg):
    """Regenerate one model from its manifest entry (seeds are deterministic)."""
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
    src_fn, w_bare = make_source_fn(10_000 + run["seed"], fmax=p["fmax_hz"])
    dt_out = m["dt_ms"] / 1000.
    tn = float(np.clip(2.6 * (2 * p["D_m"] / m["v_water"]) * 1000. + 800.,
                       1500., 4500.))
    NT = int(round(tn / 1000. / dt_out))
    g = np.zeros((n_shots, NT, len(offs)), np.float32)
    zs = p.get("z_src_m", None); zr = p.get("z_rec_m", None)
    for i in range(n_shots):
        try:
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offs,
                             z_src=zs, z_rec=zr)
        except TypeError:
            d, dtd = G.shoot(model, xs_all[i], src_fn, tn=tn, offsets=offs)
        fr = Fraction(float(dtd) / 1000. / dt_out).limit_denominator(2000)
        d = resample_poly(d, fr.numerator, fr.denominator, axis=0).astype(np.float32)
        d = d[:NT] if d.shape[0] > NT else np.pad(d, ((0, NT - d.shape[0]), (0, 0)))
        g[i] = d
    idx = np.clip(((xs_all[:n_shots] - offs[0] / 2.) / G.DH).astype(int),
                  0, vp.shape[0] - 1)
    return g, dt_out, offs, w_bare, float(np.median(kwb[idx] * G.DH))


def half_integrate(x, dt):
    """Apply the 2D far-field factor omega^{-1/2} e^{i pi/4}.

    In three dimensions the far field reproduces the source time function; in
    two dimensions it does not.  The 2D Green function is
    G ~ H_0^{(2)}(omega r / c), asymptotically proportional to
    omega^{-1/2} e^{i pi/4}, so the recorded wavefield is a HALF INTEGRATION of
    the injected signature.  A reference built as (bare wavelet) x (ghost)
    therefore does not describe what a 2D simulation records, and comparing the
    two penalises high frequencies by exactly sqrt(f).
    """
    n = len(x)
    X = np.fft.rfft(x, 2 * n)
    f = np.fft.rfftfreq(2 * n, dt)
    om = 2 * np.pi * np.maximum(f, 1e-9)
    H = np.exp(1j * np.pi / 4) / np.sqrt(om)
    H[0] = 0.0
    return np.fft.irfft(X * H, 2 * n)[:n]


def reference(w_bare, dt, wl, zs=Z_DEV, zr=Z_DEV, two_d=False):
    w = np.asarray(w_bare, np.float64)
    if abs(dt - AG_DT) > 1e-9:
        w = resample_poly(w, int(round(len(w) * AG_DT / dt)), len(w))
    wg = apply_ghost(w.astype(np.float32), dt, zs, zr).astype(np.float64)
    if two_d:
        wg = half_integrate(wg, dt)
    nyq = 0.5 / dt
    b, a = butter(4, [BAND[0] / nyq, min(BAND[1] / nyq, .95)], 'band')
    wg = filtfilt(b, a, wg)
    wg = wg[:wl] if len(wg) >= wl else np.pad(wg, (0, wl - len(wg)))
    return nrm(wg - wg.mean())


def load_stored(root):
    """Every wavelet archive under root, merged into one dictionary.

    Keys are "run_id|snr|bias" for the wavelet and the same with "|ref" appended
    for the reference it was scored against.  Both are needed: with the
    reference stored, the correlation is reproducible without the model.
    """
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(root, '*_wavelets.npz'))):
        with np.load(f) as z:
            for k in z.files:
                out[k] = z[k]
    if not out:
        raise SystemExit('no *_wavelets.npz under %s -- run the matrix with '
                         'patch_matrix_save_wavelet.py applied' % root)
    print('%d stored arrays read from %s' % (len(out), root))
    return out


def _ncc(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / d) if d > 0 else float('nan')


def ncc_pair(a, b, maxlag=0):
    """Correlation of two equal-length wavelets, over maxlag samples of slide.

    maxlag = 0 is right for the v2 and v3 archives, where the stored wavelet is
    the segment the matrix had already slid into place.  It is wrong for v5,
    where the stored wavelet starts at the estimated origin and the score is
    measured at the best shift near the origin bias: with no slide the same
    record reads 0.115 here and 0.482 in the file, and the consistency check
    then refuses to draw a figure that is not actually inconsistent.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if maxlag <= 0:
        return _ncc(a, b)
    n = b.size
    bp = np.concatenate([np.zeros(maxlag), b, np.zeros(maxlag)])
    return max(_ncc(a, bp[maxlag + L:maxlag + L + n])
               for L in range(-maxlag, maxlag + 1))


def ncc_pair_lag(a, b, maxlag):
    """As ncc_pair, and the lag it was reached at.

    The lag is the one applied to the REFERENCE: the best match is between a
    and b advanced by that many samples, so a NEGATIVE lag means the extracted
    wavelet arrives late, which is the usual sign of the origin bias.  The
    picture has to be drawn at the same lag as the number, or the figure shows
    two wavelets out of step and quotes an agreement that was measured in step.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = b.size
    bp = np.concatenate([np.zeros(maxlag), b, np.zeros(maxlag)])
    best, bl = -2.0, 0
    for L in range(-maxlag, maxlag + 1):
        v = _ncc(a, bp[maxlag + L:maxlag + L + n])
        if v > best:
            best, bl = v, L
    return best, bl


def advance(x, k):
    """x brought forward by k samples, filled with zeros, never wrapped."""
    x = np.asarray(x, float)
    if k == 0:
        return x.copy()
    y = np.zeros_like(x)
    if k > 0:
        y[:x.size - k] = x[k:]
    else:
        y[-k:] = x[:x.size + k]
    return y


def padded_visible(rec, dt_ms=2.0, kept=100, w_len_ms=200.0, tail_ms=60.0,
                   archive=None):
    """Invented samples inside the segment the correlation was computed over.

    Three sources, in order of trust.  (1) The archived segment itself: the
    extractor pads with EXACT zeros after stacking and nothing filters the kept
    segment afterwards, so its trailing zeros are the invented samples.  This
    was checked against the runner's own count on wavelet_matrix_v2_TAIL250,
    2436 of 2436 in agreement (padding_from_arrays.py).  (2) n_invented_kept,
    when the runner recorded it.  (3) A reconstruction from the origin offset,
    which hard-codes the tail and is right only for the run it was written
    for; wavelet_matrix_v2 records neither (2) nor its tail, so without the
    archive the figure would be filtering on a guess.

    extract_wavelet returns w_len + 2*pad samples and zero-fills the tail beyond

        P = origin_minus_lead / dt - tail

    The runner then keeps wp = wlong[off_p : off_p + kept], so what the score
    actually saw is

        max(0, (off_p + kept) - (Lw - P)),  off_p = (match_pos - oml) / dt

    Both quantities are needed and they differ: on wavelet_matrix_v2_TAIL250,
    which records both, 12.6 per cent of the long wavelets are padded and 4.4
    per cent of the kept segments are.  On wavelet_matrix_v2 only the second
    can be recovered, from the archive: 248 of 2480, 10.0 per cent.  (The 29
    and 8.8 per cent once quoted here came from a run that no longer exists.)
    """
    if archive is not None:
        k = "%s|%s|%s" % (rec["run_id"], rec.get("snr_db"), rec.get("t0_bias_ms"))
        if k in archive:
            w = np.asarray(archive[k])
            nz = np.nonzero(w)[0]
            return int(w.size - nz[-1] - 1) if nz.size else int(w.size)
    # Measured by the runner on the array itself, when it is there.  The
    # reconstruction below encodes the window length, the pad and the tail as
    # constants, so it goes stale silently if any of them changes in the
    # extractor; the recorded value cannot.
    if rec.get("n_invented_kept") is not None:
        return int(rec["n_invented_kept"])

    oml = rec.get("origin_minus_lead_ms")
    mp = rec.get("match_pos_ms")
    if oml is None or mp is None:
        return 0
    pad = float(rec.get("pad_ms", 120.0))
    Lw = int(round((w_len_ms + 2 * pad) / dt_ms))
    P = max(0, int(round(float(oml) / dt_ms)) - int(round(tail_ms / dt_ms)))
    off_p = int(round((float(mp) - float(oml)) / dt_ms))
    return max(0, min(kept, (off_p + kept) - (Lw - P)))


def plot_one(ax, row, lab, wv, w_ref, dt, v, rec):
    """Draw one row: waveform on the left, amplitude spectrum on the right.

    Everything it needs is either stored by the matrix or carried in the record,
    so the archive path and the rebuilding path draw identically.  The curves
    are NOT rolled to the best lag: np.roll wraps the tail onto the head, and at
    a large lag the picture would show something other than what the correlation
    compared.
    """
    wl = len(w_ref)
    t = np.arange(wl) * dt * 1e3
    ax[row, 0].plot(t, w_ref, 'k', lw=2.0, label='true')
    ax[row, 0].plot(t, wv, 'r', lw=1.3, label='extracted')
    ax[row, 0].set_ylabel(f"{lab}\nNCC = {v:.3f}", fontsize=8)
    ax[row, 0].legend(fontsize=6.6, loc='upper right')
    ax[row, 0].set_title(
        f"$D$ = {rec['D_m']:.0f} m, first reflector {rec['first_layer_m']:.0f} m, "
        f"$f_{{max}}$ = {rec['fmax_hz']:.0f} Hz", fontsize=7.5)

    nf = 1024
    fr = np.fft.rfftfreq(nf, dt)
    bm = (fr >= BAND[0]) & (fr <= BAND[1])
    for y, c, lb in ((w_ref, 'k', 'true'), (wv, 'r', 'extracted')):
        A = np.abs(np.fft.rfft(y, nf))
        A = A / (A[bm].max() + 1e-12)
        ax[row, 1].plot(fr[bm], A[bm], c, lw=1.6 if c == 'k' else 1.2, label=lb)
    ax[row, 1].set_ylabel('normalised amplitude', fontsize=8)
    ax[row, 1].legend(fontsize=6.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--snr", type=float, default=30.)
    ap.add_argument("--bias", type=float, default=0.0,
                    help="which level of the deliberate tracking error to "
                         "draw, where the matrix has a sweep")
    ap.add_argument("--results", default=None,
                    help="directory holding results.jsonl; default is the "
                         "out_root of the config")
    ap.add_argument("--score", default=None,
                    help="field to rank on: ncc_prim_true for the v2 and v3 "
                         "matrices, ncc_shape for v5.  Guessed from the file "
                         "when not given")
    ap.add_argument("--allow_boundary", action="store_true",
                    help="keep records whose origin reached the START of the "
                         "window.  The origin estimator walks back from the "
                         "envelope peak; when an earlier event (the direct "
                         "arrival, in the widened window of the matrix) keeps "
                         "the envelope above threshold, it stops only at the "
                         "window edge and the compared segment is pre-arrival "
                         "noise.  Like padding, this is visible without the "
                         "truth, so it is excluded by the same logic")
    ap.add_argument("--allow_padding", action="store_true",
                    help="keep records whose compared segment contains samples "
                         "the extractor invented.  The padded fraction is then "
                         "printed with each choice")
    ap.add_argument("--any", action="store_true",
                    help="select from every record, including the secondary "
                         "axes.  Those vary the structure or the channel count "
                         "and are not the same experiment as the grid, so the "
                         "three panels would not be comparable")
    ap.add_argument("--wavelets", default=None,
                    help="directory holding results_shard*_wavelets.npz.  With "
                         "this the figure plots what the matrix produced and "
                         "does not rebuild anything; without it the model is "
                         "regenerated, which four separate causes have shown "
                         "does not reproduce the matrix")
    ap.add_argument("--one_d", action="store_true",
                    help="compare against the reference WITHOUT the 2D far-field "
                         "factor.  The matrix uses the 2D reference, so this "
                         "makes the figure disagree with the tables and exists "
                         "only for the diagnostic printed below")
    ap.add_argument("--tol", type=float, default=0.001,
                    help="maximum accepted departure from the NCC stored in the "
                         "matrix for the same record")
    ap.add_argument("--two_d_unused", action="store_true",
                    help="build the reference with the 2D far-field factor "
                         "omega^-1/2 (half integration), which is what a 2D "
                         "simulation actually records")
    ap.add_argument("--include_shallow", action="store_true",
                    help="also consider cells below the applicability threshold")
    ap.add_argument("--d_min", type=float, default=220.,
                    help="minimum water depth for the selection")
    ap.add_argument("--out", default="fig_synthetic")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # WHICH MATRIX.  out_root is where the manifest lives, which for the v5
    # runs is still the v2 directory; the results and the archived wavelets are
    # wherever --out put them.  Without --results this figure would go on
    # showing v2 scores while the text quotes v5 ones.
    root = args.results or cfg["paths"]["out_root"]
    # The manifest stays where out_root says even when the results do not: the
    # v5 runs read it from the v2 directory and wrote their results elsewhere.
    man_root = cfg["paths"]["out_root"]
    runs = {r["run_id"]: r for r in
            (json.loads(l) for l in
             open(os.path.join(man_root, "manifest.jsonl")))}
    recs = [json.loads(l) for l in open(os.path.join(root, "results.jsonl"))]
    SC = args.score or ("ncc_shape" if any("ncc_shape" in r for r in recs[:50])
                        else "ncc_prim_true")
    print("  ranking on %s, read from %s" % (SC, root))
    ok = [r for r in recs if r.get("ok") and r["snr_db"] == args.snr
          and r.get(SC) is not None and r[SC] == r[SC]]
    # The v5 bias matrix holds the same model at nine deliberate tracking
    # errors.  Without this the selection ranks across all of them and picks
    # the worst as a record extracted with the window 100 ms off, which is the
    # bias axis and not the wavelet.
    if any(r.get("t0_bias_ms") is not None for r in ok):
        n0 = len(ok)
        ok = [r for r in ok if (r.get("t0_bias_ms") or 0.0) == args.bias]
        print("  %d records at a deliberate tracking error other than "
              "%+.0f ms excluded" % (n0 - len(ok), args.bias))
    if not args.include_shallow:
        # Select among APPLICABLE cells only.  Below the threshold the primary
        # and the direct arrival are merged, the bounded slide cannot reach the
        # alignment, and the panel shows two waveforms that do not correspond --
        # a true result, but one already covered by the threshold table, and a
        # misleading illustration of what the method does inside its domain.
        n0 = len(ok)
        ok = [r for r in ok if r.get("D_m", 1e9) >= args.d_min]
        print(f"  {n0 - len(ok)} records below D = {args.d_min:.0f} m excluded "
              f"(outside the applicability threshold)")
    if not args.any:
        # The controlled experiment is the grid: one channel throughout, no
        # fault, folds or lens, and the first reflector where the configuration
        # asked for it.  The secondary axes each change one of those, so a
        # figure drawing one of each would show three experiments as though
        # they were the spread of one.
        n1 = len(ok)
        ok = [r for r in ok if str(r["run_id"]).startswith("grid_")]
        print(f"  {n1 - len(ok)} records from the secondary axes excluded "
              f"(structure or channel count differ from the grid; --any keeps them)")
    # read the archive BEFORE filtering, so the padding filter can count the
    # invented samples on the arrays instead of reconstructing them
    archive = load_stored(args.wavelets) if args.wavelets else None
    if not args.allow_padding:
        n1 = len(ok)
        if archive is not None:
            how = "counted on the archived arrays"
        else:
            measured = sum(1 for r in ok if r.get("n_invented_kept") is not None)
            if measured == n1:
                how = "counted by the runner"
            elif measured == 0:
                how = ("RECONSTRUCTED from the origin offset with a %d ms tail, "
                       "which may not be the tail of this run: pass --wavelets"
                       % 60)
            else:
                how = "%d counted, %d reconstructed" % (measured, n1 - measured)
        ok = [r for r in ok if padded_visible(r, archive=archive) == 0]
        print("  %d records excluded whose compared segment contains invented "
              "samples (%s)" % (n1 - len(ok), how))
        print(f"  (extract_wavelet zero-fills the tail when the origin falls "
              f"late; --allow_padding keeps them)")
    if not args.allow_boundary:
        n1 = len(ok)
        ok = [r for r in ok if r["origin_minus_lead_ms"]
              > -(80. + float(r.get("pad_ms", 0.0))) + 4.]
        print("  %d records excluded whose origin sits at the start of the "
              "window (--allow_boundary keeps them)" % (n1 - len(ok)))
    if len(ok) < 3:
        raise SystemExit("fewer than three records left to choose from")
    ok.sort(key=lambda r: r[SC])
    pick = [("worst", ok[0]), ("median", ok[len(ok) // 2]), ("best", ok[-1])]
    print(f"{len(ok)} records at {args.snr:.0f} dB nominal; selected:")
    for lab, r in pick:
        print(f"  {lab:7s} {r['run_id']:28s} NCC {r[SC]:.3f} | "
              f"D {r['D_m']:.0f} m, layer {r['first_layer_m']:.0f} m, "
              f"fmax {r['fmax_hz']:.0f} Hz, n_ch {r.get('n_ch_used', '?')}"
              + (f"  [{padded_visible(r, archive=archive)}/100 samples invented]"
                 if padded_visible(r, archive=archive) else ""))
    print("\n  --only " + ",".join(r["run_id"] for _, r in pick)
          + "   <- to regenerate these three with their wavelets")

    m = cfg["meta"]
    # SIZES.  The figure was drawn 11 inches wide and reduced to a 160 mm
    # column, a factor of 1.75, which took the smallest annotations to about
    # 3.6 pt on the page.  It is now drawn at the width it will be printed at,
    # so what is set here is what the reader gets.
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.6, "legend.frameon": False,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(3, 2, figsize=(16.0 / 2.54, 15.0 / 2.54))

    for row, (lab, rec) in enumerate(pick):
        run = runs[rec["run_id"]]
        key = f"{rec['run_id']}|{rec.get('snr_db')}|{rec.get('t0_bias_ms')}"
        if archive is not None:
            if key not in archive or (key + '|ref') not in archive:
                raise SystemExit(
                    f"{key} is not in the archive.  Regenerate that record with\n"
                    f"  --only {rec['run_id']}  and patch_matrix_save_wavelet.py "
                    f"applied.")
            w, w_ref = archive[key], archive[key + '|ref']
            # The same source regenerate() uses: cfg["meta"]["dt_ms"].  Reading
            # it from run["derived"] was a guess, and there is no such field.
            dt = float(m["dt_ms"]) / 1000.
            # v5 stores the wavelet as returned, so the check has to allow
            # the same shift the score did
            klag = int(round(30e-3 / dt)) if SC == "ncc_shape" else 0
            if klag:
                v, lag = ncc_pair_lag(w, w_ref, klag)
                # drawn at the lag the score used: the stored wavelet begins at
                # the estimated origin, which is tens of milliseconds from the
                # true onset, and drawing it where it is stored shows the two
                # out of step
                w = advance(w, -lag)
                print(f"    read from the archive | NCC {v:.3f} "
                      f"(origin {-lag * dt * 1e3:+.0f} ms from the true onset, "
                      f"drawn realigned)")
            else:
                v = ncc_pair(w, w_ref)
                print(f"    read from the archive | NCC {v:.3f}")
            plot_one(ax, row, lab, w, w_ref, dt, v, rec)
            # The archive check reproduces the sliding score of the matrix,
            # so it is compared with that one; SC may be a different score
            # (ncc_shape is measured at the origin bias, not by sliding) and
            # comparing the two would fail for a reason that is not an error.
            stored_v = rec.get("ncc_prim_true", rec.get(SC))
            if stored_v is not None and abs(v - stored_v) > args.tol:
                raise SystemExit(
                    f"Refusing to draw: {key} is stored at {stored_v:.3f} and "
                    f"scores {v:.3f} here.\nThe archive and the records disagree, "
                    f"which should not be possible; the matrix was probably\n"
                    f"re-run without clearing one of the two.")
            continue

        print("    WARNING: rebuilding the model.  Four separate causes have "
              "shown that this\n    does not reproduce the matrix; use "
              "--wavelets for the figure that goes\n    in the paper.")
        g, dt, offs, w_bare, D = regenerate(run, cfg)
        # MATCHES THE MATRIX: the nominal depth of the cell, not the median of
        # the regenerated seafloor.  They differ where there is slope or
        # roughness, and D_guess sets the semblance search window.
        D_nom = float(run["params"]["D_m"])
        wl = int(round(m["w_len_ms"] * 1e-3 / dt))
        zs_ = run["params"].get("z_src_m", Z_DEV)
        zr_ = run["params"].get("z_rec_m", Z_DEV)
        # MATCHES THE MATRIX: run_matrix_v2 compares against the 2D
        # reference, and comparing against the 1D one here made the figure
        # disagree with every number in the tables.
        w_ref = reference(w_bare, dt, wl, zs_, zr_, two_d=not args.one_d)
        w_ref_1d = reference(w_bare, dt, wl, zs_, zr_, two_d=False)
        w_ref_2d = reference(w_bare, dt, wl, zs_, zr_, two_d=True)
        # The velocity, the semblance and the tracking are measured on the
        # CLEAN gather in the matrix; only the extraction sees the noise.
        vw, _, _ = estimate_vw_direct(g, dt, offs, D_guess=D_nom)
        r0 = semblance_bottom(g[0].T, dt, offs,
                              t0_min_ms=max(120, 2 * D_nom / vw * 1e3 - 200),
                              t0_max_ms=2 * D_nom / vw * 1e3 + 200)
        # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
        # were computed; it is written out so that the default of 20 does not change them.
        t0 = track_seafloor_xcorr(g, dt, offs,
                                  t0_init_s=(r0['t0'] if r0 else 2 * D_nom / vw),
                                  vw=vw, n_group=5)
        # UPSTREAM DIAGNOSTIC: everything below depends on these two, and the
        # matrix stored both.  If they disagree, the extraction is innocent.
        print(f"    vw        : figure {vw:8.1f} | matrix "
              f"{rec.get('vw_est', float('nan')):8.1f}")
        print(f"    t0 mean   : figure {np.mean(t0)*1e3:8.1f} | matrix "
              f"{rec.get('t0_track_ms', float('nan')):8.1f} ms")
        print(f"    D         : nominal {D_nom:8.1f} | regenerated median "
              f"{D:8.1f} m")

        # MATCHES THE MATRIX: the same noise, from the same deterministic seed.
        # Extracting from the clean gather while quoting a correlation measured
        # on a noisy one is not the same experiment.
        gg = g
        if args.snr < 60:
            rng_n = np.random.default_rng(run["seed"] + int(args.snr))
            ns = np.sqrt(np.mean(g ** 2)) / (10 ** (args.snr / 20.))
            gg = g + ns * rng_n.normal(0, 1, g.shape).astype(np.float32)
        # Extract LONGER than the reference so that the comparison always has
        # full overlap.  With equal lengths, an alignment lag of L samples
        # leaves only wl - |L| samples to compare, and on this matrix the lag
        # reaches 40 samples in a few per cent of cases: the correlation is then
        # computed on 60 per cent of the wavelet and is not comparable with the
        # rest.  Extracting an extra `pad_ms` on both sides removes the problem
        # without changing the method: the surplus is discarded after alignment.
        # The margin must ADAPT to the record: a fixed 120 ms pushes the window
        # start before t = 0 in shallow water and every window is then rejected.
        t_arr_min = float(np.min(np.sqrt(np.asarray(t0) ** 2 + (offs[0] / vw) ** 2)))
        lead_max = max(20., 0.90 * t_arr_min * 1e3)
        pad_ms = float(np.clip(lead_max - 80., 0., 120.))
        w, qc = extract_wavelet(gg, dt, t0, offs,
                                w_len_ms=m["w_len_ms"] + 2 * pad_ms,
                                lead_ms=min(80. + pad_ms, lead_max),
                                n_ch=run["derived"]["n_ch_used"], order=0,
                                band=BAND, vw=vw)
        if w is None:
            print(f"    extraction failed ({qc.get('error')}), panel left empty")
            ax[row, 0].text(.5, .5, "extraction failed", ha="center",
                            transform=ax[row, 0].transAxes)
            ax[row, 1].axis("off")
            continue
        if w is None:
            # the extraction can legitimately fail -- shallow cells, windows
            # that do not fit the record -- and the selected "worst" case is
            # exactly where that is most likely
            print(f"    extraction FAILED: {qc.get('error')}")
            ax[row, 0].text(.5, .5, f"extraction failed\n{qc.get('error')}",
                            ha='center', va='center', transform=ax[row, 0].transAxes)
            ax[row, 0].set_ylabel(f"{lab}", fontsize=8)
            ax[row, 1].axis('off')
            continue
        wlong = nrm(np.asarray(w, float))
        if wlong.ndim == 0 or len(wlong) < wl:
            print("    extraction too short, panel left empty")
            ax[row, 1].axis("off"); continue
        npad = int(round(pad_ms * 1e-3 / dt))
        # MATCHES THE MATRIX: the slide is bounded twice over, by the pad and by
        # half the direct-to-primary separation.  Unbounded, it latches onto a
        # later sub-bottom event, and an offset at the edge of the search is not
        # an alignment.
        sep_s = (np.hypot(2 * D_nom, offs[0]) - offs[0]) / vw
        o_lim = max(2, min(npad, int(round(0.5 * sep_s / dt))))
        o_lo = max(0, npad - o_lim)
        o_hi = min(len(wlong) - wl, npad + o_lim)
        best, bo = -2.0, npad
        for o in range(o_lo, o_hi + 1):
            seg = wlong[o:o + wl]
            x = seg - seg.mean(); y = w_ref - w_ref.mean()
            d = np.linalg.norm(x) * np.linalg.norm(y)
            c = float(x @ y / d) if d > 1e-12 else 0.
            if c > best:
                best, bo = c, o
        wv = nrm(wlong[bo:bo + wl])
        v, lag = best, bo - npad
        # decisive test: which reference does the simulation actually produce?
        def _slide(ref):
            bb = -2.0
            for o in range(o_lo, o_hi + 1):
                sg = wlong[o:o + wl]
                x = sg - sg.mean(); y = ref - ref.mean()
                d = np.linalg.norm(x) * np.linalg.norm(y)
                if d > 1e-12:
                    bb = max(bb, float(x @ y / d))
            return bb
        # --- over how much of the window should the comparison be made? -----
        # The synthetic wavelet is known, so its support can be measured: T95 is
        # where 95 per cent of the cumulative energy is reached.  Beyond it the
        # reference is essentially zero while the extraction may carry
        # sub-bottom contamination, so comparing over the full window penalises
        # a region holding 5 per cent of the true energy.  Restricting to the
        # support answers "was the wavelet recovered"; the full window also asks
        # "is the tail clean", which is what deconvolution needs.  Both are
        # reported.  Note that the support is known only because this is a
        # synthetic, so it cannot serve as a criterion in practice.
        cum = np.cumsum(w_ref ** 2) / max(np.sum(w_ref ** 2), 1e-30)
        i95 = int(np.searchsorted(cum, 0.95)) + 1

        def _slide_n(ref, n):   # bounded as above; n is the compared length
            bb = -2.0
            for o in range(0, len(wlong) - n + 1):
                sg = wlong[o:o + n]
                x = sg - sg.mean(); y = ref[:n] - ref[:n].mean()
                d = np.linalg.norm(x) * np.linalg.norm(y)
                if d > 1e-12:
                    bb = max(bb, float(x @ y / d))
            return bb

        print(f"    true support T95 = {i95*dt*1e3:.0f} ms of a {wl*dt*1e3:.0f} ms window")
        print(f"      NCC over the full window : {_slide_n(w_ref, wl):.3f}")
        print(f"      NCC over the support only: {_slide_n(w_ref, i95):.3f}")
        print(f"    NCC against reference WITHOUT the 2D factor: {_slide(w_ref_1d):.3f}")
        print(f"    NCC against reference WITH    the 2D factor: {_slide(w_ref_2d):.3f}"
              f"   <- if clearly higher, the 2D Green function is the explanation")
        # NON-CIRCULAR display shift.  np.roll wraps the tail onto the head, so
        # at a large lag the picture shows something quite different from what
        # the correlation actually compared.
        # THE CHECK.  The figure illustrates the matrix, so it must agree
        # with it.  Silently drawing a different number is how a figure ends up
        # contradicting the table beside it.
        stored = rec.get("ncc_prim_true", rec.get(SC))
        if stored is not None:
            d = abs(v - stored)
            flag = "OK" if d <= args.tol else "*** MISMATCH ***"
            print(f"    stored in the matrix {stored:.3f} | recomputed here "
                  f"{v:.3f} | difference {d:+.4f}  {flag}")
            if d > args.tol:
                raise SystemExit(
                    f"\nRefusing to draw: record {rec['run_id']} is stored at "
                    f"{stored:.3f} and recomputed at {v:.3f}.\n"
                    f"The figure and the tables would disagree.  Check that the "
                    f"reference (2D), the\nbounded slide and the extraction "
                    f"parameters match run_matrix_v2.py, or raise --tol\n"
                    f"deliberately and say so in the caption.")
        plot_one(ax, row, lab, wv, w_ref, dt, v, rec)
        sc = qc.get("align_scatter_ms", float('nan'))
        print(f"    alignment offset {lag*dt*1e3:+.0f} ms, "
              f"full {wl}/{wl} samples compared | "
              f"align scatter {sc:.1f} ms")
        if np.isfinite(sc) and sc > 0:
            # stacking windows whose timing scatters by sigma acts as
            # exp(-2 pi^2 f^2 sigma^2); this is a prediction, printed so it can
            # be checked against the measured spectra alongside
            pred = [f"{f:.0f} Hz: {np.exp(-2*np.pi**2*f**2*(sc*1e-3)**2):.2f}"
                    for f in (40., 60., 80.)]
            print(f"      predicted stack smearing -> " + ", ".join(pred))

        # --- is the low-frequency deficit in the EXTRACTION or in the DATA? ---
        # Compare the analytic reference against a RAW window of the primary,
        # with no stacking, no alignment and a single band-pass.  If the raw
        # window already lacks the low frequencies, the deficit is in the
        # simulation and not in the workflow.
        k_sf = int(round(float(np.median(t0)) / dt))
        raw = g[:, max(0, k_sf - wl // 4):max(0, k_sf - wl // 4) + wl, 0]
        if raw.shape[1] == wl:
            nyq_ = 0.5 / dt
            bb, aa = butter(4, [BAND[0] / nyq_, min(BAND[1] / nyq_, .95)], 'band')
            raw = filtfilt(bb, aa, raw, axis=1)
            Ar = np.abs(np.fft.rfft(raw * np.hanning(wl), 1024, axis=1)).mean(0)
            frr = np.fft.rfftfreq(1024, dt)
            bmr = (frr >= BAND[0]) & (frr <= BAND[1])
            lo = (frr >= 6) & (frr <= 20)
            At = np.abs(np.fft.rfft(w_ref, 1024)); Ae = np.abs(np.fft.rfft(wv, 1024))
            f_ = lambda A: (A[lo].mean() / (A[bmr].max() + 1e-30))
            print(f"    low-frequency content (6-20 Hz, relative to peak): "
                  f"true {f_(At):.3f} | extracted {f_(Ae):.3f} | "
                  f"raw window {f_(Ar):.3f}")
    ax[2, 0].set_xlabel('time (ms)'); ax[2, 1].set_xlabel('frequency (Hz)')
    plt.tight_layout(); plt.savefig(args.out + ".png", dpi=200)
    plt.savefig(args.out + ".pdf")
    print(f"\nfigure: {args.out}.png and .pdf")


if __name__ == "__main__":
    main()
