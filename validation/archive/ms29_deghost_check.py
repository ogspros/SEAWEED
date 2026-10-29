#!/usr/bin/env python3
"""
ms29_deghost_check.py -- compare the extracted MS-29 wavelet with the stack
amplitude spectra published by Brancatelli et al. (2022), before and after the
receiver deghost applied in Echos.

The two spectra are treated as data.  Their ratio is the deghost operator that
was actually applied, so the regularisation is measured rather than chosen:
the script fits

    O(f) = c / sqrt( 4 sin^2(2 pi f z_r / Vw) + eps^2 )

to the ratio, and reports z_r with its uncertainty.  It then runs three checks
on the extracted wavelet, in increasing order of what they assume:

  (1) z_r from the wavelet alone.  The receiver ghost imprints a notch on the
      extracted spectrum; the depth that best removes it is found by minimising
      the roughness of log|W| - log|G(f,z)|.  This needs no assumption about
      the reflectivity and is the cleanest comparison: three routes to the same
      number (acquisition report, Echos minimum energy, extracted wavelet).

  (2) shape of |W| against the no-deghost stack spectrum, which is the
      published-data version of the statistical check already in the paper.
      This does assume the sub-bottom reflectivity is approximately white.

  (3) shape of |W| deghosted with THEIR measured operator against their
      deghosted stack spectrum.  This is the discriminating test: if the notch
      in the extracted wavelet sat at a different frequency, applying their
      operator would over- or under-correct and the agreement would fall
      relative to (2).  The script quantifies that by repeating (3) with the
      operator displaced in depth, so the resolving power of the test is
      reported instead of assumed.

Usage
-----
python3 validation/archive/ms29_deghost_check.py ms29_wavelet.npy \
        --nodeghost Spettro_NO_deghost.txt \
        --deghost   Spettro_deghost.txt \
        --dt 0.004 --vw 1514.5 --band 8 45
"""

import argparse

import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------------------------
def load_spectrum(path):
    """Read an Echos amplitude-spectrum export: No., X [Hz], Y, Normalized Y."""
    f, y = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0].strip().isdigit():
                f.append(float(parts[1]))
                y.append(float(parts[2]))
    if not f:
        raise SystemExit("no data rows found in %s" % path)
    return np.asarray(f), np.asarray(y)


def load_wavelet(path, key=None):
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if key is None:
            keys = list(obj.keys())
            if len(keys) != 1:
                raise SystemExit("the .npz holds %s; choose one with --key"
                                 % keys)
            key = keys[0]
        obj = obj[key]
    w = np.asarray(obj, dtype=float).squeeze()
    if w.ndim != 1:
        raise SystemExit("expected a 1-D wavelet, got shape %s" % (w.shape,))
    return w


def ghost_modulus(f, z, vw):
    """|1 - exp(-i 2 pi f 2z/Vw)| for a free surface with R = -1."""
    return np.abs(2.0 * np.sin(2.0 * np.pi * f * z / vw))


def amplitude_spectrum(w, dt, nfft):
    W = np.fft.rfft(w, n=nfft)
    f = np.fft.rfftfreq(nfft, d=dt)
    return f, np.abs(W)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spectrum_df(f):
    """Frequency step, taken over the whole axis: the exported X column is
    rounded to three decimals, so f[1]-f[0] is not accurate enough."""
    return (f[-1] - f[0]) / (f.size - 1)


# --------------------------------------------------------------------------
def fit_operator(f, ratio, vw, lo, hi, z0=25.0):
    m = (f >= lo) & (f <= hi)

    def resid(p):
        c, z, eps = p
        model = c / np.sqrt(ghost_modulus(f[m], z, vw) ** 2 + eps ** 2)
        return np.log(model) - np.log(ratio[m])

    res = least_squares(resid, [1.0, z0, 0.3],
                        bounds=([1e-3, 5.0, 1e-3], [1e3, 60.0, 5.0]))
    c, z, eps = res.x
    dof = max(m.sum() - 3, 1)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * np.sum(res.fun ** 2) / dof
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(3, np.nan)
    model = c / np.sqrt(ghost_modulus(f[m], z, vw) ** 2 + eps ** 2)
    return dict(c=c, z=z, eps=eps, sd=sd, mask=m,
                r=pearson(model, ratio[m]),
                rms=float(np.sqrt(np.mean((model - ratio[m]) ** 2))))


def depth_from_wavelet(f, aw, vw, lo, hi, deg=4):
    """Receiver depth from the extracted wavelet alone.

    The extracted amplitude spectrum is modelled as a smooth envelope times
    the ghost modulus,

        log|W(f)| = P_deg(f) + log sqrt( 4 sin^2(2 pi f z/Vw) + delta^2 ),

    with P a polynomial in f.  The notch is not a true zero on real data --
    incoherent noise, cable-depth fluctuation and the finite stack all fill it
    -- so delta is fitted rather than assumed.  Fitting the envelope jointly
    is what makes the estimate insensitive to the overall shape of the source
    spectrum, which is exactly the quantity we do not want to assume.

    Several starting depths are tried because the objective has a local
    minimum wherever a harmonic of the notch could sit.
    """
    m = (f >= lo) & (f <= hi)
    x = (f[m] - lo) / (hi - lo)
    y = np.log(np.maximum(aw[m], 1e-12))

    def resid(p):
        z, delta = p[0], p[1]
        env = np.polyval(p[2:], x)
        g = np.sqrt(ghost_modulus(f[m], z, vw) ** 2 + delta ** 2)
        return env + np.log(g) - y

    lower = np.concatenate([[5.0, 1e-4], -np.inf * np.ones(deg + 1)])
    upper = np.concatenate([[60.0, 2.0], np.inf * np.ones(deg + 1)])

    best = None
    for z0 in (12.0, 18.0, 25.0, 32.0, 38.0):
        p0 = np.concatenate([[z0, 0.1], np.zeros(deg + 1)])
        res = least_squares(resid, p0, bounds=(lower, upper))
        if best is None or res.cost < best.cost:
            best = res

    z, delta = best.x[0], best.x[1]
    dof = max(m.sum() - best.x.size, 1)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
        sd = float(np.sqrt(cov[0, 0] * 2 * best.cost / dof))
    except np.linalg.LinAlgError:
        sd = float("nan")
    return z, sd, delta, float(best.cost)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavelet", help="extracted wavelet, .npy or .npz")
    ap.add_argument("--key", default=None)
    ap.add_argument("--nodeghost", required=True)
    ap.add_argument("--deghost", required=True)
    ap.add_argument("--dt", type=float, required=True,
                    help="wavelet sample interval in SECONDS")
    ap.add_argument("--vw", type=float, default=1514.5,
                    help="water velocity from the direct arrival (m/s)")
    ap.add_argument("--band", type=float, nargs=2, default=[8.0, 45.0])
    ap.add_argument("--zr_nominal", type=float, default=25.0,
                    help="receiver depth from the acquisition report (m)")
    ap.add_argument("--dz_control", type=float, default=3.0,
                    help="depth displacement used to test resolving power (m)")
    ap.add_argument("--deg", type=int, default=6,
                    help="degree of the polynomial envelope in the ghost fit; "
                         "the fitted depth should be checked for stability "
                         "against this (default 6)")
    ap.add_argument("--notch_halfwidth", type=float, default=8.0,
                    help="half-width of the window around the notch used for "
                         "the local comparison (Hz)")
    ap.add_argument("--save_npz", default=None,
                    help="write the curves used, for plotting")
    args = ap.parse_args()

    lo, hi = args.band

    # ---- the two published spectra --------------------------------------
    f_a, A = load_spectrum(args.nodeghost)
    f_b, B = load_spectrum(args.deghost)
    if not np.allclose(f_a, f_b):
        raise SystemExit("the two spectra are on different frequency axes")
    fs = f_a
    df = spectrum_df(fs)
    print("published spectra : %d points, df = %.5f Hz, f_max = %.2f Hz"
          % (fs.size, df, fs[-1]))
    print("  peak of no-deghost stack at %.2f Hz, deghost stack at %.2f Hz"
          % (fs[A.argmax()], fs[B.argmax()]))

    # ---- the operator actually applied -----------------------------------
    ratio = B / np.maximum(A, 1e-30)
    fit = fit_operator(fs, ratio, args.vw, lo, hi, z0=args.zr_nominal)
    print("\noperator recovered from the ratio of the two spectra, %g-%g Hz"
          % (lo, hi))
    print("  z_r  = %.2f +/- %.2f m   (report: %.1f m)"
          % (fit["z"], fit["sd"][1], args.zr_nominal))
    print("  notch = %.2f Hz          (predicted from the report: %.2f Hz)"
          % (args.vw / (2 * fit["z"]), args.vw / (2 * args.zr_nominal)))
    print("  eps  = %.3f  -> gain at the notch %.2f (measured max %.2f)"
          % (fit["eps"], fit["c"] / fit["eps"], ratio[fit["mask"]].max()))
    print("  r = %.4f, rms = %.4f on %d points"
          % (fit["r"], fit["rms"], int(fit["mask"].sum())))

    # ---- the extracted wavelet on the same frequency grid -----------------
    w = load_wavelet(args.wavelet, args.key)
    nfft = int(round(1.0 / (args.dt * df)))
    fw, AW = amplitude_spectrum(w, args.dt, nfft)
    k = min(fw.size, fs.size)
    # the exported X column is rounded to three decimals, so agreement to
    # 1e-3 Hz means the two grids are in fact the same one
    if not np.allclose(fw[:k], fs[:k], atol=1.1e-3):
        print("\n  note: grids differ, the wavelet spectrum is interpolated")
        AW = np.interp(fs, fw, AW)
        fw = fs
    n = min(fw.size, fs.size)
    fw, AW, A, B = fw[:n], AW[:n], A[:n], B[:n]
    print("\nwavelet: %d samples, %.1f ms, zero-padded to %d for df = %.5f Hz"
          % (w.size, w.size * args.dt * 1e3, nfft, df))

    # ---- check 1: depth from the wavelet alone ----------------------------
    z_w, z_w_sd, delta_w, cost_w = depth_from_wavelet(fw, AW, args.vw, lo, hi, args.deg)
    # the same estimator on their uncorrected stack, so that the two notch
    # depths are measured the same way and can be compared
    z_s, z_s_sd, delta_s, _ = depth_from_wavelet(fw, A, args.vw, lo, hi, args.deg)

    def notch_db(delta):
        return 20.0 * np.log10(max(delta, 1e-6) / 2.0)

    zz = [args.zr_nominal, fit["z"], z_s, z_w]
    print("\n[1] receiver depth, four routes")
    print("      acquisition report          : %.2f m" % args.zr_nominal)
    print("      Echos deghost operator      : %.2f +/- %.2f m"
          % (fit["z"], fit["sd"][1]))
    print("      their uncorrected stack     : %.2f +/- %.2f m"
          % (z_s, z_s_sd))
    print("      our extracted wavelet       : %.2f +/- %.2f m"
          % (z_w, z_w_sd))
    print("      spread over the four        : %.2f m" % (max(zz) - min(zz)))
    print("      notch frequency, stack %.2f Hz | wavelet %.2f Hz"
          % (args.vw / (2 * z_s), args.vw / (2 * z_w)))
    print("      notch depth,     stack %.1f dB | wavelet %.1f dB"
          % (notch_db(delta_s), notch_db(delta_w)))

    # ---- checks 2 and 3: spectral shape -----------------------------------
    m = (fw >= lo) & (fw <= hi)

    def norm(x):
        return x / np.max(x[m])

    r_raw = pearson(norm(AW)[m], norm(A)[m])
    r_raw_log = pearson(np.log(norm(AW)[m]), np.log(norm(A)[m]))

    def apply_operator(z):
        g = ghost_modulus(fw, z, args.vw)
        return AW * fit["c"] / np.sqrt(g ** 2 + fit["eps"] ** 2)

    AWdg = apply_operator(fit["z"])
    r_dg = pearson(norm(AWdg)[m], norm(B)[m])
    r_dg_log = pearson(np.log(norm(AWdg)[m]), np.log(norm(B)[m]))

    print("\n[2] extracted wavelet vs no-deghost stack, %g-%g Hz"
          % (lo, hi))
    print("      r = %.4f   (on log amplitude: %.4f)" % (r_raw, r_raw_log))
    print("[3] deghosted wavelet vs deghosted stack, same operator")
    print("      r = %.4f   (on log amplitude: %.4f)" % (r_dg, r_dg_log))
    print("      change relative to [2]: %+.4f" % (r_dg - r_raw))

    # the full-band correlation is dominated by the overall shape of the two
    # spectra, which the sub-bottom reflectivity and their processing both
    # affect.  Around the notch the operator is the only thing that varies
    # quickly, so a window there isolates what we are actually testing.
    f_notch = args.vw / (2 * fit["z"])
    mn = (fw >= max(lo, f_notch - args.notch_halfwidth)) \
        & (fw <= min(hi, f_notch + args.notch_halfwidth))
    print("\n    resolving power: comparison restricted to %.1f-%.1f Hz"
          % (fw[mn][0], fw[mn][-1]))
    print("    with the operator displaced in depth")

    def rlocal(z):
        a = apply_operator(z)
        return pearson(a[mn] / a[mn].max(), B[mn] / B[mn].max())

    for dz in (-args.dz_control, -1.0, 0.0, 1.0, args.dz_control):
        z = fit["z"] + dz
        print("      z_r = %5.2f m (%+4.1f)  notch %6.2f Hz   r = %.4f"
              % (z, dz, args.vw / (2 * z), rlocal(z)))

    # after deghosting with their operator the notch should no longer be
    # there; refitting the ghost model to the corrected wavelet says whether
    # a residual survives and at what depth it would sit
    z_res, z_res_sd, delta_res, _ = depth_from_wavelet(fw, AWdg, args.vw, lo, hi, args.deg)
    print("\n    residual ghost after applying their operator to the wavelet:")
    print("      best-fitting depth %.2f m, notch filled to delta = %.3f"
          % (z_res, delta_res))
    print("      (before correction: %.2f m, delta = %.3f)" % (z_w, delta_w))

    if args.save_npz:
        np.savez(args.save_npz, f=fw, A_nodeghost=A, B_deghost=B,
                 ratio=ratio[:n], W=AW, W_deghost=AWdg,
                 z_operator=fit["z"], z_wavelet=z_w, z_wavelet_sd=z_w_sd,
                 z_stack=z_s, delta_stack=delta_s,
                 delta_wavelet=delta_w, eps=fit["eps"], c=fit["c"],
                 band=[lo, hi])
        print("\ncurves written to %s" % args.save_npz)


if __name__ == "__main__":
    main()
