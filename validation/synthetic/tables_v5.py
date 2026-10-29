#!/usr/bin/env python3
"""
tables_v5.py -- the paper's synthetic tables, from the v5 matrices only.

Every number the manuscript quotes from the synthetic matrix should come from
this one script and from these two directories, so that a value in the text can
always be traced to a file.  The earlier tables mixed protocols and runs, and
that is how numbers with no provenance got in.

Prints, in the order the tables appear:

  1. the headline figures of the applicable domain, at zero bias
  2. the axes table, at zero bias
  3. the paired structural comparisons against the controls of the same seed
  4. the depth x contamination grid
  5. the noise curve, from the noise matrix
  6. the tracking-error curve, which is the new table

Records with no score (the true wavelet not inside the window returned) are
counted and reported, never silently dropped: they are the failure rate, and a
mean taken over the survivors alone is a mean conditioned on success.

    python3 validation/synthetic/tables_v5.py > tables_v5.txt
"""
import json

import numpy as np

BIAS_DIR = "wavelet_matrix_v5_bias"
NOISE_DIR = "wavelet_matrix_v5_noise"
SCORE = "ncc_shape"


def load(root):
    import glob, os
    p = os.path.join(root, "results.jsonl")
    if os.path.exists(p):
        return [json.loads(l) for l in open(p)]
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "results_shard*.jsonl")))
            for l in open(f)]


def applicable(R, bias=0.0, dmin=220.0):
    return [r for r in R if r.get("ok") and r.get("D_m", 0) >= dmin
            and not str(r.get("run_id", "")).startswith("ref_")
            and (bias is None or r.get("t0_bias_ms") == bias)]


def stat(g):
    v = np.array([r[SCORE] for r in g if r.get(SCORE) is not None
                  and r[SCORE] == r[SCORE]], float)
    off = sum(1 for r in g if r.get("origin_off_wavelet"))
    return len(g), off, v


def line(lab, g):
    n, off, v = stat(g)
    if not v.size:
        print("%-34s %5d %7d%s" % (lab, n, off, "   no scored record"))
        return
    print("%-34s %5d %6.0f%% %s%.3f \\pm %.3f$ & %.3f"
          % (lab, n, 100.0 * off / max(n, 1), "$", v.mean(), v.std(), np.median(v)))


def main():
    R = load(BIAS_DIR)
    g0 = applicable(R, 0.0)
    n, off, v = stat(g0)
    print("=" * 74)
    print("1. APPLICABLE DOMAIN, true seafloor, zero bias   [%s]" % BIAS_DIR)
    print("=" * 74)
    print("records %d | no score (wavelet not in the window) %d (%.1f%%)"
          % (n, off, 100.0 * off / n))
    print("agreement over the scored ones: median %.3f, mean %.3f +- %.3f"
          % (np.median(v), v.mean(), v.std()))
    b = np.array([r["origin_bias_ms"] for r in g0], float)
    print("origin bias: median %+.0f ms, IQR [%+.0f, %+.0f]"
          % (np.median(b), *np.percentile(b, [25, 75])))
    ok = [r for r in g0 if not r.get("origin_off_wavelet")]
    bad = [r for r in g0 if r.get("origin_off_wavelet")]
    if bad:
        f = lambda q, k: 100.0 * np.mean([(r.get(k) or 0) > 0 for r in q])
        print("failures: origin %+.0f ms past the arrival (rest %+.0f); "
              "filled samples in %.0f%% of them and %.0f%% of the rest"
              % (np.median([r["origin_minus_lead_ms"] for r in bad]),
                 np.median([r["origin_minus_lead_ms"] for r in ok]),
                 f(bad, "n_invented_kept"), f(ok, "n_invented_kept")))
        print("          coherence %.3f against %.3f"
              % (np.median([r["coh_prim"] for r in bad]),
                 np.median([r["coh_prim"] for r in ok])))

    print("\n" + "=" * 74)
    print("2. AXES (Table: axis, n, no score, mean, median)")
    print("=" * 74)
    print("%-34s %5s %7s %s" % ("axis", "n", "no score", "mean & median"))
    line("depth x contamination grid",
         [r for r in g0 if r["axis"] == "grid_D_contam"])
    for ax, lab in (("secondary_fmax_hz", "source band"),
                    ("secondary_slope_deg", "seafloor dip"),
                    ("secondary_roughness_m", "seafloor roughness"),
                    ("secondary_geom", "streamer geometry"),
                    ("secondary_n_ch", "channel count"),
                    ("secondary_p_folds", "folds"),
                    ("secondary_p_fault", "fault"),
                    ("secondary_p_lens", "anomalous body")):
        line(lab, [r for r in g0 if r["axis"] == ax])

    print("\n" + "=" * 74)
    print("3. STRUCTURAL ELEMENTS, paired against the control of the same seed")
    print("=" * 74)
    # a record with no score carries nan, not None: taking it into a mean
    # would turn the whole cell into nan
    ctl = {}
    for r in R:
        v = r.get(SCORE)
        if (r.get("ok") and str(r.get("run_id", "")).startswith("ref_")
                and r.get("t0_bias_ms") == 0.0 and v is not None and v == v):
            ctl[(r["seed"], r["snr_db"])] = v
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    for ax, lab in (("secondary_p_fault", "fault"),
                    ("secondary_p_folds", "folds"),
                    ("secondary_p_lens", "anomalous body")):
        d = [r[SCORE] - ctl[(r["seed"], r["snr_db"])] for r in g0
             if r["axis"] == ax and r.get(SCORE) is not None
             and r[SCORE] == r[SCORE] and (r["seed"], r["snr_db"]) in ctl]
        n_pairs_possible = sum(1 for r in g0 if r["axis"] == ax)
        d = np.array(d, float)
        if d.size < 5:
            print("%-18s %d usable pairs of %d records: too few"
                  % (lab, d.size, n_pairs_possible)); continue
        p = wilcoxon(d).pvalue if (wilcoxon and np.any(d != 0)) else float("nan")
        print("%-18s %3d of %3d pairs  mean %+.3f  median %+.3f  "
              "Wilcoxon p = %.4f"
              % (lab, d.size, n_pairs_possible, d.mean(), np.median(d), p))
    if not ctl:
        print("  no control records with a score: the paired comparison cannot "
              "be made from this file")

    print("\n" + "=" * 74)
    print("4. DEPTH x CONTAMINATION GRID (mean of the scored records)")
    print("=" * 74)
    gg = [r for r in R if r.get("ok") and r["axis"] == "grid_D_contam"
          and r.get("t0_bias_ms") == 0.0]
    Ds = sorted({r["D_m"] for r in gg})
    Ls = sorted({r["first_layer_m"] for r in gg})
    print("(mean of the scored records / how many of the cell had no score)")
    print("%9s  " % "D \\ layer" + "  ".join("%9.0f m" % l for l in Ls))
    for D in Ds:
        row = []
        for L in Ls:
            q = [r for r in gg if r["D_m"] == D and r["first_layer_m"] == L]
            v = [r[SCORE] for r in q
                 if r.get(SCORE) is not None and r[SCORE] == r[SCORE]]
            row.append("%6.3f/%2d" % (np.mean(v), len(q) - len(v)) if v
                       else "   -/%2d" % len(q))
        print("%9.0f  " % D + "  ".join("%11s" % c for c in row))

    print("\n" + "=" * 74)
    print("5. NOISE   [%s]" % NOISE_DIR)
    print("=" * 74)
    try:
        N = load(NOISE_DIR)
    except Exception as e:
        print("  not readable: %s" % e); N = []
    if N:
        gN = applicable(N, None)
        print("%8s %10s %5s %8s %18s %10s %10s"
              % ("nominal", "effective", "n", "no score", "NCC", "coherence",
                 "validator"))
        for s in sorted({r["snr_db"] for r in gN}, reverse=True):
            q = [r for r in gN if r["snr_db"] == s]
            n, off, v = stat(q)
            e = [r["snr_eff_db"] for r in q
                 if isinstance(r.get("snr_eff_db"), (int, float))
                 and abs(r["snr_eff_db"]) < 1e6]
            va = [r["validator"] for r in q if r.get("validator") is not None]
            print("%8.0f %10.1f %5d %7.0f%% %10.3f +- %.3f %10.3f %10.3f"
                  % (s, np.mean(e) if e else float("nan"), n,
                     100.0 * off / max(n, 1),
                     v.mean() if v.size else float("nan"),
                     v.std() if v.size else float("nan"),
                     np.median([r["coh_prim"] for r in q]),
                     np.mean(va) if va else float("nan")))

    print("\n" + "=" * 74)
    print("6. TRACKING ERROR (the new table)")
    print("=" * 74)
    print("%10s %6s %9s %18s" % ("bias (ms)", "n", "no score", "NCC"))
    for bias in sorted({r["t0_bias_ms"] for r in R if r.get("ok")}):
        q = applicable(R, bias)
        n, off, v = stat(q)
        print("%+10.0f %6d %8.0f%% %s"
              % (bias, n, 100.0 * off / max(n, 1),
                 "%10.3f +- %.3f" % (v.mean(), v.std()) if v.size
                 else "         no scored record"))
    t = np.array([r["track_pick_err_ms"] for r in applicable(R, 0.0)
                  if isinstance(r.get("track_pick_err_ms"), (int, float))
                  and r["track_pick_err_ms"] == r["track_pick_err_ms"]], float)
    if t.size:
        print("\nAUTOCORRELATION ALONE (pick_seafloor_matrix, one bracket of\n"
              "lags from 0.10 to 1.50 s) against the true seafloor.  This is the\n"
              "method Stage 1 no longer uses on its own, and the number it is\n"
              "here to justify; the consensus of Stage 1 is measured by\n"
              "eval_picker.py, not by this run.")
        print("  %d records: "
              "within 25 ms %.0f%%, within 80 ms %.0f%%, median %+.0f ms"
              % (t.size, 100 * np.mean(np.abs(t) < 25),
                 100 * np.mean(np.abs(t) < 80), np.median(t)))
        print("  the median can be near zero while the share within 80 ms is "
              "low: the errors\n  are bimodal, the picker either locks the "
              "water-bottom lobe or a sub-bottom one")
        R0 = applicable(R, 0.0)
        for D in sorted({r["D_m"] for r in R0}):
            e = np.array([r["track_pick_err_ms"] for r in R0 if r["D_m"] == D
                          and isinstance(r.get("track_pick_err_ms"), (int, float))
                          and r["track_pick_err_ms"] == r["track_pick_err_ms"]], float)
            if e.size:
                print("   D %6.0f m  n=%4d  median %+7.0f ms  within 80 ms %3.0f%%"
                      % (D, e.size, np.median(e), 100 * np.mean(np.abs(e) < 80)))


if __name__ == "__main__":
    main()
