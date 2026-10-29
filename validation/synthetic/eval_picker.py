#!/usr/bin/env python3
"""
eval_picker.py -- the three pickers against the true seafloor, on the models of
the matrix.

The v5 run measured the autocorrelation picker: within 80 ms of the truth, the
margin the extraction tolerates, on 36 per cent of records, and early by a
median of 105 ms.  This puts the consensus picker of pick_consensus.py on the
same models, next to that one and to the semblance chain of v2/v3, so the three
are compared on the same gathers rather than across runs.

Models are regenerated with the same seeds through make_fig_synthetic, as
qc_synth_case does; nothing is read from the v5 results except which runs to
take and what the truth is.

    python3 validation/synthetic/eval_picker.py --config validation/synthetic/matrix_v2.yaml --n 40 --shard 0 --n_shards 10

Each shard writes eval_picker_<shard>.jsonl; concatenate and read with --report.
"""
import argparse
import glob
import json
import os

import numpy as np

import make_fig_synthetic as F
from seaweed import pick_consensus as PC
from seaweed.wavelet_extractor_v2 import track_seafloor_xcorr, estimate_vw_direct
from seaweed.seafloor_robust import semblance_bottom


# The configurations of the matrices sit next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))

def semblance_chain(g, dt, offs, vw, D_nom):
    """The v2/v3 initialisation: semblance on the first shot, searched within
    +-200 ms of the NOMINAL depth, then cross-correlation along the line."""
    r0 = semblance_bottom(g[0].T, dt, offs,
                          t0_min_ms=max(120, 2 * D_nom / vw * 1e3 - 200),
                          t0_max_ms=2 * D_nom / vw * 1e3 + 200)
    t0i = r0["t0"] if r0 else 2 * D_nom / vw
    # n_group=5 was the default of track_seafloor_xcorr when the synthetic results
    # were computed; it is written out so that the default of 20 does not change them.
    return track_seafloor_xcorr(g, dt, offs, t0_init_s=t0i, vw=vw, n_group=5)


def run_one(run, cfg):
    g, dt, offs, w_bare, D = F.regenerate(run, cfg)
    m = cfg["meta"]
    p = run["params"]
    D_nom = float(np.median(D)) if np.ndim(D) else float(D)
    t0_true = float(np.median(np.asarray(D, float) * 2.0 / m["v_water"]))
    vw, _x, _n = estimate_vw_direct(g, dt, offs, D_guess=D_nom)
    out = dict(run_id=run["run_id"], D_m=p["D_m"], first_layer_m=p["first_layer_m"],
               slope_deg=p.get("slope_deg"), roughness_m=p.get("roughness_m"),
               axis=run["axis"], t0_true_ms=t0_true * 1e3, vw_est=float(vw))
    try:
        t0s = semblance_chain(g, dt, offs, vw, D_nom)
        out["semblance_err_ms"] = float(np.median(t0s) - t0_true) * 1e3
    except Exception as e:
        out["semblance_err_ms"] = None
        out["semblance_error"] = str(e)[:80]
    try:
        import run_matrix_v5 as V5
        t0s = V5.pick_seafloor_matrix(g, dt, offs, vw)
        out["autocorr_err_ms"] = float(np.median(t0s) - t0_true) * 1e3
    except Exception as e:
        out["autocorr_err_ms"] = None
        out["autocorr_error"] = str(e)[:80]
    try:
        t0s, rep = PC.pick_t0(g, dt, offs, vw)
        out["consensus_err_ms"] = float(np.median(t0s) - t0_true) * 1e3
        best = [d for d in rep["candidates"] if abs(d["t0"] - rep["t0"]) < 1e-9][0]
        out.update(consensus_semblance=best["semblance"],
                   consensus_multiple=best["semblance_multiple"],
                   consensus_polarity=best["multiple_polarity"],
                   consensus_stretch=best["stretch"],
                   consensus_sin_dip=best["sin_dip"],
                   consensus_agree_ms=rep["agree_ms"],
                   consensus_n_candidates=len(rep["candidates"]))
    except Exception as e:
        out["consensus_err_ms"] = None
        out["consensus_error"] = str(e)[:80]
    return out


def report(paths):
    R = [json.loads(l) for p in paths for l in open(p)]
    print("%d records\n" % len(R))
    print("%-12s %6s %9s %9s %9s %9s" % ("picker", "n", "|e|<25ms",
                                         "|e|<80ms", "median", "p90 |e|"))
    for k, lab in (("semblance_err_ms", "semblance"),
                   ("autocorr_err_ms", "autocorr"),
                   ("consensus_err_ms", "consensus")):
        e = np.array([r[k] for r in R if r.get(k) is not None], float)
        if not e.size:
            print("%-12s   none" % lab); continue
        print("%-12s %6d %8.0f%% %8.0f%% %+9.0f %9.0f"
              % (lab, e.size, 100 * np.mean(np.abs(e) < 25),
                 100 * np.mean(np.abs(e) < 80), np.median(e),
                 np.percentile(np.abs(e), 90)))
    print("\nby depth, share within the 80 ms the extraction tolerates:")
    print("%8s %6s %11s %10s %11s" % ("D (m)", "n", "semblance", "autocorr",
                                      "consensus"))
    for D in sorted({r["D_m"] for r in R}):
        g = [r for r in R if r["D_m"] == D]
        row = []
        for k in ("semblance_err_ms", "autocorr_err_ms", "consensus_err_ms"):
            e = np.array([r[k] for r in g if r.get(k) is not None], float)
            row.append("%.0f%%" % (100 * np.mean(np.abs(e) < 80)) if e.size else "-")
        print("%8.0f %6d %11s %10s %11s" % (D, len(g), *row))
    c = [r for r in R if r.get("consensus_err_ms") is not None]
    if c:
        e = np.abs(np.array([r["consensus_err_ms"] for r in c]))
        a = np.array([r.get("consensus_agree_ms", np.nan) for r in c], float)
        ok = np.isfinite(a)
        if ok.sum() > 5:
            close = a[ok] < 25
            print("\nwhen the two best candidates agree within 25 ms (%d of %d): "
                  "within 80 ms in %.0f%%; when they do not: %.0f%%"
                  % (close.sum(), ok.sum(),
                     100 * np.mean(e[ok][close] < 80) if close.any() else float('nan'),
                     100 * np.mean(e[ok][~close] < 80) if (~close).any() else float('nan')))
            print("  (an agreement flag the user can read without the truth)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "matrix_v2.yaml"))
    ap.add_argument("--n", type=int, default=40, help="runs sampled, spread over depths")
    ap.add_argument("--axis", default="grid_D_contam",
                    help="which axis to sample, or 'secondary' for every axis "
                         "except the main grid: those models were never looked "
                         "at while the picker was being written, so they are "
                         "the held-out set")
    ap.add_argument("--out_prefix", default="eval_picker")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:   # --out_prefix selects which set to read back
        report(sorted(glob.glob("%s_*.jsonl" % a.out_prefix)))
        return

    import yaml
    cfg = yaml.safe_load(open(a.config))
    root = cfg["paths"]["out_root"]
    runs = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl"))]
    if a.axis == "secondary":
        runs = [r for r in runs if r["axis"] != "grid_D_contam"]
    else:
        runs = [r for r in runs if r["axis"] == a.axis]
    if not runs:
        raise SystemExit("no runs on axis %s" % a.axis)
    # spread the sample over depths rather than taking the first n
    by = {}
    for r in runs:
        key = (r["axis"], r["params"]["D_m"]) if a.axis == "secondary" \
            else r["params"]["D_m"]
        by.setdefault(key, []).append(r)
    # take the sample SPREAD over each key, not its first few: taking the
    # first n of every (axis, depth) gave an axis whose dip was 2 degrees in
    # every record, and 5 and 10 degrees were never tested
    per = max(1, a.n // max(len(by), 1))
    sample = []
    for k in sorted(by, key=str):
        v = by[k]
        idx = np.linspace(0, len(v) - 1, min(per, len(v))).round().astype(int)
        sample += [v[i] for i in dict.fromkeys(idx)]
    sample = sample[a.shard::a.n_shards]
    out = "%s_%02d.jsonl" % (a.out_prefix, a.shard)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["run_id"] for l in open(out)}
    with open(out, "a") as fh:
        for i, r in enumerate(sample, 1):
            if r["run_id"] in done:
                continue
            try:
                rec = run_one(r, cfg)
            except Exception as e:
                rec = dict(run_id=r["run_id"], D_m=r["params"]["D_m"],
                           error=str(e)[:120])
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print("[%d] %d/%d %s" % (a.shard, i, len(sample), r["run_id"]), flush=True)


if __name__ == "__main__":
    main()
