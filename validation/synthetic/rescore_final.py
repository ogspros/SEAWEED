#!/usr/bin/env python3
"""
rescore_final.py -- the score the paper will quote, computed offline from the
archived wavelets.

Two quantities, kept apart:

  ORIGIN BIAS.  Where the returned wavelet starts, relative to the true onset,
  from the recorded times alone:
      bias = (tracked arrival + origin_minus_lead) - true arrival
  The near offset is the one of the record's own geometry, read from the
  manifest: g1500 sits at 12.5 m and the others at 125 m, and using one value
  for all of them would misplace the arrival of the 80 short-streamer records
  by tens of milliseconds.  No waveform enters it.  Measured on wavelet_matrix_v2_v3 it is 10 to 18 ms
  early at every depth, and the lag chosen independently by the correlation
  agrees with it to within a few ms, so it is a real bias of the origin
  estimator and not the correlation landing on the wrong cycle.

  SHAPE.  The agreement with the truth at the best lag within +-TOL ms of the
  bias.  TOL defaults to 8 ms, well under half the dominant period of about
  20 ms, so the comparison cannot slide from one lobe to the next.  The v3
  score allowed +-30 ms about zero, more than a full cycle, which is why it
  could not say which event had been extracted.

  OVERLAP.  The score is only defined where the true wavelet is actually
  inside the 200 ms that were returned.  When the origin lands 60 or 150 ms
  late, the shifted reference is mostly outside the window, its retained part
  is a handful of samples, and a correlation computed on it takes any value at
  all, up to +-1, which makes means and standard deviations meaningless.  A
  record keeping less than MINOVL of
  the reference energy is not given a score: it is marked as a failure, with
  error "origin off the wavelet", because the workflow did return something and
  what it returned does not contain the primary.

Writes <root>/results_rescored.jsonl: every record as it was, with

    ncc_prim_true      replaced by the shape score
    ncc_prim_true_slide  the old +-30 ms score, kept
    origin_bias_ms     the bias above
    score_lag_ms       the lag actually used, relative to the bias

To use it with the existing tools:

    python3 validation/synthetic/rescore_final.py wavelet_matrix_v2_v3 --x 125 --vwater 1500
    cd wavelet_matrix_v2_v3
    mv results.jsonl results_slide30.jsonl && mv results_rescored.jsonl results.jsonl
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--x", type=float, required=True,
                    help="near offset in m, used only for records whose "
                         "geometry cannot be read from the manifest")
    ap.add_argument("--manifest", default=None,
                    help="manifest.jsonl of the matrix; default: the same path "
                         "with the trailing _v3 removed")
    ap.add_argument("--vwater", type=float, default=1500.0)
    ap.add_argument("--dt", type=float, default=2.0, help="ms")
    ap.add_argument("--tol", type=float, default=8.0, help="ms about the bias")
    ap.add_argument("--minovl", type=float, default=0.8,
                    help="least fraction of the reference energy that must lie "
                         "inside the returned window for the score to be defined")
    a = ap.parse_args()

    p = os.path.join(a.root, "results.jsonl")
    R = [json.loads(l) for l in open(p)] if os.path.exists(p) else [
        json.loads(l)
        for f in sorted(glob.glob(os.path.join(a.root, "results_shard*.jsonl")))
        for l in open(f)]
    W = {}
    for f in sorted(glob.glob(os.path.join(a.root, "results_shard*_wavelets.npz"))):
        with np.load(f) as z:
            for k in z.files:
                W[k] = z[k]
    if not W:
        raise SystemExit("no archived wavelets in %s" % a.root)

    # near offset per record, from the manifest and the geometry table
    XS = {}
    mp = a.manifest
    if mp is None:
        base = a.root.rstrip("/")
        if base.endswith("_v3"):
            base = base[:-3]
        mp = os.path.join(base, "manifest.jsonl")
    if os.path.exists(mp):
        try:
            import generate_line2_hf as G
            for l in open(mp):
                q = json.loads(l)
                g = q.get("params", {}).get("geom")
                if g is not None:
                    XS[q["run_id"]] = float(G.geom_offsets(G.GEOMS[g])[0])
        except Exception as e:
            print("  could not read geometries (%s): falling back to --x" % e)
    if XS:
        v = sorted(set(XS.values()))
        print("  near offsets from %s: %s" % (mp, ", ".join("%.1f" % u for u in v)))
    else:
        print("  no geometries read: every record uses --x = %.1f m" % a.x)

    dt = a.dt * 1e-3
    ktol = int(round(a.tol * 1e-3 / dt))
    out, done, skipped, off = [], 0, 0, 0
    for r in R:
        rec = dict(r)
        k = "%s|%s|%s" % (r.get("run_id"), r.get("snr_db"), r.get("t0_bias_ms"))
        need = ("t0_track_ms", "t0_true_ms", "origin_minus_lead_ms", "vw_est")
        if (r.get("ok") and k in W and k + "|ref" in W
                and not any(r.get(q) is None for q in need)):
            w = W[k].astype(float); ref = W[k + "|ref"].astype(float)
            x = XS.get(r.get("run_id"), a.x)
            ta_tr = np.hypot(r["t0_track_ms"] * 1e-3, x / r["vw_est"]) * 1e3
            ta_tu = np.hypot(r["t0_true_ms"] * 1e-3, x / a.vwater) * 1e3
            rec["x_near_m"] = x
            bias = ta_tr + r["origin_minus_lead_ms"] - ta_tu
            k0 = int(round(bias * 1e-3 / dt))
            n = ref.size
            m = abs(k0) + ktol + 1
            rp = np.concatenate([np.zeros(m), ref, np.zeros(m)])
            y = w - w.mean(); ny = np.linalg.norm(y)
            best, bl = -2.0, 0
            for L in range(k0 - ktol, k0 + ktol + 1):
                seg = rp[m + L:m + L + n]
                x = seg - seg.mean(); nx = np.linalg.norm(x)
                if nx * ny < 1e-12:
                    continue
                v = float(x @ y / (nx * ny))
                if v > best:
                    best, bl = v, L
            # overlap at the lag the TIMING gives, not at the one the search
            # ended on: when the reference is almost entirely outside the
            # window every lag of the search is degenerate and bl never moves
            seg0 = rp[m + k0:m + k0 + n]
            ovl = float(seg0 @ seg0 / (ref @ ref + 1e-30))
            if best < -1.5:
                ovl = 0.0
            rec["ncc_prim_true_slide"] = r.get("ncc_prim_true")
            rec["origin_bias_ms"] = float(bias)
            rec["ref_overlap"] = ovl
            if ovl < a.minovl:
                rec["ok"] = False
                rec["error"] = "origin off the wavelet (overlap %.2f)" % ovl
                rec.pop("ncc_prim_true", None)
                off += 1
            else:
                rec["ncc_prim_true"] = best
                rec["score_lag_ms"] = float((bl - k0) * dt * 1e3)
                done += 1
        else:
            skipped += 1
        out.append(rec)

    q = os.path.join(a.root, "results_rescored.jsonl")
    with open(q, "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in out) + "\n")
    b = np.array([x["origin_bias_ms"] for x in out if "origin_bias_ms" in x])
    g = [x for x in out if x.get("ncc_prim_true") is not None
         and "origin_bias_ms" in x]
    s3 = np.array([x["ncc_prim_true"] for x in g])
    s2 = np.array([x["ncc_prim_true_slide"] for x in g], float)
    print("%s: scored %d, off the wavelet %d (%.1f%%), left alone %d  ->  %s"
          % (a.root, done, off, 100. * off / max(done + off, 1), skipped, q))
    print("  origin bias: median %+.0f ms, IQR [%+.0f, %+.0f]"
          % (np.median(b), *np.percentile(b, [25, 75])))
    print("  shape score (+-%g ms about the bias): median %.3f  mean %.3f"
          % (a.tol, np.median(s3), s3.mean()))
    print("  the +-30 ms slide score on the same records: median %.3f  mean %.3f"
          % (np.median(s2), s2.mean()))


if __name__ == "__main__":
    main()
