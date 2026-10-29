#!/usr/bin/env python3
"""The synthetic numbers of Sect. 3.2 that the table scripts do not print, and
the inputs of Fig. 2, from the records with the final recipe.

  first reflector   main grid (every depth), 30 dB, Table-2 rules: records,
                    evaluable, failures, mean and median NCC per first-reflector
                    depth, with the seafloor known (v5_bias) and picked (v6)
  failures          the known-seafloor origin failures of the evaluation domain
                    (D >= 220 m and the secondary axes), by first-reflector depth
  picking errors    median NCC and failure rate per deliberate picking error
  channel count     the reference configuration of the secondary axes (400 m of
                    water, first reflector 150 m below, 60 Hz, flat; near channel
                    only) against its channel-count axis (4, 16, 32 channels),
                    known seafloor, 30 dB; different realisations, not pairs
  multiples         records of the evaluation domain at the four SNR levels whose
                    multiple windows do not fit in the record (no multiple
                    checks), by axis
  Fig. 2            OUT/fig2/: the v5_bias records and their wavelets under the
                    keys make_fig_synthetic.py reads, then that script, unchanged

    DATA=DIR python validation/synthetic/paper_numbers.py RUNDIR OUT

DATA holds wavelet_matrix_v2/manifest.jsonl, which make_fig_synthetic.py reads
from its working directory.

RUNDIR: the output of merge_rerun.py or rerun_final.sh.  Table-2 rules: ok,
no ref_ records, zero picking error, NCC (ncc_shape) on the records whose
returned window holds at least 80 % of the true wavelet's energy.
"""
import glob
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def records(run):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(run, "results_*.jsonl"))) for l in open(f)]


def scored(g):
    return np.array([r["ncc_shape"] for r in g if not r.get("origin_off_wavelet")
                     and r.get("ncc_shape") is not None and r["ncc_shape"] == r["ncc_shape"]], float)


def table2(R, p, bias=0.0, dmin=0.0, grid=False):
    return [r for r in R if r["pass_name"] == p and r.get("ok") and r.get("D_m", 0) >= dmin
            and not str(r["run_id"]).startswith("ref_") and r.get("t0_bias_ms") == bias
            and (not grid or r.get("axis") == "grid_D_contam")]


def main():
    run, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        raise SystemExit(out + " exists")
    os.makedirs(out)
    R = records(run)
    L = ["# Synthetic numbers of Sect. 3.2 (`%s`)\n" % run,
         "## First-reflector depth: main grid, every depth, 30 dB\n",
         "| protocol | first reflector | records | evaluable | failures | mean NCC +- sd | median NCC |",
         "|---|---|---|---|---|---|---|"]
    for p, lab in (("v5_bias", "known seafloor"), ("v6", "automatic chain")):
        g = table2(R, p, grid=True)
        for fl in sorted({r["first_layer_m"] for r in g}):
            q = [r for r in g if r["first_layer_m"] == fl]; v = scored(q); f = len(q) - v.size
            L.append("| %s | %.0f m | %d | %d | %d (%.0f %%) | %.3f +- %.3f | %.3f |" % (
                lab, fl, len(q), v.size, f, 100 * f / len(q), v.mean(), v.std(), np.median(v)))
    g = table2(R, "v5_bias", dmin=220.0)
    bad = [r for r in g if r.get("origin_off_wavelet")]
    L += ["\n## Known-seafloor origin failures, evaluation domain (%d records)\n" % len(g),
          "%d failures; by first-reflector depth: %s" % (
              len(bad), ", ".join("%.0f m %d" % kv for kv in sorted(Counter(r["first_layer_m"] for r in bad).items())))]
    L += ["\n## Deliberate picking errors, evaluation domain, known seafloor\n",
          "| error (ms) | records | failures | median NCC |", "|---|---|---|---|"]
    for b in sorted({r["t0_bias_ms"] for r in R if r["pass_name"] == "v5_bias"}):
        q = table2(R, "v5_bias", bias=b, dmin=220.0); v = scored(q)
        L.append("| %+.0f | %d | %.0f %% | %s |" % (b, len(q), 100 * (len(q) - v.size) / max(len(q), 1),
                                                   "%.3f" % np.median(v) if v.size else "-"))
    L += ["\n## Channel count: reference configuration (1 channel) and its axis, known seafloor, 30 dB\n",
          "| channels | records | evaluable | mean NCC +- sd | median NCC |", "|---|---|---|---|---|"]
    base = [r for r in R if r["pass_name"] == "v5_bias" and r.get("ok") and r.get("t0_bias_ms") == 0.0]
    ref = {r["run_id"]: r for r in base if str(r["run_id"]).startswith("ref_")}.values()
    for n, q in [(1, list(ref))] + [(n, [r for r in base if r.get("axis") == "secondary_n_ch" and r.get("n_ch_used") == n])
                                    for n in (4, 16, 32)]:
        v = scored(q)
        L.append("| %d | %d | %d | %.3f +- %.3f | %.3f |" % (n, len(q), v.size, v.mean(), v.std(), np.median(v)))
    g = [r for r in R if r["pass_name"] == "v5_noise" and r.get("ok") and r.get("D_m", 0) >= 220
         and not str(r["run_id"]).startswith("ref_") and r.get("t0_bias_ms") == 0.0]
    no = [r for r in g if not r.get("n_win_mult")]
    L += ["\n## Multiple checks unavailable (evaluation domain, four SNR levels)\n",
          "%d of %d records; by axis: %s" % (len(no), len(g), ", ".join("%s %d" % kv for kv in Counter(r["axis"] for r in no).most_common()))]
    open(os.path.join(out, "paper_numbers.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    # Fig. 2: the records and wavelets of the known-seafloor pass, keys without the pass name
    fd = os.path.abspath(os.path.join(out, "fig2")); os.makedirs(os.path.join(fd, "results"))
    with open(os.path.join(fd, "results", "results.jsonl"), "w") as fh:
        for r in R:
            if r["pass_name"] == "v5_bias":
                fh.write(json.dumps(r) + "\n")
    W = {}
    for f in sorted(glob.glob(os.path.join(run, "results_*_wavelets.npz"))):
        with np.load(f) as z:
            W.update({k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("v5_bias|")})
    wp = os.path.join(fd, "results", "results_v5_bias_wavelets.npz")
    np.savez_compressed(wp, **W)
    with open(os.path.join(fd, "fig_synthetic.log"), "w") as fh:
        subprocess.run([sys.executable, os.path.join(HERE, "make_fig_synthetic.py"), "--results", os.path.join(fd, "results"),
                        "--wavelets", os.path.join(fd, "results"), "--out", os.path.join(fd, "fig_synthetic")], stdout=fh, stderr=subprocess.STDOUT,
                       cwd=os.environ.get("DATA", "."), env=dict(os.environ, PYTHONPATH=REPO))
    print("Fig. 2: %s/fig_synthetic.{pdf,png}, log fig_synthetic.log" % fd)


if __name__ == "__main__":
    main()
