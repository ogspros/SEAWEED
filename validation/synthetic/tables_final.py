#!/usr/bin/env python3
"""Tables of the synthetic matrices with the final recipe: the scripts of the
first version, unchanged (tables_v5.py, compare_v5_v6.py, tables_v5_extra.py),
run on the records of RUNDIR, plus summary_final.md for what they cannot see.
Where the chain fails, run_matrix_v6 falls back to the TRUE seafloor
(pick_failed); summary_final.md counts those records as failures, not scored.
It also reports where the onset tracking is not applicable, the
operating-domain flags and the bistability check.

    python validation/synthetic/tables_final.py RUNDIR OUTDIR

RUNDIR: the output of rerun_final.sh, or of merge_rerun.py.
"""
import glob
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(S))
DIRS = {"v5_bias": "wavelet_matrix_v5_bias", "v5_noise": "wavelet_matrix_v5_noise", "v6": "wavelet_matrix_v6",
        "shallow_v5": "wavelet_matrix_shallow_v5", "z_v5": "wavelet_matrix_z_v5", "shallow_chain": "wavelet_matrix_shallow_chain"}


def main():
    run, out = sys.argv[1], sys.argv[2]
    os.makedirs(out)
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(run, "results_*_shard*.jsonl"))) for l in open(f)]
    by = {}
    for r in R:
        by.setdefault(r["pass_name"], []).append(r)
    for p, L in by.items():
        os.makedirs(os.path.join(out, DIRS[p]))
        with open(os.path.join(out, DIRS[p], "results.jsonl"), "w") as fh:
            for r in L:
                fh.write(json.dumps(r) + "\n")
    env = dict(os.environ, PYTHONPATH=REPO)
    for name, cmd in (("tables_v5.txt", [S + "/tables_v5.py"]), ("compare_v5_v6.txt", [S + "/compare_v5_v6.py"]),
                      ("tables_v5_extra.txt", [S + "/tables_v5_extra.py", "wavelet_matrix_shallow_v5", "wavelet_matrix_z_v5"]),
                      ("tables_shallow_chain.txt", [S + "/tables_v5_extra.py", "wavelet_matrix_shallow_chain"])):
        with open(os.path.join(out, name), "w") as fh:
            subprocess.run([sys.executable] + cmd, cwd=out, stdout=fh, stderr=subprocess.STDOUT, env=env)
    # what the paper's scripts cannot see
    lines = ["# Synthetic matrices, final recipe: chain failures counted as failures (`%s`)\n" % run,
             "records per pass: " + ", ".join("%s %d" % (p, len(L)) for p, L in sorted(by.items())) + "\n"]
    for p in ("v6", "shallow_chain"):
        L = [r for r in by.get(p, []) if (r.get("t0_bias_ms") or 0.0) == 0.0]
        dom = [r for r in L if r.get("D_m", 0) >= 220 and not str(r["run_id"]).startswith("ref_")] if p == "v6" else L
        for lab, G in (("evaluation domain (D >= 220 m, no ref_)" if p == "v6" else "all", dom),):
            n = len(G); pf = [r for r in G if r.get("pick_failed")]
            notok = [r for r in G if not r.get("ok")]
            sc = [r for r in G if r.get("ok") and not r.get("pick_failed") and not r.get("origin_off_wavelet")
                  and r.get("ncc_shape") is not None and r["ncc_shape"] == r["ncc_shape"]]
            v = np.array([r["ncc_shape"] for r in sc])
            fail = n - len(sc)
            lines.append("## %s, %s\n" % (p, lab))
            lines.append("- records %d; chain (pipeline) failed %d (run_matrix_v6 then used the true seafloor: counted as failures); "
                         "not ok %d" % (n, len(pf), len(notok)))
            lines.append("- evaluable %d/%d, median NCC %s, mean %s; failures %d (%.1f %%)" % (
                len(sc), n, "%.3f" % np.median(v) if v.size else "-", "%.3f +- %.3f" % (v.mean(), v.std()) if v.size else "-",
                fail, 100 * fail / max(n, 1)))
            ok = [r for r in G if "h3_applicable" in r]
            lines.append("- onset tracking not applicable to the whole record: %d of %d (%.0f %%): %s" % (
                sum(not r["h3_applicable"] for r in ok), len(ok), 100 * sum(not r["h3_applicable"] for r in ok) / max(len(ok), 1),
                dict(Counter(r["h3_fallback_reason"] for r in ok if not r["h3_applicable"]))))
            lines.append("- operating domain: %s" % dict(Counter("%s (%s)" % (r["domain_flag"], r.get("domain_active")) for r in ok)))
            lines.append("- pipeline failures by axis: %s\n" % dict(Counter(r["axis"] for r in pf)))
    for p in ("v5_bias", "v5_noise", "v6", "shallow_v5", "z_v5", "shallow_chain"):
        L = [r for r in by.get(p, []) if r.get("ok") and "bistable" in r]
        if L:
            lines.append("- %s: bistability flagged on %d of %d primaries (20-shot records: %s partial-stack blocks)" % (
                p, sum(r["bistable"] for r in L), len(L), sorted({r["bistab_blocks"] for r in L})))
    open(os.path.join(out, "summary_final.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
