#!/usr/bin/env python3
"""seafloor_pipeline -- the final recipe in one command: stage 1, then the
onset tracking (h3_track, variant h2), then the extraction with the two-stage
alignment and the bistability check, t0_shift 0 and n_group from the 500-m
rule.

  0. first pass  run_real with the semblance anchors (--n_anchor), for dt, the
                 offsets and the water velocity; skipped with --first_pass or
                 --load_anchors
  1. anchors     anchors_from_consensus --group G --tol_ms 120 --max_shots 48,
                 G = min(max(10, round(2500 m / dxsp)), floor(N / 4)) unless
                 --group is given; skipped with --load_anchors
  2. stage 1     run_real --load_anchors, n_group = max(5, round(500 m / dxsp)),
                 --t0_shift 0, saved as stage1_track.npz
  3. onset       anchor_qc (T_dom), then h3_track --variant h2 on the stage-1
                 tracking and anchors; where the onset tracking is not
                 applicable the stage-1 tracking stands, flagged (h3_track); the
                 operating domain (separation from the direct >= max(5 T_dom,
                 end of the direct)) is flagged
  4. extraction  run_real --load_track track.npz --recipe final on the whole line

Every step writes its log in OUTDIR, which must not exist.  The flags this
script does not know (--stretch_semb, --n_ch_direct, ...) are handed to every
run_real call.  Cells, combs and blocks are extracted on the same tracking with

    python -m seaweed.run_real <line flags> --load_track OUTDIR/track.npz \\
        --recipe final --dxsp DXSP --sp_first ... --sp_last ... --sp_step ...

    python -m seaweed.seafloor_pipeline OUTDIR --segy LINE.sgy --near 262 --dx 25 \\
        --nshot 1012 --dxsp 25 [--group G] [--load_cube CUBE]

OUTDIR comes first: with it last, the value of a flag this script does not
know (--near 262) would be taken as OUTDIR.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from seaweed import code_version
from seaweed.h3_track import domain_line, validity_domain
from seaweed.run_real import n_group_rule

L_GROUP = 2500.0          # m: the --group rule


def group_rule(dxsp, nshot):
    """Shots per stage-1 group: 2500 m, at least 10 shots, at least 4 groups."""
    return min(max(10, int(round(L_GROUP / dxsp))), nshot // 4)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(name, cmd, outdir):
    """Run one step as a module of this package, its output in OUTDIR/<name>.log."""
    log = os.path.join(outdir, name + ".log")
    print(f"[{name}] " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=ROOT + (os.pathsep + os.environ["PYTHONPATH"]
                                               if os.environ.get("PYTHONPATH") else ""))
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, "-m"] + cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise SystemExit(f"step {name} failed (exit {rc}); see {log}")


def main():
    # no abbreviations: --dx of run_real would be taken for --dxsp, --n_ch for --n_ch_track
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("outdir")
    ap.add_argument("--segy", required=True)
    ap.add_argument("--nshot", type=int, required=True)
    ap.add_argument("--dxsp", type=float, required=True, help="shot spacing, m")
    ap.add_argument("--group", type=int, default=None,
                    help="shots per group of the stage-1 anchors (anchors_from_consensus --group); "
                         "default min(max(10, round(2500 m / dxsp)), floor(nshot / 4))")
    ap.add_argument("--load_cube", default=None, help="cube of run_real --save_cube; otherwise built in OUTDIR")
    ap.add_argument("--first_pass", default=None,
                    help="an existing first-pass tracking (dt, offsets, vw only) instead of step 0")
    ap.add_argument("--load_anchors", default=None, help="stage-1 anchors instead of steps 0 and 1")
    ap.add_argument("--n_ch_track", type=int, default=3)
    a, line = ap.parse_known_args()
    for k in ("--n_group", "--t0_shift", "--align", "--bistab_check", "--recipe", "--load_track",
              "--save_track", "--save_cube", "--out", "--dxsp"):
        if k in line:
            ap.error(f"{k} is set by the recipe and cannot be given here")
    if a.load_anchors is None and a.group is None:
        a.group = group_rule(a.dxsp, a.nshot)
        print(f"--group {a.group} from the rule (2500 m / {a.dxsp:g} m, 10 to nshot / 4)")
    if os.path.exists(a.outdir):
        raise SystemExit(f"{a.outdir} exists: every run goes in a new directory")
    os.makedirs(a.outdir)
    O = lambda f: os.path.join(a.outdir, f)
    ng = n_group_rule(a.dxsp)
    base = ["--segy", a.segy, "--nshot", str(a.nshot), "--n_ch_track", str(a.n_ch_track)] + line
    cube = a.load_cube or O("cube.npy")
    if (a.first_pass or a.load_anchors) and not a.load_cube:
        ap.error("--first_pass and --load_anchors skip step 0, which builds the cube: give --load_cube")

    if a.load_anchors:
        anchors = a.load_anchors
    else:
        first = a.first_pass
        if first is None:
            first = O("first_pass_track.npz")
            step("first_pass", ["seaweed.run_real"] + base + ["--n_group", str(ng), "--t0_shift", "0",
                 "--save_track", first, "--out", O("first_pass")]
                 + (["--load_cube", cube] if a.load_cube else ["--save_cube", cube]), a.outdir)
        anchors = O("stage1_anchors.npz")
        step("stage1_anchors", ["seaweed.anchors_from_consensus", cube, first, anchors, "--group", str(a.group),
                                "--tol_ms", "120", "--max_shots", "48"], a.outdir)
    lc = ["--load_cube", cube]
    step("stage1_track", ["seaweed.run_real"] + base + lc + ["--n_group", str(ng), "--t0_shift", "0",
         "--load_anchors", anchors, "--save_track", O("stage1_track.npz"), "--out", O("stage1_full_line")], a.outdir)
    step("anchor_qc", ["seaweed.anchor_qc", cube, O("stage1_track.npz"), anchors, O("qc.npz")], a.outdir)
    step("h3_track", ["seaweed.h3_track", cube, O("stage1_track.npz"), anchors, O("track.npz"), "--dxsp", str(a.dxsp),
                      "--variant", "h2", "--n_ch_track", str(a.n_ch_track), "--n_group", str(ng)], a.outdir)
    step("full", ["seaweed.run_real"] + base + lc + ["--load_track", O("track.npz"), "--recipe", "final",
                                                     "--dxsp", str(a.dxsp), "--out", O("full")], a.outdir)

    za = np.load(anchors, allow_pickle=True)
    s1p = json.loads(str(za["params"])) if "params" in za else {}
    zt = np.load(O("track.npz"), allow_pickle=True)
    tp = json.loads(str(zt["params"]))
    dom = tp["domain"]
    if not np.isfinite(dom["t_dom_ms"]):             # h3_track could not measure it: anchor_qc's
        tq = json.loads(str(np.load(O("qc.npz"), allow_pickle=True)["params"]))["t_dom_ms"]
        dom = validity_domain(zt["t0"], float(np.asarray(zt["offs"])[0]), float(zt["vw"]),
                              (tq if tq is not None else np.nan) * 1e-3, dom["direct_ms"], source="anchor_qc",
                              anchors=(za["i"], za["t"]),
                              first_onset=np.array([np.nan if x is None else x for x in dom["first_energy_s"]]),
                              autocorr_t0=np.array([np.nan if x is None else x for x in dom["autocorr_t0_s"]]))
    sel = np.load(O("full_sel.npz"), allow_pickle=True)
    summary = dict(script="seafloor_pipeline", code=code_version(), recipe="final", variant="h2",
                   dxsp=a.dxsp, n_group=ng, group=a.group, load_anchors=a.load_anchors, first_pass=a.first_pass,
                   cube=cube, line_flags=line,
                   h3_applicable=tp["applicable"], h3_fallback=tp["fallback"],
                   h3_fallback_reason=tp["fallback_reason"], h3_unreliable_fraction=tp["unreliable_fraction"],
                   h3_unreliable_stretches=tp["unreliable_stretches"], t_dom_ms=tp.get("t_dom_ms"),
                   align=str(sel["align"]), bistable=bool(sel["bistable"]), domain=dom,
                   multiple_available=bool(sel["multiple_available"]), multiple_reason=str(sel["multiple_reason"]),
                   stage1_fallback=s1p.get("stage1_fallback", "none"), stage1_n_kept=s1p.get("n_kept"),
                   stage1_single_group=s1p.get("single_group"))
    summary["domain"] = {k: v for k, v in dom.items() if k not in ("first_energy_s", "autocorr_t0_s")}  # per shot: track.npz
    with open(O("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(("ONSET TRACKING NOT APPLICABLE ON THIS UNIT (%s): stage-1 tracking used" % tp["fallback_reason"])
          if not tp["applicable"] else
          "onset tracking applicable; fallback on %.1f%% of the line" % (100 * tp["unreliable_fraction"]))
    print(domain_line(dom))
    if summary["stage1_fallback"] != "none":
        print("STAGE 1 FALLBACK: %s%s" % (summary["stage1_fallback"], " (constant drift along the whole line: the "
              "seafloor slope is ignored)" if summary["stage1_fallback"] == "single group" else
              " (%s anchors)" % summary["stage1_n_kept"]))
    if not summary["multiple_available"]:
        print(f"MULTIPLE NOT AVAILABLE ({summary['multiple_reason']}): multiple-based diagnostics not available")
    print(f"tracking {O('track.npz')}, wavelet {O('full_wavelet.npy')}, summary {O('summary.json')}")


if __name__ == "__main__":
    main()
