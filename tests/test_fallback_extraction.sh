#!/usr/bin/env bash
# The final recipe where the onset tracking is not applicable to the whole
# unit: the pipeline
# must not fail, the tracking must be the stage-1 one, flagged with the reason,
# and the extraction must still use --align two_stage --bistab_check.
#
# Two synthetic units on which h3_track h2 is not applicable, one per reason:
# a bench line at 120 m (T_dom not measurable) and a 20-shot record (fewer
# than 2 onset anchors).  GEN holds the cubes (bench/ from
# validation/synthetic/bench_lines.py, records/ from picker_records.py) and
# UNITS the stage-1 anchors of each unit, UNITS/<id>/s1_anchors.npz.
#
#   GEN=DIR UNITS=DIR bash tests/test_fallback_extraction.sh OUTDIR
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
OUT=${1:?name a new output directory}
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 64; }
GEN=${GEN:?the directory of the synthetic cubes}
UNITS=${UNITS:?the directory of the stage-1 anchors}
status=0
for spec in "bench C_R0_D120_s0p00_r2 60 125 t_dom_not_measurable" \
            "records grid_D180_L80_r010 20 25 fewer_than_2_onset_anchors"; do
  set -- $spec; sub=$1 uid=$2 n=$3 dxsp=$4 want=$5
  cube=$GEN/$sub/${uid}_cube.npy anch=$UNITS/$uid/s1_anchors.npz
  for f in "$cube" "$anch"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }; done
  mkdir -p "$OUT"
  python3 -m seaweed.seafloor_pipeline "$OUT/$uid" --segy "$uid" --nshot "$n" --dxsp "$dxsp" \
      --load_cube "$cube" --load_anchors "$anch" > "$OUT/$uid.log" 2>&1
  rc=$?
  python3 - "$OUT/$uid" "$want" "$rc" <<'PY'
import json, sys
import numpy as np
d, want, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = rc == 0
s = json.load(open(d + "/summary.json")) if ok else {}
z = np.load(d + "/track.npz", allow_pickle=True) if ok else None
s1 = np.load(d + "/stage1_track.npz", allow_pickle=True) if ok else None
sel = np.load(d + "/full_sel.npz", allow_pickle=True) if ok else None
log = open(d + "/full.log").read() if ok else ""
checks = [("pipeline exit 0", ok),
          ("onset tracking not applicable, reason %s" % want, ok and s["h3_applicable"] is False and s["h3_fallback"] == "unit"
           and s["h3_fallback_reason"] == want),
          ("tracking = stage-1 tracking", ok and np.array_equal(z["t0"], s1["t0"])),
          ("extraction: align two_stage", ok and str(sel["align"]) == "two_stage"),
          ("extraction: bistab_check on", ok and "bistab_check True" in log and "partial stacks of 50 shots" in log),
          ("flag in the extraction log and _sel.npz", ok and "ONSET TRACKING NOT APPLICABLE ON THIS UNIT" in log
           and bool(sel["h3_applicable"]) is False)]
for name, c in checks:
    print("%-58s %s" % (d.split("/")[-1] + ": " + name, "PASS" if c else "FAIL"))
sys.exit(0 if all(c for _, c in checks) else 1)
PY
  [ $? = 0 ] || status=1
done
# Two records on which the pipeline stopped before the robustness fixes,
# now built from the first pass (stage-1 anchors included): a 5-deg
# dipping record whose multiple is not extracted, and a record on which only 3
# of the 4 groups of 5 shots survive the neighbour check.
for spec in "sec_slope_deg_5p0_r005 multiple" "grid_D180_L80_r010 anchors"; do
  set -- $spec; uid=$1 what=$2
  cube=$GEN/records/${uid}_cube.npy
  [ -e "$cube" ] || { echo "missing: $cube" >&2; exit 64; }
  python3 - "$cube" "$OUT/${uid}_first_pass.npz" <<'PY'
import sys
import numpy as np
from seaweed.wavelet_extractor_v2 import estimate_vw_direct
cube, fp = sys.argv[1], sys.argv[2]
g = np.load(cube, mmap_mode="r"); m = np.load(cube + ".meta.npz", allow_pickle=True)
tr = np.load(cube.replace("_cube.npy", "_truth.npz"), allow_pickle=True)
offs = np.asarray(m["off_hdr"], float); dt = float(m["dt"])
vw = estimate_vw_direct(np.asarray(g), dt, offs, D_guess=float(np.median(tr["D"])))[0]
np.savez(fp, t0=np.zeros(g.shape[0]), vw=vw, dt=dt, offs=offs)
PY
  python3 -m seaweed.seafloor_pipeline "$OUT/${uid}_built" --segy "$uid" --nshot 20 --dxsp 25 \
      --load_cube "$cube" --first_pass "$OUT/${uid}_first_pass.npz" > "$OUT/${uid}_built.log" 2>&1
  rc=$?
  python3 - "$OUT/${uid}_built" "$what" "$rc" <<'PY'
import json, sys
import numpy as np
d, what, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
ok = rc == 0
s = json.load(open(d + "/summary.json")) if ok else {}
checks = [("pipeline exit 0 (it stopped before the fix)", ok),
          ("a wavelet and a tracking", ok and all(__import__("os").path.exists(d + f) for f in ("/full_wavelet.npy", "/track.npz"))),
          ("extraction: align two_stage", ok and s["align"] == "two_stage")]
if what == "multiple":
    sel = np.load(d + "/full_sel.npz", allow_pickle=True) if ok else None
    checks += [("multiple not available, with its reason", ok and s["multiple_available"] is False and s["multiple_reason"] != ""
                and bool(sel["multiple_available"]) is False)]
else:
    checks += [("stage-1 fallback flagged (reduced or single group)", ok and s["stage1_fallback"] in ("reduced", "single group"))]
for name, c in checks:
    print("%-58s %s" % (d.split("/")[-1] + ": " + name, "PASS" if c else "FAIL"))
sys.exit(0 if all(c for _, c in checks) else 1)
PY
  [ $? = 0 ] || status=1
done

echo; [ $status = 0 ] && echo "PASS: the fallback extraction uses the final recipe" || echo "FAIL"
exit $status
