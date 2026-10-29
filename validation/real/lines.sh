#!/usr/bin/env bash
# The final recipe on the four real lines of the paper: seaweed.seafloor_pipeline
# on each whole line, with the line flags of the paper.  Every other run on a
# real line (Viking families, archive cells) reuses the tracking written here.
#
#   DATA=DIR bash validation/real/lines.sh OUT
#
# DATA (default: the current directory) holds the inputs under the names of
# validation/real/README.md.  OUT must not exist.  The four lines run in
# parallel; IT90-65 is the longest, about 45 minutes on one core per line.
#
# The cube and the first-pass tracking of each line are those of the first
# version of the workflow (validation/viking/viking_rerun_conv.sh,
# validation/archive/run_span_experiment.sh).  The first pass supplies only the
# sampling interval, the offsets and the water velocity; without it,
# seafloor_pipeline measures them itself (step 0), and the numbers can move in
# the last digits with the water velocity.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
D=${DATA:-.}
OUT=${1:?name a new output directory}
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 64; }
S=$D/span_experiment

# line: flags of the paper; shot spacing; cube; first pass
VIKING="--segy $D/seismic.sgy --near 262 --dx 25 --nshot 1012 --stretch_semb 0.15 --dxsp 25
        --load_cube $D/viking_pick/viking_cube.npy --first_pass $D/viking_conv41/viking_track_conv.npz"
IT065="--segy $D/SGY/IT065_Raw_geom_100_3150.sgy --near 125 --dx 25 --nshot 3051 --n_ch_direct 40 --dxsp 25
       --load_cube $S/it065_cube.npy --first_pass $S/it065_track.npz"
MEPA="--segy $D/SGY/MEPA_RAW_geom_100_2031_filt.sgy --near 50 --dx 12.5 --nshot 1932 --dxsp 25
      --load_cube $S/mepa_cube.npy --first_pass $S/mepa_track.npz"
MS29="--segy $D/SGY/MS29_SP300_1157.sgy --near 300 --dx 100 --nshot 800 --n_anchor 20 --dxsp 200
      --load_cube $S/ms29_cube.npy --first_pass $S/ms29_track.npz"

for f in $VIKING $IT065 $MEPA $MS29; do
  case $f in --*|[0-9]*) ;; *) [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; };; esac
done
mkdir -p "$OUT"
for L in viking it065 mepa ms29; do
  F=${L^^}
  python3 -m seaweed.seafloor_pipeline "$OUT/$L" ${!F} > "$OUT/$L.log" 2>&1 &
done
status=0
for j in $(jobs -p); do wait "$j" || status=1; done
for L in viking it065 mepa ms29; do tail -n 1 "$OUT/$L.log"; done
exit $status
