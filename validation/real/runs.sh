#!/usr/bin/env bash
# Every extraction on the real lines behind the paper, on the trackings of
# lines.sh, with --recipe final:
#
#   viking/        the 41 selections of Table 3 (families A-G, the names of
#                  validation/viking/viking_rerun_conv.sh)
#   viking_nch*/   the C combs stacking the 4, 8 and 16 nearest channels
#   pert/<name>/   the A blocks and the C combs on ten perturbed copies of the
#                  Viking tracking (make_perturbed.py)
#   archive/       the cells of the span experiment on IT90-65, SISOC-MEPA and
#                  MS-29 (stretches of 100 to 600 shots, each sampled by a comb
#                  of 100, at most eight per size)
#
#   DATA=DIR JOBS=12 bash validation/real/runs.sh LINES OUT
#
# LINES is the output of lines.sh; OUT must not exist.  About 350 runs of a
# few seconds to a minute each.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
D=${DATA:-.}; S=$D/span_experiment; JOBS=${JOBS:-12}
LINES=${1:?the output directory of lines.sh}; OUT=${2:?name a new output directory}
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 64; }
for L in viking it065 mepa ms29; do
  [ -e "$LINES/$L/track.npz" ] || { echo "missing: $LINES/$L/track.npz" >&2; exit 64; }
done
mkdir -p "$OUT"; J=$OUT/jobs.txt; : > "$J"

# ------------------------------------------------------------------ Viking
VK="--segy $D/seismic.sgy --near 262 --dx 25 --nshot 1012 --load_cube $D/viking_pick/viking_cube.npy
    --n_ch_track 3 --stretch_semb 0.15 --recipe final --dxsp 25"
VK=$(echo $VK)
add () {  # add <dir> <name> <first> <last> <step> [flags]
  echo "python3 -m seaweed.run_real $VK --load_track $1/track.npz --sp_first $3 --sp_last $4 --sp_step $5 ${6:-} --out $1/$2 > $1/$2.log 2>&1" >> "$J"
}
families () {  # families <dir>: the 41 selections of Table 3
  local V=$1 i n L
  for i in $(seq 0 9); do add $V A_n100_L2.5km_blk$i $((100*i)) $((100*i+100)) 1; done
  for i in $(seq 0 4); do add $V D_n100_L12.4km_comb$i $i 500 5; done
  for i in $(seq 0 4); do add $V D_n100_L12.4km_comb$((i+5)) $((500+i)) 1000 5; done
  for i in $(seq 0 9); do add $V C_n100_L25km_comb$i $i 1000 10; done
  for i in $(seq 0 4); do add $V B_n200_L5km_blk$i $((200*i)) $((200*i+200)) 1; done
  add $V E_n500_L12.5km_blk0 0 500 1; add $V E_n500_L12.5km_blk1 500 1000 1
  add $V F_n1001_L25km_full 0 1001 1
  for n in 100 200 500; do
    case $n in 100) L=2.5;; 200) L=5;; 500) L=12.5;; esac
    echo "python3 -m seaweed.run_real $VK --load_track $V/track.npz --auto_select $n --out $V/G_n${n}_L${L}km_auto > $V/G_n${n}_L${L}km_auto.log 2>&1" >> "$J"
  done
}
TR=$(cd "$LINES/viking" && pwd)/track.npz
mkdir "$OUT/viking"; ln -s "$TR" "$OUT/viking/track.npz"; families "$OUT/viking"
for nc in 4 8 16; do
  N=$OUT/viking_nch$nc; mkdir "$N"; ln -s "$TR" "$N/track.npz"
  for i in $(seq 0 9); do add $N C_n100_L25km_comb$i $i 1000 10 "--n_ch $nc"; done
done
mkdir "$OUT/pert"
python3 "$HERE/make_perturbed.py" "$TR" "$OUT/pert/" > "$OUT/make_perturbed.log" 2>&1 || { cat "$OUT/make_perturbed.log"; exit 1; }
for P in "$OUT"/pert/*/; do P=${P%/}
  for i in $(seq 0 9); do add $P A_n100_L2.5km_blk$i $((100*i)) $((100*i+100)) 1; add $P C_n100_L25km_comb$i $i 1000 10; done
done

# ---------------------------------------------------------------- archive
A=$OUT/archive; mkdir "$A"
cells () {  # cells <line> <shots> <shot spacing, m> <cell sizes> <flags>
  local L=$1 N=$2 DXSP=$3 CS=$4 F="$5 --recipe final --dxsp $3 --load_track $LINES/$1/track.npz"
  local CC step nstr use span k first tag
  for CC in $CS; do step=$((CC/100)) nstr=$((N/CC)); use=$((nstr<8?nstr:8)) span=$(((CC-1)*DXSP/1000))
    for ((k=0;k<use;k++)); do first=$((k*CC))
      tag=$(printf "%s_n100_L%03dkm_S%04d_k%02d" $L $span $CC $k)
      echo "python3 -m seaweed.run_real $F --sp_first $first --sp_last $((first+CC)) --sp_step $step --out $A/$tag > $A/$tag.log 2>&1" >> "$J"
    done; done
}
cells it065 3051 25 "100 200 400 600" "--segy $D/SGY/IT065_Raw_geom_100_3150.sgy --near 125 --dx 25 --nshot 3051 --n_ch_direct 40 --n_ch_track 3 --load_cube $S/it065_cube.npy"
cells mepa 1932 25 "100 200 300" "--segy $D/SGY/MEPA_RAW_geom_100_2031_filt.sgy --near 50 --dx 12.5 --nshot 1932 --load_cube $S/mepa_cube.npy"
cells ms29 800 200 "100 200" "--segy $D/SGY/MS29_SP300_1157.sgy --near 300 --dx 100 --nshot 800 --n_anchor 20 --n_ch_track 3 --load_cube $S/ms29_cube.npy"

echo "$(wc -l < "$J") runs"
xargs -P "$JOBS" -I CMD bash -c CMD < "$J"
n_fail=$(grep -L "^figure: " "$OUT"/*/*.log "$OUT"/pert/*/*.log 2>/dev/null | grep -v make_perturbed | wc -l)
echo "$n_fail runs without a figure line"
[ "$n_fail" = 0 ]
