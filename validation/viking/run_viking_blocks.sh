#!/bin/bash
# =============================================================================
#  Viking Graben line 12 - de-confounding experiment
#
#  THE THREE EXISTING RUNS ARE NOT NESTED SUBSETS.  They were produced with
#  --auto_select, which extracts over the whole line, measures the per-shot
#  stack coherence, and then re-extracts on the contiguous block of the
#  requested length where that coherence is HIGHEST.  So the 100-shot run is
#  not "the first 100 shots": it is the 100 shots whose windows agree most with
#  each other, out of 1012.
#
#  Three things therefore vary together in those runs: fold, the length of line
#  spanned, and a selection made on coherence.  The last one is not neutral --
#  the windows agree most where the geology changes least, which is precisely
#  where the contamination is coherent from shot to shot and survives the
#  stack.  The selection may be working against accuracy.
#
#  This script separates them.  Three cells at CONSTANT FOLD (100 shots) and
#  BLIND selection, differing only in how far apart the stacked shots are:
#
#     family A   100 shots, contiguous     spans  2.5 km    x10 disjoint
#     family D   100 shots, every 5th      spans 12.4 km    x10 disjoint
#     family C   100 shots, every 10th     spans 24.8 km    x10 disjoint
#
#  A, D and C use the same shots and put the same number of windows in the
#  stack, so any difference between them is the LINE-LENGTH effect: the
#  contamination decorrelates with distance, while the stack does not care how
#  far apart the windows are.  Note that every synthetic line in the paper
#  spans at most 2.4 km, i.e. family A only -- exactly the regime where fold
#  was found to be worth +0.007 +- 0.020.
#
#     family B   fold curve, blind, contiguous: 200 (x5), 500 (x2), 1000 (x1)
#     family Z   --auto_select at 100, 200, 500: what the coherence criterion
#                buys, measured against the blind distributions of A and B
#
#  Every run reuses ONE tracked seafloor, so the shot selection is the only
#  thing that varies.  Uses the --sp_first/--sp_last/--sp_step options of run_real.
# =============================================================================
set -eu

# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ROOT=${ROOT:-$(pwd)}                # the directory that holds the data
SEGY=$ROOT/seismic.sgy               # <-- adjust
OUT=$ROOT/viking_blocks
TRACK=$OUT/track_viking.npz
mkdir -p "$OUT"

# Exactly the flags of the three existing runs, minus --auto_select, which is
# what this experiment is replacing with an explicit selection.
COMMON="--near 262 --dx 25 --nshot 1012 --D_init 375 --t0_shift -10 \
        --n_anchor 60 --n_group 20 --stretch_semb 0.15"

run () {                                     # run <tag> <first> <last> <step>
  local tag=$1 first=$2 last=$3 step=$4
  if [ -f "$OUT/${tag}_wavelet.npy" ]; then
    echo "skip $tag"; return
  fi
  echo "=== $tag : shots $first-$((last-1)) step $step ==="
  python3 -m seaweed.run_real --segy "$SEGY" $COMMON \
      --load_track "$TRACK" \
      --sp_first "$first" --sp_last "$last" --sp_step "$step" \
      --out "$OUT/$tag"
}

auto () {                                    # auto <tag> <nshots>
  local tag=$1 n=$2
  if [ -f "$OUT/${tag}_wavelet.npy" ]; then
    echo "skip $tag"; return
  fi
  echo "=== $tag : --auto_select $n ==="
  python3 -m seaweed.run_real --segy "$SEGY" $COMMON \
      --load_track "$TRACK" --auto_select "$n" --out "$OUT/$tag"
}

# --- 0. track the whole line ONCE -------------------------------------------
# No --auto_select here: this run exists to produce the tracking that all the
# others share, so that the seafloor is identical by construction.
if [ ! -f "$TRACK" ]; then
  echo "=== tracking the whole line (once) ==="
  python3 -m seaweed.run_real --segy "$SEGY" $COMMON \
      --save_track "$TRACK" --out "$OUT/Z_reference_full_line"
fi

# --- family A : fold 100, contiguous, 2.5 km, 10 disjoint -------------------
for i in $(seq 0 9); do
  run "A_n100_L2.5km_blk${i}" $((i*100)) $((i*100+100)) 1
done

# --- family D : fold 100, every 5th, 12.4 km, 10 disjoint -------------------
for h in 0 1; do
  for i in $(seq 0 4); do
    run "D_n100_L12.5km_c${h}${i}" $((h*500+i)) $((h*500+500)) 5
  done
done

# --- family C : fold 100, every 10th, 24.8 km, 10 disjoint ------------------
for i in $(seq 0 9); do
  run "C_n100_L25km_comb${i}" $i 1001 10
done

# --- family B : fold curve, blind, contiguous -------------------------------
for i in $(seq 0 4); do
  run "B_n200_L5km_blk${i}"    $((i*200)) $((i*200+200)) 1
done
for i in 0 1; do
  run "B_n500_L12.5km_blk${i}" $((i*500)) $((i*500+500)) 1
done
run "B_n1000_L25km_blk0" 0 1001 1

# --- family Z : coherence-based selection, for comparison -------------------
auto "Z_n100_L2.5km_auto"  100
auto "Z_n200_L5km_auto"    200
auto "Z_n500_L12.5km_auto" 500

echo
echo "done.  now:"
echo "  python3 validation/viking/analyse_viking_blocks.py $OUT /path/to/FarField.dat"
