#!/usr/bin/env bash
# Re-run every Viking selection of the paper on ONE tracking, the one the
# stage-1 (consensus) anchors produce.  After this the automatic-chain score and
# the fold/span experiments share the same seafloor, so only 0.823 remains to be
# quoted and the 0.827 family disappears.
#
# Step 1 runs the workflow once with the consensus anchors and saves the track.
# Steps 2 and 3 reuse that track (stages 0 to 2 are skipped), so the shot
# selection is the only thing that varies between the 41 runs.
#
#   bash validation/viking/viking_rerun_conv.sh            # everything
#   JOBS=4 bash validation/viking/viking_rerun_conv.sh     # fewer parallel runs
set -euo pipefail
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

SEGY=seismic.sgy
CUBE=viking_pick/viking_cube.npy
ANCH=viking_pick/viking_anchors_conv.npz   # the consensus anchors behind 0.823
OUT=viking_conv41
TRACK=$OUT/viking_track_conv.npz
JOBS=${JOBS:-6}

COMMON="--segy $SEGY --near 262 --dx 25 --nshot 1012 --load_cube $CUBE \
        --n_group 20 --stretch_semb 0.15 --t0_shift 0"

for f in "$SEGY" "$CUBE" "$ANCH" "$REPO/seaweed/run_real.py"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done
echo "anchors: $ANCH  ($(md5sum "$ANCH" | cut -c1-8))"
if [ -e /tmp/viking_48.npz ]; then
  echo "/tmp/viking_48.npz: $(md5sum /tmp/viking_48.npz | cut -c1-8)  \
(must match, or 0.823 was measured on other anchors)"
fi
mkdir -p "$OUT"

# ---------------------------------------------------------------- 1. tracking
if [ ! -e "$TRACK" ]; then
  echo "[1] tracking from the consensus anchors"
  python3 -m seaweed.run_real $COMMON --load_anchors "$ANCH" --save_track "$TRACK" \
      --sp_first 0 --sp_last 1000 --sp_step 10 \
      --out "$OUT/track_seed" > "$OUT/track_seed.log" 2>&1
  grep -E "anchors read from|tracked seafloor|PRIMARY" "$OUT/track_seed.log"
else
  echo "[1] tracking already present: $TRACK"
fi

# ------------------------------------------------------------- 2. job list
JOBFILE=$OUT/jobs.txt
: > "$JOBFILE"
add () {   # add <name> <sp_first> <sp_last> <sp_step>; sp_last is EXCLUSIVE
  echo "python3 -m seaweed.run_real $COMMON --load_track $TRACK \
--sp_first $2 --sp_last $3 --sp_step $4 --out $OUT/$1 > $OUT/$1.log 2>&1" >> "$JOBFILE"
}
add_auto () {  # add_auto <name> <fold>
  echo "python3 -m seaweed.run_real $COMMON --load_track $TRACK --auto_select $2 \
--out $OUT/$1 > $OUT/$1.log 2>&1" >> "$JOBFILE"
}

# fold 100, contiguous blocks, span 2.5 km (10 runs)
for i in $(seq 0 9); do add "A_n100_L2.5km_blk$i" $((100*i)) $((100*i+100)) 1; done

# fold 100, every 5th shot, span 12.4 km (10 runs: five in each half)
# NB: the family letters are the ones make_paper_figures_vik.py and
# viking_phase_test.py expect -- D is the 12.4 km comb, B the 200-shot block.
for i in $(seq 0 4); do add "D_n100_L12.4km_comb$i" "$i" 500 5; done
for i in $(seq 0 4); do add "D_n100_L12.4km_comb$((i+5))" $((500+i)) 1000 5; done

# fold 100, every 10th shot, span 24.8 km (10 runs) -- the automatic-chain family
for i in $(seq 0 9); do add "C_n100_L25km_comb$i" "$i" 1000 10; done

# fold 200, contiguous, span 5 km (5 runs)
for i in $(seq 0 4); do add "B_n200_L5km_blk$i" $((200*i)) $((200*i+200)) 1; done

# fold 500, contiguous, span 12.5 km (2 runs)
add "E_n500_L12.5km_blk0"   0 500 1
add "E_n500_L12.5km_blk1" 500 1000 1

# whole line (1 run)
add "F_n1001_L25km_full" 0 1001 1

# coherence-selected blocks (3 runs); the span is what that fold covers
add_auto "G_n100_L2.5km_auto"   100
add_auto "G_n200_L5km_auto"     200
add_auto "G_n500_L12.5km_auto"  500

echo "[2] $(wc -l < "$JOBFILE") runs queued, $JOBS at a time"
xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"

# ------------------------------------------------------------- 3. summary
echo
echo "[3] per-run summary (paste into the table check)"
for f in "$OUT"/[A-G]_*.log; do
  n=$(basename "$f" .log)
  printf "%-26s %s | %s\n" "$n" \
    "$(grep -m1 'extraction on' "$f" | sed 's/^ *//')" \
    "$(grep -m1 'PRIMARY' "$f" | sed 's/.*| //')"
done
echo
echo "check that every run shows the SAME tracked seafloor:"
grep -h "tracked seafloor" "$OUT"/[A-G]_*.log | sort -u

cat <<'NOTE'

Next, per family:
  python3 validation/viking/viking_phase_test.py FarField.dat viking_conv41 --fam A
  ... and the same for B, C, D, E, F, G
The family letter must match the regex in viking_phase_test.py and in
make_paper_figures_vik.py:  grep -n "PAT" validation/viking/viking_phase_test.py
If the pattern differs, rename the --out prefixes above instead of editing the
analysis scripts.
NOTE
