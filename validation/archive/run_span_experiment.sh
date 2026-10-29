#!/bin/bash
# =============================================================================
#  How much line does the extraction need?
#
#  Viking established that fold is irrelevant and span is not, but two things
#  it could not settle:
#
#   (i) where the improvement stops -- its line is only 25 km;
#  (ii) whether the requirement is a fixed number of kilometres or scales with
#       the DURATION of the wavelet, which sets how much sub-bottom reflectivity
#       falls inside the extraction window and contaminates it coherently.
#
#  The four wavelets measured so far differ by a factor five in duration:
#
#      MEPA      26 ms   (bubble suppressed, ~1.5 periods)
#      MS29      76 ms   (18 Hz, ~1.4 periods)
#      IT90-65   90 ms   (~3.4 periods)
#      Viking   132 ms   (long bubble train, ~5 periods)
#
#  so plotting the agreement against span/duration rather than against span is a
#  test with a definite outcome: if the three surveys collapse onto one curve
#  the scale is the wavelet duration, and if they stay apart it is something
#  else, most likely the geology.
#
#  DESIGN.  Fold is 100 throughout.  A cell is a STRETCH of line of a given
#  length, sampled by one comb of 100 shots; the realisations of a cell are
#  DIFFERENT, non-overlapping stretches.  This is the correction that the Viking
#  calibration forced: subsets interleaved over one stretch agree at 0.98-1.00
#  whatever the span, because they see the same geology a few metres apart, and
#  that number says nothing about convergence.  Only agreement between different
#  stretches does.
#
#  Requiring at least five disjoint stretches caps the longest span at about a
#  fifth of each line.  62 runs in all; the cells overlap in span/duration
#  (IT90-65 0.03-0.17, MEPA 0.10-0.29, MS29 0.26-0.52) which is what makes the
#  collapse testable.
#
#  Uses the block and cube options of run_real (--sp_first, --save_cube).
# =============================================================================
set -eu

# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
ROOT=${ROOT:-$(pwd)}                # the directory that holds the data
OUT=$ROOT/span_experiment
FOLD=100
MAXREP=8                      # disjoint stretches used per cell
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
#  Replace the flags with those used for the published extraction of each line.
#  DXSP is the SHOT-POINT spacing (used for the span); --dx is the CHANNEL
#  spacing.  DUR is the measured wavelet duration, used only for the labels.
# ---------------------------------------------------------------------------
declare -A SEGY FLAGS NSP DXSP DUR CELLS DTW NEAR VWG LAG ROUTE
# The published runs used these flags.  Everything is kept except
# --auto_select, which chose the block and is exactly what this experiment
# replaces with an explicit selection.  --t0_shift is NOT optional: it is
# per-dataset tuning and without it the extraction window sits in the wrong
# place.
#
# --n_anchor SCALES WITH THE LINE.  The published runs put 40 anchors on
# 200-300 shots, one every five to seven; carrying 40 over to 3051 shots would
# be one every 76 and the drift would be smoothed through far too few points.
# The values below keep roughly the published density, which makes the tracking
# slower, but it runs once per survey.
#
# The SEG-Y files below are the full lines.  For SISOC-MEPA the full line is
# the filtered one (_filt), filtered as the extraction of the paper.

SEGY[it065]=$ROOT/SGY/IT065_Raw_geom_100_3150.sgy                       # <-- 3051 SP, not the SP1000-1300 subset
FLAGS[it065]="--near 125 --dx 25 --nshot 3051 --t0_shift -25 \
              --n_group 9 --n_ch_track 5 --n_ch_direct 40"
NSP[it065]=3051 ; DXSP[it065]=25  ; DUR[it065]=90 ; DTW[it065]=0.002
NEAR[it065]=125 ; VWG[it065]=1453 ; LAG[it065]="0.45 1.30" ; ROUTE[it065]=pick
CELLS[it065]="100 200 400 600"

SEGY[mepa]=$ROOT/SGY/MEPA_RAW_geom_100_2031_filt.sgy                    # <-- 1932 SP, same filtering
FLAGS[mepa]="--near 50 --dx 12.5 --nshot 1932 --t0_shift -25 --n_group 20"
NSP[mepa]=1932  ; DXSP[mepa]=25   ; DUR[mepa]=26  ; DTW[mepa]=0.002
NEAR[mepa]=50   ; VWG[mepa]=1515  ; LAG[mepa]="0.70 1.20" ; ROUTE[mepa]=pick   # <-- check LAG against the section
CELLS[mepa]="100 200 300"

# MS29 already runs on the full line, so this one needs no new file.
# --ref is kept: it costs nothing and gives one more comparison per run against
# a fixed reference, which is a useful consistency check across the cells.
SEGY[ms29]=$ROOT/SGY/MS29_SP300_1157.sgy
FLAGS[ms29]="--near 300 --dx 100 --nshot 800 --t0_shift -30 \
             --n_anchor 20 --n_group 9 --n_ch_track 5 \
             --ref $ROOT/w_ms29_new.npy --ref_dt 0.004"
NSP[ms29]=800   ; DXSP[ms29]=200  ; DUR[ms29]=76  ; DTW[ms29]=0.004
NEAR[ms29]=300  ; VWG[ms29]=1515 ; LAG[ms29]="0.80 1.60" ; ROUTE[ms29]=semblance
CELLS[ms29]="100 200"

# Two modes.  "prep" builds the cube, picks the seafloor and tracks the line;
# it stops there, because the pick has to be LOOKED AT before anything is built
# on top of it.  "cells" runs the experiment against the tracking that prep
# produced.
MODE=${1:-cells}
shift || true

for S in ${*:-"it065 mepa ms29"}; do
  seg=${SEGY[$S]}; fl=${FLAGS[$S]}; n=${NSP[$S]}; dx=${DXSP[$S]}
  cube=$OUT/${S}_cube.npy ; track=$OUT/${S}_track.npz
  pick=$OUT/${S}_pick.npz ; anch=$OUT/${S}_anchors.npz

  echo; echo "######## $S : $n SP x ${dx} m = $(( n*dx/1000 )) km ########"

  if [ "$MODE" = "prep" ]; then
    # 1. cache the gather.  The tracking of this first run is expected to be
    #    poor on a long line -- one global D_init cannot centre the semblance
    #    window over hundreds of metres of relief -- and is thrown away.
    [ -f "$cube" ] || python3 -m seaweed.run_real --segy "$seg" $fl \
        --save_cube "$cube" --out "$OUT/${S}_probe"

    # 2-3. produce the tracking.  Two routes, ONE criterion for choosing:
    #      does the seafloor relief fit inside the semblance search window?
    #      That window is 2*D_init/vw +- 300 m, about 800 ms wide in two-way
    #      time, centred on a SINGLE initial depth for the whole line.  Where
    #      the relief approaches its width, or D_init is off centre, anchors
    #      are lost against the edge and the tracking is left unconstrained
    #      there.  The log says which case you are in:
    #
    #         "anchors rejected: N at the edge of the search window"
    #
    #      N ~ 0  -> ROUTE=semblance, the published procedure, unchanged.
    #      N large -> ROUTE=pick: the anchors come instead from a LOCAL
    #      autocorrelation, which needs no initial depth.  Everything after the
    #      anchors -- the 2D pilot cross-correlation, the window rejection, the
    #      drift smoothing, the t0_shift -- is identical either way.
    #
    #      Measured: IT90-65 has 731 ms of relief and lost 44 anchors of 392;
    #      MS29 has 528 ms and lost 2 of 18.
    if [ "${ROUTE[$S]}" = "pick" ]; then
      python3 -m seaweed.pick_seafloor --cube "$cube" \
          --near ${NEAR[$S]} --vw ${VWG[$S]} --dxsp $dx --group_m 800 \
          --lag ${LAG[$S]} --semblance check --dx $(echo $fl | sed -n 's/.*--dx \([0-9.]*\).*/\1/p') \
          --out "$pick" --save_anchors "$anch" --n_anchor_out 400
      python3 -m seaweed.run_real --segy "$seg" $fl \
          --load_cube "$cube" --load_anchors "$anch" \
          --save_track "$track" --out "$OUT/${S}_reference_full_line"
    else
      python3 -m seaweed.run_real --segy "$seg" $fl \
          --load_cube "$cube" \
          --save_track "$track" --out "$OUT/${S}_reference_full_line"
    fi

    echo
    echo "  LOOK AT THESE BEFORE RUNNING THE CELLS:"
    echo "    $OUT/${S}_pick_pick.png        red on the seafloor, yellow on the multiple"
    echo "    $OUT/${S}_reference_full_line.png"
    echo "  and in the log above: 'drift relative to anchors' should be tens of ms,"
    echo "  not hundreds, and the coherence at [4] should not collapse anywhere."
    continue
  fi

  if [ ! -f "$track" ]; then
    echo "  no tracking for $S: run  bash $0 prep $S  first"; continue
  fi

  for L in ${CELLS[$S]}; do
    step=$(( L / FOLD ))
    nstr=$(( n / L ))
    span=$(( (L - 1) * dx / 1000 ))
    use=$(( nstr < MAXREP ? nstr : MAXREP ))
    echo "=== $S : stretch $L SP (span ${span} km, step $step), $use of $nstr stretches ==="
    for (( k=0; k<use; k++ )); do
      first=$(( k * L ))
      last=$(( first + L ))
      tag=$(printf "%s_n%d_L%03dkm_S%04d_k%02d" "$S" "$FOLD" "$span" "$L" "$k")
      [ -f "$OUT/${tag}_wavelet.npy" ] && { echo "  skip $tag"; continue; }
      echo "  --- $tag : shots $first-$((last-1)) step $step ---"
      python3 -m seaweed.run_real --segy "$seg" $fl \
          --load_cube "$cube" --load_track "$track" \
          --sp_first "$first" --sp_last "$last" --sp_step "$step" \
          --out "$OUT/$tag"
    done
  done

  echo
  echo "read the curve for $S with:"
  for L in ${CELLS[$S]}; do
    span=$(( (L - 1) * dx / 1000 ))
    echo "  python3 validation/archive/wavelet_spread.py $OUT/${S}_n${FOLD}_L$(printf %03d $span)km_S$(printf %04d $L)_*_wavelet.npy --dt ${DTW[$S]} --dxsp $dx"
  done
done
