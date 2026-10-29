#!/usr/bin/env bash
# Re-run the ten extractions behind the paper and compare them with what was
# frozen by tests/freeze_reference.sh.
#
# This is the constraint that makes cleaning the repository safe: after every
# modification, the ten whole-line combs of family C must come out of the
# workflow byte for byte as they did when the manuscript was written.  If a
# hash changes, the modification is not cosmetic and has to be looked at again.
#
#   bash tests/smoke_test.sh                 # re-run and compare
#   JOBS=2 bash tests/smoke_test.sh          # fewer runs at a time
#   bash tests/smoke_test.sh --no-rerun DIR  # compare an existing directory
#   bash tests/smoke_test.sh --offline       # no data: compare tests/reference
#                                            # with itself, i.e. check the
#                                            # harness, not the workflow
#   DATA=DIR bash tests/smoke_test.sh        # inputs under DIR instead of
#                                            # the repository root
#   FULL=1 bash tests/smoke_test.sh          # rebuild the tracking from the
#                                            # stage-1 anchors as well, which
#                                            # is the only way stages 0 to 2
#                                            # are exercised at all
#   RECIPE=final bash tests/smoke_test.sh    # the final recipe: the
#                                            # stage-1 tracking from the same
#                                            # anchors, h3_track h2, then the ten
#                                            # combs with --recipe final, against
#                                            # tests/reference_final*.  The paper
#                                            # check above stays the
#                                            # non-regression test.  It also runs
#                                            # the stage-1 picker
#                                            # (anchors_from_consensus) on Viking
#                                            # and compares its anchors, so that a
#                                            # change to the picker shows
#
# Exit codes, meant to be read by a person and by a script alike:
#
#   0  every wavelet is bit-for-bit the one behind the paper
#   2  no wavelet moved measurably, but the bytes differ: almost always a
#      changed numpy, BLAS or thread count.  Explain it before going on.
#   1  a wavelet or a quality-control scalar has moved.  The modification
#      changes a number of the paper.
#
# Stages 0 to 2 are skipped, because the tracking is loaded from the frozen
# file: the ten runs therefore cost minutes, not hours, and the only thing that
# varies between them is the shot selection.
set -uo pipefail

cd "$(dirname "$0")/.."
# python3 -m seaweed.run_real resolves the package from this directory,
# so the test needs no installation.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

TESTS=tests
RECIPE=${RECIPE:-paper}
case "$RECIPE" in
  paper) SUF="" ;;
  final) SUF="_final" ;;
  *) echo "RECIPE is paper or final, not $RECIPE" >&2; exit 64 ;;
esac
REF=$TESTS/reference$SUF
MD5=$TESTS/reference${SUF}_md5.txt
STATS=$TESTS/reference${SUF}_stats.tsv
INPUTS=$TESTS/reference${SUF}_inputs.md5

# Directory that holds the inputs, with the layout of the experiment folder;
# the paths in $INPUTS are relative to it.
DATA=${DATA:-.}
SEGY=${SEGY:-$DATA/seismic.sgy}
CUBE=${CUBE:-$DATA/viking_pick/viking_cube.npy}
ANCH=${ANCH:-$DATA/viking_pick/viking_anchors_conv.npz}
TRACK=${TRACK:-$DATA/viking_conv41/viking_track_conv.npz}
OUT=${OUT:-$TESTS/_smoke_out}
JOBS=${JOBS:-6}
RTOL=${RTOL:-1e-6}
COMBS=${COMBS:-10}                 # lower it for a faster, partial check

RERUN=1
OFFLINE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --no-rerun) RERUN=0; shift; [ $# -gt 0 ] && { OUT=$1; shift; } ;;
    --offline)  OFFLINE=1; RERUN=0; shift ;;
    --out)      OUT=$2; shift 2 ;;
    --jobs)     JOBS=$2; shift 2 ;;
    -h|--help)  sed -n '2,33p' "$0"; exit 0 ;;
    *)          echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

# The reference is the four pieces together, not the hash list alone.  An
# md5 list on its own is what a hand-written `md5sum > tests/reference_md5.txt`
# leaves behind, and it is not enough to compare anything: freeze_reference.sh
# writes the copies and the statistics beside it.
missing=""
[ -e "$MD5" ]   || missing="$missing $MD5"
[ -d "$REF" ]   || missing="$missing $REF/"
[ -e "$STATS" ] || missing="$missing $STATS"
NOREF=0
if [ -n "$missing" ] && [ "$RECIPE" = "final" ] && [ "$RERUN" = "1" ]; then
  # the first time: run, then freeze what came out
  echo "no frozen reference for the final recipe yet (missing:$missing): the runs"
  echo "go ahead, and tests/freeze_reference.sh freezes them afterwards"
  NOREF=1; missing=""
fi
if [ -n "$missing" ]; then
  echo "the frozen reference is incomplete, missing:$missing" >&2
  echo "run tests/freeze_reference.sh on the machine that holds the data" >&2
  exit 64
fi
# A hash list with a directory in the names cannot be checked from inside a run
# directory, which is how this test uses it.
if [ "$NOREF" = "0" ] && grep -qE "^[0-9a-f]{32} +\*?[^ ]*/" "$MD5"; then
  echo "$MD5 holds path names, not bare file names." >&2
  echo "It was not written by tests/freeze_reference.sh; run that script." >&2
  exit 64
fi

# ------------------------------------------------------------------- offline
# Without the seismic data the workflow cannot run, but the frozen copies are
# in the repository, so the harness itself can still be exercised and the
# reference shown to be internally consistent.
if [ "$OFFLINE" = "1" ]; then
  echo "offline check: the frozen copies against their own hashes"
  ( cd "$REF" && md5sum -c "../../$MD5" ) || exit 1
  python3 "$TESTS/compare_wavelets.py" --against-stats "$STATS" "$REF" --rtol "$RTOL"
  exit $?
fi

# ------------------------------------------------------------ 1. the inputs
# A wavelet that moves because an input moved is not a bug in the code, and
# finding that out after a day of reading diffs is a waste.
if [ -e "$INPUTS" ]; then
  echo "[1] inputs"
  bad=0
  while read -r want path; do
    case "$want" in \#*) continue ;; esac
    [ -n "${path:-}" ] || continue
    path=$DATA/$path
    if [ ! -e "$path" ]; then
      echo "    absent   $path"; bad=1; continue
    fi
    got=$(md5sum "$path" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
      echo "    ok       $path"
    else
      echo "    CHANGED  $path (${want:0:8} -> ${got:0:8})"; bad=1
    fi
  done < "$INPUTS"
  if [ "$bad" = "1" ]; then
    echo "    an input differs from the frozen run: a wavelet that moves below"
    echo "    is not evidence about the code until this is resolved"
  fi
else
  echo "[1] no $INPUTS, inputs not verified"
fi

# ------------------------------------------------- 2. the tracking, on request
# The ten runs load the tracking, so by themselves they test stages 3 to 5 and
# say nothing about the picker.  FULL=1 rebuilds that tracking from the stage-1
# anchors, exactly as step 1 of validation/viking/viking_rerun_conv.sh does, and compares it with
# the frozen one.  The comparison is array by array: a .npz carries the time it
# was written, so its md5 changes on every run and means nothing.
if [ "$RERUN" = "1" ] && [ "${FULL:-0}" = "1" ] && [ "$RECIPE" = "paper" ]; then
  for f in "$SEGY" "$CUBE" "$ANCH" "$TRACK" seaweed/run_real.py; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }
  done
  mkdir -p "$OUT"
  echo "[2] rebuilding the tracking from $ANCH"
  if ! python3 -m seaweed.run_real --segy "$SEGY" --near 262 --dx 25 --nshot 1012 \
        --load_cube "$CUBE" --n_group 20 --stretch_semb 0.15 --t0_shift 0 \
        --load_anchors "$ANCH" --save_track "$OUT/track_check.npz" \
        --sp_first 0 --sp_last 1000 --sp_step 10 \
        --out "$OUT/track_seed" > "$OUT/track_seed.log" 2>&1; then
    echo "    the tracking run failed; see $OUT/track_seed.log" >&2
    exit 1
  fi
  python3 "$TESTS/compare_wavelets.py" --npz "$TRACK" "$OUT/track_check.npz" \
      | sed 's/^/    /'
  track_status=${PIPESTATUS[0]}
  if [ "$track_status" != "0" ]; then
    echo "    stages 0 to 2 no longer produce the seafloor of the paper" >&2
    exit 1
  fi
fi

# ------------------------------------ 2'. the final recipe: the onset tracking
# seafloor_pipeline from the stage-1 anchors of the paper: the stage-1 tracking
# (the flags of [2] above), anchor_qc, h3_track h2 and the whole-line
# extraction with --recipe final.  Its tracking is compared array by array.
if [ "$RERUN" = "1" ] && [ "$RECIPE" = "final" ]; then
  for f in "$SEGY" "$CUBE" "$ANCH" seaweed/seafloor_pipeline.py; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }
  done
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "[2'] the final recipe from $ANCH"
  if ! python3 -m seaweed.seafloor_pipeline "$OUT/pipeline" --segy "$SEGY" --near 262 --dx 25 --nshot 1012 \
        --dxsp 25 --load_cube "$CUBE" --load_anchors "$ANCH" --stretch_semb 0.15 \
        > "$OUT/pipeline.log" 2>&1; then
    echo "    the pipeline failed; see $OUT/pipeline.log" >&2
    exit 1
  fi
  # the quality-control figures of the pipeline: on the final and on the stage-1 tracking
  for f in full.png stage1_full_line.png; do
    if [ ! -s "$OUT/pipeline/$f" ]; then
      echo "    the pipeline did not write $f" >&2
      exit 1
    fi
  done
  echo "    full.png and stage1_full_line.png written"
  if [ "$NOREF" = "0" ]; then
    python3 "$TESTS/compare_wavelets.py" --npz "$REF/h3_track.npz" "$OUT/pipeline/track.npz" | sed 's/^/    /'
    if [ "${PIPESTATUS[0]}" != "0" ]; then
      echo "    the final recipe no longer produces the frozen onset tracking" >&2
      exit 1
    fi
  fi
fi

# --------------------------------- 2''. the final recipe: the stage-1 picker
# anchors_from_consensus on Viking with the group of the rule (100 shots,
# 2500 m), --tol_ms 120 --max_shots 48, from the first pass of the paper.
# Nothing else in the test runs the picker: [2] and [2'] load its anchors.
if [ "$RERUN" = "1" ] && [ "$RECIPE" = "final" ]; then
  echo "[2''] the stage-1 picker on Viking (--group 100)"
  if ! python3 -m seaweed.anchors_from_consensus "$CUBE" "$TRACK" "$OUT/stage1_anchors.npz" \
        --group 100 --tol_ms 120 --max_shots 48 > "$OUT/stage1_anchors.log" 2>&1; then
    echo "    the picker failed; see $OUT/stage1_anchors.log" >&2
    exit 1
  fi
  if [ -e "$REF/stage1_anchors.npz" ]; then
    python3 "$TESTS/compare_wavelets.py" --npz "$REF/stage1_anchors.npz" "$OUT/stage1_anchors.npz" | sed 's/^/    /'
    if [ "${PIPESTATUS[0]}" != "0" ]; then
      echo "    the stage-1 picker no longer gives the frozen anchors" >&2
      exit 1
    fi
  else
    echo "    no $REF/stage1_anchors.npz: freeze it with"
    echo "    RECIPE=final STAGE1_ONLY=1 bash tests/freeze_reference.sh $OUT"
    S1NOREF=1
  fi
fi

# ----------------------------------------------------------- 3. the ten runs
if [ "$RERUN" = "1" ]; then
  for f in "$SEGY" "$CUBE" "$TRACK" seaweed/run_real.py; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }
  done
  # The same flags as validation/viking/viking_rerun_conv.sh, which is where these runs come from.
  # They are repeated rather than sourced so that a change to that script
  # cannot silently change what the test measures.
  COMMON="--segy $SEGY --near 262 --dx 25 --nshot 1012 --load_cube $CUBE \
          --n_group 20 --stretch_semb 0.15 --t0_shift 0 --load_track $TRACK"
  [ "$RECIPE" = "final" ] && COMMON="--segy $SEGY --near 262 --dx 25 --nshot 1012 --load_cube $CUBE \
          --stretch_semb 0.15 --load_track $OUT/pipeline/track.npz --recipe final --dxsp 25"
  [ "${FULL:-0}" = "1" ] || [ "$RECIPE" = "final" ] || rm -rf "$OUT"
  mkdir -p "$OUT"
  JOBFILE=$OUT/jobs.txt; : > "$JOBFILE"
  i=0
  while [ "$i" -lt "$COMBS" ]; do
    name="C_n100_L25km_comb$i"
    echo "python3 -m seaweed.run_real $COMMON --sp_first $i --sp_last 1000 --sp_step 10 \
--out $OUT/$name > $OUT/$name.log 2>&1" >> "$JOBFILE"
    i=$((i + 1))
  done
  echo "[3] $(wc -l < "$JOBFILE") extractions, $JOBS at a time, into $OUT"
  if ! xargs -P "$JOBS" -I CMD bash -c CMD < "$JOBFILE"; then
    echo "    at least one run failed; its log is in $OUT" >&2
    exit 1
  fi
  # One tracking for the ten runs is the premise of the comparison.
  echo "    tracked seafloor seen by the runs:"
  grep -h "tracked seafloor" "$OUT"/C_*.log 2>/dev/null | sort -u | sed 's/^/      /'
else
  echo "[3] no re-run, comparing $OUT as it is"
  [ -d "$OUT" ] || { echo "$OUT does not exist" >&2; exit 64; }
fi

if [ "$NOREF" = "1" ]; then
  echo
  echo "no reference to compare with: freeze this run with"
  echo "    RECIPE=final DATA=$DATA bash tests/freeze_reference.sh $OUT"
  exit 64
fi

# ---------------------------------------------------------- 4. md5, the check
echo "[4] md5 against $MD5"
here=$(pwd)
( cd "$OUT" && md5sum -c "$here/$MD5" ) 2>&1 | sed 's/^/    /'
md5_status=${PIPESTATUS[0]}

# ------------------------------------------------- 5. what kind of difference
echo "[5] numerical comparison against $REF"
python3 "$TESTS/compare_wavelets.py" "$REF" "$OUT" --rtol "$RTOL" | sed 's/^/    /'
verdict=${PIPESTATUS[0]}

echo
if [ "$md5_status" = "0" ] && [ "$verdict" = "0" ]; then
  if [ "${S1NOREF:-0}" = "1" ]; then
    echo "the wavelets and the onset tracking match; the stage-1 anchors have no reference yet (see [2''])"
    exit 64
  fi
  [ "$RECIPE" = "final" ] && { echo "PASS: the final recipe gives the frozen stage-1 anchors, onset tracking and ten wavelets, byte for byte"; exit 0; }
  echo "PASS: the ten wavelets of the paper are unchanged, byte for byte"
  exit 0
fi
if [ "$verdict" = "2" ]; then
  echo "REVIEW: the numbers hold but the bytes do not; compare"
  echo "        $TESTS/reference_env.txt with this machine before going on"
  exit 2
fi
echo "FAIL: this modification changes a result of the paper"
exit 1
