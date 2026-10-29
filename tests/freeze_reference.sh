#!/usr/bin/env bash
# Freeze the products that produced the paper, so that any later change to the
# code can be shown to be cosmetic.
#
# Run this ONCE, on the machine that holds the data, before touching any file
# of the workflow.  It records the ten whole-line combs of family C, the ones
# behind the 0.823 of Sect. 4.4 and of the abstract, together with the inputs
# they were computed from and the environment that computed them.
#
#   bash tests/freeze_reference.sh                 # from viking_conv41/
#   bash tests/freeze_reference.sh some_other_dir  # from another run directory
#   WITH_SEGY=1 bash tests/freeze_reference.sh     # hash the SEG-Y as well
#   RECIPE=final DATA=DIR bash tests/freeze_reference.sh RUN_DIR
#                       # the final recipe, from a run of
#                       # RECIPE=final tests/smoke_test.sh: tests/reference_final*,
#                       # with the onset tracking of RUN_DIR/pipeline; the inputs are
#                       # hashed under DATA and recorded relative to it
#   RECIPE=final STAGE1_ONLY=1 bash tests/freeze_reference.sh RUN_DIR
#                       # only the stage-1 anchors of the picker on Viking
#                       # (RUN_DIR/stage1_anchors.npz), nothing else touched
#
# What is written, all of it small enough to live in git:
#
#   tests/reference_md5.txt     md5 of each frozen wavelet, bare file names, so
#                               that `md5sum -c` works from inside a run
#                               directory.  This is the check of record.
#   tests/reference/            a copy of those wavelets and of their _sel.npz,
#                               about 4 kB in total, so that a comparison is
#                               possible on a machine without the seismic data
#   tests/reference_stats.tsv   peak, r.m.s., sample of the minimum and the
#                               quality-control scalars of each run
#   tests/reference_inputs.md5  the cube, the anchors and the tracking: when a
#                               wavelet moves, this says whether an input moved
#   tests/reference_env.txt     commit, interpreter and library versions
#
# After this, `bash tests/smoke_test.sh` re-runs those ten extractions and
# compares them with what is frozen here.
set -euo pipefail

cd "$(dirname "$0")/.."                      # repository root, whatever the cwd

RUNS=${1:-viking_conv41}
RUNS=${RUNS%/}                                # a trailing slash breaks the md5 names
FAMILY=${FAMILY:-C_n100_L25km_comb}
TESTS=tests
RECIPE=${RECIPE:-paper}
case "$RECIPE" in
  paper) SUF="" ;;
  final) SUF="_final"; [ $# -ge 1 ] || { echo "RECIPE=final: name the run directory" >&2; exit 1; } ;;
  *) echo "RECIPE is paper or final, not $RECIPE" >&2; exit 1 ;;
esac
DATA=${DATA:-.}
REF=$TESTS/reference$SUF
if [ "$RECIPE" = "final" ] && [ "${STAGE1_ONLY:-0}" = "1" ]; then
  [ -e "${1%/}/stage1_anchors.npz" ] || { echo "no ${1%/}/stage1_anchors.npz" >&2; exit 1; }
  mkdir -p "$REF"; cp -f "${1%/}/stage1_anchors.npz" "$REF/stage1_anchors.npz"
  echo "stage-1 anchors frozen in $REF/stage1_anchors.npz (from ${1%/})"
  exit 0
fi
CUBE=${CUBE:-viking_pick/viking_cube.npy}
ANCH=${ANCH:-viking_pick/viking_anchors_conv.npz}
TRACK=${TRACK:-$RUNS/viking_track_conv.npz}
SEGY=${SEGY:-seismic.sgy}

shopt -s nullglob
wavelets=("$RUNS"/${FAMILY}*_wavelet.npy)
shopt -u nullglob
if [ ${#wavelets[@]} -eq 0 ]; then
  echo "no ${FAMILY}*_wavelet.npy in $RUNS" >&2
  echo "run validation/viking/viking_rerun_conv.sh first, or name the run directory as argument" >&2
  exit 1
fi
if [ ${#wavelets[@]} -ne 10 ]; then
  echo "warning: ${#wavelets[@]} wavelets found, the paper family has 10" >&2
fi

mkdir -p "$REF"

# ------------------------------------------------------- 1. the hashes of record
# Bare file names on purpose: the check runs as `cd <run dir> && md5sum -c`,
# and a path frozen here would tie the test to one directory name.
( cd "$RUNS" && md5sum "${wavelets[@]#$RUNS/}" ) > "$TESTS/reference${SUF}_md5.txt"
echo "[1] $(wc -l < "$TESTS/reference${SUF}_md5.txt") wavelets hashed in $TESTS/reference${SUF}_md5.txt"

# ------------------------------------------------------------ 2. the copies
for w in "${wavelets[@]}"; do
  cp -f "$w" "$REF/"
  sel=${w%_wavelet.npy}_sel.npz
  [ -e "$sel" ] && cp -f "$sel" "$REF/"
done
if [ "$RECIPE" = "final" ]; then
  [ -e "$RUNS/pipeline/track.npz" ] || { echo "no $RUNS/pipeline/track.npz" >&2; exit 1; }
  cp -f "$RUNS/pipeline/track.npz" "$REF/h3_track.npz"
fi
echo "[2] copied into $REF/ ($(du -sh "$REF" | cut -f1))"

# ------------------------------------------------------------ 3. the statistics
printf "[3] "
python3 "$TESTS/compare_wavelets.py" --stats "$REF" --out "$TESTS/reference${SUF}_stats.tsv"

# ---------------------------------------------------------------- 4. the inputs
{
  echo "# inputs of the frozen runs, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ins=("$CUBE" "$ANCH" "$TRACK"); [ "$RECIPE" = "final" ] && ins=("$CUBE" "$ANCH")
  for f in "${ins[@]}"; do
    if [ -e "$DATA/$f" ]; then ( cd "$DATA" && md5sum "$f" ); else echo "# absent: $f"; fi
  done
  if [ -e "$DATA/$SEGY" ]; then
    if [ "${WITH_SEGY:-0}" = "1" ]; then
      ( cd "$DATA" && md5sum "$SEGY" )
    else
      echo "# $SEGY $(stat -L -c %s "$DATA/$SEGY") bytes (hash skipped; WITH_SEGY=1 to include it)"
    fi
  else
    echo "# absent: $SEGY"
  fi
} > "$TESTS/reference${SUF}_inputs.md5"
echo "[4] inputs recorded in $TESTS/reference${SUF}_inputs.md5"

# ----------------------------------------------------------- 5. the environment
{
  echo "frozen on $(date -u +%Y-%m-%dT%H:%M:%SZ) from $RUNS"
  echo "host        $(uname -n) $(uname -sr)"
  echo "commit      $(git rev-parse HEAD 2>/dev/null || echo 'not a git repository')"
  echo "describe    $(git describe --tags --always --dirty 2>/dev/null || echo '-')"
  echo "dirty       $(git status --porcelain 2>/dev/null | wc -l) modified files"
  echo "python      $(python3 -c 'import sys; print(sys.version.split()[0])')"
  python3 - <<'PY'
for mod in ("numpy", "scipy", "matplotlib", "segyio", "obspy"):
    try:
        m = __import__(mod)
        print("%-11s %s" % (mod, getattr(m, "__version__", "?")))
    except ImportError:
        print("%-11s absent" % mod)
try:
    import numpy as np
    cfg = np.__config__.show(mode="dicts") if hasattr(np.__config__, "show") else None
    blas = (cfg or {}).get("Build Dependencies", {}).get("blas", {})
    if blas:
        print("blas        %s %s" % (blas.get("name", "?"), blas.get("version", "?")))
except Exception as exc:                      # a BLAS report is a nicety
    print("blas        unknown (%s)" % exc)
PY
  for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS; do
    echo "$v=${!v-unset}"
  done
} > "$TESTS/reference${SUF}_env.txt"
echo "[5] environment recorded in $TESTS/reference${SUF}_env.txt"

echo
echo "commit these five before the first refactoring:"
echo "  git add $TESTS/reference${SUF}_md5.txt $TESTS/reference${SUF}_stats.tsv \\"
echo "          $TESTS/reference${SUF}_inputs.md5 $TESTS/reference${SUF}_env.txt $REF"
