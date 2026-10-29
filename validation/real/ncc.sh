#!/usr/bin/env bash
# Every NCC against the recorded Viking signature behind the paper, with the
# scripts of validation/viking as they are: accuracy at --maxlag_ms 30, the
# phase comparison at 60; the single-reflector fits; the agreement by sub-band
# (where_disagreement.py); Figs. 3 and 4; and the
# summary of the Viking numbers (ncc_summary.py).
#
#   DATA=DIR bash validation/real/ncc.sh RUNS OUT
#
# RUNS is the output of runs.sh; OUT must not exist.  A few minutes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO/validation/viking:$REPO${PYTHONPATH:+:$PYTHONPATH}"
D=$(cd "${DATA:-.}" && pwd)
R=${1:?the output directory of runs.sh}; O=${2:?name a new output directory}
[ -e "$O" ] && { echo "$O exists" >&2; exit 64; }
FF=$D/FarField.dat; PT=$REPO/validation/viking/viking_phase_test.py
for f in "$FF" "$R/viking" "$D/viking_pick/viking_cube.npy"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }; done
mkdir -p "$O/families" "$O/pert" "$O/nch" "$O/figure"
for f in A B C D E F G; do python3 $PT $FF $R/viking --fam $f --maxlag_ms 30 > $O/families/$f.txt 2>&1; done
for f in A B C D E F G; do python3 $PT $FF $R/viking --fam $f --maxlag_ms 30 --band 8 18 > $O/families/${f}_8-18Hz.txt 2>&1; done
python3 $PT $FF $R/viking --fam C --maxlag_ms 60 > $O/phase_C_60ms.txt 2>&1
for nc in 4 8 16; do python3 $PT $FF $R/viking_nch$nc --fam C --maxlag_ms 30 > $O/nch/C_nch$nc.txt 2>&1; done
for P in "$R"/pert/*/; do P=${P%/}; n=$(basename $P)
  for f in A C; do python3 $PT $FF $P --fam $f --maxlag_ms 30 > $O/pert/${n}_$f.txt 2>&1; done
done
for f in A B C; do python3 $REPO/validation/viking/viking_thin_layer.py $FF $R/viking --fam $f > $O/thin_layer_$f.txt 2>&1; done
RA=$(cd "$R" && pwd)
# agreement by sub-band and by time window, per family (Sect. 3.4: the 8-18 Hz band)
( cd "$O" && python3 $REPO/validation/viking/where_disagreement.py $RA/viking $FF > where_disagreement.log 2>&1 )
( cd "$O/figure" && VIKING_CUBE=$D/viking_pick/viking_cube.npy \
    python3 $REPO/validation/viking/make_paper_figures_vik.py $RA/viking $FF > figures.log 2>&1 )
python3 $HERE/ncc_summary.py $R $O > $O/summary.md
