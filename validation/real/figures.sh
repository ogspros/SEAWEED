#!/usr/bin/env bash
# Figs. 5 and 6 of the paper, the receiver-deghost check of MS-29 and the
# variation of the earlier operator (Sect. 4.3, deghost_gain.py), with the
# scripts of validation/archive at 300 dpi, on the outputs of
# lines.sh and runs.sh.  The inputs of fig_real_surveys.py are linked file by
# file into OUT/archive.
#
#   DATA=DIR bash validation/real/figures.sh LINES RUNS OUT
#
# DATA also holds Spettro_NO_deghost.txt and Spettro_deghost.txt, the MS-29
# stack spectra before and after the receiver deghost of the earlier
# reprocessing (Brancatelli et al., 2022).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO/validation/archive:$REPO${PYTHONPATH:+:$PYTHONPATH}"
D=$(cd "${DATA:-.}" && pwd)
LINES=$(cd "${1:?the output directory of lines.sh}" && pwd)
RUNS=$(cd "${2:?the output directory of runs.sh}" && pwd)
O=${3:?name a new output directory}
[ -e "$O" ] && { echo "$O exists" >&2; exit 64; }
for f in "$D/Spettro_NO_deghost.txt" "$D/Spettro_deghost.txt"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 64; }; done
mkdir -p "$O/archive"; O=$(cd "$O" && pwd)
for L in it065 mepa ms29; do
  ln -s "$D/span_experiment/${L}_cube.npy" "$O/archive/${L}_cube.npy"
  ln -s "$D/span_experiment/${L}_cube.npy.meta.npz" "$O/archive/${L}_cube.npy.meta.npz"
  ln -s "$LINES/$L/track.npz" "$O/archive/${L}_track.npz"
  ln -s "$LINES/$L/full_wavelet.npy" "$O/archive/${L}_reference_full_line_wavelet.npy"
  for w in "$RUNS"/archive/${L}_n100_*_wavelet.npy; do ln -s "$w" "$O/archive/"; done
done
cd "$O"
python3 -c "import sys, fig_real_surveys as m; m.DPI = 300; m.main(sys.argv[1])" archive > fig_real_surveys.log 2>&1
python3 -c "import sys, fig_deghost as m; m.DPI = 300; m.main(*sys.argv[1:])" \
    "$D/Spettro_NO_deghost.txt" "$D/Spettro_deghost.txt" "$LINES/ms29/full_wavelet.npy" > fig_deghost.log 2>&1
python3 "$REPO/validation/archive/ms29_deghost_check.py" "$LINES/ms29/full_wavelet.npy" --dt 0.004 \
    --nodeghost "$D/Spettro_NO_deghost.txt" --deghost "$D/Spettro_deghost.txt" > ms29_deghost_check.log 2>&1
python3 "$HERE/deghost_gain.py" "$D/Spettro_NO_deghost.txt" "$D/Spettro_deghost.txt" > deghost_gain.log 2>&1
cat fig_real_surveys.log fig_deghost.log deghost_gain.log
