#!/usr/bin/env bash
# The synthetic matrices of the paper with the final recipe: run_matrix_v7.py
# on every model of v2 (860), shallow (400) and z (80), in 48 shards in
# proportion (31, 14, 3), one process each.  About 7 hours on 48 cores.
#
#   DATA=DIR bash validation/synthetic/rerun_final.sh OUT
#
# DATA holds the configurations and manifests of the published runs (see
# run_matrix_v7.py --data).  OUT must not exist.  The tables come from
# tables_final.py OUT TABLES.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
D=$(cd "${DATA:-.}" && pwd)
O=${1:?name a new output directory}
[ -e "$O" ] && { echo "$O exists" >&2; exit 64; }
mkdir -p "$O"; J=$O/jobs.txt; : > "$J"
for spec in "v2 31" "shallow 14" "z 3"; do set -- $spec
  for k in $(seq 0 $(($2-1))); do
    echo "python3 $HERE/run_matrix_v7.py --matrix $1 --recipe final --data $D --out $O --shard $k --n_shards $2 > $O/$1_shard$k.log 2>&1" >> "$J"
  done
done
xargs -P "${JOBS:-48}" -I CMD bash -c CMD < "$J"
