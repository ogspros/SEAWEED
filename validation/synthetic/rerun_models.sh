#!/usr/bin/env bash
# run_matrix_v7.py --recipe final on a list of models, one model per process,
# each in OUT/<run_id>.  IDS is a JSON file {"v2": [...], "shallow": [...],
# "z": [...]}; stopped_models.json is the list behind the paper (see
# merge_rerun.py).
#
#   DATA=DIR bash validation/synthetic/rerun_models.sh IDS OUT
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1
D=$(cd "${DATA:-.}" && pwd)
IDS=${1:?a JSON file of run ids per matrix}; O=${2:?name a new output directory}
[ -e "$O" ] && { echo "$O exists" >&2; exit 64; }
mkdir -p "$O"; J=$O/jobs.txt; : > "$J"
for m in v2 shallow z; do
  for id in $(python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1])).get(sys.argv[2], [])))" "$IDS" $m); do
    echo "python3 $HERE/run_matrix_v7.py --matrix $m --recipe final --data $D --out $O/$id --only $id > $O/$id.log 2>&1" >> "$J"
  done
done
xargs -P "${JOBS:-40}" -I CMD bash -c CMD < "$J"
