#!/bin/bash
# Generates the models of matrix_v2.yaml and runs the v2 extraction on them,
# in parallel shards, one Devito thread each.
#   bash validation/synthetic/run_matrix_v2.sh [n_cores] [config]
N=${1:-30}
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
CFG=${2:-$HERE/matrix_v2.yaml}
export DEVITO_LOGGING=ERROR DEVITO_ARCH=gcc DEVITO_LANGUAGE=C OMP_NUM_THREADS=1
unset DEVITO_PLATFORM

python "$HERE/build_matrix_v2.py" --config $CFG || exit 1
pids=()
for ((c=0; c<N; c++)); do
    python "$HERE/run_matrix_v2.py" --config $CFG --shard $c --n_shards $N &
    pids+=($!)
done
echo "${#pids[@]} shards started, waiting..."
for p in "${pids[@]}"; do wait $p; done

ROOT=$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['paths']['out_root'])")
cat $ROOT/results_shard*.jsonl > $ROOT/results.jsonl
echo "=== done: $ROOT/results.jsonl ($(wc -l < $ROOT/results.jsonl) records) ==="
