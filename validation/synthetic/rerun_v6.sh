#!/bin/bash
# The whole automatic chain on the synthetic matrix: seafloor from Stage 1,
# not from the generator.  One noise level, no bias sweep -- the bias curve is
# already measured and here the tracking error is whatever the picker makes.
#
# Compare against wavelet_matrix_v5_bias at zero bias: the two differ in the
# seafloor and in nothing else.
#
#   bash validation/synthetic/rerun_v6.sh 2>&1 | tee rerun_v6.log
set -e
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
NS=30
cfg=matrix_v6.yaml
out=wavelet_matrix_v6

python3 - "$HERE/matrix_v2.yaml" <<'PY'
import copy, sys, yaml
base = yaml.safe_load(open(sys.argv[1]))
c = copy.deepcopy(base)          # out_root stays: the manifest is read from it
c["posthoc"] = dict(snr_db=[30.0], t0_bias_ms=[0.0], seafloor="consensus",
                    pick_group=10, pick_tol_ms=120.0)
yaml.safe_dump(c, open("matrix_v6.yaml", "w"), sort_keys=False)
print("matrix_v6.yaml -> manifest from %s" % c["paths"]["out_root"])
PY

echo "== v6  $(date)"
mkdir -p "$out"
for k in $(seq 0 $((NS-1))); do
  python3 "$HERE/run_matrix_v6.py" --config "$cfg" --out "$out" \
    --shard $k --n_shards $NS > "$out/shard$k.log" 2>&1 &
done
wait
cat "$out"/results_shard*.jsonl > "$out/results.jsonl"
echo "   $(wc -l < "$out/results.jsonl") records  $(date)"

python3 "$HERE/compare_v5_v6.py" 2>&1 | tee tables_v6.txt
