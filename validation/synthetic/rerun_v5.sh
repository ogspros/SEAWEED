#!/bin/bash
# The threshold and towing-depth matrices, on the true seafloor.
#
# They are the two the paper still quotes from the earlier protocol, where the
# window followed a tracked seafloor and what was measured was tracking and
# extraction together.  Everything else in the synthetic section already comes
# from wavelet_matrix_v5_bias and _noise.
#
# One noise level and no bias sweep: these two axes are about the geometry, not
# about how wrong the tracking may be, and the bias curve is already measured.
#
#   bash validation/synthetic/rerun_v5.sh 2>&1 | tee rerun_v5.log
set -e
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
NS=30
for cfg in shallow z; do
  src=$HERE/matrix_${cfg}.yaml        # the source config, next to this script
  new=matrix_v5_${cfg}.yaml
  root=$(python3 - "$src" <<'PY'
import sys, copy, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["paths"]["out_root"].rstrip("/"))
PY
)
  out="$(basename "$root")_v5"
  python3 - "$src" "$new" <<'PY'
import sys, copy, yaml
base = yaml.safe_load(open(sys.argv[1]))
c = copy.deepcopy(base)
# out_root is left as it is: the manifest is read from there, the results are
# written where --out says.  Changing it is what broke the first v5 run.
c["posthoc"] = dict(snr_db=[30.0], t0_bias_ms=[0.0], measure_tracking=True)
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
print("%s -> manifest from %s" % (sys.argv[2], c["paths"]["out_root"]))
PY
  echo "== $cfg -> $out  $(date)"
  mkdir -p "$out"
  for k in $(seq 0 $((NS-1))); do
    python3 "$HERE/run_matrix_v5.py" --config "$new" --out "$out" \
      --shard $k --n_shards $NS > "$out/shard$k.log" 2>&1 &
  done
  wait
  cat "$out"/results_shard*.jsonl > "$out/results.jsonl"
  echo "   $(wc -l < "$out/results.jsonl") records  $(date)"
done
echo "done: $(date)"
