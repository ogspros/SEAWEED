#!/bin/bash
# The two extraction matrices behind the synthetic numbers of the paper.
#
# Both re-extract, with the seafloor known, the models that run_matrix_v2.py
# generated: their configurations are matrix_v2.yaml with only the posthoc
# block changed, and out_root still points to wavelet_matrix_v2 because the
# manifest of the models is read from there.  The results go where --out says,
# which is therefore not optional: without it they would land inside
# wavelet_matrix_v2.
#
#   matrix_v5_bias.yaml   30 dB; tracking error 0, +-10, +-25, +-50, +-100 ms;
#                         measure_tracking.  7740 records.
#   matrix_v5_noise.yaml  0, 10, 20 and 30 dB; no tracking error.  3440 records.
#
# The tables are then read from these two directories by tables_v5.py, and
# Fig. 2 by make_fig_synthetic.py --results wavelet_matrix_v5_bias at zero
# tracking error.
#
#   bash validation/synthetic/run_v5_bias_noise.sh 2>&1 | tee run_v5_bias_noise.log
set -e
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

NS=30
for m in bias noise; do
  cfg=$HERE/matrix_v5_$m.yaml
  out=wavelet_matrix_v5_$m
  echo "== $m -> $out  $(date)"
  mkdir -p "$out"
  for k in $(seq 0 $((NS-1))); do
    python3 "$HERE/run_matrix_v5.py" --config "$cfg" --out "$out" \
      --shard $k --n_shards $NS > "$out/shard$k.log" 2>&1 &
  done
  wait
  cat "$out"/results_shard*.jsonl > "$out/results.jsonl"
  echo "   $(wc -l < "$out/results.jsonl") records  $(date)"
done
echo "done: $(date)"
echo "tables: python3 $HERE/tables_v5.py > tables_v5.txt"
