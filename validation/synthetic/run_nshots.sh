#!/bin/bash
# Shot-count experiment, sharded by model.
#   bash validation/synthetic/run_nshots.sh [n_cores] [n_models]
# Runs at two noise levels: high SNR isolates the coherent contamination,
# low SNR isolates the sqrt(N) gain on incoherent noise.
set -u
# Run from the directory that holds the data; the code is found from here,
# and PYTHONPATH makes the seaweed package importable without installing it.
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
N=${1:-12}; M=${2:-12}
export DEVITO_LOGGING=ERROR DEVITO_ARCH=gcc DEVITO_LANGUAGE=C OMP_NUM_THREADS=1
for SNR in 30 -10; do
    echo "=== SNR ${SNR} dB nominal ==="
    rm -f nshots_snr${SNR}_shard*.json
    pids=()
    for ((c=0; c<N; c++)); do
        python "$HERE/test_nshots.py" --models "$M" --shard "$c" --n_shards "$N" \
            --snr "$SNR" > nshots_snr${SNR}_log${c}.txt 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    tail -n 12 "nshots_snr${SNR}_log$((N-1)).txt"
    echo
done
