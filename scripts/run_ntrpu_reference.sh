#!/usr/bin/env bash
# Run in the reference container, with repository mounted read-only at /workspace
# and a persistent writable host directory mounted at /work.
set -euo pipefail

source_dir="${NTRPU_SOURCE_DIR:-/workspace}"
work_dir="${NTRPU_WORK_DIR:-/work}"
device="${NTRPU_REFERENCE_DEVICE:-cuda}"
model_dir="$work_dir/models/brandon-tiny-ca6329d"

cmake -S "$source_dir" -B "$work_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DNTRANSFORMER_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build "$work_dir/build" --parallel "${NTRPU_JOBS:-8}"
ctest --test-dir "$work_dir/build" --output-on-failure --output-log "$work_dir/ctest.log"
python "$source_dir/tests/test_ntrpu_brandon.py"

hf download xaskasdf/brandon-tiny-10m-instruct config.json model.safetensors \
    --revision ca6329de991a1cfc17b2a7a1aaae2cc38c08577a --local-dir "$model_dir"
run_dir="$work_dir/results/$(date -u +%Y%m%dT%H%M%SZ)-$$"
python "$source_dir/tools/run_ntrpu_brandon.py" \
    --model-dir "$model_dir" --exe "$work_dir/build/src/ntrpu/ntrpu_ffn_fixture" \
    --output-dir "$run_dir" --device "$device"
