#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/ubuntu/ntransformer-h100
export PATH="$TASK_ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$TASK_ROOT/env/targets/x86_64-linux/lib:$TASK_ROOT/env/lib:${LD_LIBRARY_PATH:-}"
export NTRANSFORMER_TRAINING_LIB="$TASK_ROOT/build-training/libntransformer_training.so"
mkdir -p "$TASK_ROOT/training/improve-20261013"
if [ ! -f "$TASK_ROOT/build-training/libntransformer_training-precise-v2.so" ]; then
    cp "$NTRANSFORMER_TRAINING_LIB" "$TASK_ROOT/build-training/libntransformer_training-precise-v2.so"
fi
cmake --build "$TASK_ROOT/build-training" --target ntransformer_training -j 2
/home/ubuntu/brandon-mini/.venv/bin/python "$TASK_ROOT/source/tests/test_training_rms_compatible.py" \
    --brandon-root /lambda/nfs/btiny/brandon-mini/sources/stage1-v1 \
    --checkpoint /lambda/nfs/btiny/brandon-mini/stages/stage1-v1/sft/best.pt \
    --data-dir /lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered \
    --output "$TASK_ROOT/training/improve-20261013/rms-validation.json"
