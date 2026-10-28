#!/usr/bin/env bash
cd /home/ubuntu/ntransformer-h100
export DS41_PROFILE=1
export DS41_VRAM_GB=62
start=$(date +%s)
timeout 3600 ./build-release/ntransformer \
  -m models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \
  -p "The capital of France is" \
  -n 8 --temperature 0 > /tmp/real_gpu_gen.log 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc wall=$((end-start))s"
grep -E '\[prof\]|prompt:|generated|ds41-gpu' /tmp/real_gpu_gen.log
tail -3 /tmp/real_gpu_gen.log
