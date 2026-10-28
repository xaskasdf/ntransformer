#!/usr/bin/env bash
cd /home/ubuntu/ntransformer-h100
export DS41_GPU_TIMING=1
export DS41_VRAM_GB=62
timeout 600 ./build-release/ntransformer \
  -m models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \
  -p "The capital of France is" -n 2 --temperature 0 > /tmp/gputime.log 2>&1
grep '\[gputime\]' /tmp/gputime.log | head -5
