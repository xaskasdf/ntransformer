#!/usr/bin/env bash
cd /home/ubuntu/ntransformer-h100
echo "=== tiny parity (CPU fallback guaranteed by dtype guards) ==="
timeout 120 ./build-release/ntransformer -m /home/ubuntu/tiny-deepseek41.gguf \
  --token-ids 1,229,153,132,107,104,111,111,114 --dump-logits /tmp/our_logits.bin -n 1 > /dev/null 2>&1
cd /home/ubuntu
./ntransformer-h100/hf-env/bin/python ./ntransformer-h100/compare_logits.py /tmp/ref_logits.bin /tmp/our_logits.bin | head -4

echo "=== real model: GPU attention vs CPU attention logits ==="
cd /home/ubuntu/ntransformer-h100
env DS41_VRAM_GB=62 timeout 1800 ./build-release/ntransformer \
  -m models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \
  --token-ids 0 --dump-logits /tmp/our_real0_gpu.bin -n 0 > /tmp/our_real0_gpu.log 2>&1
echo "gpu-attn rc=$?"
grep -E 'ds41-gpu|top0' /tmp/our_real0_gpu.log | head -5
cd /home/ubuntu
./ntransformer-h100/hf-env/bin/python ./ntransformer-h100/compare_logits.py /tmp/our_real0.bin /tmp/our_real0_gpu.bin | head -5
