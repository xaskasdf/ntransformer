#!/usr/bin/env bash
cd /home/ubuntu/ntransformer-h100
export DS41_VRAM_GB=62
start=$(date +%s)
timeout 3600 ./build-release/ntransformer \
  -m models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \
  -p "Explain in two sentences why a mixture of experts model activates only a few experts per token." \
  -n 32 --temperature 0 > /tmp/real_final.log 2>&1
rc=$?
end=$(date +%s)
echo "rc=$rc wall=$((end-start))s"
grep -E 'prompt:|generated|ds41-gpu' /tmp/real_final.log
echo "--- output:"
sed -n 's/^.*is\.//p' /tmp/real_final.log | head -3
tail -4 /tmp/real_final.log | head -3
