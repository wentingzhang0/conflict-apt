#!/bin/bash
# D1 强证据直隔跨骨干验证：Qwen3-30B-A3B-Base 上 clean-D1 / r2-D1（50 query）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 8
ss -tln | grep -q ":8213 " && { echo "FATAL: 8213 occupied"; exit 1; }
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/Qwen3-30B-A3B-Base --served-model-name Qwen3-30B-A3B-Base \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --trust-remote-code > /tmp/vllm_30b_d1.log 2>&1 &
echo "[deploy] Qwen3-30B-A3B-Base launching ..."
ok=0
for i in $(seq 1 180); do
  if curl -s --max-time 3 http://localhost:8213/v1/models | grep -q Qwen3-30B; then
    echo "[deploy] healthy (~$((i*10))s)"; ok=1; break
  fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: deploy failed"; tail -20 /tmp/vllm_30b_d1.log; exit 1; }
sed -i 's|model_name: "[^"]*"|model_name: "Qwen3-30B-A3B-Base"|' src_new/settings.yaml

export CUDA_VISIBLE_DEVICES=6
echo "=== [$(date '+%m-%d %H:%M:%S')] clean D1 ==="
$PY experiments/run_poison.py --action run --setting clean --defense on --ablation D1 \
  > experiments/logs/d1_30b_clean.log 2>&1
grep -E "Correct" experiments/logs/d1_30b_clean.log | tail -1
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
echo "=== [$(date '+%m-%d %H:%M:%S')] r2 D1 ==="
$PY experiments/run_poison.py --action run --setting r2 --defense on --ablation D1 \
  > experiments/logs/d1_30b_r2.log 2>&1
grep -E "Correct" experiments/logs/d1_30b_r2.log | tail -1
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1
echo "########## D1-30B DONE ########## [$(date '+%m-%d %H:%M:%S')]"
