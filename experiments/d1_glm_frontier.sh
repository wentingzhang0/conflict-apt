#!/bin/bash
# 收尾链：GLM clean-D1 对照 → frontier 语料生成 → 帕累托边界测量
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 8
ss -tln | grep -q ":8213 " && { echo "FATAL: 8213 occupied"; exit 1; }
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/glm-4-9b-chat --served-model-name GLM-4-9B \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --trust-remote-code > /tmp/vllm_glm_d1.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B && { ok=1; echo "[deploy] healthy"; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: deploy failed"; exit 1; }
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml

export CUDA_VISIBLE_DEVICES=6
echo "=== [$(date '+%m-%d %H:%M:%S')] GLM clean D1 ==="
$PY experiments/run_poison.py --action run --setting clean --defense on --ablation D1 \
  > experiments/logs/d1_glm_clean.log 2>&1
grep -E "Correct" experiments/logs/d1_glm_clean.log | tail -1

echo "=== [$(date '+%m-%d %H:%M:%S')] frontier generation (6 cells × 50q × 2v) ==="
$PY experiments/gen_frontier.py > experiments/logs/gen_frontier.log 2>&1
tail -6 experiments/logs/gen_frontier.log

echo "=== [$(date '+%m-%d %H:%M:%S')] frontier evaluation ==="
$PY experiments/eval_frontier.py > experiments/logs/eval_frontier.log 2>&1
cat experiments/logs/eval_frontier.log
echo "########## GLM-D1 + FRONTIER DONE ########## [$(date '+%m-%d %H:%M:%S')]"
