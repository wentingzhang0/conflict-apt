#!/bin/bash
# CORE-2 信号审计（v3）：clean/r1/r2/adapt × 20 query，defense on，结果带 _v3audit 后缀
# 目的：实测 CORE-2 在毒/真实证据上的分离度与 clean 误标率（τ 调参闸门）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python

# 1. 部署 GLM-4-9B（审计骨干）
pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 8
ss -tln | grep -q ":8213 " && { echo "FATAL: 8213 occupied"; exit 1; }
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/glm-4-9b-chat --served-model-name GLM-4-9B \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --trust-remote-code > /tmp/vllm_glm_audit.log 2>&1 &
echo "[deploy] GLM-4-9B launching ..."
ok=0
for i in $(seq 1 90); do
  if curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B; then
    echo "[deploy] healthy (~$((i*10))s)"; ok=1; break
  fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: deploy failed"; tail -20 /tmp/vllm_glm_audit.log; exit 1; }

# 2. 审计矩阵（20 query × defense on；语料按档注入）
export CUDA_VISIBLE_DEVICES=6
run () {
  echo "=== [$(date '+%m-%d %H:%M:%S')] audit $1 ==="
  $PY experiments/run_poison.py --action run --setting "$1" --defense on \
    --end 20 --suffix v3audit > "experiments/logs/audit_$1.log" 2>&1
  grep -E "Correct .*framed|core2_flagged" "experiments/logs/audit_$1.log" | tail -2
}

run clean
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
run r1
run r2
$PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
run adapt
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1
echo "########## V3 AUDIT DONE ########## [$(date '+%m-%d %H:%M:%S')]"
