#!/bin/bash
# 补全实验链：B 同骨干攻击者端到端（GLM×frontier，50q×4）→ A D1 旁路补齐（Mistral/Qwen3-8B）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6

runf () {  # frontier setting × defense（--suffix glmatt）
  echo "=== [$(date '+%m-%d %H:%M:%S')] $1 $2 ==="
  $PY experiments/run_poison.py --action run --setting "$1" --defense "$2" --resume \
    --suffix glmatt > "experiments/logs/fin_$1_$2.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/fin_$1_$2.log" | tail -1
}
runD1 () {  # D1 旁路（clean/r2，defense on）
  echo "=== [$(date '+%m-%d %H:%M:%S')] D1 $1 ==="
  $PY experiments/run_poison.py --action run --setting "$1" --defense on --ablation D1 \
    --resume > "experiments/logs/fin_d1_$1.log" 2>&1
  grep -E "Correct" "experiments/logs/fin_d1_$1.log" | tail -1
}
deploy () {
  pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
  ss -tln | grep -q ":8213 " && { echo "FATAL: 8213 occupied"; exit 1; }
  CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" --port 8213 --gpu-memory-utilization 0.90 \
    --max-model-len 16384 --trust-remote-code > "/tmp/vllm_fin_$2.log" 2>&1 &
  local i ok=0
  for i in $(seq 1 180); do
    curl -s --max-time 3 http://localhost:8213/v1/models | grep -q "$2" && { ok=1; echo "[deploy] $2 healthy"; break; }
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    sed -i "s|model_name: \"[^\"]*\"|model_name: \"$2\"|" src_new/settings.yaml
  else
    echo "FATAL: $2 deploy failed"; tail -5 "/tmp/vllm_fin_$2.log"
  fi
  return $((1-ok))
}

# —— B：同骨干攻击者（GLM 攻击 GLM 防）——
curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B \
  || deploy /mnt/nas_share/qwen3/glm-4-9b-chat GLM-4-9B
$PY experiments/run_poison.py --action inject --tier L1_75 --n-per-query 2 >> experiments/logs/multi_inject.log 2>&1
runf f1 off; runf f1 on
$PY experiments/run_poison.py --action inject --tier L2_75 --n-per-query 2 >> experiments/logs/multi_inject.log 2>&1
runf f2 off; runf f2 on
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

# —— A：D1 旁路补齐 ——
deploy /mnt/nas_share/qwen3/Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.3 || exit 1
runD1 clean
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
runD1 r2
deploy /mnt/nas_share/qwen3/Qwen3-8B Qwen3-8B || exit 1
runD1 clean
runD1 r2
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml
echo "########## FINAL BACKFILL DONE ########## [$(date '+%m-%d %H:%M:%S')]"
