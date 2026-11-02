#!/bin/bash
# 并行链 B（GPU3 / 端口 8214，环境变量覆盖配置）：三骨干 clean s42 重跑 + src 对照
# 与链 A（8213，GLM 投毒阶段）无语料库竞争：本链只跑 clean，不触碰 APT_POISON。
# 输出与链 A 后续阶段同文件 → A 到达时 resume 跳过（B 须先完成，巡检盯）。
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6
PORT=8214

deployB () {  # path served_name
  pkill -f "port $PORT" 2>/dev/null; sleep 5
  CUDA_VISIBLE_DEVICES=3 nohup $VLLM -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" --port $PORT \
    --gpu-memory-utilization 0.90 --max-model-len 32768 --trust-remote-code \
    > "/tmp/vllmB_$2.log" 2>&1 &
  local i ok=0
  for i in $(seq 1 180); do
    curl -s --max-time 3 "http://localhost:$PORT/v1/models" | grep -q "$2" && { ok=1; echo "[deployB] $2 healthy"; break; }
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "FATAL: $2 deploy failed"; tail -5 "/tmp/vllmB_$2.log"; }
  return $((1-ok))
}
runB () {  # setting defense model_name
  local s=$1 d=$2 m=$3
  echo "=== [$(date '+%m-%d %H:%M:%S')] B: $s $d ($m) ==="
  LLM_API_BASE="http://localhost:$PORT/v1" LLM_MODEL_NAME="$m" \
    $PY experiments/run_poison.py --action run --setting "$s" --defense "$d" \
    --suffix s42 --resume > "experiments/logs/pB_${m}_${s}_${d}.log" 2>&1
  grep -E "Correct" "experiments/logs/pB_${m}_${s}_${d}.log" | tail -1
}

deployB /mnt/nas_share/qwen3/Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.3 || exit 1
runB clean off Mistral-7B-Instruct-v0.3
runB clean on  Mistral-7B-Instruct-v0.3
echo "=== [$(date '+%m-%d %H:%M:%S')] B: src 对照（Mistral）==="
LLM_API_BASE="http://localhost:$PORT/v1" LLM_MODEL_NAME="Mistral-7B-Instruct-v0.3" \
  $PY experiments/run_taa.py --output experiments/results/p0_src_mistral/taa_results.json \
  > experiments/logs/pB_src_mistral.log 2>&1
grep -E "Correct" experiments/logs/pB_src_mistral.log | tail -1

deployB /mnt/nas_share/qwen3/Qwen3-8B Qwen3-8B || exit 1
runB clean off Qwen3-8B
runB clean on  Qwen3-8B

deployB /mnt/nas_share/qwen3/Qwen3-30B-A3B-Base Qwen3-30B-A3B-Base || exit 1
runB clean off Qwen3-30B-A3B-Base
runB clean on  Qwen3-30B-A3B-Base

pkill -f "port $PORT" 2>/dev/null
echo "########## PARALLEL-B DONE ########## [$(date '+%m-%d %H:%M:%S')]"
