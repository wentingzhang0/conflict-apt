#!/bin/bash
# 多骨干全量复现（期刊版：论文 4 骨干补全，GLM-4-9B 已完成）
# 顺序：Mistral-7B → Qwen3-8B → Qwen3-30B-A3B-Base，每个骨干 12 组 run：
#   clean/r1/r2/r4/hard/adapt × off/on（50 query，--resume 幂等）
# 部署用 vllm 环境，跑实验用 pytorch 环境；vLLM 在 GPU2，实验嵌入在 GPU6（与 GLM 轮一致）
# 用法：nohup bash experiments/run_backbones.sh > experiments/logs/backbones_driver.log 2>&1 &
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
GPU=2
PORT=8213
mkdir -p experiments/logs

BACKBONES=(
  "/mnt/nas_share/qwen3/Mistral-7B-Instruct-v0.3|Mistral-7B-Instruct-v0.3"
  "/mnt/nas_share/qwen3/Qwen3-8B|Qwen3-8B"
  "/mnt/nas_share/qwen3/Qwen3-30B-A3B-Base|Qwen3-30B-A3B-Base"
)

kill_vllm() {
  pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  sleep 8
  local i
  for i in 1 2 3 4 5 6; do
    ss -tln 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 5
  done
  # SO_REUSEPORT 陷阱：仍有残留监听则强杀占用进程
  local pids
  pids=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | sort -u)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 3
  ss -tln 2>/dev/null | grep -q ":$PORT " && return 1 || return 0
}

deploy() {  # $1=model_path $2=served_name
  kill_vllm || { echo "FATAL: port $PORT occupied, abort"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU nohup $VLLM -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" --port $PORT \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --trust-remote-code \
    > "/tmp/vllm_$2.log" 2>&1 &
  echo "[deploy] $2 launching ..."
  local i
  for i in $(seq 1 180); do
    if curl -s --max-time 3 "http://localhost:$PORT/v1/models" | grep -q "$2"; then
      echo "[deploy] $2 healthy (~$((i*10))s)"
      return 0
    fi
    sleep 10
  done
  echo "[deploy] FATAL: $2 not healthy in 30min"; tail -20 "/tmp/vllm_$2.log"
  return 1
}

set_model() {
  sed -i "s|model_name: \"[^\"]*\"|model_name: \"$1\"|" src_new/settings.yaml
  echo "[settings] $(grep 'model_name' src_new/settings.yaml | head -1 | tr -d ' ')"
}

slug() { echo "$1" | tr 'A-Z' 'a-z'; }

smoke() {  # $1=served_name：2 条 clean 冒烟，预测全空则跳过该骨干
  local sl; sl=$(slug "$1")
  rm -f "experiments/results/$sl/srcnew_taa_clean_off.json"
  CUDA_VISIBLE_DEVICES=6 $PY experiments/run_poison.py --action run \
    --setting clean --defense off --end 2 > "experiments/logs/multi_smoke_$sl.log" 2>&1
  local n
  n=$($PY -c "
import json,os
p='experiments/results/$sl/srcnew_taa_clean_off.json'
d=json.load(open(p)) if os.path.exists(p) else []
print(sum(1 for r in d if r.get('prediction') and r.get('status')=='ok'))" 2>/dev/null) || n=0
  echo "[smoke] $1: $n/2 non-empty predictions"
  [ "$n" -ge 1 ]
}

run () {  # $1=setting $2=defense
  echo "=== [$(date '+%m-%d %H:%M:%S')] $1 $2 ==="
  CUDA_VISIBLE_DEVICES=6 $PY experiments/run_poison.py --action run \
    --setting "$1" --defense "$2" --resume >> "experiments/logs/multi_$1_$2.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/multi_$1_$2.log" | tail -1
}

matrix() {  # 12 组 + 语料切换（easy 全 4 变体注入，r1/r2/r4 靠 variant_cap 区分）
  run clean off; run clean on
  $PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
  run r1 off; run r1 on; run r2 off; run r2 on; run r4 off; run r4 on
  $PY experiments/run_poison.py --action inject --tier hard >> experiments/logs/multi_inject.log 2>&1
  run hard off; run hard on
  $PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
  run adapt off; run adapt on
  $PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1
}

for bb in "${BACKBONES[@]}"; do
  path="${bb%%|*}"; name="${bb##*|}"; sl=$(slug "$name")
  echo "########## BACKBONE $name ########## [$(date '+%m-%d %H:%M:%S')]"
  if ! deploy "$path" "$name"; then continue; fi
  set_model "$name"
  if ! smoke "$name"; then
    echo "[skip] $name smoke failed — results discarded"
    rm -rf "experiments/results/$sl"
    continue
  fi
  matrix
  echo "[$name] all runs done, summaries:"
  for f in "experiments/results/$sl"/srcnew_taa_*_report.json; do
    echo "  $(basename $f): $(tr -d '\n {}"' < $f | sed 's/,/ /g' | cut -c1-160)"
  done
done

kill_vllm || true
sed -i "s|model_name: \"[^\"]*\"|model_name: \"GLM-4-9B\"|" src_new/settings.yaml
echo "########## ALL BACKBONES DONE ########## [$(date '+%m-%d %H:%M:%S')]"
