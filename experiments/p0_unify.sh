#!/bin/bash
# P0 统一实验结果（回复学长 §3 承诺）：seed=42 + 投毒库统一 200 篇（easy 全 4 变体）
# 阶段：GLM（clean/r2 on/off + D0-D5b + clean-D5b）→ Mistral（src_new + src 对照）
#       → Qwen3-8B clean → 30B clean。结果带 _s42 后缀；src/ 对照写到 p0_src_mistral/
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6

runp () {  # setting defense [ablation]
  local s=$1 d=$2 a=${3:-}
  local cmd="$PY experiments/run_poison.py --action run --setting $s --defense $d --suffix s42"
  [ -n "$a" ] && cmd="$cmd --ablation $a"
  echo "=== [$(date '+%m-%d %H:%M:%S')] $s $d ${a:-} ==="
  $cmd > "experiments/logs/p0_${s}_${d}_${a:-x}.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/p0_${s}_${d}_${a:-x}.log" | tail -1
}
deploy () {
  pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
  ss -tln | grep -q ":8213 " && { echo "FATAL: 8213 occupied"; exit 1; }
  CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
    --model "$1" --served-model-name "$2" --port 8213 --gpu-memory-utilization 0.90 \
    --max-model-len 16384 --trust-remote-code > "/tmp/vllm_p0_$2.log" 2>&1 &
  local i ok=0
  for i in $(seq 1 180); do
    curl -s --max-time 3 http://localhost:8213/v1/models | grep -q "$2" && { ok=1; echo "[deploy] $2 healthy"; break; }
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    sed -i "s|model_name: \"[^\"]*\"|model_name: \"$2\"|" src_new/settings.yaml
    sed -i "s|model_name: \"[^\"]*\"|model_name: \"$2\"|" src/settings.yaml
  else
    echo "FATAL: $2 deploy failed"; tail -5 "/tmp/vllm_p0_$2.log"; fi
  return $((1-ok))
}

# ── 阶段 1：GLM（消融表与基线比较的参照系）──
deploy /mnt/nas_share/qwen3/glm-4-9b-chat GLM-4-9B || exit 1
runp clean off; runp clean on
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
runp r2 off; runp r2 on
runp r2 on D0; runp r2 on D1; runp r2 on D2; runp r2 on D3; runp r2 on D4; runp r2 on D5a; runp r2 on D5b
runp clean on D5b
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

# ── 阶段 2：Mistral（差异排查 + src 对照）──
deploy /mnt/nas_share/qwen3/Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct-v0.3 || exit 1
runp clean off; runp clean on
$PY experiments/run_taa.py --output experiments/results/p0_src_mistral/taa_results.json \
  > experiments/logs/p0_src_mistral.log 2>&1
grep -E "Correct" experiments/logs/p0_src_mistral.log | tail -1

# ── 阶段 3/4：Qwen3-8B、30B clean ──
deploy /mnt/nas_share/qwen3/Qwen3-8B Qwen3-8B || exit 1
runp clean off; runp clean on
deploy /mnt/nas_share/qwen3/Qwen3-30B-A3B-Base Qwen3-30B-A3B-Base || exit 1
runp clean off; runp clean on

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src/settings.yaml
echo "########## P0 UNIFY DONE ########## [$(date '+%m-%d %H:%M:%S')]"
