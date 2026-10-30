#!/bin/bash
# 阶段 C（防御基线）+ D（报告智能体消融）：等 P0 完成后在 GLM 上接力
# C：BDUP（重复过滤=D1+D4）/ trustrag / ppl × {clean,r2,r4,hard,adapt} + 6 会议基线 × {r2,r4,hard}
# D：R1/R2 双通道消融（clean，防御 off）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6

# 1. 等 P0 收尾
ok=0
for i in $(seq 1 2400); do
  grep -q "P0 UNIFY DONE" experiments/logs/p0_driver.log && { ok=1; break; }
  sleep 60
done
[ "$ok" = 1 ] || { echo "FATAL: P0 未在 40h 内完成"; exit 1; }
sleep 30

# 2. 部署 GLM
pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/glm-4-9b-chat --served-model-name GLM-4-9B \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 16384 \
  --trust-remote-code > /tmp/vllm_bl.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B && { ok=1; echo "[deploy] healthy"; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: deploy failed"; tail -10 /tmp/vllm_bl.log; exit 1; }
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src/settings.yaml

runp () {  # setting defense extra...（extra 为 --ablation X 或 --baseline Y）
  local s=$1 d=$2; shift 2
  echo "=== [$(date '+%m-%d %H:%M:%S')] $s $d $* ==="
  $PY experiments/run_poison.py --action run --setting "$s" --defense "$d" \
    --suffix s42 --resume "$@" > "experiments/logs/bl_${s}_${d}_${1:-plain}_$2.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/bl_${s}_${d}_${1:-plain}_$2.log" | tail -1
}
runb () {  # 会议基线
  local m=$1 s=$2
  echo "=== [$(date '+%m-%d %H:%M:%S')] baseline $m $s ==="
  $PY experiments/run_poison_baselines.py --method "$m" --setting "$s" --resume \
    > "experiments/logs/blbase_${m}_${s}.log" 2>&1
  grep -E "C=" "experiments/logs/blbase_${m}_${s}.log" | tail -1
}

# ── D：报告智能体消融（clean，防御 off；参照 = p0 的 clean_off_s42）──
runp clean off --ablation R1
runp clean off --ablation R2

# ── C-clean ──
runp clean on --ablation BDUP
runp clean off --baseline trustrag
runp clean off --baseline ppl

# ── C-easy（r2/r4）──
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
for s in r2 r4; do
  runp $s on --ablation BDUP
  runp $s off --baseline trustrag
  runp $s off --baseline ppl
done
for m in naive_rag iter_retgen arbgraph cyberrag rag_intel search_o1; do
  runb $m r2
done
for m in naive_rag iter_retgen arbgraph cyberrag rag_intel search_o1; do
  runb $m r4
done

# ── C-hard ──
$PY experiments/run_poison.py --action inject --tier hard >> experiments/logs/multi_inject.log 2>&1
runp hard on --ablation BDUP
runp hard off --baseline trustrag
runp hard off --baseline ppl
for m in naive_rag iter_retgen arbgraph cyberrag rag_intel search_o1; do
  runb $m hard
done

# ── C-adaptive ──
$PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
runp adapt on --ablation BDUP
runp adapt off --baseline trustrag
runp adapt off --baseline ppl
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "########## BASELINES+C/D DONE ########## [$(date '+%m-%d %H:%M:%S')]"
