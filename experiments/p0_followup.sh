#!/bin/bash
# P0 接力链（单一所有权，避免语料库状态竞争）：
#   E. GLM 投毒矩阵补齐 s42（r1/r4/hard/adapt × on/off）
#   D. 报告智能体消融 R1/R2（clean，防御 off）
#   C. 防御基线：BDUP/trustrag/ppl × {clean,r2,r4,hard,adapt} + 6 会议基线 × {r2,r4,hard}
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6

ok=0
for i in $(seq 1 2400); do
  grep -q "P0 UNIFY DONE" experiments/logs/p0_driver.log && { ok=1; break; }
  sleep 60
done
[ "$ok" = 1 ] || { echo "FATAL: P0 未在 40h 内完成"; exit 1; }
sleep 30

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/glm-4-9b-chat --served-model-name GLM-4-9B \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --trust-remote-code > /tmp/vllm_fu.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B && { ok=1; echo "[deploy] healthy"; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: deploy failed"; tail -10 /tmp/vllm_fu.log; exit 1; }
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src/settings.yaml

runp () {
  local s=$1 d=$2; shift 2
  local tag="${1:-plain}"
  if [ $# -gt 0 ] && [ "${1:0:2}" != "--" ]; then shift; fi
  echo "=== [$(date '+%m-%d %H:%M:%S')] $s $d $tag ==="
  $PY experiments/run_poison.py --action run --setting "$s" --defense "$d" \
    --suffix s42 --resume "$@" > "experiments/logs/fu_${s}_${d}_$tag.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/fu_${s}_${d}_$tag.log" | tail -1
}
runb () {
  local m=$1 s=$2
  echo "=== [$(date '+%m-%d %H:%M:%S')] baseline $m $s ==="
  $PY experiments/run_poison_baselines.py --method "$m" --setting "$s" --resume \
    > "experiments/logs/fu_base_${m}_${s}.log" 2>&1
  grep -E "C=" "experiments/logs/fu_base_${m}_${s}.log" | tail -1
}
BASEM="naive_rag iter_retgen arbgraph cyberrag rag_intel search_o1"

# ── E：GLM 矩阵补齐（P0 已含 clean/r2/D 系）──
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
runp r1 off; runp r1 on; runp r4 off; runp r4 on
$PY experiments/run_poison.py --action inject --tier hard >> experiments/logs/multi_inject.log 2>&1
runp hard off; runp hard on
$PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
runp adapt off; runp adapt on
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

# ── D：报告智能体消融（clean，防御 off）──
runp clean off R1 --ablation R1
runp clean off R2 --ablation R2

# ── C-clean ──
runp clean on BDUP --ablation BDUP
runp clean off trustrag --baseline trustrag
runp clean off ppl --baseline ppl

# ── C-easy ──
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
for s in r2 r4; do
  runp $s on BDUP --ablation BDUP
  runp $s off trustrag --baseline trustrag
  runp $s off ppl --baseline ppl
done
for m in $BASEM; do runb $m r2; done
for m in $BASEM; do runb $m r4; done

# ── C-hard ──
$PY experiments/run_poison.py --action inject --tier hard >> experiments/logs/multi_inject.log 2>&1
runp hard on BDUP --ablation BDUP
runp hard off trustrag --baseline trustrag
runp hard off ppl --baseline ppl
for m in $BASEM; do runb $m hard; done

# ── C-adaptive ──
$PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
runp adapt on BDUP --ablation BDUP
runp adapt off trustrag --baseline trustrag
runp adapt off ppl --baseline ppl
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1

pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "########## P0-FOLLOWUP ALL DONE ########## [$(date '+%m-%d %H:%M:%S')]"
