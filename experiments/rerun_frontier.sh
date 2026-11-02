#!/bin/bash
# 重跑 phase B（同骨干攻击者端到端）：等 final_backfill 收尾 → 部署 GLM → f1/f2 × off/on
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
VLLM=~/anaconda3/envs/vllm/bin/python
export CUDA_VISIBLE_DEVICES=6

# 1. 等当前链收尾（其末尾会 kill vllm 并恢复 GLM 配置）
ok=0
for i in $(seq 1 900); do
  grep -q "FINAL BACKFILL DONE" experiments/logs/final_backfill_driver.log && { ok=1; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: final_backfill 未在 2.5h 内完成"; exit 1; }
sleep 20

# 2. 清理失败残留（error: 'data' 的 4 个文件）
rm -f experiments/results/glm-4-9b/srcnew_taa_f1_glmatt_off.json \
      experiments/results/glm-4-9b/srcnew_taa_f1_glmatt_off_report.json \
      experiments/results/glm-4-9b/srcnew_taa_f1_glmatt_on.json \
      experiments/results/glm-4-9b/srcnew_taa_f1_glmatt_on_report.json \
      experiments/results/glm-4-9b/srcnew_taa_f2_glmatt_off.json \
      experiments/results/glm-4-9b/srcnew_taa_f2_glmatt_off_report.json \
      experiments/results/glm-4-9b/srcnew_taa_f2_glmatt_on.json \
      experiments/results/glm-4-9b/srcnew_taa_f2_glmatt_on_report.json

# 3. 部署 GLM
pkill -u zwt -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 8
CUDA_VISIBLE_DEVICES=2 nohup $VLLM -m vllm.entrypoints.openai.api_server \
  --model /mnt/nas_share/qwen3/glm-4-9b-chat --served-model-name GLM-4-9B \
  --port 8213 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  --trust-remote-code > /tmp/vllm_glm_rerun.log 2>&1 &
ok=0
for i in $(seq 1 90); do
  curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B && { ok=1; echo "[deploy] healthy"; break; }
  sleep 10
done
[ "$ok" = 1 ] || { echo "FATAL: GLM deploy failed"; tail -10 /tmp/vllm_glm_rerun.log; exit 1; }
sed -i 's|model_name: "[^"]*"|model_name: "GLM-4-9B"|' src_new/settings.yaml

# 4. 注入 + 四组 run（50q）
runf () {
  echo "=== [$(date '+%m-%d %H:%M:%S')] $1 $2 ==="
  $PY experiments/run_poison.py --action run --setting "$1" --defense "$2" --resume \
    --suffix glmatt > "experiments/logs/fin_$1_$2.log" 2>&1
  grep -E "Correct .*framed" "experiments/logs/fin_$1_$2.log" | tail -1
}
$PY experiments/run_poison.py --action inject --tier L1_75 --n-per-query 2 >> experiments/logs/multi_inject.log 2>&1
tail -2 experiments/logs/multi_inject.log | grep -v Warning | head -1
runf f1 off; runf f1 on
$PY experiments/run_poison.py --action inject --tier L2_75 --n-per-query 2 >> experiments/logs/multi_inject.log 2>&1
runf f2 off; runf f2 on
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1
echo "########## RERUN FRONTIER DONE ########## [$(date '+%m-%d %H:%M:%S')]"
