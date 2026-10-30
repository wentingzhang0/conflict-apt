#!/bin/bash
# 侧任务：clean-D5a（τ=0.80 干净回归）——τ_dup 定标证据补齐
# 等接力链 GLM 就绪后在 8213 上并行跑（clean 设定不碰 APT_POISON，与 E 阶段语料兼容）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
for i in $(seq 1 2400); do
  grep -q "P0 UNIFY DONE" experiments/logs/p0_driver.log && break
  sleep 60
done
for i in $(seq 1 360); do
  curl -s --max-time 3 http://localhost:8213/v1/models | grep -q GLM-4-9B && break
  sleep 10
done
CUDA_VISIBLE_DEVICES=6 $PY experiments/run_poison.py --action run --setting clean \
  --defense on --ablation D5a --suffix s42 --resume \
  > experiments/logs/fu_clean_on_D5a.log 2>&1
grep -E "Correct" experiments/logs/fu_clean_on_D5a.log | tail -1
echo "########## SIDE clean-D5a DONE ##########"
