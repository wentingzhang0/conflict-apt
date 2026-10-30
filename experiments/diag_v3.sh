#!/bin/bash
# CORE-2 分数分布诊断：DIAG 臂（三信号中性化）+ --audit-scores，clean/r1/adapt × 10 query
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
export CUDA_VISIBLE_DEVICES=6

run () {
  echo "=== [$(date '+%m-%d %H:%M:%S')] diag $1 ==="
  $PY experiments/run_poison.py --action run --setting "$1" --defense on \
    --end 10 --ablation DIAG --suffix diag --audit-scores \
    > "experiments/logs/diag_$1.log" 2>&1
  grep -E "Correct" "experiments/logs/diag_$1.log" | tail -1
}

run clean
$PY experiments/run_poison.py --action inject --tier easy >> experiments/logs/multi_inject.log 2>&1
run r1
$PY experiments/run_poison.py --action inject --tier adaptive >> experiments/logs/multi_inject.log 2>&1
run adapt
$PY experiments/run_poison.py --action clean >> experiments/logs/multi_inject.log 2>&1
echo "########## V3 DIAG DONE ##########"
