#!/bin/bash
# 全量实验矩阵驱动：18 个 run × 50 query（更新方案 §7）
# 用法：bash experiments/run_matrix.sh <group>
#   base    = clean/r1/r2/r4/hard/adapt × off/on（12 runs）
#   ablate  = D1-D5 on r2（6 runs）
set -u
cd /home/zwt/conflict_apt
PY=~/anaconda3/envs/pytorch/bin/python
export CUDA_VISIBLE_DEVICES=6
GRP=${1:-base}

run () {  # setting defense ablation
  local s=$1 d=$2 a=${3:-}
  local cmd="$PY experiments/run_poison.py --action run --setting $s --defense $d --resume"
  [ -n "$a" ] && cmd="$cmd --ablation $a"
  echo "=== [$(date +%H:%M:%S)] $s $d $a ==="
  $cmd >> experiments/logs/${s}_${d}${a:+_$a}.log 2>&1
  tail -6 experiments/logs/${s}_${d}${a:+_$a}.log | grep -E "Correct|framed|poison_chunks|avg"
}

case $GRP in
base)
  for s in clean r1 r2 r4 hard adapt; do
    for d in off on; do run $s $d; done
  done ;;
ablate)
  for a in D1 D2 D3 D4 D5a D5b; do run r2 on $a; done ;;
esac
echo "=== GROUP $GRP DONE ==="
