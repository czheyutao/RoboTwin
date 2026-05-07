#!/bin/bash
set -euo pipefail

PYTHON=/data/hyt/anaconda3/envs/fastwam_robotwin/bin/python
SCRIPT=script/eval_fastwam_challenge_resume_parallel.py
CKPT=third_party/FastWAM/checkpoints/fastwam_trained/step_010000.pt
DATASET_STATS=third_party/FastWAM/checkpoints/fastwam_trained/dataset_stats.json
GPU_IDS=0,1,2,3,4
TASKS=move_stapler_pad,place_fan,handover_mic,open_microwave,place_can_basket,place_dual_shoes,move_can_pot,stack_blocks_three,blocks_ranking_rgb,blocks_ranking_size
BASE_OUTDIR=/data/hyt/RoboTwin/evaluate_results/fastwam_challenge_subset_parallel/step_010000_layers16_lowfps

COMMON_ARGS=(
  --ckpt "$CKPT"
  --dataset-stats-path "$DATASET_STATS"
  --gpu-ids "$GPU_IDS"
  --tasks "$TASKS"
  --video-dit-num-layers 16
  --action-dit-num-layers 16
)

echo "===== [1/2] Clean 5ep ====="
$PYTHON $SCRIPT \
  "${COMMON_ARGS[@]}" \
  --task-config demo_clean \
  --eval-num-episodes 5 \
  --output-dir "$BASE_OUTDIR/clean5"

echo "===== [2/2] Randomized 15ep ====="
$PYTHON $SCRIPT \
  "${COMMON_ARGS[@]}" \
  --task-config demo_randomized \
  --eval-num-episodes 15 \
  --output-dir "$BASE_OUTDIR/randomized15"

echo "===== Done ====="
