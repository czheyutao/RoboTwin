#!/bin/bash
# RoboTwin 数据采集脚本
# 用法：bash collect_data.sh <task_name> <task_config> <gpu_id>
# 示例：
#   bash collect_data.sh open_microwave demo_clean 0
#   bash collect_data.sh open_microwave demo_randomized 1

CONDA_PYTHON=/data/hyt/anaconda3/envs/fastwam_robotwin/bin/python

task_name=${1}
task_config=${2}
gpu_id=${3}

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
$CONDA_PYTHON script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
