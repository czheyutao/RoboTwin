#!/bin/bash
# 作用：等待当前 FastWAM/RoboTwin 评测进程组结束后，自动启动 step_010000 的评测脚本。
# 执行示例：
#   bash script/wait_then_run_fastwam_eval_step010000.sh
#   WAIT_PGID=1797803 bash script/wait_then_run_fastwam_eval_step010000.sh

set -euo pipefail

WAIT_PGID="${WAIT_PGID:-1797803}"
ROBOTWIN_DIR="/data/hyt/RoboTwin"
TARGET_SCRIPT="$ROBOTWIN_DIR/script/run_fastwam_eval_step010000.sh"
LOG_DIR="$ROBOTWIN_DIR/logs"
mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] wait pgid=$WAIT_PGID before running $TARGET_SCRIPT"

while ps -eo pgid= | awk -v target="$WAIT_PGID" '$1 == target { found = 1 } END { exit found ? 0 : 1 }'; do
  echo "[$(timestamp)] pgid=$WAIT_PGID still running, sleep 300s"
  sleep 300
done

echo "[$(timestamp)] pgid=$WAIT_PGID finished, start target script"
cd "$ROBOTWIN_DIR"
exec bash "$TARGET_SCRIPT"
