#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/root/autodl-tmp/projects/bigquant-default/experiments/finals_pre/e5_raw23_direct/queue_e5_autodl.sh
PID_FILE=/tmp/e5_autodl_queue.pid
LOG=/tmp/e5_autodl_queue.log

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(<"$PID_FILE")
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "E5 AutoDL queue already running pid=$existing_pid"
    exit 0
  fi
fi

nohup bash "$SCRIPT" >"$LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "started E5 AutoDL queue pid=$pid log=$LOG"
