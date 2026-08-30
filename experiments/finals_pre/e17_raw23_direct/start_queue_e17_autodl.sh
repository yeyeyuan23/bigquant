#!/usr/bin/env bash
set -euo pipefail

SCRIPT=/root/autodl-tmp/projects/bigquant-default/experiments/finals_pre/e17_raw23_direct/queue_e17_autodl.sh
PID_FILE=/tmp/e17_autodl_queue.pid
LOG=/tmp/e17_autodl_queue.log

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(<"$PID_FILE")
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "E17 AutoDL queue already running pid=$existing_pid"
    exit 0
  fi
fi

nohup bash "$SCRIPT" >"$LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "started E17 AutoDL queue pid=$pid log=$LOG"
