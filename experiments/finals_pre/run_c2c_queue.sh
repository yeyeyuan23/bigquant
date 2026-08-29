#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
e15_pid_file=/tmp/e15_c2c_driver.pid
e15_log=/tmp/e15_c2c_driver.log

if [[ ! -s "$e15_pid_file" ]]; then
  echo "missing E15 pid file: $e15_pid_file" >&2
  exit 1
fi
e15_pid=$(<"$e15_pid_file")
while kill -0 "$e15_pid" 2>/dev/null; do
  sleep 60
done
if ! grep -q "\[all done\]" "$e15_log"; then
  echo "E15 ended without completion marker; refusing to start E1/E2" >&2
  exit 1
fi

cd "$root"
echo "[start E1] $(date -Is)"
bash experiments/finals_pre/e1_c2c_walkforward/run.sh
echo "[done E1] $(date -Is)"
echo "[start E2] $(date -Is)"
bash experiments/finals_pre/e2_c2c_epoch_curve/run.sh
echo "[done E2] $(date -Is)"
