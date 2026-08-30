#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/projects/bigquant-default
CODE=$ROOT/experiments/finals_pre/e17_raw23_direct
STORE=/root/bigquant_private_data/e17_raw40_2023_2024_parquet
OUT=$ROOT/reports/dependencies/finals_pre/e17_raw23_direct/c2c_autodl
C2C_QUEUE_PID=/tmp/c2c_queue.pid
C2C_QUEUE_LOG=/tmp/c2c_queue.log
PULL_PID=/tmp/e17_parquet_pull.pid
PULL_LOG=/tmp/e17_parquet_pull.log

wait_for_pid_file() {
  local pid_file=$1
  local label=$2
  if [[ ! -s "$pid_file" ]]; then
    echo "missing $label pid file: $pid_file" >&2
    return 1
  fi
  local pid
  pid=$(<"$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

echo "[wait] C2C E1/E2 queue $(date --iso-8601=seconds)"
wait_for_pid_file "$C2C_QUEUE_PID" "C2C queue"
if ! grep -q '\[done E2\]' "$C2C_QUEUE_LOG"; then
  echo "E1/E2 queue ended without E2 completion marker" >&2
  exit 1
fi

echo "[wait] E17 Parquet pull $(date --iso-8601=seconds)"
wait_for_pid_file "$PULL_PID" "Parquet pull"
if ! grep -q '\[done\]' "$PULL_LOG"; then
  echo "Parquet pull ended without completion marker" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "[audit] transferred Parquet store $(date --iso-8601=seconds)"
/root/autodl-tmp/conda-envs/quant/bin/python \
  "$CODE/e17_audit_parquet_store.py" \
  --store "$STORE" \
  --output "$OUT/parquet_store_audit.json"

echo "[smoke] CUDA loader/model $(date --iso-8601=seconds)"
/root/autodl-tmp/conda-envs/quant/bin/python \
  "$CODE/e17_autodl_smoke.py" \
  --root "$CODE" \
  --store "$STORE" \
  --device cuda \
  >"$OUT/cuda_smoke.json"

echo "[train] E17 formal AutoDL runs $(date --iso-8601=seconds)"
bash "$CODE/run_e17_autodl.sh"
