#!/usr/bin/env bash
# Keep TARGET trainers busy until the queue drains, then finalise.
#
# Replaces the fixed "3 workers then wait" orchestrator, which had two flaws:
#   - concurrency decayed as its own workers exited while orphans still ran;
#   - its `wait` only covered its own children, so it could power off the box
#     while a job started by some other worker was still training.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
D=$P/experiments/finals_pre/e9_seed_and_label_matrix
OUT=$P/reports/dependencies/finals_pre/e9_seed_and_label_matrix
TARGET=6
cd "$P"
# pgrep -fc PRINTS 0 and also EXITS non-zero when nothing matches, so a
# `|| echo 0` fallback emits "0\n0" and every [ ] comparison then explodes --
# which is what made the supervisor declare the queue drained while jobs were
# still queued. Take the count and default only an empty string.
running() { local n; n=$(pgrep -fc "evaluate_unifie[d]|evaluate_ablatio[n]" 2>/dev/null); echo "${n:-0}"; }
claimed() { local n; n=$(ls "$OUT/claims" 2>/dev/null | grep -vc '^\.'); echo "${n:-0}"; }

echo "[SUPERVISE START] $(date -Is) target=$TARGET"
TOTAL=$(grep -c . "$D/jobs.txt")
while [ "$(claimed)" -lt "$TOTAL" ] || [ "$(running)" -gt 0 ]; do
  n=$(running); w=$(pgrep -fc "worker.s[h]" 2>/dev/null || echo 0)
  if [ "$(claimed)" -lt 27 ] && [ "$w" -lt "$TARGET" ]; then
    i=$((w + 1))
    nohup "$D/worker.sh" > "$OUT/worker_x$i.$(date +%s).log" 2>&1 &
    echo "[topup] worker -> $((w+1))  (running=$n claimed=$(claimed)) $(date -Is)"
  fi
  sleep 60
done
echo "[QUEUE DRAINED] $(date -Is)"
# A failed job kept its claim, so it would silently vanish from the queue.
# Sweep twice: release every claim with no result, and let the workers retry.
for pass in 1 2; do
  while [ "$(running)" -gt 0 ]; do sleep 30; done
  missing=0
  for f in "$OUT"/claims/*; do
    n=$(basename "$f"); [ "$n" = ".lock" ] && continue
    if [ ! -s "$OUT/$n/oos_metrics.json" ]; then rm -f "$f"; missing=$((missing+1)); fi
  done
  [ "$missing" -eq 0 ] && { echo "[retry pass $pass] 无缺口"; break; }
  echo "[retry pass $pass] 释放 $missing 个失败任务重跑 $(date -Is)"
  for i in 1 2 3 4 5 6; do
    nohup "$D/worker.sh" > "$OUT/worker_r${pass}_$i.log" 2>&1 &
  done
  sleep 90
  TOTAL=$(grep -c . "$D/jobs.txt")
while [ "$(claimed)" -lt "$TOTAL" ] || [ "$(running)" -gt 0 ]; do sleep 60; done
done
# belt and braces: nothing may still be training when we commit
while [ "$(running)" -gt 0 ]; do sleep 30; done

ok=$(cat "$OUT"/worker_*.log 2>/dev/null | grep -c "^\[done ")
bad=$(cat "$OUT"/worker_*.log 2>/dev/null | grep -c "^\[FAIL ")
echo "[E9 END] $(date -Is) done=$ok fail=$bad"
git add -A
git commit -q -m "E10: o2o 核组与补种子 ($TOTAL jobs, done=$ok fail=$bad)" || echo "[commit] 无改动"
if git push -q origin feature/unified-alpha-fusion; then
  echo "[push ok] $(git rev-parse --short HEAD) $(date -Is)"
  sync; echo "[shutdown] $(date -Is)"; shutdown -h now
else
  echo "[push FAILED] 保持开机 $(date -Is)"
fi
