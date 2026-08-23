#!/usr/bin/env bash
# Run the whole E9 matrix on 3 workers, then commit + push, then power off.
# Shutdown is gated on a SUCCESSFUL push: if the push fails the box stays up so
# nothing is stranded on a disk nobody can reach.
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
D=$P/experiments/finals_pre/e9_seed_and_label_matrix
OUT=$P/reports/dependencies/finals_pre/e9_seed_and_label_matrix
cd "$P"
echo "[E9 START] $(date -Is)  27 jobs / 3 workers"
for i in 1 2 3; do "$D/worker.sh" > "$OUT/worker_$i.log" 2>&1 & done
wait
ok=$(grep -h "^\[done " "$OUT"/worker_*.log 2>/dev/null | wc -l)
bad=$(grep -h "^\[FAIL " "$OUT"/worker_*.log 2>/dev/null | wc -l)
echo "[E9 END] $(date -Is)  done=$ok fail=$bad"

echo "[commit] $(date -Is)"
git add -A
git commit -q -m "E9: 种子与标签矩阵（27 个 run）

P1 E3 家族用 o2o 重训三种子（完整版那一支用已有的 E6b）——检验架构结论
    是否依赖训练标签，而不只是评估标签。
P2 (2,10,45) 用 o2o 重训三种子，与 E6b 匹配对比——把唯一有争议的核尺度
    结论放到训练/评估口径一致的条件下判。
P3 其余 o2c 消融补到 3 种子——deck 里所有单种子数字改为均值 ± 散布。

done=$ok fail=$bad" || echo "[commit] 无改动"
if git push -q origin feature/unified-alpha-fusion; then
  echo "[push ok] $(git rev-parse --short HEAD)  $(date -Is)"
  echo "[shutdown] $(date -Is)"
  sync; shutdown -h now
else
  echo "[push FAILED] 机器保持开机，结果还在磁盘上 $(date -Is)"
fi
