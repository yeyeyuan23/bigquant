#!/usr/bin/env bash
# 给每个 o2o 消融臂算 N 分第一层。基座已缓存，所以每个候选只要几十秒。
# 之前只有 M_raw 和几个对照有 N —— 「每个实验一行 A/B/N」缺的就是这一块。
set -uo pipefail
# 用法：run_nscore_all_arms.sh [isotonic|linear] [输出子目录名]
# 默认 isotonic（2026-08-26 起的现行口径）；linear 用于复现更早的数。
RESIDUAL=${1:-isotonic}
OUTNAME=${2:-nscore_all_arms}
R=/root/autodl-tmp/projects/bigquant-default
P=/root/autodl-tmp/conda-envs/quant/bin/python
FP=$R/reports/dependencies/finals_pre
NAME=unified_microstructure_full_oos.parquet

args=()
for d in $FP/e9_seed_and_label_matrix/o2o_*/; do
  n=$(basename $d)
  [ -f "$d/$NAME" ] && args+=(--candidate "$n=$d/$NAME")
done
for s in 20260801 20260812 20260823; do
  f=$FP/e6b_o2o_label/seed$s/$NAME
  [ -f "$f" ] && args+=(--candidate "BASE_full_o2o_$s=$f")
done
args+=(--candidate "linear85_o2o=$FP/e3_pathway_ablation/linear85_o2o/$NAME")
# E11：21 通道臂。跑在另一条 lane 上，产物目录不同，所以单列一段。
for s in 20260801 20260812 20260823 20260904 20260915; do
  f=$FP/e11_book_orders/seed$s/$NAME
  [ -f "$f" ] && args+=(--candidate "e11_book_orders_$s=$f")
done
for a in daily6 trade book; do for s in s01 s12 s23; do
  f=$FP/e10_channel_ablation/${a}_${s}/$NAME
  [ -f "$f" ] && args+=(--candidate "e10_${a}_${s}=$f")
done; done

echo "候选数 $(( ${#args[@]} / 2 ))"
$P $R/experiments/finals_pre/e7_nscore/e7_layer1_ric.py \
  --base $FP/e7_nscore/base/y_pool_oos.parquet \
  --data-root /root/autodl-tmp/data --years 2024 \
  --label-column ret_open_to_open \
  --extra-labels $FP/shared/o2o_labels.parquet \
  --residual "$RESIDUAL" \
  --output-dir $FP/full_neutralization/$OUTNAME \
  "${args[@]}"
