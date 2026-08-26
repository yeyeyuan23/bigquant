#!/usr/bin/env bash
# E10：通道组消融 —— 拿掉整组通道后从零重训。
#
# 已有的 E1「置换重要性」测的是**冻结模型有多依赖**某组通道；它无法回答
# 「拿掉之后模型重新学，能不能从别的通道补回来」。同一份置换数据里已经能看到
# 冗余的迹象：每组整体的影响都远大于组内各通道单独之和，个别通道打乱后 IC 还变好。
# 所以置换通常**高估**必要性，必须用重训来验。
#
# 口径与 o2o 主臂完全一致（label 用 ret_open_to_open），才能和完整双通路配对比较。
set -uo pipefail
P=/root/autodl-tmp/projects/bigquant-default
PY=/root/autodl-tmp/conda-envs/quant/bin/python
DATA=/root/autodl-tmp/projects/bigquant/data
STORE=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
O2O=$P/reports/dependencies/finals_pre/shared/o2o_labels.parquet
EXP=$P/experiments/finals_pre/e3_pathway_ablation
OUT=$P/reports/dependencies/finals_pre/e10_channel_ablation
mkdir -p "$OUT"; cd "$P"

run() {
  name="$1"; seed="$2"; shift 2
  echo "[start] $name seed=$seed $(date -Is)"
  if "$PY" "$EXP/evaluate_ablation.py" \
      --data-root "$DATA" --micro-store "$STORE" --output-dir "$OUT/$name" \
      --years 2024 --train-start-year 2019 --prediction-days 999 --epochs 3 \
      --model-dim 96 --kernels 3 15 60 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 --seed "$seed" \
      --fast-pack --label-column ret_open_to_open --extra-labels "$O2O" \
      "$@" > "$OUT/$name.log" 2>&1
  then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}
"$@"
