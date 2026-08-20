#!/usr/bin/env bash
# GPU lane 1: seed robustness for (3,15,60) full and (2,10,45).
set -uo pipefail
project_root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
out_root=$project_root/reports/dependencies/finals_pre_20260819
cd "$project_root"
run_variant() {
  name="$1"; shift
  outdir="$out_root/e2c_seed_robustness/$name"
  if [ -s "$outdir/unified_microstructure_full_oos.parquet" ]; then echo "[skip ] $name"; return 0; fi
  echo "[start] $name $(date -Is)"
  if "$python_bin" scripts/evaluate_unified_microstructure.py \
      --data-root "$data_root" --micro-store "$micro_store" --output-dir "$outdir" \
      --years 2024 --train-start-year 2019 --training-mode expanding --prediction-days 999 \
      --epochs 3 --model-dim 96 --tcn-blocks 3 --tail-minutes 30 --max-minutes 242 \
      --max-stocks 1200 --min-train-days 900 --learning-rate 4e-4 \
      "$@" > "$out_root/e2c_${name}.log" 2>&1; then echo "[done ] $name $(date -Is)"; else echo "[FAIL ] $name $(date -Is)"; fi
}
mkdir -p "$out_root/e2c_seed_robustness"
run_variant full_seed12   --kernels 3 15 60 --seed 20260812
run_variant full_seed23   --kernels 3 15 60 --seed 20260823
run_variant k21045_seed12 --kernels 2 10 45 --seed 20260812
run_variant k21045_seed23 --kernels 2 10 45 --seed 20260823
echo "[SEED LANE DONE] $(date -Is)"
