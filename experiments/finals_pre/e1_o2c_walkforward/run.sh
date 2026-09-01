#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
raw=/root/bigquant_private_data/bigalpha_2026_stock_bar1m_private_20250101_20260828
private_store=/root/autodl-tmp/unified_microstructure_store_private1m_2025_2026
out=$root/reports/dependencies/finals_pre/e1_o2c_walkforward

mkdir -p "$out"
cd "$root"
if [[ ! -s "$raw/manifest.json" ]]; then
  echo "missing private bar1m manifest: $raw/manifest.json" >&2
  exit 1
fi
parts=("$raw"/part_*.parquet)
if [[ ! -e "${parts[0]}" ]]; then
  echo "private bar1m dataset contains no parquet parts" >&2
  exit 1
fi
if [[ ! -s "$private_store/manifest.json" ]]; then
  "$python_bin" scripts/prepare_unified_microstructure_store.py \
    --input "${parts[@]}" \
    --output-dir "$private_store" \
    --source-profile canonical \
    --resume
fi
"$python_bin" experiments/finals_pre/common/build_private_o2c_labels.py
"$python_bin" experiments/finals_pre/e1_o2c_walkforward/train.py
"$python_bin" experiments/finals_pre/e1_o2c_walkforward/score.py
