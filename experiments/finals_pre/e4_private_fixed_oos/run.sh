#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/projects/bigquant-default
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
experiment=$root/experiments/finals_pre/e4_private_fixed_oos
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$root"
"$python_bin" experiments/finals_pre/common/build_private_c2c_labels.py
"$python_bin" "$experiment/infer_fixed.py"
"$python_bin" "$experiment/score_fixed.py"
