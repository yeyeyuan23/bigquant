#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/projects/bigquant-unified-integration
python_bin=/root/autodl-tmp/conda-envs/quant/bin/python
data_root=/root/autodl-tmp/projects/bigquant/data
micro_store=/root/autodl-tmp/unified_microstructure_store_v2_2019_2024
run_root="$project_root/reports/m_expanding_history_retrain_20260803"

mkdir -p "$run_root"
cd "$project_root"

run_holdout() {
  local epochs="$1"
  local output_dir="$run_root/holdout_2024_e${epochs}"
  local checkpoint="$output_dir/unified_microstructure_block_00_checkpoint.pt"
  local predictions="$output_dir/unified_microstructure_full_oos.parquet"
  local manifest="$output_dir/run_manifest.json"
  local metrics="$output_dir/oos_metrics.json"
  if [[ -s "$checkpoint" && -s "$predictions" && -s "$manifest" && -s "$metrics" ]]; then
    echo "holdout_2024_e${epochs}=complete; reusing existing artifacts"
    return 0
  fi
  "$python_bin" scripts/evaluate_unified_microstructure.py \
    --data-root "$data_root" \
    --micro-store "$micro_store" \
    --output-dir "$output_dir" \
    --years 2024 \
    --train-start-year 2019 \
    --training-mode expanding \
    --prediction-days 999 \
    --epochs "$epochs" \
    --model-dim 96 \
    --kernels 3 15 60 \
    --tcn-blocks 3 \
    --tail-minutes 30 \
    --max-minutes 242 \
    --max-stocks 1200 \
    --min-train-days 900 \
    --learning-rate 4e-4 \
    --seed 20260801
}

run_holdout 3
run_holdout 6

"$python_bin" scripts/score_submission_j_stability.py \
  "$run_root/holdout_2024_e3/unified_microstructure_full_oos.parquet" \
  "$run_root/holdout_2024_e6/unified_microstructure_full_oos.parquet" \
  --years 2024 \
  --data-dir "$data_root" \
  --reports-dir "$project_root/reports" \
  --cache-dir "$run_root/score_cache" \
  --output "$run_root/holdout_j_scores.json" \
  --summary-csv "$run_root/holdout_j_scores.csv"

best_epochs=$(
  "$python_bin" -c '
import json
from pathlib import Path

path = Path("/root/autodl-tmp/projects/bigquant-unified-integration/reports/m_expanding_history_retrain_20260803/holdout_j_scores.json")
summaries = json.loads(path.read_text())["summaries"]
winner = max(summaries, key=lambda row: (row["J_stable"], row["J_worst"]))
print(6 if "_e6/" in winner["version"] else 3)
'
)

"$python_bin" -c '
import json
from pathlib import Path

run_root = Path("/root/autodl-tmp/projects/bigquant-unified-integration/reports/m_expanding_history_retrain_20260803")
scores = json.loads((run_root / "holdout_j_scores.json").read_text())
winner = max(scores["summaries"], key=lambda row: (row["J_stable"], row["J_worst"]))
payload = {
    "selection_protocol": "2019_2023_train_2024_untouched_holdout",
    "selected_epochs": 6 if "_e6/" in winner["version"] else 3,
    "winner": winner,
    "evidence_boundary": "Local J proxy; final checkpoint is not an official platform score.",
}
(run_root / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
'

final_output="$run_root/final_full_history_e${best_epochs}"
"$python_bin" scripts/evaluate_unified_microstructure.py \
  --data-root "$data_root" \
  --micro-store "$micro_store" \
  --output-dir "$final_output" \
  --years 2023 2024 \
  --train-start-year 2019 \
  --training-mode expanding \
  --prediction-days 20 \
  --epochs "$best_epochs" \
  --model-dim 96 \
  --kernels 3 15 60 \
  --tcn-blocks 3 \
  --tail-minutes 30 \
  --max-minutes 242 \
  --max-stocks 1200 \
  --min-train-days 1200 \
  --learning-rate 4e-4 \
  --seed 20260801 \
  --block-indices 25

"$python_bin" -c '
import hashlib
import json
from pathlib import Path

run_root = Path("/root/autodl-tmp/projects/bigquant-unified-integration/reports/m_expanding_history_retrain_20260803")
selection = json.loads((run_root / "selection.json").read_text())
epochs = selection["selected_epochs"]
output_dir = run_root / f"final_full_history_e{epochs}"
checkpoint = output_dir / "unified_microstructure_block_25_checkpoint.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
metrics = json.loads((output_dir / "oos_metrics.json").read_text())
payload = {
    "status": "complete",
    "selected_epochs": epochs,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
    "final_block_diagnostics": metrics[0],
    "next_step": "Build a separate AIStudio candidate from this checkpoint; do not overwrite the prior M submission before validation.",
}
(run_root / "final_checkpoint.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
'
