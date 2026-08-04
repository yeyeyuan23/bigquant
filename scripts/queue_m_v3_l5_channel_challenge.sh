#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/projects/bigquant-m-v3-l5-channels-20260803
m_v2_root=/root/autodl-tmp/projects/bigquant-m-v2-l5-gated-20260803/reports/m_v2_flow_gated_20260803
run_root="$project_root/reports/m_v3_l5_channels_20260803"
mkdir -p "$run_root"

while [[ ! -s "$m_v2_root/j_scores.json" ]]; do
  timestamp=$(date -Iseconds)
  gpu_state=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)
  echo "$timestamp status=waiting_for_m_v2_control gpu=$gpu_state"
  sleep 60
done

echo "$(date -Iseconds) status=starting_m_v3_l5_channels"
exec bash "$project_root/scripts/run_m_v3_l5_channel_challenge.sh"
