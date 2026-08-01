#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATE_STORE="/root/autodl-tmp/candidate454_completion_full_2019_2024/candidate454_store"
MANIFEST="$CANDIDATE_STORE/candidate454_manifest.json"
VALIDATION="$CANDIDATE_STORE/candidate454_validation.json"
STATE_ROOT="$PROJECT_ROOT/reports/unified_alpha_fusion_gate"
MAX_POLLS="${UNIFIED_GATE_MAX_POLLS:-720}"

mkdir -p "$STATE_ROOT"
cd "$PROJECT_ROOT"

poll=0
while [[ ! -s "$MANIFEST" || ! -s "$VALIDATION" ]]; do
  poll=$((poll + 1))
  if ((poll > MAX_POLLS)); then
    echo "[$(date '+%F %T')] timed out waiting for validated candidate454 store" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] waiting for candidate454 manifest and validation ($poll/$MAX_POLLS)"
  sleep 60
done

echo "[$(date '+%F %T')] candidate454 artifacts detected; starting unified suite"
exec bash run_unified_alpha_fusion_suite.sh
