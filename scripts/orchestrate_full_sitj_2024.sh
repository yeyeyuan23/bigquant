#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/projects/bigquant"
PYTHON="/root/autodl-tmp/conda-envs/quant/bin/python"
cd "$ROOT"

mkdir -p logs profiles
RUN_ID="full_sitj_$(date +%Y%m%d_%H%M%S)"
echo "$RUN_ID" > logs/latest_full_sitj_run.txt

wait_external_workers() {
  local worker_file="$1"
  while read -r _name pid _start _end; do
    while kill -0 "$pid" 2>/dev/null; do
      sleep 10
    done
  done < "$worker_file"
}

validate_worker_logs() {
  local worker_file="$1"
  local expected_text="$2"
  while read -r name _pid _start _end; do
    if ! grep -q "$expected_text" "logs/${name}.log"; then
      echo "worker failed or incomplete: $name" >&2
      tail -80 "logs/${name}.log" >&2
      return 1
    fi
  done < "$worker_file"
}

echo "===== WAIT COMPONENT WORKERS ====="
CICC_WORKERS="$(cat logs/latest_cicc_component_workers.txt)"
FZ_WORKERS="$(cat logs/latest_fz_component_workers.txt)"
wait_external_workers "$CICC_WORKERS" &
WAIT_CICC=$!
wait_external_workers "$FZ_WORKERS" &
WAIT_FZ=$!
wait "$WAIT_CICC"
wait "$WAIT_FZ"
validate_worker_logs "$CICC_WORKERS" "CICC component preparation complete"
validate_worker_logs "$FZ_WORKERS" "FZ component preparation complete"

CICC_COMPONENTS="$(
  find data/runtime/e2e_generation_internal_v2_20260730/components/cicc34 \
    -maxdepth 1 -type f -name '*.parquet' | wc -l
)"
FZ_COMPONENTS="$(
  find data/runtime/e2e_generation_internal_v2_20260730/components/fz76_minute_daily \
    -maxdepth 1 -type f -name '*.parquet' | wc -l
)"
test "$CICC_COMPONENTS" -eq 36
test "$FZ_COMPONENTS" -eq 48
echo "components ready: CICC=$CICC_COMPONENTS FZ=$FZ_COMPONENTS"

echo "===== FINALIZE TEAMMATE FACTORS ====="
CICC_DELTA="data/factors/candidate_pool_cicc34_202201_202412_internal_v2_delta.parquet"
FZ_DELTA="data/factors/candidate_pool_fz76_202201_202412_internal_v2_delta.parquet"
if [[ -s "$CICC_DELTA" && -s "$FZ_DELTA" ]]; then
  echo "reusing completed teammate deltas"
else
  env \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=96 \
    OMP_NUM_THREADS=48 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    "$PYTHON" -m cProfile \
    -o "profiles/${RUN_ID}_cicc_finalize.prof" \
    data/runtime/e2e_generation_20260729/build_teammate_delta_2022_2023.py \
    --which cicc --start 2022-01 --end 2024-12 \
    > "logs/${RUN_ID}_cicc_finalize.log" 2>&1 &
  P_CICC=$!

  env \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=96 \
    OMP_NUM_THREADS=48 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    "$PYTHON" -m cProfile \
    -o "profiles/${RUN_ID}_fz_finalize.prof" \
    data/runtime/e2e_generation_20260729/build_teammate_delta_2022_2023.py \
    --which fz --start 2022-01 --end 2024-12 --fz-start 2021-01 \
    > "logs/${RUN_ID}_fz_finalize.log" 2>&1 &
  P_FZ=$!
  wait "$P_CICC"
  wait "$P_FZ"
fi

echo "===== BUILD BASE46 2024 DELTA ====="
BASE46_DELTA="data/factors/candidate_pool_base46_2024_delta.parquet"
if [[ -s "$BASE46_DELTA" ]]; then
  echo "reusing completed base46 2024 delta"
else
  env \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=96 \
    OMP_NUM_THREADS=48 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    "$PYTHON" -m cProfile \
    -o "profiles/${RUN_ID}_base46_2024.prof" \
    scripts/build_base46_2024_delta.py \
    > "logs/${RUN_ID}_base46_2024.log" 2>&1
fi
"$PYTHON" - <<'PY'
import polars as pl
old_path = "data/factors/candidate_pool.parquet"
new_path = "data/factors/candidate_pool_base46_2024_delta.parquet"
old = pl.scan_parquet(old_path).select(
    pl.col("candidate_id").n_unique().alias("candidates"),
    pl.col("date").max().alias("date_max"),
).collect().row(0, named=True)
new = pl.scan_parquet(new_path).select(
    pl.col("candidate_id").n_unique().alias("candidates"),
    pl.col("date").min().alias("date_min"),
    pl.col("date").max().alias("date_max"),
).collect().row(0, named=True)
if old["candidates"] != 46 or old["date_max"].year != 2023:
    raise RuntimeError(f"frozen base46 history changed unexpectedly: {old}")
if (
    new["candidates"] != 46
    or new["date_min"].year != 2024
    or new["date_max"].year != 2024
):
    raise RuntimeError(f"base46 2024 delta incomplete: {new}")
print(f"base46 ready: old={old}, new={new}")
PY

echo "===== BUILD CORRECTED RUNTIME ====="
RUNTIME="data/runtime/all156_full_2019_2024_corrected"
if [[ -s "$RUNTIME/factors/candidate_pool.parquet" && -s "$RUNTIME/manifest_candidate_pool.json" ]]; then
  echo "reusing completed corrected runtime"
else
  env \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=208 \
    OMP_NUM_THREADS=208 \
    "$PYTHON" -m cProfile \
    -o "profiles/${RUN_ID}_runtime.prof" \
    scripts/build_full156_runtime_2024.py \
    > "logs/${RUN_ID}_runtime.log" 2>&1
fi

"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

runtime = Path("data/runtime/all156_full_2019_2024_corrected")
pool = runtime / "factors" / "candidate_pool.parquet"
manifest = json.loads((runtime / "manifest_candidate_pool.json").read_text())
digest = hashlib.sha256()
with pool.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
expected = str(manifest.get("sha256", ""))
if actual != expected:
    raise RuntimeError(
        f"corrected runtime SHA mismatch: actual={actual}, expected={expected}"
    )
if int(manifest.get("duplicate_keys", -1)) != 0:
    raise RuntimeError("corrected runtime manifest does not certify duplicate_keys=0")
print(f"runtime manifest verified: sha256={actual}")
PY

REPORTS="reports/${RUN_ID}"
mkdir -p "$REPORTS"
if [[ ! -e "$REPORTS/first_round" ]]; then
  ln -s ../first_round "$REPORTS/first_round"
fi

echo "===== RUN S I T AND THREE J ROUTES ====="
env \
  PYTHONUNBUFFERED=1 \
  POLARS_MAX_THREADS=208 \
  OMP_NUM_THREADS=208 \
  BIGALPHA_LIGHTGBM_NUM_THREADS=208 \
  MKL_NUM_THREADS=32 \
  OPENBLAS_NUM_THREADS=32 \
  NUMEXPR_MAX_THREADS=208 \
  CUDA_VISIBLE_DEVICES=0 \
  BIGALPHA_LIGHTGBM_DEVICE_TYPE=cuda \
  BIGALPHA_TRUST_CANDIDATE_POOL_MANIFEST=1 \
  "$PYTHON" -m cProfile \
  -o "profiles/${RUN_ID}_sitj.prof" \
  scripts/run_combinations.py \
  --admission-routes sit \
  --data-dir "$RUNTIME" \
  --reports-dir "$REPORTS" \
  --single-factor-cache-dir "$RUNTIME/cache/single_factor_v5_full2024" \
  --incremental-cache-dir "$RUNTIME/cache/incremental_v9_s_full2024" \
  --tree-cache-dir "$RUNTIME/cache/tree_v8_i_full2024" \
  > "logs/${RUN_ID}_sitj.log" 2>&1

echo "$REPORTS" > logs/latest_full_sitj_reports.txt
echo "status=complete run_id=$RUN_ID reports=$REPORTS"
