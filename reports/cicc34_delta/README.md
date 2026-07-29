# CICC34 candidate delta

This report is built from the teammate release package `data-cicc34-20260729-v1`.

The package covers 2019-2021 only, so it is treated as a development-window
candidate-pool delta. Do not overwrite `data/factors/candidate_pool.parquet`,
which is the formal 2019-2023 pool.

The CICC34 package combines new delta Feather files with the existing
`MICRO_DAILY_FULL` base data for `OB-006` and the `INT-004` spread member.

Rebuild on AutoDL after extracting the release asset:

```bash
/root/autodl-tmp/conda-envs/quant/bin/python scripts/build_cicc34_candidate_pool_delta.py
/root/autodl-tmp/conda-envs/quant/bin/python scripts/screen_fz76_candidate_delta.py \
  --pool data/factors/candidate_pool_cicc34_delta.parquet \
  --manifest data/manifest_candidate_pool_cicc34_delta.json \
  --output reports/cicc34_delta/development_rank_ic.csv \
  --daily-output reports/cicc34_delta/development_daily_rank_ic.csv
```

Generated local data:

- `data/factors/candidate_pool_cicc34_delta.parquet` is ignored by Git.
- `data/manifest_candidate_pool_cicc34_delta.json` records the local data hash.
- `development_rank_ic.csv` is the compact 2019-2021 Rank IC summary.
- `development_daily_rank_ic.csv` is the daily Rank IC detail and is ignored.
