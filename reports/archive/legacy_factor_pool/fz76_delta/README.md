# FZ76 candidate delta

This report is built from the teammate release package `data-fz76-20260729-v1`.

The package covers 2019-2021 only, so it is treated as a development-window
candidate-pool delta. Do not overwrite `data/factors/candidate_pool.parquet`,
which is the formal 2019-2023 pool.

Rebuild on AutoDL after extracting the release asset:

```bash
/root/autodl-tmp/conda-envs/quant/bin/python scripts/build_fz76_candidate_pool_delta.py
/root/autodl-tmp/conda-envs/quant/bin/python scripts/screen_fz76_candidate_delta.py
```

Generated local data:

- `data/factors/candidate_pool_fz76_delta.parquet` is ignored by Git.
- `data/manifest_candidate_pool_fz76_delta.json` records the local data hash.
- `development_rank_ic.csv` is the compact 2019-2021 Rank IC summary.
- `development_daily_rank_ic.csv` is the daily Rank IC detail.
