# Factor Wiki remaining component generators

This directory contains the executable formulas that were missing from the
candidate-only PR. Candidate modules remain small wrappers, while these
scripts rebuild their named daily component columns from the agreed E2E
one-minute files. No minute data or generated Parquet is committed.

## Scope

- `build_daily_base.py` builds the adjusted daily OHLCV base used by long-window
  formulas.
- `build_changjiang_components.py` is the exact implementation used for the
  Changjiang sandbox. It emits all locally implementable columns and asserts
  that the 123 submitted component columns in
  `submission_manifest_changjiang_123.csv` are present.
- `build_haitong_components.py` is the exact implementation used for the
  Haitong sandbox. It emits 47 frozen Haitong columns and asserts the 11 still
  missing from the current `main` against
  `submission_manifest_haitong_11.csv`.
- `build_cicc_remaining_components.py` preserves the exact CICC remaining-batch
  construction.  With `BIGALPHA_CICC_RAW_ONLY=1`, it emits the nine components
  constructed directly from minute close/volume without requiring prior
  component panels; eight map directly to submitted candidates and `CICC-043`
  is the paired lag-correlation helper retained for reproducibility.  Full mode
  additionally combines the previously generated Haitong and CICC shape
  primitives and covers all 13 rows in `submission_manifest_cicc_13.csv`.
- `changjiang_static_dedup_352.csv` is the frozen formula/data-routing audit read
  by the Changjiang builder.  It is metadata, not market data.

## Exact remaining-132 handoff

Against `origin/main` at `0183338`, the generation gap is exactly:

- 120 Changjiang HF components (`HF-105` through `HF-224`);
- 11 Haitong HF components (`HF-092` through `HF-102`);
- one Changjiang PV component (`PV-219`, source `CJ-G112-V01`).

The Changjiang manifest also contains `PV-217` and `PV-218`, so the generator
remains a complete reproduction of its sandbox. Current `main` already builds
those two. Current `main` also already contains executable builders for the
CICC 13 and 14 eligible Haitong PV components. `PV-216` is not part of the
remaining 132 because its 252-day history exceeds the active 126-observation
submission contract and it has been retired from the candidate tree.

The code resolves the repository and project roots at runtime.  Input/output
locations can be overridden without editing formulas:

```bash
export BIGALPHA_PROJECT_ROOT=/path/to/project
export BIGALPHA_E2E_BAR1M_DIR=/path/to/bigalpha_2026_e2e_bar1m
export BIGALPHA_INSTRUMENT_MAP=/path/to/instrument_map.csv
export BIGALPHA_FACTOR_WIKI_WORK=/path/to/generated/factor_wiki_remaining

python scripts/factor_wiki_remaining/build_daily_base.py
python scripts/factor_wiki_remaining/build_haitong_components.py
python scripts/factor_wiki_remaining/build_changjiang_components.py
```

For the directly generated CICC block:

```bash
BIGALPHA_CICC_RAW_ONLY=1 \
python scripts/factor_wiki_remaining/build_cicc_remaining_components.py
```

For full CICC mode, also set:

```bash
export BIGALPHA_HAITONG_PRIMITIVES=/path/to/haitong_minute_primitives_2019_2021.parquet
export BIGALPHA_CICC_DIRECT_COMPONENTS=/path/to/cicc79_remaining19_daily_2019_2021.parquet
export BIGALPHA_CICC_PILOT_COMPONENTS=/path/to/cicc79_pilot_daily_2019_2021.parquet
python scripts/factor_wiki_remaining/build_cicc_remaining_components.py
```

## Previously abbreviated semantics

- Changjiang `method 1` is the pooled 20-day coefficient of variation computed
  from the underlying 5-minute observations, not the mean of daily CV values.
- `local statistic` means the within-stock-day 5-minute empirical quintile
  statistic.  Ties use pandas average ranks; each submitted variant fixes the
  selector and quintile in code.
- `CJ-G112-V01` is the trailing-21-day Spearman correlation of daily volume and
  adjusted close.  Ranks are recomputed inside each trailing window.  The old
  sandbox wording `global TS rank` was removed because full-history ranks would
  allow future observations to alter past values.

All minute returns are computed separately for the morning and afternoon
sessions, so no return crosses the lunch break.  Generated values use only the
current or earlier trading dates and are available after the current close.
