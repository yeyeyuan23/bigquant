# EN454 + Residual T 20% submission

This is a formal `main(datasources, start_date, end_date)` bundle. It rebuilds
Candidate454 once, trains the 60-day rolling ElasticNet causally, runs the
frozen residual-T checkpoint, and returns their daily-rank blend:

`factor = daily_rank(0.80 * EN454 + 0.20 * residual_T)`

## Upload exactly these four files together

- `unified_en454_t_residual.ipynb`
- `unified_en454_t_residual.py`
- `unified_en454_t_residual_weights.py`
- `unified_candidate454_runtime.py`

## AIStudio validation

Run this from the directory containing the four files:

```bash
set -o pipefail
jupyter nbconvert --to script --stdout unified_en454_t_residual.ipynb \
  > /tmp/en454_t_residual_notebook_export.py
python -m py_compile \
  unified_en454_t_residual.py \
  unified_en454_t_residual_weights.py \
  unified_candidate454_runtime.py \
  /tmp/en454_t_residual_notebook_export.py \
  2>&1 | tee en454_t_residual_compile.log
```

Then put the repository's `aistudio_submission_lookahead_probe.py` beside this
directory and run:

```bash
python aistudio_submission_lookahead_probe.py \
  en454_t_residual_020/unified_en454_t_residual.py \
  --start 2024-01-02 \
  --cutoff 2024-01-05 \
  --end 2024-01-08 \
  --bar1m bigalpha_2026_stock_bar1m \
  --financial bigalpha_2026_financial \
  2>&1 | tee en454_t_residual_prefix.json
```

Use the competition's actual table names if they differ. Do not shorten the
history window: Candidate454 needs its own long lookbacks, EN454 needs 60
training days, and residual-T needs 60 temporal days.

## Exact pass criteria

- compilation exits with code 0;
- probe `status` and `full_output_contract.status` are `ok`;
- every cutoff has `difference_rows=0`, `max_abs_diff=0.0`, and
  `prefix_invariant=true`;
- columns are exactly `date, instrument, factor`;
- keys are unique, factor values are finite, and every day is nonconstant;
- all loaded submission modules resolve inside this four-file directory.

The local A/B/J evidence used to select 20% is a proxy, not an official score.
Only an accepted competition run supplies an official platform score.
