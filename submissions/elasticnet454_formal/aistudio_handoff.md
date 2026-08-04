# Candidate454 rolling ElasticNet formal submission

This is a formal `main(datasources, start_date, end_date)` submission bundle,
not a probe notebook.  It causally reconstructs the frozen 454-feature schema,
trains a rolling ElasticNet online, and returns exactly
`date, instrument, factor`.

## Upload exactly these six files together

- `unified_candidate454_elasticnet.ipynb`
- `unified_candidate454_elasticnet.py`
- `unified_candidate454_spec.py`
- `unified_candidate454_components.py`
- `unified_candidate454_direct.py`
- `unified_candidate454_gtja.py`

The notebook has one executable cell:
`from unified_candidate454_elasticnet import main`.

## Frozen model contract

- Candidate schema: exactly 454 unique factors.
- Target: next-trading-day open-to-close return, ranked cross-sectionally.
- Rolling model: 60 trading days train, one trading day label isolation,
  then at most 20 trading days predict.
- ElasticNet: `alpha=0.001`, `l1_ratio=0.5`, `max_iter=20000`, cyclic
  coordinate descent.
- Prediction-day ranking uses every stock in the current stock-pool keys.  It
  deliberately does not consult next-day label availability.

## AIStudio copy-paste validation

Run this in the directory containing the six uploaded files.  The notebook
export goes to `/tmp` so it cannot overwrite the canonical model module.

```bash
set -o pipefail
jupyter nbconvert --to script --stdout unified_candidate454_elasticnet.ipynb \
  > /tmp/candidate454_elasticnet_notebook_export.py
python -m py_compile \
  unified_candidate454_elasticnet.py \
  unified_candidate454_spec.py \
  unified_candidate454_components.py \
  unified_candidate454_direct.py \
  unified_candidate454_gtja.py \
  /tmp/candidate454_elasticnet_notebook_export.py \
  2>&1 | tee candidate454_elasticnet_compile.log
```

Then copy the repository's current `aistudio_submission_lookahead_probe.py`
beside the upload directory and run:

```bash
python aistudio_submission_lookahead_probe.py \
  elasticnet454_formal/unified_candidate454_elasticnet.py \
  --start 2024-01-02 \
  --cutoff 2024-01-05 \
  --end 2024-01-08 \
  --bar1m bigalpha_2026_stock_bar1m \
  --financial bigalpha_2026_financial \
  2>&1 | tee candidate454_elasticnet_prefix.json
```

Use the actual table names shown by the competition if they differ.  This
route reconstructs more than 300 calendar days of minute history so that both
the 126-trading-day candidate lookbacks and the 60-day model window are real;
do not shorten the history or stock pool to make the check pass.

## Exact pass criteria

- compilation exits with code 0;
- `status` is `ok`;
- `full_output_contract.status` is `ok`;
- every cutoff has `difference_rows=0`, `max_abs_diff=0.0`, and
  `prefix_invariant=true`;
- output columns are exactly `date, instrument, factor`;
- output keys match `bigalpha_2026_instruments`, are unique, and all factor
  values are finite with a nonconstant cross-section each day;
- `submission_import_root` is the uploaded `elasticnet454_formal` directory
  and every `loaded_submission_modules` path is inside it.

Only an accepted competition run supplies an official score.  Local OOS,
AutoDL replay, compilation, and the prefix probe are separate evidence.
