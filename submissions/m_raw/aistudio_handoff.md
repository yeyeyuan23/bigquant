# M expanding-history e3 AIStudio handoff

This bundle is a frozen inference-only candidate. Local J/A/B values are not platform scores.

## Upload together

- `unified_m_raw.ipynb`
- `unified_m_raw.py`
- `unified_m_base.py`
- `unified_m_temporal.py`
- `unified_m_microstructure.py`
- `unified_m_checkpoint.py`

The notebook must contain exactly one code cell: `from unified_m_raw import main`.

## Evidence already passed

- Python compilation for all five modules.
- Embedded checkpoint SHA-256 equals `252c39baf946f494898fcd0e2c3a0aeca4de2ece378b9de6386a5200378e1dcb`.
- Frozen checkpoint decoding and synthetic 3-stock inference are finite with shape `[1, 3]`.
- Bundle manifest and training-window manifest are present.

## AIStudio checks still required

1. Run a real five-trading-day window using platform `bar1m` and `bigalpha_2026_instruments`.
2. Require exactly the stock-pool keys for every requested day; expected benchmark is 5,000 rows when the pool has 1,000 stocks per day.
3. Require columns exactly `date, instrument, factor`, unique keys, finite factor values, and a nonconstant cross-section every day.
4. Run the same-start full/cutoff prefix probe. Pass only when common-prefix `different_rows=0`.
5. Save elapsed time and peak memory. Treat timeout, OOM, missing rows, extra rows, or a constant day as failure.
6. Record the official platform score only after an actual competition submission; do not infer it from this validation.

Current status: local static validation passed; real AIStudio output-contract and prefix validation pending.
