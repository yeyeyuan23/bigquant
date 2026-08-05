# Current reports

The canonical current J result is the continuous six-year Candidate454
joint-v2 comparison over 2019-2024 with 18 candidate routes.

## Files to read

- `J_2019_2024_18_ROUTES_FULL.csv`: the main route table. This is the only
  compact table needed for comparing routes. It contains all 27 columns from
  the audited `joint_official_proxy_ab_details.csv` schema.
- `J_2019_2024_18_ROUTES_DETAILED.json`: complete route provenance, A/B
  internals, exposure audit, and materialized-route hashes.
- `J_2019_2024_18_ROUTES_MANIFEST.json`: exact 18-route joint pool.
- `J_2019_2024_18_ROUTES_VALIDATION.json`: schema, completeness, uniqueness,
  and formula checks for the canonical CSV.

## Shared run metadata

| Field | Value |
|---|---:|
| Period | 2019-2024 continuous |
| Candidate454 reference factors | 454 |
| Joint candidate routes | 18 |
| Common scored rows | 1,391,994 |
| Scored trading days | 1,394 |
| B rolling weight windows | 67 |

`reference_factor_count`, `joint_route_count`, `joint_common_rows`,
`score_days`, and `score_weight_windows` are run-level metadata repeated on
every route row for a self-contained export.

## Full CSV schema

The canonical CSV contains these 27 columns, in order:

```text
group, route, family, years,
J, A, B, score_proxy, a_proxy, b_proxy,
a_rank_ic_mean, a_rank_ic_ir, a_long_short_sharpe, a_stress_ic_ir,
a_rank_ic_mean_percentile, a_rank_ic_ir_percentile,
a_long_short_sharpe_percentile, a_stress_ic_ir_percentile,
b_model_score, b_mean_abs_weight, b_std_abs_weight,
b_nonzero_window_ratio, score_days, score_weight_windows,
reference_factor_count, joint_route_count, joint_common_rows
```

## Evidence boundary

The score is `J = 0.3 * A + 0.7 * B`, with no `J_stable`. Frozen predictions
are scored over a period that includes checkpoint training dates. The result
is not strict OOS evidence and is not an official platform score.

The reproducibility source remains under
`reports/candidate454_joint_combo_audit_2019_2024_20260805/`; frozen replay and
checkpoint dependencies remain in their original directories and must not be
removed merely to simplify the report view.
