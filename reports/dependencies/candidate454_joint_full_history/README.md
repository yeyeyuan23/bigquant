# Candidate454 joint-v2 frozen full-history J (2019–2024)

## Decision

Under the requested full-history proxy, `x_tree_frozen` ranks first.  This is
the strongest submission candidate in this table, followed by `m_raw_frozen`,
`x_mlp_frozen`, `t_frozen`, and `m_l5_frozen`.

This ranking includes checkpoint training dates.  It is neither strict OOS
evidence nor an official platform score, so the saturated top scores must not
be presented as expected platform scores.

## Scoring contract

- One continuous score over 2019–2024; yearly J values are not averaged.
- Common scored interval: 1,394 trading days and 1,391,994 rows.  The start is
  determined by the 60-day EN/T warm-up.
- `J = 0.3 * A + 0.7 * B`; no extra `J_stable` or second standard-deviation
  penalty is applied.
- B uses 67 rolling weight windows and already contains the weight-standard-
  deviation term through ModelScore.
- All ten candidate routes are fitted simultaneously with Candidate454, so B
  measures incremental weight in this exact route pool.

The default-branch scorer was rerun on 2026-08-05 without training.  Both CSV
reports and all ten materialized route parquets were byte-identical to this
report; the detailed JSON differed only in its temporary output paths.

## Ranking

| Rank | Route | J | A | B | B ModelScore | B mean | B std | B nonzero |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `x_tree_frozen` | 1.000000 | 1.000000 | 1.000000 | 3.882431 | 0.377105 | 0.097131 | 1.000000 |
| 2 | `m_raw_frozen` | 0.998491 | 1.000000 | 0.997845 | 3.370645 | 0.036165 | 0.010729 | 1.000000 |
| 3 | `x_mlp_frozen` | 0.995499 | 0.995055 | 0.995690 | 3.197956 | 0.124523 | 0.038938 | 0.970149 |
| 4 | `t_frozen` | 0.990973 | 0.995055 | 0.989224 | 2.446580 | 0.033236 | 0.013585 | 1.000000 |
| 5 | `m_l5_frozen` | 0.977725 | 0.996154 | 0.969828 | 1.766792 | 0.027343 | 0.015476 | 0.925373 |
| 6 | `en454_existing_causal` | 0.974085 | 0.953846 | 0.982759 | 1.965503 | 0.029453 | 0.014985 | 0.940299 |
| 7 | `t_residual_frozen` | 0.506105 | 0.882418 | 0.344828 | 0.910057 | 0.004766 | 0.005237 | 0.641791 |
| 8 | `x_en_tres_w020` | 0.310091 | 0.993407 | 0.017241 | 0.476585 | 0.004007 | 0.008408 | 0.268657 |
| 9 | `x_en_w050` | 0.308582 | 0.993407 | 0.015086 | 0.464079 | 0.005416 | 0.011671 | 0.268657 |
| 10 | `x_en_tres_w010` | 0.301039 | 0.993407 | 0.004310 | 0.248460 | 0.001576 | 0.006344 | 0.074627 |

The X+EN combinations retain high A but have very low incremental B in this
joint pool.  They are therefore not preferred over the individual frozen
routes by this comparison.

## Exposure neutralization audit

The consistent six-year panel uses `SIZE`, `LIQUIDTY`, and
`industry_level1_code`; redundant `float_market_cap` is retained in the source
audit but excluded from regression because SIZE is present.

- Rows/days: 1,456,000 / 1,456
- Missing rows per exposure column: 1,437 (0.098695%)
- Exposure panel SHA256:
  `2cd890e0f2dfba0c9242728cb55b10d63b31c61801cb31b469317fd10aac2788`
- Processing: daily 1%/99% winsorization, daily z-score, then daily OLS
  residualization with intercept and industry fixed effects.

The downloaded ten-style platform exposure panel exists only for 2024 and is
kept as a separate 2024 diagnostic.  It is not mixed into this six-year table,
and this six-year neutralization is not described as proprietary BARRA.

## Frozen checkpoint dependencies

Only the checkpoints and checkpoint manifests referenced by the six frozen
base routes are retained.  Each replay manifest records the repository-relative
checkpoint path and SHA256; `tests/test_frozen_six_year_artifacts.py` verifies
that all six files exist, match their declared hashes, and are inference-only.
The shared Candidate454/exposure and microstructure stores remain external data
dependencies and must not be deleted as part of branch cleanup.

## Artifacts

- `joint_official_proxy_summary.csv`: concise ranking.
- `joint_official_proxy_ab_details.csv`: A four raw components and percentiles,
  B ModelScore/mean/std/nonzero, common rows, days, and windows.
- `joint_official_proxy_detailed.json`: route provenance, input hashes, full
  exposure column coverage and yearly exposure source hashes.
- `validation.json`: formula, schema, evidence-boundary, and artifact-hash QA.
- `routes_manifest.json`: exact ten-route joint pool.
