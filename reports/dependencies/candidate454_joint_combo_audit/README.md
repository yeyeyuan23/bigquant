# Candidate454 joint-v2 combination audit (2019-2024)

## Scope

- One continuous 2019-2024 score; yearly scores are not averaged.
- `J = 0.3 * A + 0.7 * B`; no `J_stable`.
- Frozen current predictions only; no retraining.
- Includes checkpoint training dates. This is not strict OOS evidence and not
  an official platform score.

## Shared run metadata

| Field | Value |
|---|---:|
| Candidate454 reference factors | 454 |
| Joint candidate routes | 18 |
| Common scored rows | 1,391,994 |
| Scored trading days | 1,394 |
| B rolling weight windows | 67 |

These fields describe the joint run and therefore must not be interpreted as
route-specific metrics.

## Ranking

| Rank | Route | J | A | B | B mean | B std | B nonzero |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `x_tree_frozen` | 1.000000 | 1.000000 | 1.000000 | 0.375992 | 0.096586 | 1.000000 |
| 2 | `x_mlp_frozen` | 0.997033 | 0.995055 | 0.997881 | 0.125687 | 0.040155 | 0.970149 |
| 3 | `t_frozen` | 0.989618 | 0.995055 | 0.987288 | 0.033797 | 0.015596 | 1.000000 |
| 4 | `m_l5_frozen` | 0.981050 | 0.996154 | 0.974576 | 0.027984 | 0.016001 | 0.925373 |
| 5 | `en454_existing_causal` (`X_en454`) | 0.931281 | 0.953846 | 0.921610 | 0.025246 | 0.017392 | 0.805970 |
| 6 | `m_raw_frozen` | 0.577331 | 1.000000 | 0.396186 | 0.024583 | 0.026031 | 0.701493 |
| 7 | `t_residual_frozen` | 0.433793 | 0.882418 | 0.241525 | 0.005336 | 0.006453 | 0.626866 |
| 8 | `full_boosting_current_rebuilt` | 0.343832 | 0.997802 | 0.063559 | 0.008064 | 0.013416 | 0.462687 |
| 9 | `m75_xtree25` | 0.326695 | 1.000000 | 0.038136 | 0.010693 | 0.020427 | 0.313433 |
| 10 | `x_en_tres_w020` | 0.318785 | 0.993407 | 0.029661 | 0.003937 | 0.008494 | 0.268657 |
| 11 | `m50_t50` | 0.314501 | 0.998901 | 0.021186 | 0.005672 | 0.013367 | 0.238806 |
| 12 | `x_en_w050` | 0.311369 | 0.993407 | 0.019068 | 0.004721 | 0.011476 | 0.223881 |
| 13 | `m50_xmlp50` | 0.310052 | 0.998901 | 0.014831 | 0.005450 | 0.017256 | 0.149254 |
| 14 | `m75_en25` | 0.308898 | 1.000000 | 0.012712 | 0.002354 | 0.007592 | 0.149254 |
| 15 | `m75_xmlp25` | 0.305932 | 1.000000 | 0.008475 | 0.001824 | 0.007019 | 0.104478 |
| 16 | `en_tres_w020_uploaded` | 0.301480 | 0.965385 | 0.016949 | 0.002868 | 0.008664 | 0.149254 |
| 17 | `x_en_tres_w010` | 0.300988 | 0.993407 | 0.004237 | 0.001211 | 0.004931 | 0.074627 |
| 18 | `en_tres_w010` | 0.295712 | 0.960989 | 0.010593 | 0.002600 | 0.009139 | 0.089552 |

## Interpretation and missing route

The timed-out uploaded route is `en_tres_w020_uploaded`. Its six-year local J
is `0.301480`; its high A is offset by very low incremental B, so it is not a
good submission candidate under this pool.

`full_boosting_current_rebuilt` applies the historical orthogonal fusion
formula to the current X-MLP, T-residual, and M-raw frozen outputs. It is a
current-component reconstruction, not the byte-identical historical package,
whose embedded checkpoint hashes differ.

The user's pure `X_en454` route is `en454_existing_causal`: the causal
ElasticNet combination regression over Candidate454. It is already included in
this six-year joint pool. It is distinct from both `x_en_w050` (a 50/50 blend
of X-MLP and X_en454) and the historical `tree_res_s3` experiment (an X-tree
trained on an EN454 residual target).

Because B is fitted jointly, adding correlated X/M/residual blends can split
weights and change every route's B. Scores from this 18-route table must not be
mixed with the earlier 10-route table.
