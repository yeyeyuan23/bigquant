# EN454 stability J before M-multiaxis

This is a local Candidate454 joint proxy over 2019-04-04 through 2024-12-30.
The two new EN routes and the existing EN454 route are continuous causal OOS
after a 60-day warmup. Frozen X/T/M-raw routes include checkpoint training
dates, so the absolute pool ranking is not strict full-pool OOS and is not an
official platform score.

`m_multiaxis_full` is intentionally excluded. Only `m_raw_frozen` is retained
among old M routes; M-L5, M residual, and old M blends are excluded.

| Rank | Route | J | A | B | IC | ICIR | Sharpe | Stress IR |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `x_tree_frozen` | 1.000000 | 1.000000 | 1.000000 | 0.229434 | 2.566562 | 30.696820 | 2.239611 |
| 2 | `x_mlp_frozen` | 0.997001 | 0.995055 | 0.997835 | 0.074548 | 1.107686 | 7.440338 | 0.762261 |
| 3 | `t_frozen` | 0.989426 | 0.995055 | 0.987013 | 0.079696 | 1.028791 | 7.332433 | 0.736353 |
| 4 | `m_raw_frozen` | 0.984848 | 1.000000 | 0.978355 | 0.074957 | 1.377373 | 13.814226 | 1.112577 |
| 5 | `en454_existing_causal` | 0.980093 | 0.953846 | 0.991342 | 0.040233 | 0.583082 | 5.062684 | 0.405381 |
| 6 | `t_residual_frozen` | 0.634422 | 0.882418 | 0.528139 | 0.036868 | 0.448972 | 1.962400 | 0.357382 |
| 7 | `pema_full454` | 0.428933 | 0.924725 | 0.216450 | 0.040520 | 0.536267 | 3.783228 | 0.306822 |
| 8 | `clean_pema` | 0.324254 | 0.919231 | 0.069264 | 0.041043 | 0.535018 | 3.504413 | 0.291924 |

## EN decision

Neither new route improves the A composite. Relative to the existing EN454,
P-EMA changes A by -0.029121 and Clean-P-EMA changes A by -0.034615. Both have
slightly higher mean IC but lower ICIR, Sharpe, and stress IR.

The very low B values for the new routes are coexistence evidence: all three
highly related EN routes are in the same joint regression, where the existing
EN454 is selected more strongly. Those B values should not be interpreted as a
controlled one-for-one replacement score.

The maximum absolute error in `J = 0.3*A + 0.7*B` is
`1.1102230246251565e-16`.
