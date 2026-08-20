# Finals-pre experiment suite (2026-08-19 → )

Protocol unless noted: expanding history from 2019-01-02, 1-trading-day label
isolation (asserted), predict untouched 2024; seed 20260801; RTX 4090D.
Artifacts live under `reports/dependencies/finals_pre_20260819/`; every
checkpoint sha256 is recorded in the sibling `oos_metrics.json`.

| ID | Question | Status | Headline result |
|---|---|---|---|
| E0 | Rebuild lost e3/e6 holdout evidence | done (`aeaf1a6`) | e3 OOS RankIC 0.0418 vs e6 0.0373 — deeper fit degrades OOS; proxy J/A/B in `holdout_j_scores.json` |
| E1 | Pathway ablation + Linear-85 baseline | done 08-20 evening | Mean IC clusters 0.042-0.044 for every variant except stats_only (0.0334), but the full model wins BOTH stability metrics: ICIR 0.391 / LS-Sharpe 3.73 vs seq_only 0.379/3.48, no_deepsets 0.354/3.30, stats_only 0.345/2.49, Linear-85 ridge 0.273/1.07. Every component buys stability, not mean IC — matching what the contest actually scores. N-framework pool-increment (E7 Layer-1, same seed/window): all variants clear noise (t 3.3-5.0), pairwise dN vs full all insignificant (worst -seq -0.0053, paired t -1.28) — full table in e1_pathway_ablation/N_ABLATION_RESULTS.md |
| E2 | Single-scale kernel ablation k3/k15/k60 | done (`aeaf1a6`) | 0.0444 / 0.0453 / 0.0425 vs full 0.0418; cross-scale corr 0.83–0.89 — scales carry unshared info, 1x1 fusion leans to k60 |
| E2b | Kernel-value sensitivity (5,30,120)/(2,10,45) | done (`fda450a`) | 0.0421 / 0.0458 — narrow band, log-spaced design robust to exact values |
| E2c | Seed robustness (3 seeds × full, k21045) | done 08-20 | full 0.0419±0.0004, (2,10,45) 0.0451±0.0006 — the gap is 5-8× seed spread, the smaller-kernel edge is real |
| E4 | Multi-block walk-forward, 26 blocks 2023-24 | running (2 workers: w1 blocks 00-15, w2 16-25) | per-block dirs `e4_walkforward/block_XX/`, blocks 00-05 done: IC 0.041/0.086/…/0.043 |
| E7 | N score: pool-incremental value (`e7_incremental_score/DESIGN.md`) | done (`bb1bf80`) | Layer-1 residual RIC vs LightGBM-454 base (own OOS IC 0.063): noise +0.001 (t 0.5), in-pool −0.004 (t −1.5), EN454 +0.014 (t 1.6 ns), **M_raw +0.0246 (t 4.9)**. Layer-2 paired ΔLGBM demoted to backup (no power in 60d windows). FR-001 LOO addendum aborted by request |
| E3 | Main-result analysis pack (A sub-metrics, neutralized stress split, deciles, rolling IC) | next (08-21 daytime) | stress split MUST use platform-style neutralization |
| E5 | EN blend dose-response vs EN454 | next (after E4 merge) | — |
| — | Channel permutation importance (frozen e3 checkpoint) | next (after E4 frees GPU) | supports the "book/trade-structure channels carry the increment" claim |

Environment note (08-20): the conda env carried a stale editable install of
`bigalpha_2026_factors` pointing at the retired main worktree src; the
single-worktree cleanup broke ambient `import bigalpha2026` for scripts
without their own sys.path bootstrap. Re-installed editable from this
worktree — ambient imports now resolve to `bigquant-default/src`.

Presentation-side documents (deck, speaker notes, study guide, audits,
platform-score record) are maintained locally in `BigAlpha2026_Pre/`; the
repository is now private, so they may be merged here after the finals.
