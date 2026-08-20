# Finals-pre experiment suite (2026-08-19 → )

Protocol unless noted: expanding history from 2019-01-02, 1-trading-day label
isolation (asserted), predict untouched 2024; seed 20260801; RTX 4090D.
Artifacts live under `reports/dependencies/finals_pre_20260819/`; every
checkpoint sha256 is recorded in the sibling `oos_metrics.json`.

| ID | Question | Status | Headline result |
|---|---|---|---|
| E0 | Rebuild lost e3/e6 holdout evidence | done (`aeaf1a6`) | e3 OOS RankIC 0.0418 vs e6 0.0373 — deeper fit degrades OOS; proxy J/A/B in `holdout_j_scores.json` |
| E2 | Single-scale kernel ablation k3/k15/k60 | done (`aeaf1a6`) | 0.0444 / 0.0453 / 0.0425 vs full 0.0418; cross-scale corr 0.83–0.89 (`e2_scale_correlation_matrix.csv`) — scales carry unshared info, 1x1 fusion leans to k60 |
| E2b | Kernel-value sensitivity (5,30,120)/(2,10,45) | done (`fda450a`) | 0.0421 / 0.0458 — narrow band, log-spaced design robust to exact values |
| E2c | Seed robustness (2 extra seeds × full, k21045) | running (`lane_seeds.sh`) | first point: full seed12 = 0.0424 (vs 0.0418) |
| E4 | Multi-block walk-forward, 26 blocks 2023-24 | running (`lane_e4_walkforward.sh`) | per-block dirs `e4_walkforward/block_XX/` each with own checkpoint; blocks 0-1: 0.0411, 0.0855 |
| E7 | N score: pool-incremental value (see `e7_incremental_score/DESIGN.md`) | done (`bb1bf80`) | Layer-1 residual RIC vs LightGBM-454 base (base OOS IC 0.063): noise +0.001 (t 0.5), in-pool −0.004 (t −1.5), EN454 +0.014 (t 1.6 ns), **M_raw +0.0246 (t 4.9)**. Layer-2 paired ΔLGBM demoted to backup (no power in 60d windows; M +0.0026 t 1.9 vs noise +0.0020 t 1.4). FR-001 true-LOO addendum aborted by request |
| E1 | Pathway ablation (SeqOnly/StatsOnly/Linear-85/No-DeepSets) | not started | — |
| E3 | Main-result analysis pack (A sub-metrics, neutralized stress split, deciles, rolling IC) | not started | stress split MUST use platform-style neutralization |
| E5 | EN blend dose-response vs EN454 | not started | — |

Presentation-side documents (deck, speaker notes, red-team audit, N-score
design, final platform scores) are maintained in the local
`BigAlpha2026_Pre/` workspace (repo is now private; scores live in the root README).
