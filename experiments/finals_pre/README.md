# Finals-pre experiment suite (2026-08-19 → 08-21, COMPLETE)

Protocol unless noted: expanding history from 2019-01-02, 1-trading-day label
isolation (asserted), predict untouched 2024; seed 20260801; RTX 4090D.
Artifacts live under `reports/dependencies/finals_pre/`; every
checkpoint sha256 is recorded in the sibling `oos_metrics.json`.

| ID | Question | Headline result |
|---|---|---|
| E0 | e3/e6 holdout rebuild | e3 OOS RankIC 0.0418 vs e6 0.0373 — deeper fit degrades OOS |
| E1 | Pathway ablation + Linear-85 | Mean IC clusters 0.042-0.044 (stats_only 0.0334), but the full model wins BOTH stability metrics: ICIR 0.391 / LS-Sharpe 3.73 vs 3.48/3.30/2.49/1.07 — every component buys stability, matching what the contest scores |
| E2 | Single-scale kernels k3/k15/k60 | 0.0444/0.0453/0.0425 vs full 0.0418; cross-scale corr 0.83-0.89 |
| E2b | Kernel-value sensitivity | (5,30,120) 0.0421, (2,10,45) 0.0458 — log-spaced design robust to values |
| E2c | Seed robustness (3 seeds x 2 configs) | full 0.0419±0.0004, (2,10,45) 0.0451±0.0006 — the smaller-kernel edge is real (5-8x seed spread) |
| E3 | Main-result analysis (platform-style neutralization) | Neutralized IC 0.0523 (ICIR 0.91, t 14.1), LS-Sharpe(20%) 6.98, all four A percentiles 1.000; stress-day IC 0.033 (IR 0.53) positive and significant — degrades least, not strongest; top deciles monotone |
| E4 | Walk-forward 26 blocks 2023-24 | mean per-block IC 0.0483, 25/26 positive; merged series + per_block_ic.csv. Param-drift metric is uninformative by design (fresh seed per block -> permutation symmetry); functional stability is the 25/26 |
| E5 | EN blend dose-response (two-year J) | M x EN454 daily Spearman 0.18; J: M 0.983 > baseline 0.954; +0.006-0.007 at 10-25% blend, 50% blend falls back (B_2024 diluted to 0.842) — credit-reallocation again |
| E7 | N score (residual RIC vs LightGBM-454 base) | M_raw +0.0246 (t 4.9); controls: noise 0.001/t0.5, in-pool -0.004/t-1.5, EN454 ns; Layer-2 delta-LGBM demoted (no power in 60d windows) |
| Perm | Channel permutation importance (frozen e3, 61 days) | group_trade -36%, group_price -32%, group_book -26%; clock group immune to cross-stock shuffle by construction; redundant microprice_gap shuffle IMPROVES IC +3.8% |

Environment notes: (1) the stale editable install of bigalpha_2026_factors
pointed at the retired main worktree; re-installed from this worktree.
(2) Two scorer contracts every new route script must honor: ambient
bigalpha2026 import, and full label-universe coverage with neutral-0 fill.
(3) Orchestrating bash processes were killed several times by an unknown
external reaper (pythons survived); per-block resumable outputs made every
recovery lossless.

Presentation documents live locally in `BigAlpha2026_Pre/` (deck, study
guide, speaker notes, audits, platform-score record); repo is private.
