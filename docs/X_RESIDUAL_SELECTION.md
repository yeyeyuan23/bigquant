# X nonlinear residual selection

## Outcome

The retained challenger from the Candidate454 nonlinear residual experiment is
`tree_res_s3`, trained with seed `20260733`. This is an experiment selection,
not an AIStudio result or an official platform score.

The route is frozen at:

`reports/x_residual_retrain_v1/tree_seed_20260733/unified_tree_residual_full_oos.parquet`

Route SHA-256:

`f65aef525de83371e0311cd6e6b3e0abb12c588f7e61e7649e1fb892a32f2f8e`

## Why this route was retained

Selection was lexicographic rather than based on one headline metric:

1. Causal unseen-fold residual and total-return improvements had to be positive
   in every fold.
2. Worst-fold delta RankIC and standalone `J_stable` had to be competitive.
3. Candidate454 + EN454 + challenger joint Elastic Net had to keep both EN454
   and the challenger nonzero without a challenger-weight collapse.

`tree_res_s3` passed all three gates:

| Metric | Value |
|---|---:|
| Candidate RankIC mean | 0.04773394 |
| Residual RankIC mean | 0.04331133 |
| Residual RankIC worst fold | 0.03970169 |
| Delta RankIC mean | 0.02600774 |
| Delta RankIC worst fold | 0.02178718 |
| Positive delta folds | 3/3 |
| Mean daily Spearman vs EN454 | 0.22918172 |
| Standalone J mean | 0.85502439 |
| Standalone J worst | 0.83565317 |
| Standalone J stable | 0.84533878 |

Joint EN local proxy:

| Year | Challenger nonzero | EN454 nonzero | Mean abs weight | Std abs weight | Challenger J |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.90 | 1.00 | 0.00778775 | 0.00562708 | 0.82986707 |
| 2024 | 1.00 | 1.00 | 0.00593081 | 0.00393999 | 0.84562657 |

`mlp_res_s2` was the closest new residual comparator. It had slightly higher
joint J in each year, but weaker unseen-fold mean and worst delta RankIC, lower
standalone `J_stable`, and only 0.90 challenger nonzero ratio in both years.
The selection therefore favors the more stable tree residual route.

Old absolute Tree and MLP routes were comparators only. Their formal artifacts
are not removed or superseded by this experiment.

## Frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| run_manifest.json | `84ed92d3943c63a73bb1a2ceb07548268656f007cbffbc62abee81f2559cde33` |
| oos_metrics.csv | `3438d5320ae5071146c4f2f164c3e9e7037533a15b67201d3cd747574a2d9573` |
| feature_importance.csv | `ae1e92e20087a2ab08fe6816a619fb3c2ef57082c623de65e3829d26d6b71f33` |
| 2023 H1 checkpoint | `d2069ce2e56dab29067cad9d0402e5a487cdcfc2d33c3f385f12be91865ae05a` |
| 2023 H2 checkpoint | `26d5bd42e068b8da79ed87e090d5f6ff9d2c6ea6f9d076ca866af1d45790de61` |
| 2024 H1 checkpoint | `9da1e9dee54e03852f3773929ebf811e63ffe4254b850f943809b3f00531b662` |
| 2024 H2 checkpoint | `71addcab8ec49d39b2eeb6eb763be3ab1ec9d5f232d73f8256ea15fc502410b7` |

Training source commit: `b93de3a`. The same code was cherry-picked onto the
publication worktree as `22ce859` after rebasing onto the latest target branch.

Detailed machine-readable evidence is in `docs/x_residual_selection.json` and
the remote experiment directory. Model artifacts are intentionally not stored
in Git.
