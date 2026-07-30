# Submission artifacts

This directory intentionally contains only:

- historical submissions with explicit platform-score evidence;
- the historical platform-top LightGBM reconstruction;
- the latest generated I/T candidate submissions.

Each submission is stored as an exact `.py` / `.ipynb` pair. The notebook has
one code cell whose contents must exactly match the corresponding Python file.

| Version | Status | Evidence |
| --- | --- | --- |
| `smoke_v01` | Historical scored submission | Public score `0.57416` |
| `rule_v03` | Historical scored submission | Public score `0.60478` |
| `lgbm_platform_top_v01` | Historical platform-top reconstruction | Highest historical total/B mechanism; exact score not recorded here |
| `enet_i_54_candidate` | Current candidate | Latest frozen I pool; not yet platform-scored |
| `lgbm_t_orthogonal_28_candidate` | Current candidate | Latest frozen orthogonal T pool; not yet platform-scored |

Generate and validate both current candidate pairs with:

```bash
python scripts/build_latest_submission_notebooks.py
```

Use `--reports-dir PATH` to select a specific completed I/T run. Without it,
the builder chooses the newest reports directory containing both
`latest/i_only_result.json` and `latest/t_orthogonal_only_result.json`, and
prints the selected path.

Local J is diagnostic and must not be presented as an official platform score.
