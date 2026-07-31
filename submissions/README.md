# Submission artifacts

This directory intentionally contains only:

- historical submissions with explicit platform-score evidence;
- the historical platform-top LightGBM reconstruction;
- retained candidate snapshots that still need platform validation.

Each submission is stored as an exact `.py` / `.ipynb` pair. The notebook has
one code cell whose contents must exactly match the corresponding Python file.

| Version | Status | Evidence |
| --- | --- | --- |
| `smoke_v01` | Historical scored submission | Public score `0.57416` |
| `rule_v03` | Historical scored submission | Public score `0.60478` |
| `lgbm_platform_top_v01` | Historical platform-top reconstruction | Highest historical total/B mechanism; exact score not recorded here |
| `enet_i_51_candidate` | Retained candidate snapshot | Superseded locally; not platform-scored |
| `lgbm_t_orthogonal_26_candidate` | Retained candidate snapshot | Superseded locally; not platform-scored |

Generate all three current S/I/T candidate pairs with:

```bash
python scripts/build_latest_submission_notebooks.py
```

Use `--reports-dir PATH` to select a specific completed S/I/T run. Without it,
the builder chooses the newest compatible reports directory and prints the
selected path. New, unverified outputs are written to
`remote_submission_notebooks/`; they must pass real AIStudio execution and
prefix/look-ahead probes before promotion into this directory.

Local J is diagnostic and must not be presented as an official platform score.

Candidate notebooks use normal Python imports. Upload the `.ipynb` together
with its generated sibling `*_deps.py` file; AIStudio does not need a directory
upload. Each dependency file contains only that notebook's frozen candidates.
Generated notebooks must not embed module source strings or install modules
through `exec`/`sys.modules`.

`PV-009` and `HF-048` are hard-excluded from generated submissions because
their historical implementations rely on disallowed data.
