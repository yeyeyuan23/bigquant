# M-l5 seed 20260803 final AIStudio handoff

Upload exactly the seven files listed in `submission_bundle_manifest.json`.
The notebook contains one code cell: `from unified_m_l5 import main`.

```bash
cd /home/aiuser/work
jupyter nbconvert --to script unified_m_l5.ipynb --output unified_m_l5_notebook
python -m py_compile unified_m_l5_notebook.py unified_m_l5.py unified_m_base.py unified_m_temporal.py unified_m_microstructure.py unified_m_microstructure_v2.py unified_m_l5_checkpoint.py
python /home/aiuser/work/aistudio_submission_lookahead_probe.py /home/aiuser/work/unified_m_l5.py --start 2024-12-23 --cutoff 2024-12-27 --end 2024-12-30 --bar1m bigalpha_2026_stock_bar1m | tee /home/aiuser/work/unified_m_l5_probe.log
```

Pass only when status is `ok`, `total_difference_rows=0`, every cutoff has
`prefix_invariant=true`, the output contract has no missing dates or unexpected
keys, and the accepted submission file list contains all seven files.
