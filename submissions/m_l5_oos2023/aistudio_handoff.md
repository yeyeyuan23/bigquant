# M-l5 OOS-2023 seed 20260803 AIStudio handoff

Upload exactly the seven files listed in `submission_bundle_manifest.json`
into one directory. The commands default to `/home/aiuser/work/sub_m_l5`;
set `M_L5_DIR` first if your directory has a different name.
The notebook contains one code cell: `from unified_m_l5 import main`.

```bash
M_L5_DIR="${M_L5_DIR:-/home/aiuser/work/sub_m_l5}"
M_L5_PROBE="${M_L5_PROBE:-/home/aiuser/work/aistudio_submission_lookahead_probe.py}"
cd "$M_L5_DIR"
test -f unified_m_l5.py
test -f "$M_L5_PROBE"

python -m jupyter nbconvert --to notebook --execute unified_m_l5.ipynb --output unified_m_l5_executed.ipynb --ExecutePreprocessor.timeout=180
python -m py_compile unified_m_l5.py unified_m_base.py unified_m_temporal.py unified_m_microstructure.py unified_m_microstructure_v2.py unified_m_l5_checkpoint.py
PYTHONPATH="$M_L5_DIR" python "$M_L5_PROBE" \
  "$M_L5_DIR/unified_m_l5.py" \
  --start 2024-12-23 \
  --cutoff 2024-12-27 \
  --end 2024-12-30 \
  --bar1m bigalpha_2026_stock_bar1m \
  | tee "$M_L5_DIR/unified_m_l5_probe.log"
```

Pass only when status is `ok`, `total_difference_rows=0`, every cutoff has
`prefix_invariant=true`, the output contract has no missing dates or unexpected
keys, and the accepted submission file list contains all seven files.
