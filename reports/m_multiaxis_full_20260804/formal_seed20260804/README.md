# M multi-axis formal training artifact

- Default-branch source commit: `b6a112f`
- Training range: `2019-01-02` through `2024-12-26`
- Protocol: tail-60 one-minute main path plus 24 time, volatility, and turnover tokens
- Epochs: 3/3
- Parameters: 542,338
- Final checkpoint SHA256: `4bfcba4a5d3b0d9e02380b234c0521dacb416000d94ee103260698c2777e48d5`
- Epoch mean IC: `0.02660255`, `0.03295071`, `0.03934963`
- Epoch IC standard deviation: `0.14160970`, `0.12563117`, `0.11640960`
- Existing-M guard: all 93 files matched the pre-training hashes

The epoch statistics are training diagnostics, not OOS J or an official platform
score. No J evaluation has been run for this checkpoint yet.

The directly runnable upload bundle is mirrored under
`submissions/m_multiaxis_full_20260804` and contains exactly `predict.ipynb`,
`train.py`, and `weights.json`. The local build and `py_compile` checks passed;
AIStudio execution, prefix/look-ahead validation, submission acceptance, and the
official score remain separate evidence.
