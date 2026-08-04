# Validation report

Status as of 2026-08-04:

- Selected formula: `daily_rank(0.80 * EN454 + 0.20 * residual_T)`.
- Upload count: four files, total size about 3.6 MB.
- Python compilation: passed for all three modules.
- Notebook structure: valid nbformat 4.5; its only code cell imports `main`.
- Frozen Candidate454 schema: exactly 454 unique factors.
- Residual-T checkpoint SHA-256:
  `f08859257c3b56bbcf2bf226a8982529bfcea708026322cdc953cfa5293f74e8`.
- Checkpoint strict load and temporal forward pass: passed.
- Synthetic 60-day causal ElasticNet fit/predict contract: passed.
- 2023 paired OOS evidence: 241,854 common rows over 242 days.
- Local proxy at 20%: A `0.95759912`, B `0.99340659`,
  J `0.98266435`, raw B model score `5.11884518`, Rank IC
  `0.04592564`.
- Relative to EN454 alone: delta A `+0.00660793`, delta B
  `+0.00219780`, delta J `+0.00352084`.

The AutoDL host cannot execute the true AIStudio prefix probe because it does
not provide the platform-only `dai` module; the attempt stopped at
`import dai` before entering submission code. Run the copy-paste probe in
`aistudio_handoff.md` inside AIStudio. Local proxy scores and static/runtime
checks are not an official platform score.
