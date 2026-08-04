# M-v3 five-level flow-gated challenger

This experiment is isolated from the current M champion. It does not modify or
overwrite the checkpoint, submission bundle, or platform evidence associated
with the reported `0.83488` score.

## Input contract

- Minute sequence: the audited compressed L1-L3 archive.
- Dynamic channels: session-safe first differences and trailing five-minute
  means derived from the existing 17-channel L1-L3 panel.
- Five-level context channels: four same-day, per-stock values from the audited
  `MICRO_DAILY_FULL` platform export: outer-imbalance proxy, tail deep-pressure
  residual, post-shock replenishment pressure, and pressure persistence. They
  are broadcast over the minute sequence and masked when five-level coverage is
  absent.
- Fusion: learned per-stock gate between the causal TCN summary and explicit
  statistics summary.
- `OB-009` and `OB-010` remain separate factor candidates and are not inserted
  into Candidate454 by this experiment.
- Evaluation: three fixed seeds, 2019-2023 expanding-history training, one 2024
  prediction block, followed by the shared local J scorer. Local metrics are
  not official platform evidence.

## Remote locations

- Worktree: `/root/autodl-tmp/projects/bigquant-m-v3-l5-channels-20260803`
- Branch: `feat/m-v3-l5-channels-20260803`
- Output: `reports/m_v3_l5_channels_20260803`

## Entry point

```bash
bash scripts/run_m_v3_l5_channel_challenge.sh
```

Promotion requires three-seed OOS evidence, output-contract and prefix checks,
AIStudio runtime/RSS validation, and an official score above the frozen
champion. No challenger may replace the champion solely on local proxy results.
