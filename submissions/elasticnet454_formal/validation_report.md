# Validation report

Status as of 2026-08-03:

- Local unit and probe-isolation suite: 11 tests passed.
- AutoDL submission, competition-score-proxy, and probe-isolation suites: 18
  tests passed; Ruff passed on the new model and regression test.
- Python compilation: passed after using a writable bytecode cache.
- Frozen candidate specification: 454 unique IDs, 266 component specs, 149
  GTJA specs.
- Candidate runtime dependency digests match the current Candidate454 X/T
  submission runtime from which they were copied.
- AutoDL real-store replay for the first 20 prediction dates of 2024 produced
  20,000 keys with zero key mismatches against the stored research route.
- Daily factor rank correlation against that research route was
  `0.999748797`.

The replay is intentionally not bitwise identical to the old research OOS
file.  Its evaluator ranked only stocks whose following-day target was finite
and then neutral-filled omitted rows.  In the first 20-day block, all 21 rows
with unavailable following-day targets were forced to zero; changing that
rank denominator also changed many other floating-point ranks.  A formal
submission cannot know following-day target availability at prediction time,
so this bundle ranks all current stock-pool rows and contains a regression test
that prediction output is unchanged when all gap/prediction-date targets are
made unavailable.

No real AIStudio execution or competition score has been claimed here.
