"""Read-only completion checks and timing extraction for the sealed TCN grid."""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.finals_pre.tcn_architecture_o2c.prepare import verify_seal
from experiments.finals_pre.tcn_architecture_o2c.protocol import file_hash, object_hash, write_json, now
from experiments.finals_pre.tcn_architecture_o2c.score import verify_run, actual_key_hash

OUT = ROOT / "reports/dependencies/finals_pre/tcn_architecture_o2c"
meta = verify_seal(OUT / "prepared", deep=True)
status = json.loads((OUT / "status.json").read_text())
assert status["state"] == "complete" and status["completed"] == 45
original_audit = json.loads((OUT / "results/audit.json").read_text())
for name, digest in original_audit["files"].items():
    assert file_hash(OUT / "results" / name) == digest, name
arms = meta["protocol"]["arms"]
seeds = meta["protocol"]["seeds"]
summaries = json.loads((OUT / "results/summary.json").read_text())
summary_by_arm = {r["arm"]: r for r in summaries}
cube = np.empty((15, 3, 241))
timings = []
for ai, arm in enumerate(arms):
    for si, seed in enumerate(seeds):
        tag = f"seed{si+1}"
        run = OUT / "runs" / tag / arm["name"]
        _, manifest = verify_run(OUT / "prepared", run)
        assert manifest["epochs"] == 3 and manifest["seed"] == seed
        assert manifest["config"]["kernels"] == arm["kernels"]
        assert manifest["config"]["tcn_blocks"] == arm["depth"]
        assert manifest["parameter_count"] == arm["parameter_count"]
        assert manifest["receptive_field"] == arm["receptive_field"]
        assert manifest["initial_hashes"] == meta["initial_hashes"][str(seed)][arm["name"]]
        assert manifest["checkpoint_reused"] is False
        assert manifest["usable_training_days"] == 1213 and not manifest["skipped_data_days"]
        factor = pd.read_parquet(run / "factor.parquet").sort_values(["date", "instrument"])
        assert not factor.duplicated(["date", "instrument"]).any()
        assert np.isfinite(factor.factor).all()
        assert actual_key_hash(factor) == meta["prediction_keys_hash"]
        assert object_hash(factor.observed.tolist()) == meta["prediction_availability_hash"]
        assert len(factor) == manifest["rows"] == 242000
        assert int((~factor.observed).sum()) == manifest["missing_data_rows"] == 230
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["scoring_keys_hash"] == meta["scoring_keys_hash"]
        assert metrics["scored_rows"] == meta["scoring_rows"] == 240567
        assert file_hash(run / "daily_metrics.csv") == metrics["daily_metrics_sha256"]
        daily = pd.read_csv(run / "daily_metrics.csv")
        assert daily.date.tolist() == meta["scoring_dates"]
        assert np.isfinite(daily[["rank_ic", "long_short_return"]]).all().all()
        assert abs(daily.rank_ic.mean() - metrics["rank_ic"]) < 1e-12
        assert abs(daily.rank_ic.mean()/daily.rank_ic.std() - metrics["rank_ic_ir"]) < 1e-12
        stress = daily.loc[daily.stress, "rank_ic"]
        assert len(stress) == metrics["stress_days"] == 61
        assert abs(stress.mean()/stress.std() - metrics["stress_ic_ir"]) < 1e-12
        cube[ai, si] = daily.rank_ic.to_numpy()
        logs = sorted((OUT / "logs").glob(f"{tag}_{arm['name']}_attempt*.log"),
                      key=lambda f: f.stat().st_mtime)
        ends = []
        for line in logs[-1].read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("phase") == "training" and event.get("step") == 3639 and event.get("epoch") == 3:
                ends.append(event)
        assert len(ends) == 1
        timings.append({
            "seed_tag": tag, "arm": arm["name"],
            "startup_through_training_seconds": ends[0]["elapsed_seconds"],
            "startup_through_factor_write_seconds": manifest["elapsed_seconds"],
            "rank_ic": metrics["rank_ic"],
            "parameters": manifest["parameter_count"],
            "peak_gpu_allocated_gib": manifest["peak_cuda_allocated_bytes"] / 2**30,
            "peak_gpu_reserved_gib": manifest["peak_cuda_reserved_bytes"] / 2**30,
        })
    print("verified", arm["name"], flush=True)
saved = np.load(OUT / "results/daily_rankic_cube.npz")
assert np.allclose(cube, saved["rank_ic"], rtol=0, atol=1e-15)
assert saved["arms"].tolist() == [a["name"] for a in arms]
assert saved["seeds"].tolist() == seeds
assert saved["dates"].tolist() == meta["scoring_dates"]
# Independent resampling implementation: construct every synchronous date draw,
# then compute each candidate separately rather than using the production batching.
deltas = cube[1:] - cube[0]
daily_delta = deltas.mean(axis=1)
points = daily_delta.mean(axis=1)
rng = np.random.default_rng(meta["protocol"]["bootstrap"]["seed"])
starts = rng.integers(0, 232, size=(10000, 25))
indices = (starts[:, :, None] + np.arange(10)).reshape(10000, 250)[:, :241]
intervals, raw_p = [], []
for values, point in zip(daily_delta, points):
    draws = values[indices].mean(axis=1)
    intervals.append(np.quantile(draws, [0.025, 0.975]))
    raw_p.append((1 + np.count_nonzero(np.abs(draws-point) >= abs(point))) / 10001)
order = sorted(range(14), key=lambda i: raw_p[i])
adjusted = np.empty(14)
last = 0.0
for rank, j in enumerate(order):
    last = max(last, (14-rank)*raw_p[j])
    adjusted[j] = min(1.0, last)
for ai, arm in enumerate(arms):
    row = summary_by_arm[arm["name"]]
    assert abs(cube[ai].mean() - row["rank_ic"]) < 1e-12
    if ai:
        j = ai-1
        assert abs(points[j] - row["delta_rank_ic"]) < 1e-12
        assert np.allclose(intervals[j], [row["ci_low"], row["ci_high"]], rtol=0, atol=1e-12)
        assert abs(raw_p[j] - row["p_raw"]) < 1e-12
        assert abs(adjusted[j] - row["p_holm"]) < 1e-12
        assert int((deltas[j].mean(axis=1) > 0).sum()) == row["positive_seeds"]
timing_frame = pd.DataFrame(timings)
timing_frame.to_csv(OUT / "results/timing_per_run.csv", index=False)
cost_rows = []
for arm in arms:
    group = timing_frame[timing_frame.arm == arm["name"]]
    cost_rows.append({
        "arm": arm["name"],
        "mean_startup_through_training_minutes": group.startup_through_training_seconds.mean()/60,
        "mean_total_minutes": group.startup_through_factor_write_seconds.mean()/60,
        "peak_gpu_allocated_gib": group.peak_gpu_allocated_gib.max(),
        "peak_gpu_reserved_gib": group.peak_gpu_reserved_gib.max(),
        "parameters": int(group.parameters.iloc[0]),
    })
pd.DataFrame(cost_rows).to_csv(OUT / "results/timing_summary.csv", index=False)
cost_by_arm = {r["arm"]: r for r in cost_rows}
baseline, five = cost_by_arm["baseline"], cost_by_arm["branches5"]
audit = {
    "state": "passed", "checked_at": now(), "runs": 45,
    "grid": [15, 3, 241], "training_dates": 1213, "scoring_rows": 240567,
    "minute_partitions_deep_hashed": len(meta["minute_file_hashes"]),
    "source_files_hashed": len(meta["source_hashes"]),
    "verification_source": str(Path(__file__).relative_to(ROOT)),
    "verification_source_sha256": file_hash(Path(__file__)),
    "checks": ["all original summary artifacts", "source and input deep checksums",
               "all 45 checkpoints and factors", "sample order and initialization",
               "prediction keys, availability and finite predictions",
               "scoring key manifests and daily calendars", "RankIC, ICIR and stress ICIR arithmetic",
               "complete saved cube", "independent synchronous 10-day bootstrap",
               "independent Holm adjustment", "training endpoint logs and timings"],
    "total_wall_hours": status["elapsed_seconds"]/3600,
    "total_training_inference_hours": timing_frame.startup_through_factor_write_seconds.sum()/3600,
    "baseline_cost": baseline, "five_branch_cost": five,
    "five_training_overhead_percent": 100*(five["mean_startup_through_training_minutes"]/baseline["mean_startup_through_training_minutes"]-1),
    "five_total_overhead_percent": 100*(five["mean_total_minutes"]/baseline["mean_total_minutes"]-1),
    "timing_definition": "startup through final training step includes validation, initialization and data loading; total includes checkpoint, inference and factor writes, excludes scoring; not isolated GPU compute",
    "assessment": "Share with caveats",
    "caveats": ["historical validation period previously used", "three epochs only",
                "three fixed seeds; date bootstrap is not all initialization uncertainty",
                "95 percent intervals are unadjusted; decisions use Holm p-values",
                "parameter counts differ; no isolated mechanism proof", "Sharpe excludes costs"],
}
write_json(OUT / "audits/completion_audit.json", audit)
print(json.dumps(audit), flush=True)
