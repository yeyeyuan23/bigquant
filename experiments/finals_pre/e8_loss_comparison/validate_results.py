"""Recompute the compact E8 archive without importing the training/scoring code."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("smooth_l1", "l1", "l2_half")
SEEDS = (20260801, 20260812, 20260823)
METRICS = ("rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(actual, expected, *, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def validate(output):
    receipt = read(output / "collection.json")
    for name, item in receipt["files"].items():
        path = output / name
        assert path.stat().st_size == item["bytes"], name
        assert digest(path) == item["sha256"], name
    for name, expected in receipt["source_hashes"].items():
        assert digest(Path(__file__).parent / name) == expected, name
    meta = read(output / "prepared/prepared.json")
    prepared_hash = digest(output / "prepared/prepared.json")
    for name, expected in meta["source_hashes"].items():
        assert digest(ROOT / name) == expected, name
    audit = read(output / "results/audit.json")
    assert audit["runs"] == 9 and audit["prepared_sha256"] == prepared_hash
    for name, expected in audit["files"].items():
        assert digest(output / "results" / name) == expected, name
    status = read(output / "status.json")
    assert (status["state"], status["completed"], status["total"]) == ("complete", 9, 9)
    summary = read(output / "results/summary.json")
    assert [r["arm"] for r in summary] == list(ARMS)
    csv_summary = pd.read_csv(output / "results/summary.csv")
    per_run = pd.read_csv(output / "results/per_run.csv").set_index(["arm", "seed"])
    pairs = pd.read_csv(output / "results/paired_deltas.csv").set_index(["arm", "seed"])
    assert len(per_run) == len(pairs) == 9
    cube = np.empty((3, 3, 241))
    metrics = np.empty((3, 3, 4))
    manifests, checkpoints, stress_mask = {}, set(), None
    sharpe_rounding_error = 0.0
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            run = output / "runs" / f"seed{s + 1}" / arm
            m, scores = read(run / "manifest.json"), read(run / "metrics.json")
            assert (m["arm"], m["seed"], m["loss_name"]) == (arm, seed, arm)
            assert read(run / "status.json")["state"] == "complete"
            assert m["prepared_sha256"] == prepared_hash == scores["prepared_sha256"]
            assert m["epochs"] == 3 and m["auxiliary_weight"] == 0.05
            assert not m["checkpoint_reused"] and not m["skipped_data_days"]
            assert m["usable_training_days"] == meta["usable_training_days"] == 1213
            assert len(m["epoch_losses"]) == len(m["epoch_diagnostics"]) == 3
            assert m["actual_sample_hash"] == meta["actual_sample_hashes"][str(seed)]
            assert m["initial_hashes"] == meta["initial_hashes"][str(seed)][arm]
            assert m["prediction_keys_hash"] == meta["prediction_keys_hash"]
            assert m["prediction_availability_hash"] == meta["prediction_availability_hash"]
            assert scores["scoring_keys_hash"] == meta["scoring_keys_hash"]
            assert scores["scored_rows"] == meta["scoring_rows"] == 240567
            assert scores["scored_days"] == 241 and scores["stress_days"] == 61
            assert scores["factor_sha256"] == m["factor_sha256"]
            assert scores["daily_metrics_sha256"] == digest(run / "daily_metrics.csv")
            assert m["parameter_count"] == 217953 and m["config"]["max_minutes"] == 240
            checkpoints.add(m["checkpoint_sha256"])
            manifests[arm, seed] = m
            daily = pd.read_csv(run / "daily_metrics.csv")
            assert daily.date.tolist() == meta["scoring_dates"]
            assert daily.date.is_unique and np.isfinite(daily.rank_ic).all()
            if stress_mask is None:
                stress_mask = daily.stress.to_numpy()
            np.testing.assert_array_equal(daily.stress.to_numpy(), stress_mask)
            ic = daily.rank_ic.to_numpy()
            ls = daily.long_short_return.to_numpy()
            stress = ic[stress_mask]
            recomputed = [ic.mean(), ic.mean() / ic.std(ddof=1),
                          ls.mean() / ls.std(ddof=1) * np.sqrt(252),
                          stress.mean() / stress.std(ddof=1)]
            for k, metric in enumerate(METRICS):
                # Frozen scorer reduces float32 returns; CSV recomputation uses float64.
                tolerance = 2e-6 if metric == "long_short_sharpe" else 1e-12
                close(recomputed[k], scores[metric], atol=tolerance)
                close(per_run.loc[(arm, seed), metric], scores[metric])
                metrics[a, s, k] = scores[metric]
            sharpe_rounding_error = max(sharpe_rounding_error,
                                       abs(recomputed[2] - scores["long_short_sharpe"]))
            cube[a, s] = ic
    assert len(checkpoints) == 9
    for seed in SEEDS:
        for arm in ARMS[1:]:
            for key in ("config", "initial_hashes", "actual_sample_hash"):
                assert manifests[arm, seed][key] == manifests[ARMS[0], seed][key]
    assert len({json.dumps(manifests[ARMS[0], s]["initial_hashes"], sort_keys=True)
                for s in SEEDS}) == 3
    with np.load(output / "results/daily_rankic_cube.npz", allow_pickle=False) as archive:
        close(archive["rank_ic"], cube)
        assert archive["arms"].tolist() == list(ARMS)
        assert archive["seeds"].tolist() == list(SEEDS)
        assert archive["dates"].tolist() == meta["scoring_dates"]
    for a, arm in enumerate(ARMS):
        m = [manifests[arm, seed] for seed in SEEDS]
        for k, metric in enumerate(METRICS):
            close(summary[a][metric], metrics[a, :, k].mean())
            close(summary[a][metric + "_seed_sd"], metrics[a, :, k].std(ddof=1))
            for s, seed in enumerate(SEEDS):
                close(pairs.loc[(arm, seed), "delta_" + metric],
                      metrics[a, s, k] - metrics[0, s, k])
        close(summary[a]["mean_minutes"], np.mean([v["elapsed_seconds"] for v in m]) / 60)
        close(summary[a]["peak_gpu_gib"], max(v["peak_cuda_allocated_bytes"] for v in m) / 2**30)
        for key, value in summary[a].items():
            if isinstance(value, (int, float)):
                close(csv_summary.iloc[a][key], value)
    # Independent vectorized reconstruction of the predeclared moving-block draws.
    delta = (cube[1:] - cube[:1]).mean(axis=1)
    rng = np.random.default_rng(20260906)
    starts = rng.integers(0, 232, size=(10000, 25))
    indices = (starts[..., None] + np.arange(10)).reshape(10000, 250)[:, :241]
    means = delta[:, indices].mean(axis=2)
    points = delta.mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975], axis=1)
    pvalues = (1 + (np.abs(means - points[:, None]) >= np.abs(points[:, None])).sum(1)) / 10001
    order = np.argsort(pvalues)
    adjusted = np.empty(2)
    adjusted[order[0]] = min(1, 2 * pvalues[order[0]])
    adjusted[order[1]] = min(1, max(adjusted[order[0]], pvalues[order[1]]))
    for a in range(2):
        for key, values in (("delta_rank_ic", points), ("ci_low", ci[0]),
                            ("ci_high", ci[1]), ("p_raw", pvalues), ("p_holm", adjusted)):
            close(summary[a + 1][key], values[a])
        assert summary[a + 1]["positive_seeds"] == int(
            ((cube[a + 1] - cube[0]).mean(axis=1) > 0).sum())
        assert ci[0, a] < 0 < ci[1, a] and adjusted[a] >= 0.05
    return {
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "runs": 9, "scoring_days": 241, "scoring_rows_per_run": 240567,
        "collected_files_verified": len(receipt["files"]),
        "sealed_source_files_verified": len(meta["source_hashes"]),
        "collection_source_files_verified": len(receipt["source_hashes"]),
        "paired_initialization_samples_and_config": True,
        "distinct_checkpoint_hashes": len(checkpoints),
        "daily_metrics_summary_pairs_bootstrap_holm_recomputed": True,
        "max_sharpe_float32_to_float64_difference": sharpe_rounding_error,
        "scope": "Local verification begins with archived daily scores. Full factors, labels, "
                 "exposures and weights remain on AutoDL; stock-level neutralization and training "
                 "are not rerun locally. Their original checksums and remote audit are preserved.",
        "interpretation": "No reliable RankIC difference under three epochs and auxiliary weight "
                          "0.05. Not equivalence or evidence that auxiliary loss is necessary.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = validate(args.output)
    (args.output / "local_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
