"""Seed/protocol safeguards for the paired minute-grid experiment orchestration."""

from __future__ import annotations

import json
import sys

import pytest

from experiments.finals_pre.minute_grid_240 import monitor, run_pair, summarize_pairs


def test_monitor_does_not_shutdown_when_only_one_seed_has_finished(tmp_path, monkeypatch):
    commands = []

    def remote(command):
        commands.append(command)
        if command.startswith("cat "):
            return json.dumps({"state": "running", "pid": 9876, "completed_seeds": [20260812]})
        assert command == "kill -0 9876"
        return ""

    monkeypatch.setattr(monitor, "ssh", remote)
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not collect or shutdown"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "monitor.py",
            "--shutdown",
            "--remote-root",
            "/new/results",
            "--local-root",
            str(tmp_path),
        ],
    )
    assert monitor.main() == 0
    assert commands == ["cat /new/results/status.json", "kill -0 9876"]


def fake_pair(tmp_path, seed):
    pair = tmp_path / f"seed{seed}"
    pair.mkdir()
    sources = {
        arm: {key: key for key in ("model_sha256", "trainer_sha256", "fastpack_sha256")}
        for arm in ("clock240", "clock242")
    }
    manifest = {
        "seed": seed,
        "epochs": 3,
        "train": "train",
        "test": "test",
        "store_manifest_sha256": "store",
        "label_sha256": {"2024": "labels"},
        "exposure_sha256": "exposure",
        "arms": sources,
    }
    (pair / "pair_manifest.json").write_text(json.dumps(manifest))
    return pair


def test_three_seed_summary_preserves_pairing_and_uses_sample_std(tmp_path):
    pairs = []
    for seed, value in zip(summarize_pairs.EXPECTED_SEEDS, (1.0, 3.0, 5.0), strict=True):
        pair = fake_pair(tmp_path, seed)
        scores = [
            {
                "arm": arm,
                "seed": seed,
                "epochs": 3,
                "days": 241,
                **{metric: number for metric in summarize_pairs.METRICS},
            }
            for arm, number in (("clock240", value), ("clock242", 2.0))
        ]
        (pair / "comparison.json").write_text(json.dumps({"scores": scores}))
        pairs.append(pair)
    result = summarize_pairs.summarize(list(reversed(pairs)), tmp_path / "summary")
    for row in result["summary"]:
        assert row["mean_240"] == 3
        assert row["std_240"] == 2
        assert row["mean_242"] == 2
        assert row["std_242"] == 0
        assert row["mean_paired_delta"] == 1
        assert row["std_paired_delta"] == 2
        assert row["positive_pairs"] == 2
    manifest = json.loads((pairs[-1] / "pair_manifest.json").read_text())
    manifest["label_sha256"] = {"2024": "different"}
    (pairs[-1] / "pair_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different data"):
        summarize_pairs.summarize(pairs, tmp_path / "bad")


def test_requested_seed_reaches_both_training_commands_and_manifest(tmp_path, monkeypatch):
    seed = 20260823
    current, control, data, store = [tmp_path / p for p in ("current", "control", "data", "store")]
    for root in (current, control):
        for relative in (
            "src/alpha_models/microstructure.py",
            "scripts/evaluate_unified_microstructure.py",
            "experiments/finals_pre/common/fastpack.py",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
    for year in range(2019, 2025):
        path = data / f"labels/year={year}/part-{year}.parquet"
        path.parent.mkdir(parents=True)
        path.write_text("fixture")
    store.mkdir()
    (store / "manifest.json").write_text("{}")
    exposure = tmp_path / "exposure.parquet"
    exposure.write_text("fixture")
    output = tmp_path / "results"
    output.mkdir()
    for minutes in (240, 242):
        (output / f"preflight{minutes}.json").write_text(
            json.dumps({"minutes": minutes, "seed": seed})
        )
    argv = ["run_pair.py", "--seed", str(seed)]
    for name, path in (
        ("current-root", current),
        ("control-root", control),
        ("data-root", data),
        ("micro-store", store),
        ("exposure", exposure),
        ("output-dir", output),
    ):
        argv += ["--" + name, str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    commands = []
    monkeypatch.setattr(
        run_pair.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(run_pair, "score", lambda args: None)
    assert run_pair.main() == 0
    assert len(commands) == 2
    assert [cmd[cmd.index("--max-minutes") + 1] for cmd in commands] == ["240", "242"]
    assert [cmd[cmd.index("--seed") + 1] for cmd in commands] == [str(seed), str(seed)]
    assert json.loads((output / "pair_manifest.json").read_text())["seed"] == seed
    # Refuse an audit from another seed before starting either training arm.
    (output / "status.json").unlink()
    (output / "preflight242.json").write_text(json.dumps({"minutes": 242, "seed": 20260801}))
    commands.clear()
    with pytest.raises(RuntimeError, match="preflight seed differs"):
        run_pair.main()
    assert commands == []
