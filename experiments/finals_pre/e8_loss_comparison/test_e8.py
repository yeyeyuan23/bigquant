"""E8 regression checks for loss semantics, paired training and complete scoring."""

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent / "_runtime"
sys.path[:0] = [str(ROOT), str(RUNTIME / "src"), str(RUNTIME / "scripts")]
from evaluate_unified_temporal import correlation_loss

from experiments.finals_pre.e8_loss_comparison import prepare, protocol, score, train
from experiments.finals_pre.e8_loss_comparison.losses import auxiliary_loss, loss_components


@pytest.fixture(autouse=True, scope="module")
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_smooth_l1_preserves_original_objective_and_gradient():
    p = torch.tensor([-2.0, -0.3, 0.4, 3.0], requires_grad=True)
    y = torch.tensor([-0.8, 0.2, -0.1, 0.9])
    expected = correlation_loss(p, y)
    actual, diagnostic = loss_components(p, y, "smooth_l1")
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0])
    assert diagnostic["large_error_fraction"].item() == 0.5


@pytest.mark.parametrize(
    "name,expected",
    [
        ("smooth_l1", [-0.2, -0.04, 0.0, 0.04, 0.2]),
        ("l1", [-0.2, -0.2, 0.0, 0.2, 0.2]),
        ("l2_half", [-0.4, -0.04, 0.0, 0.04, 0.4]),
    ],
)
def test_auxiliary_gradient_contract(name, expected):
    p = torch.tensor([-2.0, -0.2, 0.0, 0.2, 2.0], requires_grad=True)
    auxiliary_loss(p, torch.zeros_like(p), name).backward()
    torch.testing.assert_close(p.grad, torch.tensor(expected))


def test_small_error_smooth_and_half_squared_are_identical():
    p = torch.tensor([-0.5, -0.1, 0.2, 0.9])
    y = torch.zeros_like(p)
    torch.testing.assert_close(auxiliary_loss(p, y, "smooth_l1"), auxiliary_loss(p, y, "l2_half"))
    with pytest.raises(ValueError, match="unknown loss"):
        auxiliary_loss(p, y, "typo")


def test_paired_initialization_is_exact_for_every_loss_and_seed():
    seeds = []
    for seed in protocol.SEEDS:
        hashes = [
            protocol.initial_hashes(protocol.initialize_model(a, seed), a) for a in protocol.ARMS
        ]
        assert hashes[0] == hashes[1] == hashes[2]
        seeds.append(hashes[0])
    assert seeds[0] != seeds[1] != seeds[2]


@pytest.mark.parametrize("arm", protocol.ARMS, ids=lambda a: a.name)
def test_actual_training_loop_uses_requested_loss_and_exports_audit(tmp_path, monkeypatch, arm):
    seed = protocol.SEEDS[0]
    codes = ("a", "b", "c", "d")
    days = pd.to_datetime(["2023-12-28", "2024-01-02", "2024-01-03"])
    labels = pd.DataFrame(
        [(d, s, 0.01 * i) for d in days for i, s in enumerate(codes)],
        columns=["date", "instrument", protocol.LABEL],
    )
    monkeypatch.setattr(train, "load_labels", lambda *a: labels.copy())
    monkeypatch.setattr(prepare, "verify_seal", lambda *a, **k: None)
    called = []

    def observed_loss(p, y, name):
        called.append(name)
        return loss_components(p, y, name)

    monkeypatch.setattr(train, "loss_components", observed_loss)

    def batch(*args, **kwargs):
        values = np.random.default_rng(42).normal(size=(1, 4, 12, 17)).astype(np.float32)
        return SimpleNamespace(
            values=values,
            observed_mask=np.ones_like(values, dtype=bool),
            minute_mask=np.ones((1, 4, 12), dtype=bool),
            stock_mask=np.array([[True, True, True, False]]),
        )

    monkeypatch.setattr(train, "load_microstructure_day_fast", batch)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    schedule = [{"epoch": e, "date": "2023-12-28", "instruments": codes} for e in (1, 2, 3)]
    schedule_path = prepared / f"schedule_{protocol.seed_tag(seed)}.json.gz"
    with gzip.open(schedule_path, "wt") as handle:
        json.dump(schedule, handle)
    actual = hashlib.sha256()
    for item in schedule:
        actual.update(json.dumps([item["epoch"], item["date"], list(codes[:3])]).encode())
    meta = {
        "source_hashes": {},
        "small_data_hashes": {},
        "micro_store": "unused",
        "data_root": "unused",
        "schedule_files": {str(seed): protocol.file_hash(schedule_path)},
        "initial_hashes": {
            str(seed): {
                arm.name: protocol.initial_hashes(protocol.initialize_model(arm, seed), arm)
            }
        },
        "usable_training_days": 1,
        "actual_sample_hashes": {str(seed): actual.hexdigest()},
        "prediction_keys_hash": train.actual_key_hash(labels[labels.date.dt.year == 2024]),
        "prediction_availability_hash": protocol.object_hash([True, True, True, False] * 2),
    }
    protocol.write_json(prepared / "prepared.json", meta)
    output = tmp_path / "run"
    train.train_arm(prepared, arm.name, seed, output, "cpu")
    assert called == [arm.name] * 3
    _, manifest = score.verify_run(prepared, output)
    assert manifest["loss_name"] == arm.name and not manifest["checkpoint_reused"]
    assert len(manifest["epoch_diagnostics"]) == 3
    factor = pd.read_parquet(output / "factor.parquet")
    assert len(factor) == 8 and factor.observed.sum() == 6
    assert (factor.loc[~factor.observed, "factor"] == 0).all()
    (output / "factor.parquet").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed factor"):
        score.verify_run(prepared, output)


def test_summary_requires_all_nine_runs_and_compares_two_losses(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=241).strftime("%Y-%m-%d").tolist()
    protocol.write_json(tmp_path / "prepared/prepared.json", {"scoring_dates": dates})
    with pytest.raises(FileNotFoundError):
        score.aggregate(tmp_path)
    monkeypatch.setattr(score, "verify_run", lambda *a: ({}, {}))
    base = np.random.default_rng(19).normal(0.02, 0.04, (3, 241))
    for i, arm in enumerate(protocol.ARMS):
        for j, seed in enumerate(protocol.SEEDS):
            directory = tmp_path / "runs" / protocol.seed_tag(seed) / arm.name
            directory.mkdir(parents=True)
            increment = (0.0, 0.02, -0.02)[i]
            daily = pd.DataFrame({"date": dates, "rank_ic": base[j] + increment})
            daily.to_csv(directory / "daily_metrics.csv", index=False)
            protocol.write_json(
                directory / "metrics.json",
                {
                    "arm": arm.name,
                    "seed": seed,
                    "elapsed_seconds": 1800.0,
                    "peak_cuda_allocated_bytes": 2**30,
                    **{m: float(daily.rank_ic.mean()) for m in protocol.METRICS},
                    "daily_metrics_sha256": protocol.file_hash(directory / "daily_metrics.csv"),
                },
            )
    summary = score.aggregate(tmp_path)
    assert len(summary) == 3
    assert summary[1]["delta_rank_ic"] == pytest.approx(0.02)
    assert summary[2]["delta_rank_ic"] == pytest.approx(-0.02)
    assert summary[1]["p_holm"] < 0.05 and summary[2]["p_holm"] < 0.05
    assert len(pd.read_csv(tmp_path / "results/paired_deltas.csv")) == 9
    np.testing.assert_allclose(score.holm_adjust([0.01, 0.04]), [0.02, 0.04])
