"""Contract tests for the sealed TCN experiment, including its actual training loop."""

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
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.finals_pre.tcn_architecture_o2c import (
    prepare,
    protocol,
    score,
    train,
)


@pytest.fixture(autouse=True, scope="module")
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("arm", protocol.ARMS, ids=lambda a: a.name)
def test_all_arms_keep_time_length_and_have_finite_backward(arm):
    model = protocol.initialize_model(arm, protocol.SEEDS[0]).train()
    assert sum(p.numel() for p in model.parameters()) == arm.parameter_count
    lengths = []
    hooks = [
        block.register_forward_hook(lambda m, i, o: lengths.append(o.shape[1]))
        for block in model.tcn
    ]
    values = torch.randn(1, 4, 240, 17)
    observed = torch.ones_like(values, dtype=torch.bool)
    minutes = torch.ones(1, 4, 240, dtype=torch.bool)
    minutes[:, :, 80:85] = False
    values[:, :, 80:85] = float("nan")
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    loss, norm = train.checked_update(
        model,
        (values, observed, minutes, torch.ones(1, 4, dtype=torch.bool)),
        torch.tensor([-1.0, -0.3, 0.4, 1.0]),
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
    )
    assert lengths == [240] * arm.depth
    assert np.isfinite([loss, norm]).all()
    for hook in hooks:
        hook.remove()


@pytest.mark.parametrize("name", ["baseline", "depth2", "depth4", "rf179_depth2", "rf177_depth4"])
def test_direct_receptive_field_by_input_gradient(name):
    arm = protocol.ARM_BY_NAME[name]
    model = protocol.initialize_model(arm, protocol.SEEDS[0]).double().eval()
    count = arm.receptive_field + 9
    values = torch.randn(1, count, 96, dtype=torch.float64, requires_grad=True)
    sequence = values
    for block in model.tcn:
        sequence = block(sequence, torch.ones(1, count, dtype=torch.bool))
    position = count - 3
    sequence[0, position, 0].backward()
    strength = values.grad.abs().sum(dim=-1)[0]
    earliest = position - arm.receptive_field + 1
    assert torch.equal(strength[:earliest], torch.zeros_like(strength[:earliest]))
    assert strength[earliest] > 0
    assert torch.equal(strength[position + 1 :], torch.zeros_like(strength[position + 1 :]))


def test_semantic_shared_initialization_and_independent_repeated_branches():
    baseline = protocol.initialize_model(protocol.ARMS[0], protocol.SEEDS[0])
    expected = protocol.initial_hashes(baseline, protocol.ARMS[0])
    for arm in protocol.ARMS:
        actual = protocol.initial_hashes(protocol.initialize_model(arm, protocol.SEEDS[0]), arm)
        assert all(actual[k] == expected[k] for k in actual.keys() & expected.keys())
    arm = protocol.ARM_BY_NAME["same_scale60"]
    repeated = protocol.initialize_model(arm, protocol.SEEDS[0])
    branches = repeated.tcn[0].branches
    assert len({b.weight.data_ptr() for b in branches}) == 3
    assert not torch.equal(branches[0].weight, branches[1].weight)
    assert not torch.equal(branches[1].weight, branches[2].weight)
    other = protocol.initialize_model(protocol.ARMS[0], protocol.SEEDS[1])
    assert not torch.equal(other.head[0].weight, baseline.head[0].weight)


def test_date_shuffle_and_stock_sampling_do_not_consume_each_others_rng():
    dates = pd.date_range("2023-01-01", periods=4)
    targets = {
        d: pd.Series(np.arange(1500), index=[f"s{i:04}" for i in range(1500)]) for d in dates
    }
    schedule = protocol.training_schedule(dates, targets, np.arange(4), protocol.SEEDS[0])
    smaller = protocol.training_schedule(dates, targets, np.arange(4), protocol.SEEDS[0], 500)
    assert [x["date"] for x in schedule] == [x["date"] for x in smaller]
    assert all(len(x["instruments"]) == 1200 for x in schedule)
    assert schedule == protocol.training_schedule(dates, targets, np.arange(4), protocol.SEEDS[0])
    assert schedule != protocol.training_schedule(dates, targets, np.arange(4), protocol.SEEDS[1])
    assert len(schedule) == 12


def test_nonfinite_gradient_stops_before_weight_update():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return self.weight * x

    model = Toy()
    model.weight.register_hook(lambda grad: grad * float("nan"))
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="non-finite"):
        train.checked_update(
            model,
            (torch.tensor([[-1.0, -0.3, 0.4, 1.0]]),),
            torch.tensor([-0.8, -0.5, 0.2, 1.0]),
            optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            torch.device("cpu"),
        )
    assert model.weight.item() == 1


def test_block_bootstrap_shares_dates_and_holm_adjusts_full_family():
    daily = np.random.default_rng(5).normal(0.01, 0.02, (3, 241))
    point, low, high, p = score.paired_bootstrap(np.stack([daily, daily * 2]), replicates=2000)
    assert point[1] == pytest.approx(point[0] * 2)
    assert low[1] == pytest.approx(low[0] * 2)
    assert high[1] == pytest.approx(high[0] * 2)
    assert p[0] == p[1]
    adjusted = score.holm_adjust([0.001, 0.004, 0.04] + [0.9] * 11)
    np.testing.assert_allclose(adjusted[:3], [0.014, 0.052, 0.48])
    assert (adjusted[3:] == 1).all()
    _, low, high, p = score.paired_bootstrap(np.zeros((14, 3, 241)), replicates=500)
    assert (low == 0).all() and (high == 0).all() and (p == 1).all()
    with pytest.raises(ValueError):
        score.paired_bootstrap(np.full((14, 3, 241), np.nan))


def test_seal_rejects_modified_minute_data(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    source = tmp_path / "model.py"
    source.write_text("frozen")
    partition = tmp_path / "store/data/trade_date=2024-01-02/part.parquet"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"original")
    meta = {
        "protocol_hash": protocol.object_hash(protocol.protocol_document()),
        "source_hashes": {"model.py": protocol.file_hash(source)},
        "small_data_hashes": {},
        "micro_store": str(tmp_path / "store"),
        "input_dates": ["2024-01-02"],
        "minute_file_hashes": {str(partition): protocol.file_hash(partition)},
        "minute_file_stats": {str(partition): prepare.stat_identity(partition)},
    }
    protocol.write_json(tmp_path / "prepared.json", meta)
    prepare.verify_seal(tmp_path, deep=True)
    partition.write_bytes(b"modified")
    # Filesystems can retain the same size/mtime for a rapid same-length rewrite.
    # The full checksum audit must still detect changed data.
    monkeypatch.setattr(prepare, "stat_identity", lambda p: meta["minute_file_stats"][str(p)])
    with pytest.raises(RuntimeError, match="minute data checksum mismatch"):
        prepare.verify_seal(tmp_path, deep=True)


@pytest.mark.parametrize("tamper_mask", [False, True])
def test_real_train_loop_exports_audit_or_records_failure(tmp_path, monkeypatch, tamper_mask):
    arm, seed = protocol.ARMS[1], protocol.SEEDS[0]
    codes = ("a", "b", "c", "d")
    days = pd.to_datetime(["2023-12-28", "2024-01-02", "2024-01-03"])
    labels = pd.DataFrame(
        [(d, s, 0.01 * i) for d in days for i, s in enumerate(codes)],
        columns=["date", "instrument", protocol.LABEL],
    )
    monkeypatch.setattr(train, "load_labels", lambda *a: labels.copy())
    monkeypatch.setattr(prepare, "verify_seal", lambda *a, **k: None)

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
    schedule = [{"epoch": epoch, "date": "2023-12-28", "instruments": codes} for epoch in (1, 2, 3)]
    with gzip.open(prepared / f"schedule_{protocol.seed_tag(seed)}.json.gz", "wt") as handle:
        json.dump(schedule, handle)
    actual = hashlib.sha256()
    for item in schedule:
        actual.update(json.dumps([item["epoch"], item["date"], list(codes[:3])]).encode())
    expected = labels[labels.date.dt.year == 2024]
    meta = {
        "source_hashes": {},
        "small_data_hashes": {},
        "micro_store": "unused",
        "data_root": "unused",
        "schedule_files": {
            str(seed): protocol.file_hash(prepared / f"schedule_{protocol.seed_tag(seed)}.json.gz")
        },
        "initial_hashes": {
            str(seed): {
                arm.name: protocol.initial_hashes(protocol.initialize_model(arm, seed), arm)
            }
        },
        "usable_training_days": 1,
        "actual_sample_hashes": {str(seed): actual.hexdigest()},
        "prediction_keys_hash": train.actual_key_hash(expected),
        "prediction_availability_hash": "bad"
        if tamper_mask
        else protocol.object_hash([True, True, True, False] * 2),
    }
    protocol.write_json(prepared / "prepared.json", meta)
    output = tmp_path / "run"
    if tamper_mask:
        with pytest.raises(RuntimeError, match="data mask differs"):
            train.train_arm(prepared, arm.name, seed, output, "cpu")
        assert json.loads((output / "status.json").read_text())["state"] == "failed"
        assert not (output / "factor.parquet").exists()
    else:
        train.train_arm(prepared, arm.name, seed, output, "cpu")
        factor = pd.read_parquet(output / "factor.parquet")
        assert len(factor) == 8 and factor.observed.sum() == 6
        assert (factor.loc[~factor.observed, "factor"] == 0).all()
        assert len(pd.read_csv(output / "missing_data.csv")) == 2
        _, manifest = score.verify_run(prepared, output)
        assert manifest["checkpoint_reused"] is False
        assert len(manifest["epoch_losses"]) == 3
        (output / "factor.parquet").write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="changed factor"):
            score.verify_run(prepared, output)


def test_incomplete_grid_cannot_produce_summary(tmp_path):
    protocol.write_json(tmp_path / "prepared/prepared.json", {})
    with pytest.raises(FileNotFoundError):
        score.aggregate(tmp_path)
    assert not (tmp_path / "results/summary.csv").exists()


def test_complete_grid_reports_paired_findings_and_four_groups(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=241).strftime("%Y-%m-%d").tolist()
    protocol.write_json(tmp_path / "prepared/prepared.json", {"scoring_dates": dates})
    monkeypatch.setattr(score, "verify_run", lambda *a: ({}, {}))
    base = np.random.default_rng(19).normal(0.02, 0.04, (3, 241))
    for index, arm in enumerate(protocol.ARMS):
        for j, seed in enumerate(protocol.SEEDS):
            directory = tmp_path / "runs" / protocol.seed_tag(seed) / arm.name
            directory.mkdir(parents=True)
            increment = 0.02 if index == 1 else (-0.02 if index == 2 else 0.0)
            daily = pd.DataFrame({"date": dates, "rank_ic": base[j] + increment})
            daily.to_csv(directory / "daily_metrics.csv", index=False)
            metrics = {
                "arm": arm.name,
                "seed": seed,
                "elapsed_seconds": 1800.0,
                "peak_cuda_allocated_bytes": 2**30,
                **{m: float(daily.rank_ic.mean()) for m in protocol.METRICS},
                "daily_metrics_sha256": protocol.file_hash(directory / "daily_metrics.csv"),
            }
            protocol.write_json(directory / "metrics.json", metrics)
    summary = score.aggregate(tmp_path)
    assert summary[1]["conclusion"] == "在本协议下支持该配置改善 RankIC"
    assert summary[2]["conclusion"] == "在本协议下支持该配置降低 RankIC"
    assert summary[3]["conclusion"] == "现有实验不足以区分"
    assert summary[1]["delta_rank_ic"] == pytest.approx(0.02)
    assert len(pd.read_csv(tmp_path / "results/paired_deltas.csv")) == 45
    for group in ("depth", "branches", "kernels", "controls"):
        assert "baseline" in pd.read_csv(tmp_path / f"results/{group}.csv").arm.tolist()
