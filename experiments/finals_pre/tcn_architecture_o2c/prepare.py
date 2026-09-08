"""Seal inputs and paired randomness, then exercise all arms on real data."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent / "_runtime"
sys.path[:0] = [
    str(ROOT),
    str(RUNTIME / "src"),
    str(RUNTIME / "scripts"),
    str(RUNTIME / "experiments/finals_pre/common"),
]

import numpy as np
import pandas as pd
import torch
from evaluate_unified_microstructure import (
    _tensor_inputs,
    apply_training_history,
    prepare_label_panel,
    validate_micro_store,
)
from evaluate_unified_temporal import load_labels
from fastpack import CHANNELS, load_microstructure_day_fast

from alpha_models import rolling_oos_blocks
from experiments.finals_pre.tcn_architecture_o2c.protocol import (
    ARMS,
    LABEL,
    SEEDS,
    file_hash,
    initial_hashes,
    initialize_model,
    make_scaler,
    now,
    object_hash,
    protocol_document,
    seed_tag,
    training_schedule,
    write_json,
)
from experiments.finals_pre.tcn_architecture_o2c.train import actual_key_hash, checked_update


def source_files():
    # Frozen runtime dependencies keep the experiment independent of older repository defaults.
    names = [
        "__init__",
        "base",
        "data_contract",
        "feature_bundle",
        "microstructure",
        "microstructure_data",
        "tabular",
        "temporal",
        "training_data",
    ]
    paths = {RUNTIME / f"src/alpha_models/{name}.py" for name in names}
    paths.update(
        RUNTIME / name
        for name in (
            "src/competition_score_proxy.py",
            "scripts/evaluate_unified_microstructure.py",
            "scripts/evaluate_unified_temporal.py",
            "experiments/finals_pre/common/fastpack.py",
            "experiments/finals_pre/e3_progressive_add/score.py",
        )
    )
    paths.update(Path(__file__).resolve().parent.glob("*.py"))
    return sorted(paths)


def data_inventory(store, dates):
    return sorted(
        path
        for day in dates
        for path in (Path(store) / "data" / f"trade_date={day}").glob("*.parquet")
    )


def stat_identity(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def verify_seal(prepared, *, deep=False):
    meta = json.loads((Path(prepared) / "prepared.json").read_text())
    if meta["protocol_hash"] != object_hash(protocol_document()):
        raise RuntimeError("protocol changed")
    for name, expected in meta["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise RuntimeError(f"source changed: {name}")
    for name, expected in meta["small_data_hashes"].items():
        if file_hash(name) != expected:
            raise RuntimeError(f"data changed: {name}")
    inventory = data_inventory(meta["micro_store"], meta["input_dates"])
    if [str(p) for p in inventory] != sorted(meta["minute_file_hashes"]):
        raise RuntimeError("minute partition inventory changed")
    for path in inventory:
        if stat_identity(path) != meta["minute_file_stats"][str(path)]:
            raise RuntimeError(f"minute file changed: {path}")
    if deep:
        with ThreadPoolExecutor(max_workers=4) as pool:
            hashes = list(pool.map(file_hash, inventory))
        if dict(zip(map(str, inventory), hashes, strict=True)) != meta["minute_file_hashes"]:
            raise RuntimeError("minute data checksum mismatch")
    return meta


def prepare(data_root, micro_store, exposure, output, device_name="cuda"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preparation requires an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {"state": "running", "phase": "input_audit", "at": now()})
    data_root, micro_store, exposure = (
        Path(p).resolve() for p in (data_root, micro_store, exposure)
    )
    validate_micro_store(micro_store)
    if len(CHANNELS) != 17:
        raise RuntimeError("expected exactly 17 input channels")
    labels = load_labels(data_root, 2019, 2024)
    labels["date"] = pd.to_datetime(labels.date).dt.normalize()
    labels["instrument"] = labels.instrument.astype(str)
    dates, targets = prepare_label_panel(labels, LABEL)
    training, validation = apply_training_history(
        rolling_oos_blocks(
            dates,
            (2024,),
            train_days=480,
            prediction_days=1000,
        ),
        mode="expanding",
    )[0]
    if (len(training), str(dates[training[0]].date()), str(dates[training[-1]].date())) != (
        1213,
        "2019-01-02",
        "2023-12-28",
    ) or len(validation) != 241:
        raise RuntimeError("training/validation calendar differs from agreed protocol")
    expected = (
        labels.loc[labels.date.dt.year == 2024, ["date", "instrument"]]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    if expected.duplicated().any():
        raise RuntimeError("duplicate prediction keys")
    scoring = (
        labels.loc[labels.date.dt.year == 2024]
        .dropna(subset=[LABEL])
        .sort_values(["date", "instrument"])
    )
    if not np.isfinite(scoring[LABEL]).all():
        raise RuntimeError("nonfinite validation returns")
    score_dates = [str(d.date()) for d in pd.DatetimeIndex(scoring.date.unique())]
    if len(score_dates) != 241 or score_dates != [str(d.date()) for d in dates[validation]]:
        raise RuntimeError("score date mismatch")
    date_codes = {str(d.date()): tuple(targets[d].index.astype(str)) for d in dates[training]}
    date_codes.update({str(d.date()): tuple(g.instrument) for d, g in expected.groupby("date")})
    inventory = data_inventory(micro_store, sorted(date_codes))
    print(f"Hashing {len(inventory)} minute partitions", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(file_hash, inventory))
    small = [data_root / f"labels/year={year}/part-{year}.parquet" for year in range(2019, 2025)]
    small += [exposure, micro_store / "manifest.json"]
    meta = {
        "created_at": now(),
        "protocol": protocol_document(),
        "protocol_hash": object_hash(protocol_document()),
        "data_root": str(data_root),
        "micro_store": str(micro_store),
        "exposure": str(exposure),
        "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()},
        "small_data_hashes": {str(p): file_hash(p) for p in small},
        "minute_file_hashes": dict(zip(map(str, inventory), hashes, strict=True)),
        "minute_file_stats": {str(p): stat_identity(p) for p in inventory},
        "input_dates": sorted(date_codes),
        "channels": CHANNELS,
        "training_dates": [str(d.date()) for d in dates[training]],
        "isolated_date": str(dates[validation[0] - 1].date()),
        "scoring_dates": score_dates,
        "scoring_keys_hash": actual_key_hash(scoring),
        "prediction_keys_hash": actual_key_hash(expected),
        "schedule_files": {},
        "actual_sample_hashes": {},
        "initial_hashes": {},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device_name,
            "gpu": torch.cuda.get_device_name() if device_name == "cuda" else None,
        },
    }
    coverage, packed_checks = {}, {}
    probes = (meta["training_dates"][0], meta["training_dates"][-1], score_dates[0])
    for index, (day, codes) in enumerate(sorted(date_codes.items()), 1):
        batch = load_microstructure_day_fast(micro_store, pd.Timestamp(day), codes, max_minutes=240)
        available = [] if batch is None else np.flatnonzero(batch.stock_mask[0]).tolist()
        coverage[day] = [codes[i] for i in available]
        if batch is not None and (
            batch.values.shape != (1, len(codes), 240, 17)
            or not np.isfinite(batch.values[batch.observed_mask]).all()
        ):
            raise RuntimeError(f"invalid packed input: {day}")
        if day in probes and batch is not None:
            packed_checks[day] = {
                name: hashlib.sha256(getattr(batch, name).tobytes()).hexdigest()
                for name in ("values", "observed_mask", "minute_mask", "stock_mask")
            }
        if index % 100 == 0:
            print(f"Audited inputs {index}/{len(date_codes)}", flush=True)
    write_json(output / "coverage.json", coverage)
    meta["coverage_sha256"] = file_hash(output / "coverage.json")
    meta["packed_input_probes"] = packed_checks
    meta["usable_training_days"] = sum(len(coverage[d]) >= 2 for d in meta["training_dates"])
    observed = []
    for day, group in expected.groupby("date", sort=True):
        codes = set(coverage[str(day.date())])
        observed.extend([code in codes and len(codes) >= 2 for code in group.instrument])
    meta["prediction_availability_hash"] = object_hash(observed)
    from experiments.finals_pre.tcn_architecture_o2c.score import expected_scoring_keys

    meta["raw_label_keys_hash"] = meta["scoring_keys_hash"]
    meta["scoring_keys_hash"], meta["scoring_rows"] = expected_scoring_keys(meta, expected)
    for seed in SEEDS:
        schedule = training_schedule(dates, targets, training, seed)
        schedule_path = output / f"schedule_{seed_tag(seed)}.json.gz"
        with gzip.open(schedule_path, "wt") as handle:
            json.dump(schedule, handle, separators=(",", ":"))
        meta["schedule_files"][str(seed)] = file_hash(schedule_path)
        actual = hashlib.sha256()
        for item in schedule:
            available = set(coverage[item["date"]])
            actual.update(
                json.dumps(
                    [
                        item["epoch"],
                        item["date"],
                        [s for s in item["instruments"] if s in available],
                    ]
                ).encode()
            )
        meta["actual_sample_hashes"][str(seed)] = actual.hexdigest()
        common = {}
        per_arm = {}
        for arm in ARMS:
            model = initialize_model(arm, seed)
            per_arm[arm.name] = initial_hashes(model, arm)
            for name, digest in per_arm[arm.name].items():
                if name in common and common[name] != digest:
                    raise RuntimeError(f"common initialization mismatch: {name}")
                common[name] = digest
        meta["initial_hashes"][str(seed)] = per_arm
    preflight = []
    device = torch.device(device_name)
    for arm in ARMS:
        model = initialize_model(arm, SEEDS[0]).to(device).train()
        torch.manual_seed(SEEDS[0])
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
        scaler = make_scaler(device)
        for day in probes:
            codes = tuple(s for s in coverage[day] if s in targets[pd.Timestamp(day)].index)[:32]
            if len(codes) < 2:
                raise RuntimeError(f"insufficient real preflight input: {day}")
            batch = load_microstructure_day_fast(micro_store, pd.Timestamp(day), codes)
            target = torch.tensor(
                targets[pd.Timestamp(day)].reindex(codes).to_numpy(np.float32), device=device
            )
            loss, norm = checked_update(
                model,
                _tensor_inputs(batch, np.arange(len(codes)), device),
                target,
                optimizer,
                scaler,
                device,
            )
            preflight.append(
                {
                    "arm": arm.name,
                    "date": day,
                    "loss": loss,
                    "gradient_norm": norm,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "receptive_field": arm.receptive_field,
                }
            )
        print(f"Preflight passed: {arm.name}", flush=True)
        del model, optimizer
    write_json(output / "preflight.json", preflight)
    meta["preflight_sha256"] = file_hash(output / "preflight.json")
    write_json(output / "prepared.json", meta)
    verify_seal(output)
    write_json(output / "status.json", {"state": "complete", "at": now()})
    return meta
