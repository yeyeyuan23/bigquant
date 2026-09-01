from __future__ import annotations

"""Run full-Barra neutralization and the production E5 A-score path."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competition_score_proxy import (
    FULL_BARRA_INDUSTRY_COLUMNS,
    FULL_BARRA_REGRESSORS,
    FULL_BARRA_STYLE_COLUMNS,
    prepare_full_barra_exposures,
    preprocess_factor,
)


def synthetic_score_inputs(days: int = 241, stocks: int = 50):
    rng = np.random.default_rng(20260901)
    dates = pd.bdate_range("2024-01-02", periods=days)
    instruments = np.array([f"S{index:04d}" for index in range(stocks)])
    exposure_rows = []
    factor_rows = []
    label_rows = []
    for day_index, day in enumerate(dates):
        styles = rng.normal(size=(stocks, len(FULL_BARRA_STYLE_COLUMNS)))
        industries = np.zeros((stocks, len(FULL_BARRA_INDUSTRY_COLUMNS)))
        industries[np.arange(stocks), np.arange(stocks) % industries.shape[1]] = 1.0
        hidden = np.sin(np.arange(stocks) * 0.37 + day_index * 0.13)
        hidden += rng.normal(scale=0.08, size=stocks)
        label = hidden + 0.2 * np.cos(np.arange(stocks) * 0.17 + day_index * 0.07)
        factor = 1.7 * styles[:, 0] - 0.9 * styles[:, 1] + hidden
        for stock_index, instrument in enumerate(instruments):
            exposure = {
                "date": day,
                "instrument": instrument,
                **dict(zip(FULL_BARRA_STYLE_COLUMNS, styles[stock_index], strict=True)),
                "industry_level1_code": f"I{stock_index % 32:02d}",
                "float_market_cap": float(np.exp(styles[stock_index, 0] + 10.0)),
                "weights": 1.0,
                "ret": 0.0,
                **dict(
                    zip(
                        FULL_BARRA_INDUSTRY_COLUMNS,
                        industries[stock_index],
                        strict=True,
                    )
                ),
            }
            exposure_rows.append(exposure)
            factor_rows.append(
                {"date": day, "instrument": instrument, "factor": factor[stock_index]}
            )
            label_rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "ret_next_open_to_close": label[stock_index],
                }
            )
    return (
        pd.DataFrame(factor_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(exposure_rows),
    )


def test_full_barra_names_order_and_residualization_are_executed():
    factor, _, raw_exposures = synthetic_score_inputs(days=2, stocks=64)
    exposures = prepare_full_barra_exposures(raw_exposures)
    assert tuple(exposures.columns[2:]) == FULL_BARRA_REGRESSORS
    neutral = preprocess_factor(factor, exposures)
    merged = neutral.merge(exposures, on=["date", "instrument"], validate="one_to_one")
    for _, day in merged.groupby("date"):
        residual = day["factor"].to_numpy(float)
        design = day[list(FULL_BARRA_REGRESSORS)].to_numpy(float)
        np.testing.assert_allclose(design.T @ residual, 0.0, atol=1e-10)


@pytest.mark.parametrize("damage", ["missing", "reordered", "extra", "duplicate"])
def test_broken_full_barra_inputs_fail_closed(damage):
    _, _, exposures = synthetic_score_inputs(days=1, stocks=50)
    if damage == "missing":
        exposures = exposures.drop(columns="LIQUIDTY")
    elif damage == "reordered":
        columns = list(exposures.columns)
        left, right = columns.index("SIZE"), columns.index("BETA")
        columns[left], columns[right] = columns[right], columns[left]
        exposures = exposures[columns]
    elif damage == "extra":
        exposures["UNDECLARED_STYLE"] = 0.0
    else:
        exposures = pd.concat([exposures, exposures.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="full-Barra|duplicate"):
        prepare_full_barra_exposures(exposures)


def write_e5_artifacts(output: Path, factor: pd.DataFrame, score_module) -> None:
    for arm in score_module.ARMS:
        for seed in score_module.SEEDS:
            run = output / f"{arm}_seed{seed}"
            run.mkdir(parents=True)
            factor.to_parquet(run / "factor_2024.parquet", index=False)
            checkpoint = run / "checkpoint.pt"
            checkpoint.write_bytes(f"{arm}:{seed}".encode())
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            metadata = {
                "checkpoint_sha256": digest,
                "label": score_module.LABEL,
                "prediction_days": int(factor["date"].nunique()),
                "factor_rows": len(factor),
                "device": "cuda",
            }
            (run / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")


def run_e5_score(
    monkeypatch,
    score_module,
    output: Path,
    labels: Path,
    exposures: Path,
    source_dir: Path,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_autodl.py",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output),
            "--labels",
            str(labels),
            "--exposures",
            str(exposures),
        ],
    )
    return score_module.main()


def test_complete_e5_a_score_pipeline_and_negative_barra_guard(
    load_experiment_module, monkeypatch, tmp_path
):
    score = load_experiment_module("e5_raw23_direct/score_autodl.py", "pre_e5_score_pipeline")
    factor, labels, exposures = synthetic_score_inputs()
    output = tmp_path / "runs"
    write_e5_artifacts(output, factor, score)
    labels_path = tmp_path / "labels.parquet"
    exposures_path = tmp_path / "exposures.parquet"
    labels.to_parquet(labels_path, index=False)
    exposures.to_parquet(exposures_path, index=False)
    source_dir = Path(__file__).resolve().parents[2] / "src"

    assert run_e5_score(monkeypatch, score, output, labels_path, exposures_path, source_dir) == 0
    per_seed = pd.read_csv(output / "a4_per_seed.csv")
    assert len(per_seed) == len(score.ARMS) * len(score.SEEDS)
    assert set(score.A_COLUMNS).issubset(per_seed.columns)
    assert np.isfinite(per_seed[list(score.A_COLUMNS)].to_numpy()).all()
    audit = json.loads((output / "score_audit.json").read_text(encoding="utf-8"))
    assert audit["full_barra_regressor_count"] == 42
    assert audit["test_days"] == 241

    exposures.drop(columns="LIQUIDTY").to_parquet(exposures_path, index=False)
    with pytest.raises(RuntimeError, match="full-Barra schema mismatch"):
        run_e5_score(monkeypatch, score, output, labels_path, exposures_path, source_dir)


def test_actual_autodl_e5_artifacts_run_the_complete_score_chain(
    load_experiment_module, monkeypatch, tmp_path
):
    canonical = Path("/root/autodl-tmp/projects/bigquant-default")
    actual_runs = canonical / "reports/dependencies/finals_pre/e5_raw23_direct/o2c_autodl"
    labels = Path("/root/autodl-tmp/data/labels/year=2024/part-2024.parquet")
    exposures = Path("/root/autodl-tmp/exposure_2024_full.parquet")
    required = (actual_runs, labels, exposures)
    if not all(path.exists() for path in required):
        pytest.skip("AutoDL formal E5 inputs are not mounted in this environment")

    score = load_experiment_module(
        "e5_raw23_direct/score_autodl.py", "pre_e5_actual_score_pipeline"
    )
    output = tmp_path / "actual_e5_score"
    output.mkdir()
    for arm in score.ARMS:
        for seed in score.SEEDS:
            source = actual_runs / f"{arm}_seed{seed}"
            assert source.is_dir()
            (output / source.name).symlink_to(source, target_is_directory=True)

    source_dir = Path(__file__).resolve().parents[2] / "src"
    assert run_e5_score(monkeypatch, score, output, labels, exposures, source_dir) == 0
    per_seed = pd.read_csv(output / "a4_per_seed.csv")
    assert len(per_seed) == 6
    assert per_seed["days"].eq(241).all()
    assert np.isfinite(per_seed[list(score.A_COLUMNS)].to_numpy()).all()
