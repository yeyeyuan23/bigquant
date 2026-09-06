"""Run one matched 240/242-clock pair, then score both with industry/Barra removal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(args):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(args.current_root / "src"))
    from competition_score_proxy import prepare_full_barra_exposures, preprocess_factor

    keys = ["date", "instrument"]
    label = "ret_next_open_to_close"
    labels = pd.read_parquet(args.data_root / "labels/year=2024/part-2024.parquet")
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["instrument"] = labels["instrument"].astype(str)
    labels = labels.dropna(subset=[label])
    exposure = prepare_full_barra_exposures(pd.read_parquet(args.exposure))
    dispersion = labels.groupby("date")[label].std()
    stress = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)
    scores = []
    factors = {}
    for arm in ("clock240", "clock242"):
        directory = args.output_dir / arm
        factor = pd.read_parquet(directory / "unified_microstructure_full_oos.parquet")
        factor["date"] = pd.to_datetime(factor["date"]).dt.normalize()
        factor["instrument"] = factor["instrument"].astype(str)
        factors[arm] = factor.sort_values(keys).reset_index(drop=True)
        neutral = preprocess_factor(factor, exposure).rename(columns={"factor": "neutral_factor"})
        merged = neutral.merge(labels[keys + [label]], on=keys, validate="one_to_one").dropna()
        ic = (
            merged.groupby("date")
            .apply(
                lambda g: g["neutral_factor"].corr(g[label], method="spearman"),
                include_groups=False,
            )
            .dropna()
        )
        if len(ic) != 241:
            raise RuntimeError(f"{arm}: expected 241 scoreable O2C dates, got {len(ic)}")
        ic.rename("neutral_rank_ic").to_csv(directory / "daily_neutral_rank_ic.csv")
        merged["bucket"] = merged.groupby("date")["neutral_factor"].transform(
            lambda v: pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop")
        )
        spread = (
            merged[merged.bucket == 4].groupby("date")[label].mean()
            - merged[merged.bucket == 0].groupby("date")[label].mean()
        ).dropna()
        stress_ic = ic[ic.index.isin(stress)]
        scores.append(
            {
                "arm": arm,
                "seed": args.seed,
                "epochs": 3,
                "days": len(ic),
                "neutral_rank_ic": float(ic.mean()),
                "neutral_rank_ic_ir": float(ic.mean() / ic.std()),
                "long_short_sharpe": float(spread.mean() / spread.std() * np.sqrt(252)),
                "stress_ic_ir": float(stress_ic.mean() / stress_ic.std()),
            }
        )
    pd.testing.assert_frame_equal(factors["clock240"][keys], factors["clock242"][keys])
    table = pd.DataFrame(scores)
    table.to_csv(args.output_dir / "comparison.csv", index=False)
    a, b = table.set_index("arm").loc["clock240"], table.set_index("arm").loc["clock242"]
    metrics = ["neutral_rank_ic", "neutral_rank_ic_ir", "long_short_sharpe", "stress_ic_ir"]
    write_json(
        args.output_dir / "comparison.json",
        {
            "scores": scores,
            "delta_240_minus_242": {key: float(a[key] - b[key]) for key in metrics},
            "scope": "One seed; current fixed-clock 242 control, not frozen tail-padded submission.",
            "evaluation": "Same industry plus Barra neutralization as E2; O2C; 241 days in 2024.",
            "factor_rows": len(factors["clock240"]),
        },
    )
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "current-root",
        "control-root",
        "output-dir",
        "data-root",
        "micro-store",
        "exposure",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    if status_path.exists():
        raise RuntimeError("output already has a run; use a fresh output directory")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        POLARS_MAX_THREADS="8",
        PYTHONUNBUFFERED="1",
    )
    preflight = {}
    for minutes in (240, 242):
        item = json.loads((args.output_dir / f"preflight{minutes}.json").read_text())
        assert item.pop("minutes") == minutes
        if item.get("seed", 20260801) != args.seed:
            raise RuntimeError("preflight seed differs from requested training seed")
        preflight[minutes] = item
    if preflight[240] != preflight[242]:
        raise RuntimeError("paired preflight differs beyond the minute count")
    manifest = {
        "started_at": now(),
        "seed": args.seed,
        "epochs": 3,
        "paired_preflight": preflight[240],
        "control": "pre-fix c0dcd21941e30739e3135997e7928875bca022aa fixed 242 grid",
        "train": "2019-2023, expanding with one trading date label isolation",
        "test": "2024 O2C, 241 scoreable dates",
        "preserved": "original submissions bundles and existing experiment outputs",
        "environment": {key: env[key] for key in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS")},
        "data_root": str(args.data_root),
        "micro_store": str(args.micro_store),
        "store_manifest_sha256": digest(args.micro_store / "manifest.json"),
        "label_sha256": {
            str(y): digest(args.data_root / f"labels/year={y}/part-{y}.parquet")
            for y in range(2019, 2025)
        },
        "exposure_sha256": digest(args.exposure),
        "arms": {},
    }
    write_json(args.output_dir / "pair_manifest.json", manifest)
    started = time.monotonic()
    status = {
        "state": "running",
        "seed": args.seed,
        "started_at": now(),
        "pid": os.getpid(),
        "completed_arms": [],
    }
    write_json(status_path, status)
    try:
        for arm, root, minutes in [
            ("clock240", args.current_root, 240),
            ("clock242", args.control_root, 242),
        ]:
            destination = args.output_dir / arm
            command = [
                sys.executable,
                "-u",
                str(root / "scripts/evaluate_unified_microstructure.py"),
                "--data-root",
                str(args.data_root),
                "--micro-store",
                str(args.micro_store),
                "--output-dir",
                str(destination),
                "--years",
                "2024",
                "--train-start-year",
                "2019",
                "--training-mode",
                "expanding",
                "--prediction-days",
                "999",
                "--epochs",
                "3",
                "--model-dim",
                "96",
                "--kernels",
                "3",
                "15",
                "60",
                "--tcn-blocks",
                "3",
                "--tail-minutes",
                "30",
                "--max-minutes",
                str(minutes),
                "--max-stocks",
                "1200",
                "--min-train-days",
                "900",
                "--learning-rate",
                "4e-4",
                "--seed",
                str(args.seed),
                "--label-column",
                "ret_next_open_to_close",
                "--fast-pack",
            ]
            manifest["arms"][arm] = {
                "command": command,
                "root": str(root),
                "model_sha256": digest(root / "src/alpha_models/microstructure.py"),
                "trainer_sha256": digest(root / "scripts/evaluate_unified_microstructure.py"),
                "fastpack_sha256": digest(root / "experiments/finals_pre/common/fastpack.py"),
            }
            write_json(args.output_dir / "pair_manifest.json", manifest)
            status.update(arm=arm, arm_started_at=now(), phase="train_and_infer")
            write_json(status_path, status)
            arm_started = time.monotonic()
            with (args.output_dir / f"{arm}.log").open("w", buffering=1) as log:
                subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            status["completed_arms"].append({"arm": arm, "seconds": time.monotonic() - arm_started})
        status.update(phase="neutralized_scoring")
        write_json(status_path, status)
        score(args)
        status.update(state="completed", phase="complete", finished_at=now())
    except BaseException as error:
        status.update(state="failed", error=repr(error), finished_at=now())
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
