"""Replay frozen X/T submission checkpoints on the existing Candidate454 store.

This is inference only.  It never fits or updates model parameters.  The
resulting history may include dates used to train the frozen checkpoint and
must therefore be reported as full-history/in-sample-inclusive evidence.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS = ("date", "instrument")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--candidate454-store", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route",
        choices=(
            "x_tree",
            "x_mlp_current",
            "t_protocol_best",
            "t_residual_best",
            "m_raw_expanding_e3",
            "m_l5_final",
        ),
        required=True,
    )
    parser.add_argument("--micro-store", type=Path)
    parser.add_argument("--deep-book-store", type=Path)
    parser.add_argument("--submission-package", type=Path)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--day-batch-size", type=int, default=5)
    return parser.parse_args()


def load_manifest(store: Path) -> tuple[tuple[str, ...], Path]:
    path = store / "candidate454_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_ids = tuple(map(str, payload["candidate_ids"]))
    if len(candidate_ids) != 454 or len(set(candidate_ids)) != 454:
        raise ValueError("Candidate454 manifest must contain 454 unique candidate_ids")
    return candidate_ids, path


def load_year(store: Path, year: int, candidate_ids: tuple[str, ...]) -> pd.DataFrame:
    path = store / "features" / f"year={year}" / f"part-{year}.parquet"
    frame = pd.read_parquet(path, columns=[*KEY_COLUMNS, *candidate_ids])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"duplicate Candidate454 keys in {path}")
    return frame


def rank_daily(frame: pd.DataFrame, raw: np.ndarray) -> pd.DataFrame:
    result = frame.loc[:, list(KEY_COLUMNS)].copy()
    result["factor"] = pd.Series(raw, index=result.index).groupby(
        result["date"], sort=False
    ).rank(method="average", pct=True) * 2.0 - 1.0
    if not np.isfinite(result["factor"].to_numpy(dtype="float64")).all():
        raise ValueError("frozen replay produced non-finite factors")
    return result


def import_submission(repo_root: Path, route: str, submission_package: Path | None):
    if route == "x_mlp_current":
        package = repo_root / "submissions"
        module_name = "unified_x"
    elif route == "m_raw_expanding_e3":
        package = repo_root / "submissions" / "m_expanding_history_e3_20260803"
        module_name = "unified_m_raw"
    elif route == "t_residual_best":
        package = repo_root / "submissions" / "en454_t_residual_020"
        module_name = "unified_en454_t_residual"
    elif route == "m_l5_final":
        if submission_package is None:
            raise ValueError("--submission-package is required for m_l5_final")
        package = submission_package
        module_name = "unified_m_l5"
    else:
        package = repo_root / "submissions" / route
        module_name = "unified_x_tree" if route == "x_tree" else "unified_t"
    sys.path.insert(0, str(package))
    return importlib.import_module(module_name)


def replay_x(
    *,
    submission,
    store: Path,
    output_dir: Path,
    candidate_ids: tuple[str, ...],
    years: range,
) -> dict[str, object]:
    model, _candidate_spec, frozen_ids = submission._load_model()
    if tuple(frozen_ids) != candidate_ids:
        raise ValueError("X frozen Candidate454 order differs from store manifest")
    outputs = []
    for year in years:
        destination = output_dir / f"x_tree_frozen_full_history_{year}.parquet"
        if destination.is_file():
            outputs.append(str(destination))
            continue
        panel = load_year(store, year, candidate_ids)
        parts = []
        for _day, block in panel.groupby("date", sort=True):
            matrix = block.loc[:, list(candidate_ids)].to_numpy(dtype="float32", copy=True)
            raw = model.predict(matrix, num_iteration=model.best_iteration or -1)
            parts.append(rank_daily(block, np.asarray(raw)))
        result = pd.concat(parts, ignore_index=True)
        result.to_parquet(destination, index=False)
        outputs.append(str(destination))
        del panel, parts, result
        gc.collect()
    return {"route": "x_tree", "files": outputs}


def choose_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def replay_x_mlp(
    *,
    submission,
    store: Path,
    output_dir: Path,
    candidate_ids: tuple[str, ...],
    years: range,
    requested_device: str,
) -> dict[str, object]:
    import torch

    model, candidate_spec = submission._load_model(torch)
    if tuple(candidate_spec["candidate_ids"]) != candidate_ids:
        raise ValueError("X-MLP frozen Candidate454 order differs from store manifest")
    device = choose_device(torch, requested_device)
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for year in years:
            destination = output_dir / f"x_mlp_current_frozen_full_history_{year}.parquet"
            if destination.is_file():
                outputs.append(str(destination))
                continue
            panel = load_year(store, year, candidate_ids)
            parts = []
            for _day, block in panel.groupby("date", sort=True):
                matrix = block.loc[:, list(candidate_ids)].to_numpy(
                    dtype="float32", copy=True
                )
                values = torch.from_numpy(matrix).unsqueeze(0).to(device)
                observed = torch.isfinite(values)
                stocks = torch.ones(
                    values.shape[:2], dtype=torch.bool, device=device
                )
                raw = model(values, observed, stocks).squeeze(0).float().cpu().numpy()
                parts.append(rank_daily(block, raw))
                del matrix, values, observed, stocks, raw
            pd.concat(parts, ignore_index=True).to_parquet(destination, index=False)
            outputs.append(str(destination))
            del panel, parts
            gc.collect()
    return {"route": "x_mlp_current", "device": str(device), "files": outputs}


def replay_t(
    *,
    submission,
    store: Path,
    output_dir: Path,
    candidate_ids: tuple[str, ...],
    years: range,
    requested_device: str,
    route_label: str,
) -> dict[str, object]:
    import torch

    frozen_ids = tuple(submission.FROZEN_CANDIDATE_SPEC["candidate_ids"])
    if frozen_ids != candidate_ids:
        raise ValueError("T frozen Candidate454 order differs from store manifest")
    payload = submission._load_checkpoint(torch)
    config = submission.CandidateTemporalConfig(**payload["config"])
    model = submission.CandidateTemporalNetwork(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    device = choose_device(torch, requested_device)
    model.to(device).eval()

    # Stream years and retain only the rolling history.  The store starts in
    # 2019, hence its first 59 trading dates are intentionally unavailable.
    history: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    outputs = []
    skipped = []
    with torch.inference_mode():
        for year in years:
            destination = output_dir / f"{route_label}_frozen_full_history_{year}.parquet"
            panel = load_year(store, year, candidate_ids)
            parts = []
            for day, block in panel.groupby("date", sort=True):
                current = block.set_index("instrument").sort_index()
                history.append((pd.Timestamp(day), current))
                if len(history) > config.lookback:
                    history.pop(0)
                if len(history) < config.lookback:
                    skipped.append(str(pd.Timestamp(day).date()))
                    continue
                instruments = tuple(current.index.astype(str))
                window = np.stack(
                    [
                        past.reindex(instruments)
                        .loc[:, list(candidate_ids)]
                        .to_numpy(dtype="float32", copy=True)
                        for _past_day, past in history
                    ],
                    axis=1,
                )
                values = torch.from_numpy(window).unsqueeze(0).to(device)
                observed = torch.isfinite(values)
                stocks = torch.ones(
                    (1, len(instruments)), dtype=torch.bool, device=device
                )
                raw = model(values, observed, stocks).squeeze(0).float().cpu().numpy()
                keys = current.reset_index().loc[:, ["instrument"]]
                keys.insert(0, "date", pd.Timestamp(day))
                parts.append(rank_daily(keys, raw))
                del window, values, observed, stocks, raw
            if not parts:
                del panel
                continue
            pd.concat(parts, ignore_index=True).to_parquet(destination, index=False)
            outputs.append(str(destination))
            del panel, parts
            gc.collect()
    return {
        "route": route_label,
        "device": str(device),
        "files": outputs,
        "lookback_trading_days": int(config.lookback),
        "skipped_insufficient_history_dates": skipped,
    }


def replay_m(
    *,
    submission,
    data_dir: Path,
    micro_store: Path,
    output_dir: Path,
    years: range,
    requested_device: str,
    day_batch_size: int,
) -> dict[str, object]:
    import base64
    import io
    import zlib

    import torch
    from unified_m_microstructure import pack_microstructure_days

    device = choose_device(torch, requested_device)
    model = submission._load_frozen_model(torch, io, base64, zlib, device)
    outputs = []
    for year in years:
        destination = output_dir / f"m_raw_expanding_e3_frozen_full_history_{year}.parquet"
        if destination.is_file():
            outputs.append(str(destination))
            continue
        universe_path = data_dir / "universe" / f"year={year}" / f"part-{year}.parquet"
        universe = pd.read_parquet(universe_path, columns=list(KEY_COLUMNS))
        universe["date"] = pd.to_datetime(universe["date"], errors="raise").dt.normalize()
        universe["instrument"] = universe["instrument"].astype(str)
        pool_by_day = {
            pd.Timestamp(day): tuple(sorted(block["instrument"].unique()))
            for day, block in universe.groupby("date", sort=True)
        }
        parts = []
        days = list(pool_by_day)
        for offset in range(0, len(days), day_batch_size):
            batch_days = days[offset : offset + day_batch_size]
            frames = []
            valid_days = []
            for day in batch_days:
                source = micro_store / "data" / f"trade_date={day.date()}" / "part-000.parquet"
                if not source.is_file():
                    continue
                features = pd.read_parquet(source)
                features["trade_date"] = day
                features = features.loc[
                    features["instrument"].astype(str).isin(pool_by_day[day])
                ]
                frames.append(features)
                valid_days.append(day)
            if not frames:
                continue
            features = pd.concat(frames, ignore_index=True)
            instruments = tuple(
                sorted({value for day in valid_days for value in pool_by_day[day]})
            )
            batch = pack_microstructure_days(
                features,
                dates=valid_days,
                instruments=instruments,
                max_minutes=242,
            )
            values = torch.from_numpy(batch.values).to(device)
            observed = torch.from_numpy(batch.observed_mask).to(device)
            minutes = torch.from_numpy(batch.minute_mask).to(device)
            stocks = torch.from_numpy(batch.stock_mask).to(device)
            with torch.inference_mode():
                raw = (
                    model.predict((values, observed, minutes, stocks))
                    .float()
                    .cpu()
                    .numpy()
                )
            instrument_index = {value: index for index, value in enumerate(instruments)}
            for day_index, day in enumerate(valid_days):
                day_instruments = pool_by_day[day]
                positions = np.asarray(
                    [instrument_index[value] for value in day_instruments], dtype=int
                )
                available = batch.stock_mask[day_index, positions]
                if int(available.sum()) < 2:
                    continue
                ranked = (
                    pd.Series(raw[day_index, positions][available])
                    .rank(method="average", pct=True)
                    .to_numpy()
                    * 2.0
                    - 1.0
                )
                factor = np.zeros(len(day_instruments), dtype="float32")
                factor[available] = ranked.astype("float32")
                parts.append(
                    pd.DataFrame(
                        {
                            "date": day,
                            "instrument": np.asarray(day_instruments),
                            "factor": factor,
                        }
                    )
                )
            del frames, features, batch, values, observed, minutes, stocks, raw
        if not parts:
            raise RuntimeError(f"M replay produced no rows for {year}")
        result = pd.concat(parts, ignore_index=True)
        result.to_parquet(destination, index=False)
        outputs.append(str(destination))
        del universe, pool_by_day, parts, result
        gc.collect()
    return {
        "route": "m_raw_expanding_e3",
        "device": str(device),
        "files": outputs,
        "micro_store": str(micro_store),
        "checkpoint_training_window": submission.FROZEN_MANIFEST[
            "checkpoint_training_window"
        ],
    }


def replay_m_l5(
    *,
    submission,
    data_dir: Path,
    micro_store: Path,
    deep_book_store: Path,
    output_dir: Path,
    years: range,
    requested_device: str,
    day_batch_size: int,
) -> dict[str, object]:
    import base64
    import io
    import zlib

    import torch
    from unified_m_microstructure_v2 import (
        add_deep_book_context_channels,
        add_dynamic_microstructure_channels,
        pack_microstructure_v2_days,
    )

    device = choose_device(torch, requested_device)
    model = submission._load_frozen_model(torch, io, base64, zlib, device)
    outputs = []
    for year in years:
        destination = output_dir / f"m_l5_final_frozen_full_history_{year}.parquet"
        if destination.is_file():
            outputs.append(str(destination))
            continue
        universe_path = data_dir / "universe" / f"year={year}" / f"part-{year}.parquet"
        universe = pd.read_parquet(universe_path, columns=list(KEY_COLUMNS))
        universe["date"] = pd.to_datetime(universe["date"], errors="raise").dt.normalize()
        universe["instrument"] = universe["instrument"].astype(str)
        pool_by_day = {
            pd.Timestamp(day): tuple(sorted(block["instrument"].unique()))
            for day, block in universe.groupby("date", sort=True)
        }
        deep_path = deep_book_store / f"year={year}" / f"part-{year}.parquet"
        deep = pd.read_parquet(deep_path)
        deep["date"] = pd.to_datetime(deep["date"], errors="raise").dt.normalize()
        deep["instrument"] = deep["instrument"].astype(str)
        parts = []
        days = list(pool_by_day)
        for offset in range(0, len(days), day_batch_size):
            batch_days = days[offset : offset + day_batch_size]
            frames = []
            valid_days = []
            for day in batch_days:
                source = micro_store / "data" / f"trade_date={day.date()}" / "part-000.parquet"
                if not source.is_file():
                    continue
                base = pd.read_parquet(source)
                base["trade_date"] = day
                base = base.loc[base["instrument"].astype(str).isin(pool_by_day[day])]
                frames.append(base)
                valid_days.append(day)
            if not frames:
                continue
            base = pd.concat(frames, ignore_index=True)
            dynamic = add_dynamic_microstructure_channels(base)
            deep_batch = deep.loc[deep["date"].isin(valid_days)]
            features = add_deep_book_context_channels(dynamic, deep_batch)
            instruments = tuple(
                sorted({value for day in valid_days for value in pool_by_day[day]})
            )
            batch = pack_microstructure_v2_days(
                features,
                dates=valid_days,
                instruments=instruments,
                max_minutes=242,
            )
            values = torch.from_numpy(batch.values).to(device)
            observed = torch.from_numpy(batch.observed_mask).to(device)
            minutes = torch.from_numpy(batch.minute_mask).to(device)
            stocks = torch.from_numpy(batch.stock_mask).to(device)
            with torch.inference_mode():
                raw = (
                    model.predict((values, observed, minutes, stocks))
                    .float()
                    .cpu()
                    .numpy()
                )
            instrument_index = {value: index for index, value in enumerate(instruments)}
            for day_index, day in enumerate(valid_days):
                day_instruments = pool_by_day[day]
                positions = np.asarray(
                    [instrument_index[value] for value in day_instruments], dtype=int
                )
                available = batch.stock_mask[day_index, positions]
                if int(available.sum()) < 2:
                    continue
                ranked = (
                    pd.Series(raw[day_index, positions][available])
                    .rank(method="average", pct=True)
                    .to_numpy()
                    * 2.0
                    - 1.0
                )
                factor = np.zeros(len(day_instruments), dtype="float32")
                factor[available] = ranked.astype("float32")
                parts.append(
                    pd.DataFrame(
                        {
                            "date": day,
                            "instrument": np.asarray(day_instruments),
                            "factor": factor,
                        }
                    )
                )
            del frames, base, dynamic, features, batch, values, observed, minutes, stocks, raw
        if not parts:
            raise RuntimeError(f"M-l5 replay produced no rows for {year}")
        result = pd.concat(parts, ignore_index=True)
        result.to_parquet(destination, index=False)
        outputs.append(str(destination))
        del universe, deep, pool_by_day, parts, result
        gc.collect()
    return {
        "route": "m_l5_final",
        "device": str(device),
        "files": outputs,
        "micro_store": str(micro_store),
        "deep_book_store": str(deep_book_store),
        "checkpoint_training_window": submission.FROZEN_MANIFEST[
            "checkpoint_training_window"
        ],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir if args.data_dir is not None else args.repo_root / "data"
    candidate_ids, manifest_path = load_manifest(args.candidate454_store)
    submission = import_submission(args.repo_root, args.route, args.submission_package)
    years = range(args.start_year, args.end_year + 1)
    common = {
        "submission": submission,
        "store": args.candidate454_store,
        "output_dir": args.output_dir,
        "candidate_ids": candidate_ids,
        "years": years,
    }
    if args.route == "x_tree":
        result = replay_x(**common)
    elif args.route == "x_mlp_current":
        result = replay_x_mlp(
            **common,
            requested_device=args.device,
        )
    elif args.route in {"t_protocol_best", "t_residual_best"}:
        result = replay_t(
            **common,
            requested_device=args.device,
            route_label=args.route,
        )
    elif args.route == "m_raw_expanding_e3":
        if args.micro_store is None:
            raise ValueError("--micro-store is required for m_raw_expanding_e3")
        result = replay_m(
            submission=submission,
            data_dir=data_dir,
            micro_store=args.micro_store,
            output_dir=args.output_dir,
            years=years,
            requested_device=args.device,
            day_batch_size=args.day_batch_size,
        )
    else:
        if args.micro_store is None or args.deep_book_store is None:
            raise ValueError(
                "--micro-store and --deep-book-store are required for m_l5_final"
            )
        result = replay_m_l5(
            submission=submission,
            data_dir=data_dir,
            micro_store=args.micro_store,
            deep_book_store=args.deep_book_store,
            output_dir=args.output_dir,
            years=years,
            requested_device=args.device,
            day_batch_size=args.day_batch_size,
        )
    result.update(
        {
            "protocol": "frozen_checkpoint_full_history_inference_v1",
            "training_performed": False,
            "includes_checkpoint_training_period": True,
            "candidate454_manifest": str(manifest_path),
            "years_requested": list(years),
        }
    )
    (args.output_dir / f"{args.route}_replay_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
