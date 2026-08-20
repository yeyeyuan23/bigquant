"""Run the resumable X/T protocol repair as one sequential GPU job.

Selection uses 2023 OOS only. The selected configuration is then audited on
untouched 2024 OOS and finally refit on all eligible 2019-2024 labels. Existing
routes and checkpoints are never overwritten because every run has its own
directory under a new report root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TemporalSpec:
    model_dim: int
    transformer_layers: int
    attention_heads: int
    feedforward_dim: int
    history_mode: str
    train_stride: int
    epochs: int = 8

    @property
    def run_id(self) -> str:
        return (
            f"d{self.model_dim}_l{self.transformer_layers}_"
            f"{self.history_mode}_s{self.train_stride}_e{self.epochs}"
        )


@dataclass(frozen=True)
class MlpSpec:
    epochs: int
    train_stride: int

    @property
    def run_id(self) -> str:
        return f"mlp_s{self.train_stride}_e{self.epochs}"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_metrics(run_dir: Path) -> dict[str, float]:
    rows = json.loads((run_dir / "oos_metrics.json").read_text(encoding="utf-8"))
    rank_ic = np.asarray([float(row["rank_ic_mean"]) for row in rows], dtype=float)
    if len(rank_ic) != 2 or not np.isfinite(rank_ic).all():
        raise RuntimeError(f"expected two finite H1/H2 metrics in {run_dir}")
    return {
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_ic_worst": float(rank_ic.min()),
        "rank_ic_std": float(rank_ic.std()),
        "rank_ic_stable": float(rank_ic.mean() - 0.5 * rank_ic.std()),
    }


def run_complete(run_dir: Path) -> bool:
    try:
        read_metrics(run_dir)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    return True


def run_command(command: list[str], run_dir: Path, *, dry_run: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "command.json", {"argv": command})
    if dry_run:
        print("DRY-RUN", " ".join(command), flush=True)
        return
    with (run_dir / "run.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit {result.returncode}; see {run_dir / 'run.log'}"
        )


def common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root",
        str(args.data_root),
        "--candidate-pool",
        str(args.candidate_pool),
        "--candidate-manifest",
        str(args.candidate_manifest),
        "--expected-candidate-count",
        "454",
    ]


def temporal_command(
    args: argparse.Namespace,
    spec: TemporalSpec,
    *,
    year: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        "scripts/evaluate_unified_temporal.py",
        *common_args(args),
        "--output-dir",
        str(output_dir),
        "--years",
        str(year),
        "--train-start-year",
        "2019",
        "--epochs",
        str(spec.epochs),
        "--train-stride",
        str(spec.train_stride),
        "--model-dim",
        str(spec.model_dim),
        "--transformer-layers",
        str(spec.transformer_layers),
        "--attention-heads",
        str(spec.attention_heads),
        "--feedforward-dim",
        str(spec.feedforward_dim),
        "--history-mode",
        spec.history_mode,
        "--kernels",
        "3",
        "5",
        "15",
        "--dropout",
        "0.1",
        "--learning-rate",
        "0.0005",
        "--max-stocks",
        "1200",
        "--seed",
        str(seed),
    ]


def mlp_command(
    args: argparse.Namespace,
    spec: MlpSpec,
    *,
    year: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        str(args.python),
        "scripts/evaluate_unified_mlp.py",
        *common_args(args),
        "--output-dir",
        str(output_dir),
        "--years",
        str(year),
        "--train-start-year",
        "2019",
        "--hidden-dims",
        "512",
        "256",
        "--dropout",
        "0.12",
        "--epochs",
        str(spec.epochs),
        "--train-stride",
        str(spec.train_stride),
        "--max-stocks",
        "1200",
        "--learning-rate",
        "0.0004",
        "--seed",
        str(seed),
    ]


def run_oos(
    command: list[str],
    run_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, float] | None:
    if run_complete(run_dir):
        print(f"reuse complete run: {run_dir}", flush=True)
    else:
        run_command(command, run_dir, dry_run=dry_run)
    return None if dry_run else read_metrics(run_dir)


def score_routes(
    args: argparse.Namespace,
    routes: list[Path],
    *,
    year: int,
    output: Path,
) -> dict[str, dict[str, float]]:
    command = [
        str(args.python),
        "scripts/score_route_periods.py",
        *(str(route) for route in routes),
        "--years",
        str(year),
        "--data-dir",
        str(args.data_root),
        "--reports-dir",
        str(ROOT / "reports"),
        "--output",
        str(output),
    ]
    if not output.is_file():
        run_command(command, output.parent, dry_run=args.dry_run)
    if args.dry_run:
        return {}
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {str(row["route"]): row for row in payload["results"]}


def choose_screen_top(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["rank_ic_worst"]),
            float(row["rank_ic_stable"]),
            float(row["rank_ic_mean"]),
        ),
        reverse=True,
    )[:count]


def aggregate_confirm(
    rows: list[dict[str, object]],
    *,
    spec_field: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[spec_field]), []).append(row)
    summaries = []
    for run_id, members in grouped.items():
        j_stable = np.asarray([float(row["J_stable"]) for row in members])
        j_worst = np.asarray([float(row["J_worst"]) for row in members])
        rank_stable = np.asarray([float(row["rank_ic_stable"]) for row in members])
        summaries.append(
            {
                spec_field: run_id,
                "seeds": [int(row["seed"]) for row in members],
                "J_stable_mean": float(j_stable.mean()),
                "J_stable_worst_seed": float(j_stable.min()),
                "J_stable_seed_std": float(j_stable.std()),
                "J_worst_across_seeds": float(j_worst.min()),
                "rank_ic_stable_mean": float(rank_stable.mean()),
                "rank_ic_stable_worst_seed": float(rank_stable.min()),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            float(row["J_stable_worst_seed"]),
            float(row["J_stable_mean"]),
            float(row["rank_ic_stable_worst_seed"]),
        ),
        reverse=True,
    )


def final_refit_command(
    args: argparse.Namespace,
    *,
    model: str,
    output_dir: Path,
    epochs: int,
    train_stride: int,
    temporal: TemporalSpec | None = None,
) -> list[str]:
    command = [
        str(args.python),
        "scripts/train_unified_final_checkpoint.py",
        "--model",
        model,
        *common_args(args),
        "--output-dir",
        str(output_dir),
        "--train-start",
        "2019-01-01",
        "--train-end",
        "2024-12-31",
        "--epochs",
        str(epochs),
        "--train-stride",
        str(train_stride),
        "--max-stocks",
        "1200",
        "--seed",
        "20260803",
    ]
    if model == "mlp":
        return [
            *command,
            "--hidden-dims",
            "512",
            "256",
            "--dropout",
            "0.12",
            "--learning-rate",
            "0.0004",
        ]
    assert temporal is not None
    return [
        *command,
        "--model-dim",
        str(temporal.model_dim),
        "--transformer-layers",
        str(temporal.transformer_layers),
        "--attention-heads",
        str(temporal.attention_heads),
        "--feedforward-dim",
        str(temporal.feedforward_dim),
        "--history-mode",
        temporal.history_mode,
        "--dropout",
        "0.1",
        "--learning-rate",
        "0.0005",
    ]


def run_mlp_protocol(args: argparse.Namespace) -> dict[str, object]:
    root = args.output_root / "x_mlp"
    screen_specs = [MlpSpec(epochs=e, train_stride=s) for s in (1, 2) for e in (8, 12)]
    screen_rows = []
    for spec in screen_specs:
        run_dir = root / "screen_2023" / spec.run_id / "seed_20260803"
        metrics = run_oos(
            mlp_command(args, spec, year=2023, seed=20260803, output_dir=run_dir),
            run_dir,
            dry_run=args.dry_run,
        )
        if metrics:
            screen_rows.append({"spec": asdict(spec), "run_id": spec.run_id, **metrics})
    if args.dry_run:
        return {}
    top = choose_screen_top(screen_rows, 2)
    confirm_specs = [MlpSpec(**row["spec"]) for row in top]
    confirm_rows = []
    for spec in confirm_specs:
        for seed in args.seeds:
            run_dir = root / "confirm_2023" / spec.run_id / f"seed_{seed}"
            metrics = run_oos(
                mlp_command(args, spec, year=2023, seed=seed, output_dir=run_dir),
                run_dir,
                dry_run=False,
            )
            confirm_rows.append(
                {
                    "run_id": spec.run_id,
                    "spec": asdict(spec),
                    "seed": seed,
                    "route": str(run_dir / "unified_mlp_full_oos.parquet"),
                    **metrics,
                }
            )
    j_by_route = score_routes(
        args,
        [Path(row["route"]) for row in confirm_rows],
        year=2023,
        output=root / "confirm_2023" / "j_scores.json",
    )
    for row in confirm_rows:
        row.update({key: value for key, value in j_by_route[row["route"]].items() if key.startswith("J_")})
    ranking = aggregate_confirm(confirm_rows, spec_field="run_id")
    winner_id = str(ranking[0]["run_id"])
    winner = next(spec for spec in confirm_specs if spec.run_id == winner_id)
    audit_rows = []
    for seed in args.seeds:
        run_dir = root / "audit_2024" / winner.run_id / f"seed_{seed}"
        metrics = run_oos(
            mlp_command(args, winner, year=2024, seed=seed, output_dir=run_dir),
            run_dir,
            dry_run=False,
        )
        audit_rows.append({"seed": seed, "route": str(run_dir / "unified_mlp_full_oos.parquet"), **metrics})
    score_routes(
        args,
        [Path(row["route"]) for row in audit_rows],
        year=2024,
        output=root / "audit_2024" / "j_scores.json",
    )
    final_dir = root / "final_full_history"
    if not (final_dir / "final_checkpoint.json").is_file():
        run_command(
            final_refit_command(
                args,
                model="mlp",
                output_dir=final_dir,
                epochs=winner.epochs,
                train_stride=winner.train_stride,
            ),
            final_dir,
            dry_run=False,
        )
    result = {
        "selection_year": 2023,
        "audit_year": 2024,
        "screen": screen_rows,
        "confirm": confirm_rows,
        "ranking": ranking,
        "winner": asdict(winner),
        "audit": audit_rows,
        "final_checkpoint": str(final_dir / "final_checkpoint.json"),
    }
    write_json(root / "selection.json", result)
    return result


def run_temporal_protocol(args: argparse.Namespace) -> dict[str, object]:
    root = args.output_root / "t_temporal"
    widths = ((128, 2, 8, 384), (256, 4, 8, 768), (384, 6, 8, 1024))
    screen_specs = [
        TemporalSpec(dim, layers, heads, ff, history, stride)
        for dim, layers, heads, ff in widths
        for history in ("dense", "stride2", "multiscale")
        for stride in (1, 2)
    ]
    screen_rows = []
    for spec in screen_specs:
        run_dir = root / "screen_2023" / spec.run_id / "seed_20260803"
        metrics = run_oos(
            temporal_command(args, spec, year=2023, seed=20260803, output_dir=run_dir),
            run_dir,
            dry_run=args.dry_run,
        )
        if metrics:
            screen_rows.append({"spec": asdict(spec), "run_id": spec.run_id, **metrics})
    if args.dry_run:
        return {}
    top = choose_screen_top(screen_rows, 4)
    confirm_specs = [
        replace(TemporalSpec(**row["spec"]), epochs=epochs)
        for row in top
        for epochs in (8, 12)
    ]
    confirm_rows = []
    for spec in confirm_specs:
        for seed in args.seeds:
            run_dir = root / "confirm_2023" / spec.run_id / f"seed_{seed}"
            metrics = run_oos(
                temporal_command(args, spec, year=2023, seed=seed, output_dir=run_dir),
                run_dir,
                dry_run=False,
            )
            confirm_rows.append(
                {
                    "run_id": spec.run_id,
                    "spec": asdict(spec),
                    "seed": seed,
                    "route": str(run_dir / "unified_temporal_2023_full_oos.parquet"),
                    **metrics,
                }
            )
    j_by_route = score_routes(
        args,
        [Path(row["route"]) for row in confirm_rows],
        year=2023,
        output=root / "confirm_2023" / "j_scores.json",
    )
    for row in confirm_rows:
        row.update({key: value for key, value in j_by_route[row["route"]].items() if key.startswith("J_")})
    ranking = aggregate_confirm(confirm_rows, spec_field="run_id")
    winner_id = str(ranking[0]["run_id"])
    winner = next(spec for spec in confirm_specs if spec.run_id == winner_id)
    audit_rows = []
    for seed in args.seeds:
        run_dir = root / "audit_2024" / winner.run_id / f"seed_{seed}"
        metrics = run_oos(
            temporal_command(args, winner, year=2024, seed=seed, output_dir=run_dir),
            run_dir,
            dry_run=False,
        )
        audit_rows.append(
            {
                "seed": seed,
                "route": str(run_dir / "unified_temporal_2024_full_oos.parquet"),
                **metrics,
            }
        )
    score_routes(
        args,
        [Path(row["route"]) for row in audit_rows],
        year=2024,
        output=root / "audit_2024" / "j_scores.json",
    )
    final_dir = root / "final_full_history"
    if not (final_dir / "final_checkpoint.json").is_file():
        run_command(
            final_refit_command(
                args,
                model="temporal",
                output_dir=final_dir,
                epochs=winner.epochs,
                train_stride=winner.train_stride,
                temporal=winner,
            ),
            final_dir,
            dry_run=False,
        )
    result = {
        "selection_year": 2023,
        "audit_year": 2024,
        "screen": screen_rows,
        "confirm": confirm_rows,
        "ranking": ranking,
        "winner": asdict(winner),
        "audit": audit_rows,
        "final_checkpoint": str(final_dir / "final_checkpoint.json"),
    }
    write_json(root / "selection.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/autodl-tmp/conda-envs/quant/bin/python"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/projects/bigquant/data"),
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=Path(
            "/root/autodl-tmp/candidate454_completion_full_2019_2024/"
            "candidate454_store/features"
        ),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path(
            "/root/autodl-tmp/candidate454_completion_full_2019_2024/"
            "candidate454_store/candidate454_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "reports" / "xt_protocol_retrain_20260803",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260801, 20260802, 20260803])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / "run.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another XT protocol run holds {lock_path}") from exc
        write_json(
            args.output_root / "run_manifest.json",
            {
                "protocol": "2023_select_2024_audit_2019_2024_final_refit",
                "selection_uses_2024": False,
                "candidate_count": 454,
                "allowed_sources": ["bar1m", "financial"],
                "seeds": args.seeds,
                "pid": os.getpid(),
            },
        )
        x_result = run_mlp_protocol(args)
        t_result = run_temporal_protocol(args)
        if not args.dry_run:
            write_json(
                args.output_root / "complete.json",
                {
                    "status": "complete",
                    "x_selection": str(args.output_root / "x_mlp" / "selection.json"),
                    "t_selection": str(args.output_root / "t_temporal" / "selection.json"),
                    "x_winner": x_result["winner"],
                    "t_winner": t_result["winner"],
                    "evidence_boundary": "Local OOS/proxy and final checkpoints only.",
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
