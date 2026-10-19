"""Run the two additional predeclared seeds; keep a single overall completion status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_pair import digest, now, write_json
from summarize_pairs import summarize

SEEDS = (20260812, 20260823)


def verify_original_protocol(args):
    original = json.loads((args.original_pair / "pair_manifest.json").read_text())
    for arm, root in (("clock240", args.current_root), ("clock242", args.control_root)):
        for filename, key in (
            ("src/alpha_models/microstructure.py", "model_sha256"),
            ("scripts/evaluate_unified_microstructure.py", "trainer_sha256"),
            ("experiments/finals_pre/common/fastpack.py", "fastpack_sha256"),
        ):
            if digest(root / filename) != original["arms"][arm][key]:
                raise RuntimeError(f"original training source differs: {arm}/{filename}")
    actual = {
        "store_manifest_sha256": digest(args.micro_store / "manifest.json"),
        "exposure_sha256": digest(args.exposure),
        "label_sha256": {
            str(y): digest(args.data_root / f"labels/year={y}/part-{y}.parquet")
            for y in range(2019, 2025)
        },
    }
    for key, value in actual.items():
        if value != original[key]:
            raise RuntimeError(f"original data differs: {key}")
    return {
        "training_sources_match": True,
        "data_hashes_match": True,
        "original_seed": original["seed"],
    }


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "current-root",
        "control-root",
        "output-dir",
        "data-root",
        "micro-store",
        "exposure",
        "original-pair",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    if status_path.exists():
        raise RuntimeError("output already has a run; use a fresh output directory")
    started = time.monotonic()
    status = {
        "state": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "seeds": list(SEEDS),
        "completed_seeds": [],
    }
    write_json(status_path, status)
    code = Path(__file__).resolve().parent
    env = dict(
        os.environ,
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        POLARS_MAX_THREADS="8",
    )
    try:
        write_json(args.output_dir / "protocol_check.json", verify_original_protocol(args))
        pairs = [args.original_pair]
        for seed in SEEDS:
            destination = args.output_dir / f"seed{seed}"
            destination.mkdir()
            status.update(seed=seed, phase="preflight", seed_started_at=now())
            write_json(status_path, status)
            for minutes, root in ((240, args.current_root), (242, args.control_root)):
                with (destination / f"preflight{minutes}.json").open("w") as output:
                    subprocess.run(
                        [
                            sys.executable,
                            str(code / "preflight.py"),
                            str(root),
                            "--seed",
                            str(seed),
                        ],
                        stdout=output,
                        check=True,
                        env=env,
                    )
            status.update(phase="paired_training")
            write_json(status_path, status)
            command = [sys.executable, "-u", str(code / "run_pair.py"), "--seed", str(seed)]
            for key in ("current_root", "control_root", "data_root", "micro_store", "exposure"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            command += ["--output-dir", str(destination)]
            subprocess.run(command, check=True, env=env)
            status["completed_seeds"].append(seed)
            pairs.append(destination)
        status.update(phase="three_seed_summary")
        write_json(status_path, status)
        summarize(pairs, args.output_dir)
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
