"""Serial 45-run driver; restart completed arms safely after infrastructure interruption."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent / "_runtime"
sys.path[:0] = [
    str(ROOT),
    str(RUNTIME / "src"),
    str(RUNTIME / "scripts"),
    str(RUNTIME / "experiments/finals_pre/common"),
]

# Apply limits before importing any numerical runtime, including in children.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ[variable] = "8"

from experiments.finals_pre.tcn_architecture_o2c.prepare import prepare, verify_seal
from experiments.finals_pre.tcn_architecture_o2c.protocol import (
    ARMS,
    SEEDS,
    now,
    seed_tag,
    write_json,
)
from experiments.finals_pre.tcn_architecture_o2c.score import aggregate, score_run, verify_run


def next_attempt_path(parent, stem, suffix=""):
    index = 1
    while (parent / f"{stem}_attempt{index}{suffix}").exists():
        index += 1
    return parent / f"{stem}_attempt{index}{suffix}"


def run(args):
    with ExitStack() as resources:
        return run_locked(args, resources)


def run_locked(args, resources):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            "use a fresh output directory, or --resume for this sealed experiment"
        )
    output.mkdir(parents=True, exist_ok=True)
    lock = resources.enter_context((output / "run.lock").open("w"))
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = None
    if args.device == "cuda":
        gpu_lock_path = Path("/tmp/bigquant_tcn_gpu0.lock")
        gpu_lock = resources.enter_context(gpu_lock_path.open("w"))
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if jobs:
            raise RuntimeError(f"GPU already has compute processes: {jobs}")
    started = time.monotonic()
    status = {
        "state": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "total": len(ARMS) * len(SEEDS),
        "completed": 0,
    }
    write_json(output / "status.json", status)
    try:
        prepared = output / "prepared"
        if (prepared / "prepared.json").is_file():
            meta = verify_seal(prepared, deep=True)
            requested = (
                str(args.data_root.resolve()),
                str(args.micro_store.resolve()),
                str(args.exposure.resolve()),
            )
            if requested != (meta["data_root"], meta["micro_store"], meta["exposure"]):
                raise RuntimeError("resume data paths differ from sealed experiment")
        else:
            prepare(args.data_root, args.micro_store, args.exposure, prepared, args.device)
        if args.preflight_only:
            status.update(state="preflight_complete", finished_at=now())
            write_json(output / "status.json", status)
            return
        # Preparation used the GPU; release its cached allocator before starting a child.
        import torch

        if args.device == "cuda":
            torch.cuda.empty_cache()
        durations = []
        for seed in SEEDS:
            for arm in ARMS:
                arm_output = output / "runs" / seed_tag(seed) / arm.name
                complete = False
                if (arm_output / "manifest.json").exists():
                    _, manifest = verify_run(prepared, arm_output)
                    complete = True
                    durations.append(manifest["elapsed_seconds"])
                elif arm_output.exists() and any(arm_output.iterdir()):
                    old = json.loads((arm_output / "status.json").read_text())
                    if old["state"] == "failed":
                        raise RuntimeError(f"recorded failure requires investigation: {arm_output}")
                    if not args.resume or old["state"] != "running":
                        raise RuntimeError(f"incomplete arm cannot be resumed: {arm_output}")
                    # An interrupted arm is restarted from scratch, never from partial weights.
                    archived = next_attempt_path(
                        output / "interruptions", f"{seed_tag(seed)}_{arm.name}"
                    )
                    archived.parent.mkdir(exist_ok=True)
                    arm_output.rename(archived)
                status.update(
                    current_seed=seed,
                    current_arm=arm.name,
                    phase="scoring" if complete else "training",
                    updated_at=now(),
                )
                write_json(output / "status.json", status)
                if not complete:
                    log = next_attempt_path(output / "logs", f"{seed_tag(seed)}_{arm.name}", ".log")
                    log.parent.mkdir(exist_ok=True)
                    command = [
                        sys.executable,
                        "-u",
                        str(Path(__file__).with_name("train.py")),
                        "--prepared",
                        str(prepared),
                        "--output",
                        str(arm_output),
                        "--seed",
                        str(seed),
                        "--arm",
                        arm.name,
                        "--device",
                        args.device,
                    ]
                    print(f"Starting seed={seed} arm={arm.name} log={log}", flush=True)
                    with log.open("w") as handle:
                        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
                    _, manifest = verify_run(prepared, arm_output)
                    durations.append(manifest["elapsed_seconds"])
                metrics = score_run(prepared, arm_output)
                status["completed"] += 1
                status.update(
                    last_metrics=metrics,
                    updated_at=now(),
                    estimated_remaining_hours=(45 - status["completed"])
                    * sum(durations)
                    / len(durations)
                    / 3600,
                )
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
            # Timing/correctness checkpoint only. No candidate is selected out by performance.
            write_json(
                output / f"{seed_tag(seed)}_complete.json",
                {
                    "seed": seed,
                    "at": now(),
                    "arms_completed": [a.name for a in ARMS],
                    "candidate_elimination": False,
                },
            )
        verify_seal(prepared, deep=True)
        aggregate(output)
        status.update(
            state="complete", finished_at=now(), elapsed_seconds=time.monotonic() - started
        )
        write_json(output / "status.json", status)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
        write_json(output / "status.json", status)
        raise
    finally:
        lock.close()
        if gpu_lock is not None:
            gpu_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
