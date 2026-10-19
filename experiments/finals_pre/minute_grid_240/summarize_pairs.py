"""Summarize predeclared paired seeds without selecting a winning initialization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRICS = ("neutral_rank_ic", "neutral_rank_ic_ir", "long_short_sharpe", "stress_ic_ir")
EXPECTED_SEEDS = (20260801, 20260812, 20260823)


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol_signature(manifest):
    return {
        key: manifest[key]
        for key in (
            "epochs",
            "train",
            "test",
            "store_manifest_sha256",
            "label_sha256",
            "exposure_sha256",
        )
    } | {
        "sources": {
            arm: {
                key: manifest["arms"][arm][key]
                for key in ("model_sha256", "trainer_sha256", "fastpack_sha256")
            }
            for arm in ("clock240", "clock242")
        }
    }


def summarize(pairs, output):
    records = {}
    common = None
    for pair in pairs:
        manifest = json.loads((pair / "pair_manifest.json").read_text())
        signature = protocol_signature(manifest)
        if common is None:
            common = signature
        elif signature != common:
            raise ValueError("cannot pool runs with different data, sources or training protocols")
        seed = manifest["seed"]
        if seed in records:
            raise ValueError("duplicate seed")
        scores = json.loads((pair / "comparison.json").read_text())["scores"]
        if len(scores) != 2 or {row["arm"] for row in scores} != {"clock240", "clock242"}:
            raise ValueError("every seed must have one 240 and one 242 arm")
        for row in scores:
            if row["seed"] != seed or row["epochs"] != 3 or row["days"] != 241:
                raise ValueError("unexpected seed, epoch count or evaluation period")
            if not all(math.isfinite(row[key]) for key in METRICS):
                raise ValueError("nonfinite metric")
        records[seed] = {row["arm"]: row for row in scores}
    if sorted(records) != list(EXPECTED_SEEDS):
        raise ValueError("summary requires exactly the three predeclared seeds")
    per_run = [records[seed][arm] for seed in EXPECTED_SEEDS for arm in ("clock240", "clock242")]
    deltas = [
        {
            "seed": seed,
            **{
                key: records[seed]["clock240"][key] - records[seed]["clock242"][key]
                for key in METRICS
            },
        }
        for seed in EXPECTED_SEEDS
    ]
    summary = []
    for metric in METRICS:
        a = [records[s]["clock240"][metric] for s in EXPECTED_SEEDS]
        b = [records[s]["clock242"][metric] for s in EXPECTED_SEEDS]
        d = [r[metric] for r in deltas]
        summary.append(
            {
                "metric": metric,
                "mean_240": statistics.mean(a),
                "std_240": statistics.stdev(a),
                "mean_242": statistics.mean(b),
                "std_242": statistics.stdev(b),
                "mean_paired_delta": statistics.mean(d),
                "std_paired_delta": statistics.stdev(d),
                "positive_pairs": sum(v > 0 for v in d),
                "pairs": len(d),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "three_seed_per_run.csv", per_run)
    save_csv(output / "three_seed_paired_deltas.csv", deltas)
    save_csv(output / "three_seed_summary.csv", summary)
    payload = {
        "seeds": list(EXPECTED_SEEDS),
        "summary": summary,
        "per_run": per_run,
        "protocol": common,
        "scope": "Three fixed seeds on the same 2024 O2C evaluation period; no significance claim.",
    }
    (output / "three_seed_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs=3, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.pairs, args.output_dir)


if __name__ == "__main__":
    main()
