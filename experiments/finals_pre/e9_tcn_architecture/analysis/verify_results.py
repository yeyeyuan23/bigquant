"""Verify published E9 sources and daily statistics without private data or a GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.finals_pre.tcn_architecture_o2c.protocol import ARMS, SEEDS, file_hash
from experiments.finals_pre.tcn_architecture_o2c.score import holm_adjust, paired_bootstrap


def verify():
    output = ROOT / "reports/dependencies/finals_pre/tcn_architecture_o2c"
    meta = json.loads((output / "prepared/prepared.json").read_text())
    audit = json.loads((output / "results/audit.json").read_text())
    for name, expected in meta["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Sealed source mismatch: {name}")
    if file_hash(output / "prepared/prepared.json") != audit["prepared_sha256"]:
        raise ValueError("Prepared manifest mismatch")
    for name, expected in audit["files"].items():
        if file_hash(output / "results" / name) != expected:
            raise ValueError(f"Result hash mismatch: {name}")
    completion = json.loads((output / "audits/completion_audit.json").read_text())
    if file_hash(ROOT / completion["verification_source"]) != completion["verification_source_sha256"]:
        raise ValueError("Historical completion audit source changed")
    cube = np.empty((15, 3, 241))
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            run = output / "runs" / f"seed{si+1}" / arm.name
            metrics = json.loads((run / "metrics.json").read_text())
            if file_hash(run / "daily_metrics.csv") != metrics["daily_metrics_sha256"]:
                raise ValueError(f"Daily hash mismatch: {arm.name}/{seed}")
            daily = pd.read_csv(run / "daily_metrics.csv")
            if daily.date.tolist() != meta["scoring_dates"]:
                raise ValueError("Scoring date mismatch")
            if metrics["scoring_keys_hash"] != meta["scoring_keys_hash"]:
                raise ValueError("Scoring key mismatch")
            if (metrics["arm"], metrics["seed"]) != (arm.name, seed):
                raise ValueError("Run identity mismatch")
            cube[ai, si] = daily.rank_ic.to_numpy()
            np.testing.assert_allclose(daily.rank_ic.mean(), metrics["rank_ic"], atol=1e-12, rtol=0)
    with np.load(output / "results/daily_rankic_cube.npz") as saved:
        np.testing.assert_allclose(cube, saved["rank_ic"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(saved["arms"], [arm.name for arm in ARMS])
        np.testing.assert_array_equal(saved["seeds"], SEEDS)
        np.testing.assert_array_equal(saved["dates"], meta["scoring_dates"])
    point, low, high, raw = paired_bootstrap(cube[1:] - cube[0:1])
    adjusted = holm_adjust(raw)
    summary = json.loads((output / "results/summary.json").read_text())
    if [row["arm"] for row in summary] != [arm.name for arm in ARMS]:
        raise ValueError("Summary arm order mismatch")
    for ai, row in enumerate(summary):
        np.testing.assert_allclose(row["rank_ic"], cube[ai].mean(), atol=1e-12, rtol=0)
        if ai:
            actual = [point[ai-1], low[ai-1], high[ai-1], raw[ai-1], adjusted[ai-1]]
            expected = [row[k] for k in ("delta_rank_ic", "ci_low", "ci_high", "p_raw", "p_holm")]
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    print("PASS: 20 sealed sources, 45 daily series, result hashes, bootstrap and Holm.")
    print("Private minute data, checkpoints and full factors require the separate AutoDL audit.")


if __name__ == "__main__":
    verify()
