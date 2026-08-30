"""Smoke-test the AutoDL E17 Parquet loader and model forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--day", default="2023-01-03")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.root))
    from e17_train_autodl import load_contract, load_day, tensor_inputs
    from model_raw23 import ProgressiveConfig, ProgressiveModel

    channels, max_minutes = load_contract(args.store)
    batch = load_day(
        pd.Timestamp(args.day),
        store=args.store,
        channels=channels,
        max_minutes=max_minutes,
    )
    if batch is None:
        raise RuntimeError("smoke-test day is missing")
    selection = np.flatnonzero(batch[4])[:64]
    if len(selection) != 64:
        raise RuntimeError(f"expected at least 64 usable stocks, found {len(selection)}")

    device = torch.device(args.device)
    outputs = {}
    for arm, keep in (("baseline17", tuple(range(17))), ("raw40", ())):
        config = ProgressiveConfig(
            input_dim=40,
            keep_channels=keep,
            model_dim=96,
            max_minutes=max_minutes,
            kernels=(3, 15, 60),
            tcn_blocks=3,
            tail_minutes=30,
            dropout=0.1,
        )
        model = ProgressiveModel(**asdict(config)).network.to(device).eval()
        with torch.inference_mode():
            prediction = model(
                *tensor_inputs(batch, selection, device=device)
            ).squeeze(0)
        outputs[arm] = {
            "shape": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all().item()),
        }

    result = {
        "day": args.day,
        "rows": len(batch[0]),
        "values_shape": list(batch[1].shape),
        "values_dtype": str(batch[1].dtype),
        "channels": len(channels),
        "max_minutes": max_minutes,
        "device": str(device),
        "outputs": outputs,
    }
    print(json.dumps(result, indent=2))
    if not all(item["finite"] for item in outputs.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
