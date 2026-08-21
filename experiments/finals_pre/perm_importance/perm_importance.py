"""Channel permutation importance on the frozen e3 checkpoint.

For each channel (and each of the four channel groups), shuffle that channel's
values ACROSS STOCKS within the day at inference time and measure how much the
daily RankIC drops versus the untouched baseline. No retraining: the shuffle
keeps the marginal distribution realistic while severing the channel-to-stock
link, so the IC drop is attributable to the destroyed information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluate_unified_temporal import load_labels
from evaluate_unified_microstructure import load_microstructure_day, prepare_label_panel
from bigalpha2026.alpha_models import MICROSTRUCTURE_CHANNELS
from bigalpha2026.alpha_models.microstructure import MicrostructureModel

GROUPS = {
    "group_price(3)": [0, 1, 2],
    "group_book(5)": [3, 4, 5, 6, 7],
    "group_trade(6)": [8, 9, 10, 11, 12, 13],
    "group_clock(3)": [14, 15, 16],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--micro-store", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--day-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicrostructureModel.load(args.checkpoint, map_location=device)
    network = model.network.to(device).eval()

    labels = load_labels(args.data_root, args.year, args.year)
    dates, targets = prepare_label_panel(labels)
    year_dates = [d for d in dates if d.year == args.year][:: args.day_stride]
    print(f"evaluating {len(year_dates)} days (stride {args.day_stride})", flush=True)

    configs: dict[str, list[int] | None] = {"baseline": None}
    for index, channel in enumerate(MICROSTRUCTURE_CHANNELS):
        configs[channel] = [index]
    configs.update(GROUPS)
    sums = {name: [] for name in configs}
    rng = np.random.default_rng(args.seed)

    with torch.inference_mode():
        for day in year_dates:
            day_target = targets[day].dropna()
            instruments = tuple(day_target.index.astype(str))
            batch = load_microstructure_day(
                args.micro_store, day, instruments, max_minutes=242
            )
            if batch is None:
                continue
            available = np.flatnonzero(batch.stock_mask[0])
            if len(available) < 100:
                continue
            values = torch.from_numpy(batch.values[:, available]).to(device)
            observed = torch.from_numpy(batch.observed_mask[:, available]).to(device)
            minutes = torch.from_numpy(batch.minute_mask[:, available]).to(device)
            stocks = torch.from_numpy(batch.stock_mask[:, available]).to(device)
            target = pd.Series(day_target.to_numpy(np.float32)[available])
            permutation = torch.from_numpy(rng.permutation(len(available))).to(device)
            for name, channel_indices in configs.items():
                if channel_indices is None:
                    v, o = values, observed
                else:
                    v = values.clone()
                    o = observed.clone()
                    idx = torch.tensor(channel_indices, device=device)
                    v[0][:, :, idx] = values[0][permutation][:, :, idx]
                    o[0][:, :, idx] = observed[0][permutation][:, :, idx]
                scores = network(v, o, minutes, stocks).squeeze(0).float().cpu().numpy()
                ic = pd.Series(scores).rank().corr(target.rank())
                sums[name].append(float(ic))
            print(f"{day.date()} done", flush=True)

    baseline = float(np.mean(sums["baseline"]))
    rows = []
    for name, series in sums.items():
        mean_ic = float(np.mean(series))
        rows.append(
            {
                "config": name,
                "ic_mean": mean_ic,
                "delta_vs_baseline": mean_ic - baseline,
                "drop_pct": (baseline - mean_ic) / abs(baseline) * 100 if name != "baseline" else 0.0,
            }
        )
    table = pd.DataFrame(rows).sort_values("delta_vs_baseline")
    table.to_csv(args.output_dir / "perm_importance.csv", index=False)
    (args.output_dir / "perm_summary.json").write_text(
        json.dumps(
            {"baseline_ic": baseline, "days": len(sums["baseline"]), "rows": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
