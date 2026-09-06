"""Compare the real inputs, training calendar and initial weights across code snapshots."""
# Select the requested snapshot before importing any model or loader modules.

import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--seed", type=int, default=20260801)
args = parser.parse_args()
root = args.root.resolve()
sys.path[:0] = [
    str(root / "src"),
    str(root / "scripts"),
    str(root / "experiments/finals_pre/common"),
]
import numpy as np
import torch
from evaluate_unified_microstructure import apply_training_history, prepare_label_panel
from evaluate_unified_temporal import load_labels
from fastpack import load_microstructure_day_fast

from alpha_models import rolling_oos_blocks
from alpha_models.microstructure import MicrostructureConfig, MicrostructureNetwork

config = MicrostructureConfig()
torch.manual_seed(args.seed)
model = MicrostructureNetwork(config)
weight_hash = hashlib.sha256()
for name, value in model.state_dict().items():
    weight_hash.update(name.encode())
    weight_hash.update(value.numpy().tobytes())
labels = load_labels(Path("/root/autodl-tmp/data"), 2019, 2024)
dates, targets = prepare_label_panel(labels)
blocks = apply_training_history(
    rolling_oos_blocks(dates, (2024,), train_days=60, prediction_days=999), mode="expanding"
)
assert len(blocks) == 1
train, test = blocks[0]
assert len(test) == 241
rng = np.random.default_rng(args.seed)
order_hash = hashlib.sha256()
for _ in range(3):
    shuffled = train.copy()
    rng.shuffle(shuffled)
    order_hash.update(shuffled.tobytes())
    # Mirror stock subsampling RNG too, even if this universe stays below the cap.
    for index in shuffled:
        count = len(targets[dates[int(index)]].dropna())
        if count > 1200:
            order_hash.update(np.sort(rng.choice(count, 1200, replace=False)).tobytes())
checks = []
for day in (dates[train[0]], dates[train[-1]], dates[test[0]]):
    instruments = tuple(targets[day].index.astype(str))
    batch = load_microstructure_day_fast(
        Path("/root/autodl-tmp/unified_microstructure_store_v2_2019_2024"),
        day,
        instruments,
        max_minutes=config.max_minutes,
    )
    positions = np.arange(240) if config.max_minutes == 240 else np.r_[1:121, 122:242]
    fields = {}
    for field in ("values", "observed_mask", "minute_mask"):
        canonical = np.take(getattr(batch, field), positions, axis=2)
        fields[field] = hashlib.sha256(canonical.tobytes()).hexdigest()
    checks.append({"date": str(day.date()), "stocks": len(instruments), "hashes": fields})
print(
    json.dumps(
        {
            "seed": args.seed,
            "minutes": config.max_minutes,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "initial_weights_sha256": weight_hash.hexdigest(),
            "training_order_sha256": order_hash.hexdigest(),
            "train_days": len(train),
            "test_days": len(test),
            "train_start": str(dates[train[0]].date()),
            "train_end": str(dates[train[-1]].date()),
            "sample_inputs": checks,
        },
        indent=2,
    )
)
