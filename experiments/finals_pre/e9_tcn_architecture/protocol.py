"""Frozen, independently auditable TCN experiment definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SEEDS = (20260801, 20260812, 20260823)
EPOCHS = 3
AMP_INITIAL_SCALE = 1024.0
LABEL = "ret_next_open_to_close"
BASELINE = "baseline"
METRICS = ("rank_ic", "rank_ic_ir", "long_short_sharpe", "stress_ic_ir")
BOOTSTRAP_SEED = 20260906
BOOTSTRAP_REPLICATES = 10000
BLOCK_LENGTH = 10


@dataclass(frozen=True)
class Arm:
    name: str
    group: str
    depth: int
    kernels: tuple[int, ...]

    @property
    def receptive_field(self):
        return 1 + self.depth * (max(self.kernels) - 1)

    @property
    def parameter_count(self):
        return 111681 + self.depth * (96 * sum(self.kernels) + 9216 * len(self.kernels) + 288)


ARMS = (
    Arm(BASELINE, "baseline", 3, (3, 15, 60)),
    Arm("depth1", "depth", 1, (3, 15, 60)),
    Arm("depth2", "depth", 2, (3, 15, 60)),
    Arm("depth4", "depth", 4, (3, 15, 60)),
    Arm("single15", "branches", 3, (15,)),
    Arm("single60", "branches", 3, (60,)),
    Arm("branches2", "branches", 3, (15, 60)),
    Arm("branches4", "branches", 3, (3, 9, 15, 60)),
    Arm("branches5", "branches", 3, (3, 9, 15, 30, 60)),
    Arm("kernels_2_10_45", "kernels", 3, (2, 10, 45)),
    Arm("kernels_5_30_120", "kernels", 3, (5, 30, 120)),
    Arm("kernels_5_60_120", "kernels", 3, (5, 60, 120)),
    Arm("same_scale60", "controls", 3, (60, 60, 60)),
    Arm("rf179_depth2", "controls", 2, (4, 22, 90)),
    Arm("rf177_depth4", "controls", 4, (2, 11, 45)),
)
ARM_BY_NAME = {arm.name: arm for arm in ARMS}


def make_scaler(device):
    import torch

    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=AMP_INITIAL_SCALE)


def seed_tag(seed):
    return f"seed{SEEDS.index(seed) + 1}"


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % 2**63


def config_for(arm):
    from alpha_models.microstructure import MicrostructureConfig

    return MicrostructureConfig(
        model_dim=96,
        max_minutes=240,
        kernels=arm.kernels,
        tcn_blocks=arm.depth,
        tail_minutes=30,
        dropout=0.1,
        industry_context=False,
    )


def module_identity(name, arm):
    """Match actual semantics, including repeated kernels and mixer column order."""
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "tcn" and parts[2] == "branches":
        branch = int(parts[3])
        kernel = arm.kernels[branch]
        occurrence = arm.kernels[:branch].count(kernel)
        return f"tcn.{parts[1]}.kernel{kernel}.copy{occurrence}"
    if len(parts) >= 3 and parts[0] == "tcn" and parts[2] == "mix":
        return name + ":" + ",".join(map(str, arm.kernels))
    return name


def initialize_model(arm, seed):
    import torch
    from torch import nn

    from alpha_models.microstructure import MicrostructureNetwork

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MicrostructureNetwork(config_for(arm))
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LayerNorm)):
                torch.manual_seed(stable_seed(seed, module_identity(name, arm)))
                module.reset_parameters()
    if sum(p.numel() for p in model.parameters()) != arm.parameter_count:
        raise RuntimeError("parameter count changed; review and version the experiment protocol")
    return model


def initial_hashes(model, arm):
    hashes = {}
    for name, module in model.named_modules():
        params = dict(module.named_parameters(recurse=False))
        if params:
            digest = hashlib.sha256()
            for key, value in params.items():
                array = value.detach().cpu().numpy()
                digest.update(key.encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            hashes[module_identity(name, arm)] = digest.hexdigest()
    return hashes


def training_schedule(dates, targets, training, seed, max_stocks=1200):
    """Stock sampling cannot consume either date-shuffle or model RNG streams."""
    schedule = []
    for epoch in range(EPOCHS):
        order = np.array(training, dtype=np.int64, copy=True)
        np.random.default_rng(stable_seed(seed, f"dates:{epoch}")).shuffle(order)
        for index in order:
            day = dates[int(index)]
            instruments = tuple(targets[day].dropna().sort_index().index.astype(str))
            if len(instruments) > max_stocks:
                rng = np.random.default_rng(stable_seed(seed, f"stocks:{epoch}:{day.date()}"))
                chosen = np.sort(rng.choice(len(instruments), max_stocks, replace=False))
                instruments = tuple(instruments[i] for i in chosen)
            schedule.append(
                {"epoch": epoch + 1, "date": str(day.date()), "instruments": instruments}
            )
    return schedule


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def protocol_document():
    return {
        "schema_version": 2,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "label": LABEL,
        "minutes": 240,
        "model_dim": 96,
        "max_stocks": 1200,
        "train": "2019-2023; one trading date label isolation",
        "validation": "2024; 241 dates; previously used historical validation period",
        "learning_rate": 4e-4,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "gradient_clip": 1.0,
        "amp": "cuda float16; float32 correlation loss",
        "amp_initial_scale": AMP_INITIAL_SCALE,
        "amp_growth_interval": 2000,
        "initialization": "semantic-module stable seeds; not legacy baseline initialization",
        "inference_missing": "fixed data-only mask; explicit neutral factor for missing data only",
        "bootstrap": {
            "length": BLOCK_LENGTH,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "kind": "overlapping non-circular moving blocks",
        },
        "tests": "two-sided centered bootstrap; Holm across 14 baseline contrasts; alpha=.05",
        "arms": [
            dict(asdict(a), receptive_field=a.receptive_field, parameter_count=a.parameter_count)
            for a in ARMS
        ],
    }
