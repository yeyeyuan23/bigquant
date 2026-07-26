"""Registered factor candidates grouped by their four base data families."""

from .pv import (
    aggregate_minute_daily,
    build_pv_001_factor,
    build_pv_002_factor,
    compute_pv_001_features,
    compute_pv_002_daily,
)
from .hf import (
    build_hf_001_factor,
    build_hf_002_factor,
    compute_hf_001_daily,
    compute_hf_002_daily,
)
from .ob import (
    build_ob_001_factor,
    build_ob_002_factor,
    compute_ob_001_daily,
    compute_ob_002_daily,
)
from .fr import (
    build_fr_001_factor,
    build_fr_002_factor,
    compute_fr_001_events,
    compute_fr_002_events,
)

__all__ = [
    "aggregate_minute_daily",
    "build_hf_001_factor",
    "build_hf_002_factor",
    "build_fr_001_factor",
    "build_fr_002_factor",
    "build_ob_001_factor",
    "build_ob_002_factor",
    "build_pv_001_factor",
    "build_pv_002_factor",
    "compute_hf_001_daily",
    "compute_hf_002_daily",
    "compute_fr_001_events",
    "compute_fr_002_events",
    "compute_ob_001_daily",
    "compute_ob_002_daily",
    "compute_pv_001_features",
    "compute_pv_002_daily",
]
