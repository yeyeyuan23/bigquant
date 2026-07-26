"""Financial-report (FR) factor candidates."""

from .fr_001 import build_fr_001_factor, compute_fr_001_events
from .fr_002 import build_fr_002_factor, compute_fr_002_events
from .fr_003 import build_fr_003_factor, compute_fr_003_events
from .fr_004 import build_fr_004_factor, compute_fr_004_events
from .fr_005 import build_fr_005_factor, compute_fr_005_events

__all__ = [
    "build_fr_001_factor",
    "build_fr_002_factor",
    "build_fr_003_factor",
    "build_fr_004_factor",
    "build_fr_005_factor",
    "compute_fr_001_events",
    "compute_fr_002_events",
    "compute_fr_003_events",
    "compute_fr_004_events",
    "compute_fr_005_events",
]
