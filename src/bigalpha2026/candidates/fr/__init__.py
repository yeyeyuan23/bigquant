"""Financial-report (FR) factor candidates."""

from .fr_001 import build_fr_001_factor, compute_fr_001_events
from .fr_002 import build_fr_002_factor, compute_fr_002_events

__all__ = [
    "build_fr_001_factor",
    "build_fr_002_factor",
    "compute_fr_001_events",
    "compute_fr_002_events",
]
