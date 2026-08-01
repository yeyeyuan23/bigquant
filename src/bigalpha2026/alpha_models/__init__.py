"""Unified, model-agnostic Alpha generation framework."""

from .bar_features import BAR1M_BASE_COLUMNS, ROLLING_WINDOWS, build_bar1m_all156
from .base import AlphaModel, ModelFactory, register_model
from .data_contract import (
    ALLOWED_FACTOR_SOURCES,
    DEFAULT_SUBMISSION_DATA_CONTRACT,
    SubmissionDataContract,
)
from .feature_bundle import (
    ALL618,
    BAR156,
    CANDIDATE462,
    FeatureBundleConfig,
    get_feature_bundle,
    merge_feature_bundle,
)
from .tabular import All618MLPConfig, All618MLPModel, All618MLPNetwork
from .temporal import (
    All156TemporalConfig,
    All156TemporalModel,
    All156TemporalNetwork,
    CandidateFactorTower,
)
from .training_data import (
    PanelArrays,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
)

__all__ = [
    "ALL618",
    "ALLOWED_FACTOR_SOURCES",
    "BAR1M_BASE_COLUMNS",
    "BAR156",
    "CANDIDATE462",
    "DEFAULT_SUBMISSION_DATA_CONTRACT",
    "ROLLING_WINDOWS",
    "All156TemporalConfig",
    "All156TemporalModel",
    "All156TemporalNetwork",
    "All618MLPConfig",
    "All618MLPModel",
    "All618MLPNetwork",
    "AlphaModel",
    "CandidateFactorTower",
    "FeatureBundleConfig",
    "ModelFactory",
    "PanelArrays",
    "SubmissionDataContract",
    "build_bar1m_all156",
    "candidate_ids_from_manifest",
    "get_feature_bundle",
    "load_candidate_feature_panel",
    "merge_feature_bundle",
    "panel_arrays",
    "register_model",
]
