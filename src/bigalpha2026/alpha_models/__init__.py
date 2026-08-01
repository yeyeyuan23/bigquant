"""Candidate462 model-agnostic Alpha generation framework."""

from .base import AlphaModel, ModelFactory, register_model
from .data_contract import (
    ALLOWED_FACTOR_SOURCES,
    DEFAULT_SUBMISSION_DATA_CONTRACT,
    SubmissionDataContract,
)
from .feature_bundle import (
    CANDIDATE462,
    FeatureBundleConfig,
    get_feature_bundle,
)
from .microstructure import (
    BOOK_LEVELS,
    MICROSTRUCTURE_CHANNELS,
    RAW_MICROSTRUCTURE_COLUMNS,
    MicrostructureConfig,
    MicrostructureDayBatch,
    MicrostructureModel,
    MicrostructureNetwork,
    build_microstructure_features,
    pack_microstructure_days,
)
from .tabular import CandidateMLPConfig, CandidateMLPModel, CandidateMLPNetwork
from .temporal import (
    CandidateTemporalConfig,
    CandidateTemporalModel,
    CandidateTemporalNetwork,
)
from .training_data import (
    PanelArrays,
    candidate_ids_from_manifest,
    load_candidate_feature_panel,
    panel_arrays,
    rolling_oos_blocks,
)

__all__ = [
    "ALLOWED_FACTOR_SOURCES",
    "BOOK_LEVELS",
    "CANDIDATE462",
    "DEFAULT_SUBMISSION_DATA_CONTRACT",
    "MICROSTRUCTURE_CHANNELS",
    "RAW_MICROSTRUCTURE_COLUMNS",
    "AlphaModel",
    "CandidateMLPConfig",
    "CandidateMLPModel",
    "CandidateMLPNetwork",
    "CandidateTemporalConfig",
    "CandidateTemporalModel",
    "CandidateTemporalNetwork",
    "FeatureBundleConfig",
    "ModelFactory",
    "MicrostructureConfig",
    "MicrostructureDayBatch",
    "MicrostructureModel",
    "MicrostructureNetwork",
    "PanelArrays",
    "SubmissionDataContract",
    "candidate_ids_from_manifest",
    "build_microstructure_features",
    "get_feature_bundle",
    "load_candidate_feature_panel",
    "panel_arrays",
    "pack_microstructure_days",
    "register_model",
    "rolling_oos_blocks",
]
