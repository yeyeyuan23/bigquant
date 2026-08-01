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
from .microstructure_data import (
    PRICE_COLUMNS,
    VOLUME_COLUMNS,
    MicrostructureSourceConfig,
    audit_canonical_microstructure,
    canonicalize_microstructure_input,
    required_source_columns,
    validate_instrument_map,
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
    "PRICE_COLUMNS",
    "RAW_MICROSTRUCTURE_COLUMNS",
    "VOLUME_COLUMNS",
    "AlphaModel",
    "CandidateMLPConfig",
    "CandidateMLPModel",
    "CandidateMLPNetwork",
    "CandidateTemporalConfig",
    "CandidateTemporalModel",
    "CandidateTemporalNetwork",
    "FeatureBundleConfig",
    "MicrostructureConfig",
    "MicrostructureDayBatch",
    "MicrostructureModel",
    "MicrostructureNetwork",
    "MicrostructureSourceConfig",
    "ModelFactory",
    "PanelArrays",
    "SubmissionDataContract",
    "audit_canonical_microstructure",
    "build_microstructure_features",
    "candidate_ids_from_manifest",
    "canonicalize_microstructure_input",
    "get_feature_bundle",
    "load_candidate_feature_panel",
    "pack_microstructure_days",
    "panel_arrays",
    "register_model",
    "required_source_columns",
    "rolling_oos_blocks",
    "validate_instrument_map",
]
