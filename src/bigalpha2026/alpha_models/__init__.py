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
)

__all__ = [
    "ALLOWED_FACTOR_SOURCES",
    "CANDIDATE462",
    "DEFAULT_SUBMISSION_DATA_CONTRACT",
    "AlphaModel",
    "CandidateMLPConfig",
    "CandidateMLPModel",
    "CandidateMLPNetwork",
    "CandidateTemporalConfig",
    "CandidateTemporalModel",
    "CandidateTemporalNetwork",
    "FeatureBundleConfig",
    "ModelFactory",
    "PanelArrays",
    "SubmissionDataContract",
    "candidate_ids_from_manifest",
    "get_feature_bundle",
    "load_candidate_feature_panel",
    "panel_arrays",
    "register_model",
]
