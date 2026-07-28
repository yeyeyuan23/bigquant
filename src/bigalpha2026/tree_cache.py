"""Content-addressed cache for strict walk-forward tree predictions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TREE_CACHE_SCHEMA_VERSION = "tree-prediction-cache-v6-orthogonal-entry"
PREDICTION_COLUMNS = ("date", "instrument", "factor")
LIGHTGBM_IMPORTANCE_COLUMNS = (
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "feature",
    "split_importance",
    "gain_importance",
)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_digest(payload: Mapping[str, object]) -> str:
    """Return the stable SHA-256 digest of a JSON-compatible payload."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def frame_column_fingerprint(
    frame: pd.DataFrame,
    column: str,
) -> str:
    """Fingerprint one keyed feature from its actual values, including nulls."""

    required = {"date", "instrument", column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"feature fingerprint is missing columns: {missing}")
    keyed = frame.loc[:, ["date", "instrument", column]].copy()
    keyed["date"] = pd.to_datetime(keyed["date"], errors="coerce").dt.normalize()
    keyed["instrument"] = keyed["instrument"].astype(str)
    if keyed[["date", "instrument"]].isna().any().any():
        raise ValueError("feature fingerprint contains invalid keys")
    if keyed.duplicated(["date", "instrument"]).any():
        raise ValueError("feature fingerprint contains duplicate keys")
    keyed = keyed.sort_values(["date", "instrument"]).reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(
        keyed,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(str(len(keyed)).encode("ascii"))
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def feature_fingerprints(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, str]:
    """Fingerprint every model feature independently."""

    return {
        column: frame_column_fingerprint(frame, column)
        for column in columns
    }


class TreePredictionCache:
    """Persist predictions only when every model input has the same digest."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        feature_fingerprints: Mapping[str, str],
        label_fingerprint: str,
        model_config: Mapping[str, object],
        refresh: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.feature_fingerprints = dict(feature_fingerprints)
        self.label_fingerprint = str(label_fingerprint)
        self.model_config = dict(model_config)
        self.refresh = bool(refresh)
        self.hits = 0
        self.misses = 0

    def payload(
        self,
        feature_columns: Sequence[str],
        *,
        prediction_years: Sequence[int],
        label_column: str,
        train_window_days: int,
        test_window_days: int,
    ) -> dict[str, object]:
        missing = sorted(
            set(feature_columns).difference(self.feature_fingerprints)
        )
        if missing:
            raise ValueError(f"tree cache lacks feature fingerprints: {missing}")
        return {
            "schema_version": TREE_CACHE_SCHEMA_VERSION,
            "feature_columns": list(feature_columns),
            "feature_fingerprints": {
                feature: self.feature_fingerprints[feature]
                for feature in feature_columns
            },
            "label_column": label_column,
            "label_fingerprint": self.label_fingerprint,
            "prediction_years": [int(year) for year in prediction_years],
            "train_window_days": int(train_window_days),
            "test_window_days": int(test_window_days),
            "model_config": self.model_config,
        }

    def pool_state_digest(
        self,
        candidate_columns: Sequence[str],
    ) -> str:
        """Digest current candidate membership and contents.

        Adding, removing, renaming, or changing any candidate changes this
        digest even when reusable public-baseline predictions remain valid.
        """

        return content_digest(
            {
                "schema_version": TREE_CACHE_SCHEMA_VERSION,
                "candidate_columns": list(candidate_columns),
                "candidate_fingerprints": {
                    feature: self.feature_fingerprints[feature]
                    for feature in candidate_columns
                },
            }
        )

    def get_or_compute(
        self,
        feature_columns: Sequence[str],
        *,
        prediction_years: Sequence[int],
        label_column: str,
        train_window_days: int,
        test_window_days: int,
        force_refresh: bool = False,
        compute: Callable[[], pd.DataFrame],
    ) -> tuple[pd.DataFrame, bool, str]:
        """Load one exact prediction or atomically materialize it."""

        payload = self.payload(
            feature_columns,
            prediction_years=prediction_years,
            label_column=label_column,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
        )
        key = content_digest(payload)
        parquet_path = self.cache_dir / f"{key}.parquet"
        metadata_path = self.cache_dir / f"{key}.json"
        if (
            not self.refresh
            and not force_refresh
            and parquet_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == payload:
                cached = pd.read_parquet(parquet_path)
                self._validate_prediction(cached)
                self.hits += 1
                return cached, True, key

        prediction = compute()
        self._validate_prediction(prediction)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_partial = parquet_path.with_suffix(".parquet.partial")
        metadata_partial = metadata_path.with_suffix(".json.partial")
        prediction.to_parquet(parquet_partial, index=False)
        metadata_partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        parquet_partial.replace(parquet_path)
        metadata_partial.replace(metadata_path)
        self.misses += 1
        return prediction, False, key

    def get_or_compute_with_importance(
        self,
        feature_columns: Sequence[str],
        *,
        prediction_years: Sequence[int],
        label_column: str,
        train_window_days: int,
        test_window_days: int,
        force_refresh: bool = False,
        compute: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
        """Load or materialize predictions plus LightGBM importance rows."""

        payload = self.payload(
            feature_columns,
            prediction_years=prediction_years,
            label_column=label_column,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
        )
        key = content_digest(payload)
        parquet_path = self.cache_dir / f"{key}.parquet"
        importance_path = self.cache_dir / f"{key}.importance.parquet"
        metadata_path = self.cache_dir / f"{key}.json"
        if (
            not self.refresh
            and not force_refresh
            and parquet_path.exists()
            and importance_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == payload:
                cached = pd.read_parquet(parquet_path)
                importance = pd.read_parquet(importance_path)
                self._validate_prediction(cached)
                self._validate_importance(importance)
                self.hits += 1
                return cached, importance, True, key

        prediction, importance = compute()
        self._validate_prediction(prediction)
        self._validate_importance(importance)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_partial = parquet_path.with_suffix(".parquet.partial")
        importance_partial = importance_path.with_suffix(".parquet.partial")
        metadata_partial = metadata_path.with_suffix(".json.partial")
        prediction.to_parquet(parquet_partial, index=False)
        importance.to_parquet(importance_partial, index=False)
        metadata_partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        parquet_partial.replace(parquet_path)
        importance_partial.replace(importance_path)
        metadata_partial.replace(metadata_path)
        self.misses += 1
        return prediction, importance, False, key

    @staticmethod
    def _validate_prediction(frame: pd.DataFrame) -> None:
        if list(frame.columns) != list(PREDICTION_COLUMNS):
            raise ValueError(
                "cached tree prediction columns do not match contract; "
                f"expected={list(PREDICTION_COLUMNS)}, actual={list(frame.columns)}"
            )
        if frame.empty:
            raise ValueError("cached tree prediction is empty")
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError("cached tree prediction contains duplicate keys")
        factor = pd.to_numeric(frame["factor"], errors="coerce")
        if factor.isna().any() or not np.isfinite(factor).all():
            raise ValueError("cached tree prediction contains invalid factor values")

    @staticmethod
    def _validate_importance(frame: pd.DataFrame) -> None:
        missing = sorted(set(LIGHTGBM_IMPORTANCE_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"cached tree importance columns missing: {missing}")
        split = pd.to_numeric(frame["split_importance"], errors="coerce")
        gain = pd.to_numeric(frame["gain_importance"], errors="coerce")
        if split.isna().any() or gain.isna().any():
            raise ValueError("cached tree importance contains invalid values")
        if not np.isfinite(split).all() or not np.isfinite(gain).all():
            raise ValueError("cached tree importance contains non-finite values")
