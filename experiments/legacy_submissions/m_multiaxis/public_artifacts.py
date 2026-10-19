"""Load explicitly supplied trusted artifacts for source-only submission examples.

No model, fallback weights, training data, download, or import-time file access.
Keep private artifacts outside the repository. See ARTIFACT_CONTRACT.md.
"""
from collections.abc import Mapping
import json
import os
from pathlib import Path


def artifact_path(variable, *, explicit_path=None):
    """Resolve a user-supplied file path, failing clearly when it is absent."""
    value = explicit_path if explicit_path is not None else os.environ.get(variable)
    if value is None or not os.fspath(value).strip():
        raise RuntimeError(
            f"Set {variable} to an external trusted artifact file. "
            "This source-only release does not include trained artifacts."
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{variable} must identify an existing artifact file: {path}")
    return path.resolve()


def _validate_payload(payload, required_keys):
    if not isinstance(payload, Mapping):
        raise ValueError("External artifact must contain a mapping")
    missing = set(required_keys).difference(payload)
    if missing:
        raise ValueError(f"External artifact is missing required keys: {sorted(missing)}")
    return payload


def load_json_artifact(variable, *, required_keys=(), explicit_path=None):
    """Read a JSON mapping from an explicit local path, never a bundled fallback."""
    path = artifact_path(variable, explicit_path=explicit_path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return _validate_payload(payload, required_keys)


def load_torch_checkpoint(torch, variable, *, required_keys=("config", "state_dict"), explicit_path=None):
    """Require weights-only loading; never retry with unrestricted pickle."""
    path = artifact_path(variable, explicit_path=explicit_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "A PyTorch version supporting weights_only=True is required; "
            "unrestricted pickle loading is disabled."
        ) from error
    return _validate_payload(payload, required_keys)
