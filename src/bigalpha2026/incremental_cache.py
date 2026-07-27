"""Content-addressed cache for Elastic Net incremental summaries."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from .tree_cache import content_digest

INCREMENTAL_CACHE_SCHEMA_VERSION = "elastic-net-incremental-cache-v3-J"


class IncrementalSummaryCache:
    """Persist an evaluation summary only under its exact input digest."""

    def __init__(self, cache_dir: Path, *, refresh: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.refresh = bool(refresh)
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        payload: Mapping[str, object],
        compute: Callable[[], Mapping[str, object]],
    ) -> tuple[dict[str, object], bool, str]:
        cache_payload = {
            "schema_version": INCREMENTAL_CACHE_SCHEMA_VERSION,
            **dict(payload),
        }
        key = content_digest(cache_payload)
        path = self.cache_dir / f"{key}.json"
        if not self.refresh and path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("payload") == cache_payload:
                summary = artifact.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError("incremental cache summary must be an object")
                self.hits += 1
                return summary, True, key

        summary = dict(compute())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".json.partial")
        partial.write_text(
            json.dumps(
                {"payload": cache_payload, "summary": summary},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        partial.replace(path)
        self.misses += 1
        return summary, False, key
