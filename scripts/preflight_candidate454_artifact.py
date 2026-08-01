"""Fail fast unless the upstream candidate artifact is complete for 2019-2024."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=454)
    parser.add_argument("--required-start", default="2019-01-02")
    parser.add_argument("--required-end", default="2024-12-31")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidate_rows = payload.get("candidate_rows")
    if not isinstance(candidate_rows, dict):
        raise TypeError("candidate manifest has no candidate_rows mapping")
    candidate_ids = sorted(str(value) for value in candidate_rows)
    errors: list[str] = []
    if not args.pool.exists():
        errors.append(f"candidate pool does not exist: {args.pool}")
    if len(candidate_ids) != args.expected_count:
        errors.append(f"candidate_count={len(candidate_ids)}, expected={args.expected_count}")
    empty_ids = sorted(
        candidate_id
        for candidate_id, rows in candidate_rows.items()
        if not isinstance(rows, int) or rows <= 0
    )
    if empty_ids:
        errors.append(f"non-positive candidate rows: {empty_ids[:10]}")
    if payload.get("duplicate_keys") not in (None, 0):
        errors.append(f"duplicate_keys={payload['duplicate_keys']}")
    date_range = payload.get("date_range")
    if not isinstance(date_range, list) or len(date_range) != 2:
        errors.append(f"invalid date_range={date_range!r}")
    else:
        start, end = map(pd.Timestamp, date_range)
        if start > pd.Timestamp(args.required_start):
            errors.append(f"date_start={start.date()} > {args.required_start}")
        if end < pd.Timestamp(args.required_end):
            errors.append(f"date_end={end.date()} < {args.required_end}")

    result = {
        "ready": not errors,
        "pool": str(args.pool),
        "manifest": str(args.manifest),
        "candidate_count": len(candidate_ids),
        "expected_candidate_count": args.expected_count,
        "date_range": date_range,
        "required_date_range": [args.required_start, args.required_end],
        "errors": errors,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(
            "candidate454 artifact is incomplete; formal unified training was not started"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
