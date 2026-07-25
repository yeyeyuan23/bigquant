#!/usr/bin/env python3
"""Run both factors and the disclosed metrics on deterministic synthetic data."""

from __future__ import annotations

import json

from bigalpha2026.evaluation import (
    build_return_labels,
    daily_prices_from_bar,
    evaluate_single_factor,
)
from bigalpha2026.hf_pressure import main as hf_main
from bigalpha2026.quality_interaction import main as interaction_main
from bigalpha2026.synthetic import make_synthetic_datasources


def main() -> None:
    datasources = make_synthetic_datasources()
    universe = datasources["bigalpha_2026_instruments"]
    start = universe["date"].min()
    end = universe["date"].max()
    labels = build_return_labels(
        daily_prices_from_bar(datasources["bigalpha_2026_stock_bar1m"])
    )
    exposures = datasources["bigalpha_2026_exposure"]
    factors = {
        "hf_pressure": hf_main(datasources, start, end),
        "quality_interaction": interaction_main(datasources, start, end),
    }
    report = {
        name: {
            "rows": len(frame),
            "coverage": float(frame["factor"].notna().mean()),
            "metrics": evaluate_single_factor(frame, labels, exposures),
        }
        for name, frame in factors.items()
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

