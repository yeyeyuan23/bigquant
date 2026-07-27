from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.candidates.fr.fr_001 import (
    build_fr_001_factor_from_panel,
    compute_fr_001_events,
    compute_fr_001_events_from_panel,
)
from bigalpha2026.candidates.fr.fr_002 import build_fr_002_factor_from_panel
from bigalpha2026.candidates.hf.hf_001 import build_hf_001_factor_from_daily
from bigalpha2026.candidates.hf.hf_002 import build_hf_002_factor_from_daily
from bigalpha2026.candidates.ob.ob_001 import build_ob_001_factor_from_daily
from bigalpha2026.candidates.ob.ob_002 import build_ob_002_factor_from_daily


class DailyPanelAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"] * 3),
                "instrument": ["A", "B", "C"],
            }
        )

    def test_hf_daily_adapters(self) -> None:
        daily = self.pool.assign(
            shock_q90_active_count=[4, 5, 2],
            shock_q90_recovery_5m_median=[-0.2, 0.3, 9.0],
            avg_trade_value=[1.0, 2.0, np.nan],
            avg_trade_volume=[3.0, 2.0, 1.0],
            directional_efficiency=[0.1, 0.2, 0.3],
            tail_trade_value_ratio=[1.1, 1.0, 0.9],
        )
        for result in (
            build_hf_001_factor_from_daily(daily, self.pool),
            build_hf_002_factor_from_daily(daily, self.pool),
        ):
            self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
            self.assertTrue(np.isfinite(result["factor"]).all())

    def test_ob_daily_adapters(self) -> None:
        daily = self.pool.assign(
            tail_60_relative_spread_median=[0.1, 0.2, np.nan],
            tail_60_depth_completeness_median=[0.8, 0.9, 1.0],
            tail_60_bid_depth_imbalance_median=[-0.1, 0.0, 0.1],
            negative_mid_shock_q10_bid_depth_recovery_5m_median=[0.0, 0.1, 0.2],
            full_day_depth_shape_median=[-0.2, 0.0, 0.2],
            tail_60_depth_shape_median=[-0.3, 0.0, 0.3],
            tail_60_shape_sign_consistency=[-0.5, 0.0, 0.5],
        )
        for result in (
            build_ob_001_factor_from_daily(daily, self.pool),
            build_ob_002_factor_from_daily(daily, self.pool),
        ):
            self.assertEqual(list(result.columns), ["date", "instrument", "factor"])
            self.assertTrue(np.isfinite(result["factor"]).all())

    def test_fr_contract_disclosure_date_is_accepted(self) -> None:
        financial = pd.DataFrame(
            {
                "disclosure_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-02-02", "2024-02-02"]
                ),
                "instrument": ["A"] * 4,
                "report_date": pd.to_datetime(
                    ["2023-12-31", "2023-12-31", "2024-01-31", "2024-01-31"]
                ),
                "category": ["ttm", "lf", "ttm", "lf"],
                "shift": [0] * 4,
                "net_cffoa": [10.0, np.nan, 12.0, np.nan],
                "net_profit": [8.0, np.nan, 9.0, np.nan],
                "total_assets": [np.nan, 100.0, np.nan, 100.0],
            }
        )
        events = compute_fr_001_events(
            financial,
            pd.bdate_range("2024-01-02", "2024-02-09"),
        )
        self.assertFalse(events.empty)

    def test_fr_panel_adapters_preserve_effective_date(self) -> None:
        financial = pd.DataFrame(
            {
                "disclosure_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-05", "2024-02-10", "2024-02-10"]
                ),
                "effective_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-08", "2024-02-19", "2024-02-19"]
                ),
                "instrument": ["A"] * 4,
                "report_date": pd.to_datetime(
                    ["2023-12-31", "2023-12-31", "2024-01-31", "2024-01-31"]
                ),
                "category": ["ttm", "lf", "ttm", "lf"],
                "shift": [0] * 4,
                "net_cffoa": [10.0, np.nan, 12.0, np.nan],
                "net_profit": [8.0, np.nan, 9.0, np.nan],
                "operating_revenue": [50.0, np.nan, 55.0, np.nan],
                "total_assets": [np.nan, 100.0, np.nan, 100.0],
            }
        )
        events = compute_fr_001_events_from_panel(financial)
        self.assertTrue(
            events.loc[
                events["disclosure_date"].eq(pd.Timestamp("2024-02-10")),
                "effective_date",
            ].eq(pd.Timestamp("2024-02-19")).all()
        )
        pool = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-02-16", "2024-02-19"]),
                "instrument": ["A", "A"],
            }
        )
        for result in (
            build_fr_001_factor_from_panel(financial, pool),
            build_fr_002_factor_from_panel(financial, pool),
        ):
            self.assertEqual(len(result), 2)
            self.assertTrue(np.isfinite(result["factor"]).all())


if __name__ == "__main__":
    unittest.main()
