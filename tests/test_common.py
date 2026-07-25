from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from bigalpha2026.common import (
    attach_pit_quality,
    cumulative_to_increment,
    validate_factor_output,
)


class CommonTests(unittest.TestCase):
    def test_cumulative_fields_reset_each_day_and_after_vendor_reset(self) -> None:
        values = pd.Series([10.0, 25.0, 4.0, 9.0, 3.0])
        instruments = pd.Series(["A"] * 5)
        days = pd.to_datetime(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
        )
        increments = cumulative_to_increment(values, instruments, pd.Series(days))
        np.testing.assert_allclose(increments.to_numpy(), [10.0, 15.0, 4.0, 5.0, 3.0])

    def test_pit_financial_record_is_not_visible_before_disclosure(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "instrument": ["A", "A", "A"],
            }
        )
        financial = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-03"]),
                "instrument": ["A"],
                "net_cffoa": [120.0],
                "net_profit": [100.0],
                "total_assets": [1000.0],
                "cash_received_from_sales_and_services": [210.0],
                "operating_revenue": [200.0],
            }
        )
        attached = attach_pit_quality(panel, financial)
        before = attached.loc[attached["date"].eq(pd.Timestamp("2024-01-02"))]
        on_date = attached.loc[attached["date"].eq(pd.Timestamp("2024-01-03"))]
        self.assertTrue(before["quality_accrual"].isna().all())
        self.assertAlmostEqual(float(on_date["quality_accrual"].iloc[0]), 0.02)
        self.assertEqual(float(on_date["report_age"].iloc[0]), 0.0)

    def test_official_category_uses_ttm_flows_and_lf_assets(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-30"]),
                "instrument": ["A"],
            }
        )
        financial = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-30", "2024-04-30", "2024-04-30"]),
                "report_date": pd.to_datetime(["2024-03-31"] * 3),
                "instrument": ["A", "A", "A"],
                "category": ["ttm", "lf", "mrq"],
                "shift": [0, 0, 0],
                "net_cffoa": [140.0, np.nan, 9999.0],
                "net_profit": [100.0, np.nan, 9999.0],
                "total_assets": [np.nan, 2000.0, 1.0],
                "cash_received_from_sales_and_services": [220.0, np.nan, 9999.0],
                "operating_revenue": [200.0, np.nan, 1.0],
            }
        )
        attached = attach_pit_quality(panel, financial)
        self.assertAlmostEqual(float(attached["quality_accrual"].iloc[0]), 0.02)
        self.assertAlmostEqual(float(attached["quality_cash"].iloc[0]), 1.1)

    def test_output_schema_rejects_duplicates(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "instrument": ["A", "A"],
                "factor": [0.1, 0.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_factor_output(frame)


if __name__ == "__main__":
    unittest.main()
