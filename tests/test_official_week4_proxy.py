import unittest
from datetime import date, timedelta

import numpy as np
import polars as pl

from bigalpha2026.official_week4_proxy import (
    barra_style_exposure_profile,
    index_enhancement_metrics,
    industry_rank_ic_profile,
    neutralize_factor_against_styles_and_industry,
    paired_index_enhancement_delta,
    prepare_available_barra_styles,
)


class OfficialWeek4ProxyTest(unittest.TestCase):
    def test_prepares_available_size_and_liquidity_proxies(self):
        exposures = pl.DataFrame(
            {
                "date": [date(2024, 1, 2)],
                "instrument": ["A"],
                "float_market_cap": [100.0],
                "turn": [0.5],
            }
        ).with_columns(pl.col("date").cast(pl.Datetime))
        prepared = prepare_available_barra_styles(exposures)
        self.assertIn("SIZE", prepared.columns)
        self.assertIn("LIQUIDTY", prepared.columns)
        self.assertAlmostEqual(prepared["SIZE"][0], np.log(100.0))
        self.assertAlmostEqual(prepared["LIQUIDTY"][0], np.log1p(0.5))

    def test_barra_profile_recovers_multivariate_coefficients(self):
        dates = []
        instruments = []
        size = []
        liquidity = []
        factor_values = []
        for day in range(3):
            for instrument_index in range(40):
                dates.append(f"2024-01-{day + 2:02d}")
                instruments.append(f"S{instrument_index:03d}")
                size_value = instrument_index - 20.0
                liquidity_value = (instrument_index % 7) - 3.0
                size.append(size_value)
                liquidity.append(liquidity_value)
                factor_values.append(2.0 * size_value - liquidity_value)
        factor = pl.DataFrame(
            {
                "date": dates,
                "instrument": instruments,
                "factor": factor_values,
            }
        ).with_columns(pl.col("date").str.to_datetime())
        exposures = pl.DataFrame(
            {
                "date": dates,
                "instrument": instruments,
                "SIZE": size,
                "LIQUIDTY": liquidity,
            }
        ).with_columns(pl.col("date").str.to_datetime())
        result = barra_style_exposure_profile(
            factor,
            exposures,
            standardize_daily=False,
            minimum_rows=30,
        )
        summary = {row["style"]: row for row in result.summary.to_dicts()}
        self.assertAlmostEqual(summary["SIZE"]["mean_coefficient"], 2.0, places=8)
        self.assertAlmostEqual(
            summary["LIQUIDTY"]["mean_coefficient"],
            -1.0,
            places=8,
        )
        self.assertIn("BETA", result.missing_styles)

    def test_industry_rank_ic_reports_cross_industry_consistency(self):
        dates = []
        instruments = []
        industries = []
        factors = []
        labels = []
        for day in range(4):
            for industry in ("A", "B"):
                for rank in range(8):
                    dates.append(f"2024-02-{day + 1:02d}")
                    instruments.append(f"{industry}{rank}")
                    industries.append(industry)
                    factors.append(float(rank))
                    labels.append(float(rank + day * 0.01))
        keys = {
            "date": dates,
            "instrument": instruments,
        }
        factor = pl.DataFrame({**keys, "factor": factors}).with_columns(
            pl.col("date").str.to_datetime()
        )
        label = pl.DataFrame({**keys, "ret_close_to_close": labels}).with_columns(
            pl.col("date").str.to_datetime()
        )
        exposure = pl.DataFrame(
            {**keys, "industry_level1_code": industries}
        ).with_columns(pl.col("date").str.to_datetime())
        result = industry_rank_ic_profile(factor, label, exposure)
        summary = result.summary.row(0, named=True)
        self.assertEqual(summary["industry_coverage"], 2)
        self.assertAlmostEqual(summary["industry_ic_same_sign_ratio"], 1.0)
        self.assertAlmostEqual(summary["industry_ic_worst_oriented"], 1.0)

    def test_neutralization_removes_industry_and_style_components(self):
        dates = []
        instruments = []
        industries = []
        sizes = []
        factors = []
        for day in range(3):
            for industry_index, industry in enumerate(("A", "B")):
                for rank in range(20):
                    dates.append(f"2024-02-{day + 1:02d}")
                    instruments.append(f"{industry}{rank:02d}")
                    industries.append(industry)
                    size = float(rank - 9.5)
                    signal = float(((rank * 7) % 11) - 5)
                    sizes.append(size)
                    factors.append(
                        4.0 * industry_index
                        + 2.5 * size
                        + signal
                    )
        keys = {"date": dates, "instrument": instruments}
        factor = pl.DataFrame(
            {**keys, "factor": factors}
        ).with_columns(pl.col("date").str.to_datetime())
        exposures = pl.DataFrame(
            {
                **keys,
                "industry_level1_code": industries,
                "SIZE": sizes,
            }
        ).with_columns(
            pl.col("date").str.to_datetime().cast(pl.Datetime("ns"))
        )

        neutralized = neutralize_factor_against_styles_and_industry(
            factor,
            exposures,
        ).join(
            exposures,
            on=["date", "instrument"],
            how="inner",
            validate="1:1",
        )
        daily_industry_means = neutralized.group_by(
            ["date", "industry_level1_code"]
        ).agg(
            pl.col("factor").mean().abs().alias("absolute_mean")
        )
        daily_style_correlation = neutralized.group_by(
            "date"
        ).agg(
            pl.corr("factor", "SIZE")
            .abs()
            .alias("absolute_correlation")
        )
        self.assertLess(
            float(daily_industry_means["absolute_mean"].max()),
            1e-10,
        )
        self.assertLess(
            float(
                daily_style_correlation[
                    "absolute_correlation"
                ].max()
            ),
            1e-10,
        )

    def test_index_enhancement_delta_uses_identical_keys(self):
        dates = []
        instruments = []
        baseline_values = []
        augmented_values = []
        returns = []
        market_cap = []
        for day in range(80):
            date_value = date(2024, 3, 1) + timedelta(days=day)
            for rank in range(20):
                dates.append(date_value)
                instruments.append(f"S{rank:03d}")
                centered = rank - 9.5
                baseline_values.append(-centered)
                augmented_values.append(centered)
                returns.append((0.001 + 0.0003 * np.sin(day)) * centered)
                market_cap.append(float(rank + 1))
        keys = {"date": dates, "instrument": instruments}
        baseline = pl.DataFrame({**keys, "factor": baseline_values})
        augmented = pl.DataFrame({**keys, "factor": augmented_values})
        labels = pl.DataFrame({**keys, "ret_close_to_close": returns})
        exposures = pl.DataFrame({**keys, "float_market_cap": market_cap})
        result = paired_index_enhancement_delta(
            baseline,
            augmented,
            labels,
            exposures=exposures,
        )
        summary = result.row(0, named=True)
        self.assertEqual(summary["benchmark_source"], "market_cap_universe_proxy")
        self.assertGreater(summary["delta_information_ratio"], 0)
        self.assertGreater(summary["augmented_annualized_excess_return"], 0)

        single = index_enhancement_metrics(
            augmented,
            labels,
            exposures=exposures,
        )
        self.assertEqual(single.summary["date_count"][0], 80)

        with self.assertRaisesRegex(ValueError, "identical keys"):
            paired_index_enhancement_delta(
                baseline.slice(1),
                augmented,
                labels,
                exposures=exposures,
            )


if __name__ == "__main__":
    unittest.main()
