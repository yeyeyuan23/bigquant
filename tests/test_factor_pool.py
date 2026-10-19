import unittest

import numpy as np
import pandas as pd

from evaluation import ElasticNetConfig
from factor_pool import (
    PUBLIC_PREFIX,
    SELF_PREFIX,
    apply_feature_directions,
    build_feature_panel,
    family_balanced_factor,
    screen_public_factors,
    validate_candidate_pool,
)
from factorlib import FACTORLIB_FEATURE_COLUMNS


class FactorPoolTest(unittest.TestCase):
    def test_left_join_keeps_universe_and_neutral_fills_missing_features(self):
        dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
        instruments = ["A", "B", "C"]
        universe = pd.DataFrame(
            [
                {"date": date, "instrument": instrument}
                for date in dates
                for instrument in instruments
            ]
        )
        factorlib_rows = []
        for row_index, row in universe.iloc[:-1].reset_index(drop=True).iterrows():
            factorlib_rows.append(
                {
                    "date": row["date"],
                    "instrument": row["instrument"],
                    **{
                        column: float(row_index + feature_index)
                        for feature_index, column in enumerate(
                            FACTORLIB_FEATURE_COLUMNS
                        )
                    },
                }
            )
        candidate_pool = universe.iloc[:3].copy()
        candidate_pool["candidate_id"] = "HF-003"
        candidate_pool["factor_version"] = "v1"
        candidate_pool["factor"] = [1.0, 2.0, 3.0]

        panel, public_columns, self_columns, coverage = build_feature_panel(
            universe,
            pd.DataFrame(
                factorlib_rows,
                columns=("date", "instrument", *FACTORLIB_FEATURE_COLUMNS),
            ),
            candidate_pool[
                [
                    "date",
                    "instrument",
                    "candidate_id",
                    "factor_version",
                    "factor",
                ]
            ],
            admitted_candidates=("HF-003",),
        )

        self.assertEqual(len(panel), len(universe))
        self.assertEqual(len(public_columns), 36)
        self.assertEqual(self_columns, (f"{SELF_PREFIX}HF-003",))
        missing_row = panel.loc[
            panel["date"].eq(pd.Timestamp("2022-01-05"))
            & panel["instrument"].eq("C")
        ].iloc[0]
        self.assertTrue((missing_row[list(public_columns)] == 0).all())
        self.assertEqual(missing_row[self_columns[0]], 0)
        self.assertLess(float(coverage["coverage"].min()), 1.0)
        with self.assertRaisesRegex(ValueError, "no admitted"):
            build_feature_panel(
                universe,
                pd.DataFrame(
                    factorlib_rows,
                    columns=("date", "instrument", *FACTORLIB_FEATURE_COLUMNS),
                ),
                candidate_pool[
                    [
                        "date",
                        "instrument",
                        "candidate_id",
                        "factor_version",
                        "factor",
                    ]
                ],
                admitted_candidates=("FR-999",),
            )

        factor = family_balanced_factor(
            panel,
            (public_columns[0], self_columns[0]),
        )
        self.assertEqual(list(factor.columns), ["date", "instrument", "factor"])
        self.assertTrue(factor["factor"].between(-1, 1).all())

    def test_public_screening_uses_development_signal_and_freezes_direction(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2019-01-02", periods=80)
        rows = []
        labels = []
        for date in dates:
            for stock in range(20):
                signal = rng.normal()
                noise = rng.normal()
                rows.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        f"{PUBLIC_PREFIX}signal": signal,
                        f"{PUBLIC_PREFIX}noise": noise,
                    }
                )
                labels.append(
                    {
                        "date": date,
                        "instrument": f"S{stock:03d}",
                        "ret_close_to_close": signal + rng.normal(scale=0.1),
                    }
                )
        panel = pd.DataFrame(rows)
        screening = screen_public_factors(
            panel,
            pd.DataFrame(labels),
            (f"{PUBLIC_PREFIX}signal", f"{PUBLIC_PREFIX}noise"),
            development_years=(2019,),
            elastic_net_config=ElasticNetConfig(window_days=40, step_days=20),
        )
        signal = screening.loc[
            screening["feature"].eq(f"{PUBLIC_PREFIX}signal")
        ].iloc[0]
        self.assertTrue(signal["selected"])
        self.assertEqual(signal["direction"], 1.0)
        oriented = apply_feature_directions(panel, screening)
        self.assertEqual(len(oriented), len(panel))

    def test_candidate_pool_contract(self):
        pool = pd.DataFrame(
            {
                "date": pd.to_datetime(["2022-01-04"]),
                "instrument": ["A"],
                "candidate_id": ["FR-002"],
                "factor_version": ["v1"],
                "factor": [0.2],
            }
        )
        validate_candidate_pool(pool)


if __name__ == "__main__":
    unittest.main()
