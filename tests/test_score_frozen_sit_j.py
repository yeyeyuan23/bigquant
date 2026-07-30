import unittest

from scripts.score_frozen_sit_j import (
    J_INCLUDE_EXPOSURES,
    REQUIRED_J_EXPOSURE_COLUMNS,
    _continuous_input_years,
    _validate_funnel,
    _validated_prediction_years,
)


class FrozenSitJContractTest(unittest.TestCase):
    def test_official_proxy_requires_risk_exposures(self):
        self.assertTrue(J_INCLUDE_EXPOSURES)
        self.assertEqual(
            REQUIRED_J_EXPOSURE_COLUMNS,
            {"SIZE", "LIQUIDTY", "industry_level1_code"},
        )

    def test_2024_only_request_still_loads_continuous_history(self):
        prediction_years = _validated_prediction_years([2024])
        self.assertEqual(prediction_years, (2024,))
        self.assertEqual(
            _continuous_input_years(prediction_years),
            (2019, 2020, 2021, 2022, 2023, 2024),
        )

    def test_development_year_cannot_be_scored_as_J(self):
        with self.assertRaisesRegex(ValueError, "official evaluation years"):
            _validated_prediction_years([2022])

    def test_frozen_routes_must_form_a_strict_funnel(self):
        _validate_funnel(
            {
                "S": ("self__A", "self__B", "self__C"),
                "I": ("self__A", "self__B"),
                "T": ("self__A",),
            }
        )
        with self.assertRaisesRegex(ValueError, "I is not a subset"):
            _validate_funnel(
                {
                    "S": ("self__A",),
                    "I": ("self__B",),
                    "T": ("self__B",),
                }
            )
        with self.assertRaisesRegex(ValueError, "T is not a subset"):
            _validate_funnel(
                {
                    "S": ("self__A", "self__B"),
                    "I": ("self__A",),
                    "T": ("self__B",),
                }
            )


if __name__ == "__main__":
    unittest.main()
