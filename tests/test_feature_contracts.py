import unittest

import pandas as pd

from bigalpha2026.feature_contracts import (
    FEATURE_CONTRACTS,
    get_feature_contract,
    validate_feature_columns,
    validate_feature_frame,
)


class FeatureContractsTest(unittest.TestCase):
    def test_four_base_families_have_fixed_frequency_and_keys(self):
        self.assertEqual(set(FEATURE_CONTRACTS), {"PV", "HF", "OB", "FR"})
        self.assertEqual(get_feature_contract("pv").source_frequency, "1d")
        self.assertEqual(get_feature_contract("HF").panel_frequency, "1d")
        self.assertEqual(get_feature_contract("OB").panel_frequency, "1d")
        self.assertEqual(get_feature_contract("FR").panel_frequency, "event")
        self.assertEqual(get_feature_contract("PV").key, ("date", "instrument"))

    def test_contracts_do_not_persist_labels_or_candidate_outputs(self):
        prohibited = ("factor", "label", "target", "next_", "forward_")
        for contract in FEATURE_CONTRACTS.values():
            for column in contract.columns:
                self.assertFalse(
                    any(token in column.lower() for token in prohibited),
                    msg=f"{contract.family} contains prohibited column {column}",
                )

    def test_strict_columns_reject_candidate_specific_or_label_columns(self):
        columns = list(get_feature_contract("PV").columns)
        validate_feature_columns(columns, "PV")
        with self.assertRaisesRegex(ValueError, "extra=.*pv001_raw"):
            validate_feature_columns(columns + ["pv001_raw"], "PV")
        with self.assertRaisesRegex(ValueError, "extra=.*next_return"):
            validate_feature_columns(columns + ["next_return"], "PV")

    def test_frame_validation_rejects_duplicate_storage_keys(self):
        contract = get_feature_contract("PV")
        row = {
            column: 1.0
            for column in contract.columns
            if column not in {"date", "instrument"}
        }
        frame = pd.DataFrame(
            [
                {"date": "2022-01-04", "instrument": "000001.SZ", **row},
                {"date": "2022-01-04", "instrument": "000001.SZ", **row},
            ],
            columns=contract.columns,
        )
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            validate_feature_frame(frame, "PV")


if __name__ == "__main__":
    unittest.main()
