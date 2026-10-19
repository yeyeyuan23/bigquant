from datetime import date

import pytest

from alpha_models.data_contract import SubmissionDataContract


def test_contract_accepts_only_bar1m_and_financial_factor_sources() -> None:
    contract = SubmissionDataContract()
    contract.validate_factor_manifest(
        {
            "intraday_reversal": {"bar1m"},
            "cashflow_price_interaction": {"bar1m", "financial"},
        }
    )


@pytest.mark.parametrize("forbidden", ["factorlib", "exposure", "industry", "basic"])
def test_contract_rejects_non_whitelisted_factor_sources(forbidden: str) -> None:
    contract = SubmissionDataContract()
    with pytest.raises(ValueError, match="source whitelist violation"):
        contract.validate_factor_manifest({"bad_feature": {"bar1m", forbidden}})


def test_contract_allows_full_public_training_history_and_rejects_future() -> None:
    contract = SubmissionDataContract()
    prediction_start = date(2026, 7, 1)
    contract.validate_observation_dates(
        [date(2019, 1, 1), date(2024, 12, 31), prediction_start], prediction_start
    )
    with pytest.raises(ValueError, match="future"):
        contract.validate_observation_dates([date(2026, 7, 2)], prediction_start)


def test_contract_locks_final_training_dates_to_2019_2024() -> None:
    contract = SubmissionDataContract()
    contract.validate_training_dates([date(2019, 1, 1), date(2024, 12, 31)])
    with pytest.raises(ValueError, match="2019-2024"):
        contract.validate_training_dates([date(2025, 1, 1)])
