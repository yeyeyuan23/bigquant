from __future__ import annotations

"""Execute the production O2C label builder on boundary-date fixtures."""

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def label_builder(load_experiment_module):
    pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    return load_experiment_module("common/build_private_o2c_labels.py", "pre_private_o2c_labels")


def market_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-12"])
    instruments = ("000001.SZ", "600000.SH")
    daily_rows = []
    pool_rows = []
    for day_index, day in enumerate(dates):
        for stock_index, instrument in enumerate(instruments):
            daily_rows.append(
                {
                    "trade_date": day,
                    "instrument": instrument,
                    "daily_o2c": 0.01 * (day_index + 1) + 0.001 * stock_index,
                }
            )
            pool_rows.append({"date": day, "instrument": instrument})
    return pd.DataFrame(pool_rows), pd.DataFrame(daily_rows)


def test_friday_uses_monday_and_skips_non_trading_days(label_builder):
    pool, daily = market_fixture()
    labels = label_builder.build_next_day_o2c_labels(pool, daily)
    friday = labels[labels["date"] == pd.Timestamp("2024-06-07")]
    assert friday["label_date"].eq(pd.Timestamp("2024-06-10")).all()
    assert friday["ret_next_open_to_close"].tolist() == pytest.approx([0.02, 0.021])
    label_builder.validate_next_day_o2c_labels(labels, pool, daily)


def test_future_prices_cannot_change_an_existing_next_day_label(label_builder):
    pool, daily = market_fixture()
    before = label_builder.build_next_day_o2c_labels(pool, daily)
    changed = daily.copy()
    changed.loc[changed["trade_date"] == pd.Timestamp("2024-06-12"), "daily_o2c"] += 100.0
    after = label_builder.build_next_day_o2c_labels(pool, changed)
    key = before["date"] == pd.Timestamp("2024-06-07")
    pd.testing.assert_frame_equal(
        before.loc[key].reset_index(drop=True), after.loc[key].reset_index(drop=True)
    )


def test_same_day_or_two_day_ahead_labels_are_rejected(label_builder):
    pool, daily = market_fixture()
    correct = label_builder.build_next_day_o2c_labels(pool, daily)

    same_day = correct.copy()
    same_day["label_date"] = same_day["date"]
    same_day = same_day.drop(columns="ret_next_open_to_close").merge(
        daily.rename(
            columns={
                "trade_date": "label_date",
                "daily_o2c": "ret_next_open_to_close",
            }
        ),
        on=["label_date", "instrument"],
        validate="many_to_one",
    )
    with pytest.raises(RuntimeError, match="exact next market day"):
        label_builder.validate_next_day_o2c_labels(same_day, pool, daily)

    two_days_ahead = correct.copy()
    friday = two_days_ahead["date"] == pd.Timestamp("2024-06-07")
    two_days_ahead.loc[friday, "label_date"] = pd.Timestamp("2024-06-12")
    two_days_ahead.loc[friday, "ret_next_open_to_close"] = [0.03, 0.031]
    with pytest.raises(RuntimeError, match="exact next market day"):
        label_builder.validate_next_day_o2c_labels(two_days_ahead, pool, daily)
