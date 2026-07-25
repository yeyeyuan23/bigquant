"""Deterministic synthetic data for local tests; never uses competition data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_datasources(
    days: int = 80,
    instruments: int = 12,
    minutes_per_day: int = 12,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range("2023-01-02", periods=days)
    symbols = [f"{index:06d}.SZ" for index in range(1, instruments + 1)]
    universe = pd.MultiIndex.from_product(
        [trading_days, symbols], names=["date", "instrument"]
    ).to_frame(index=False)

    bars: list[dict[str, object]] = []
    latent = rng.normal(0.0, 0.6, size=(days, instruments))
    quality_state = np.linspace(-0.8, 0.8, instruments)
    base_prices = np.linspace(8.0, 20.0, instruments)
    previous_close = base_prices.copy()
    for day_index, day in enumerate(trading_days):
        for instrument_index, symbol in enumerate(symbols):
            pressure = latent[day_index, instrument_index]
            quality = quality_state[instrument_index]
            previous_pressure = latent[max(day_index - 1, 0), instrument_index]
            previous_confirmation = quality * max(
                np.sign(quality) * previous_pressure,
                0.0,
            )
            next_period_component = (
                0.0010 * previous_pressure + 0.0020 * previous_confirmation
                if day_index > 0
                else 0.0
            )
            open_price = previous_close[instrument_index] * (
                1.0 + rng.normal(0.0, 0.002)
            )
            minute_returns = (
                rng.normal(0.0, 0.0009, minutes_per_day)
                + 0.00025 * pressure
                + next_period_component / minutes_per_day
            )
            prices = open_price * np.cumprod(1.0 + minute_returns)
            cumulative_volume = 0.0
            cumulative_amount = 0.0
            for minute_index in range(minutes_per_day):
                timestamp = day + pd.Timedelta(hours=14, minutes=minute_index)
                price = float(prices[minute_index])
                minute_volume = float(rng.integers(1000, 8000))
                cumulative_volume += minute_volume
                cumulative_amount += minute_volume * price
                spread = max(0.01, price * 0.0008)
                bid = price - spread / 2.0
                ask = price + spread / 2.0
                row: dict[str, object] = {
                    "date": timestamp,
                    "instrument": symbol,
                    "time": 140000000 + minute_index * 100000,
                    "open": open_price if minute_index == 0 else prices[minute_index - 1],
                    "high": price * (1.0 + abs(rng.normal(0.0, 0.0005))),
                    "low": price * (1.0 - abs(rng.normal(0.0, 0.0005))),
                    "close": price,
                    "volume": cumulative_volume,
                    "amount": cumulative_amount,
                    "bid_price1": bid,
                    "ask_price1": ask,
                }
                for level in range(1, 6):
                    base_depth = 5000.0 + 500.0 * level
                    row[f"bid_volume{level}"] = max(
                        100.0,
                        base_depth * (1.0 + 0.20 * pressure + rng.normal(0.0, 0.08)),
                    )
                    row[f"ask_volume{level}"] = max(
                        100.0,
                        base_depth * (1.0 - 0.20 * pressure + rng.normal(0.0, 0.08)),
                    )
                bars.append(row)
            previous_close[instrument_index] = prices[-1]

    financial_rows: list[dict[str, object]] = []
    for instrument_index, symbol in enumerate(symbols):
        quality = quality_state[instrument_index]
        for day_index in range(0, days, 20):
            report_day = trading_days[day_index]
            assets = 1e9 * (1.0 + 0.1 * instrument_index)
            profit = 2e7 * (1.0 + quality + rng.normal(0.0, 0.05))
            cfo = profit + assets * (0.02 * quality + rng.normal(0.0, 0.002))
            revenue = 3e8 * (1.0 + 0.1 * instrument_index)
            report_period = report_day - pd.offsets.QuarterEnd()
            financial_rows.extend(
                [
                    {
                        "date": report_day,
                        "instrument": symbol,
                        "category": "ttm",
                        "shift": 0,
                        "report_date": report_period,
                        "net_cffoa": cfo,
                        "net_profit": profit,
                        "total_assets": np.nan,
                        "cash_received_from_sales_and_services": revenue
                        * (1.0 + 0.2 * quality),
                        "operating_revenue": revenue,
                        "total_operating_revenue": revenue,
                    },
                    {
                        "date": report_day,
                        "instrument": symbol,
                        "category": "lf",
                        "shift": 0,
                        "report_date": report_period,
                        "net_cffoa": np.nan,
                        "net_profit": np.nan,
                        "total_assets": assets,
                        "cash_received_from_sales_and_services": np.nan,
                        "operating_revenue": np.nan,
                        "total_operating_revenue": np.nan,
                    },
                ]
            )

    exposures = universe.copy()
    exposures["SIZE"] = np.tile(np.linspace(-1, 1, instruments), days)
    exposures["RESVOL"] = rng.normal(0, 1, len(exposures))
    exposures["LIQUIDTY"] = rng.normal(0, 1, len(exposures))
    return {
        "bigalpha_2026_instruments": universe,
        "bigalpha_2026_stock_bar1m": pd.DataFrame(bars),
        "bigalpha_2026_financial": pd.DataFrame(financial_rows),
        "bigalpha_2026_exposure": exposures,
    }
