"""Daily self-financing accounts with raw scores and next-open execution.

The signal loader deliberately accepts no risk exposures or future-return labels.
The accounting follows the existing E7 execution/fee assumptions, with returns
and turnover assigned to their actual calendar trading dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {
    "baseline_daily": "每日五分位",
    "buffer_20_30": "排名缓冲区",
    "rank_mean_5d": "五日平均排名",
}


@dataclass(frozen=True)
class Costs:
    scenario: str
    buy_bps: float
    sell_bps: float
    slippage_bps: float = 0.0

    def charge(self, trades: np.ndarray) -> float:
        buys = np.maximum(trades, 0).sum()
        sells = np.maximum(-trades, 0).sum()
        return float((buys * self.buy_bps + sells * self.sell_bps
                      + (buys + sells) * self.slippage_bps) / 10000)


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    instruments: np.ndarray
    scores: np.ndarray
    opens: np.ndarray
    closes: np.ndarray
    tradable: np.ndarray
    missing_closes: np.ndarray


def load_panel(raw_scores: Path, daily_prices: Path, start: str, end: str) -> Panel:
    scores = pd.read_parquet(raw_scores)[["date", "instrument", "factor"]].copy()
    prices = pd.read_parquet(daily_prices).copy()
    for frame in (scores, prices):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError("Duplicate date/instrument input keys")
    prices = prices[prices.date.between(start, end)]
    scores = scores[scores.date.between(start, end)]
    dates = pd.DatetimeIndex(sorted(prices.date.unique()))
    instruments = np.array(sorted(set(prices.instrument) | set(scores.instrument)))
    if len(dates) < 3 or set(scores.date) - set(dates):
        raise ValueError("Invalid signal/price calendar")

    def pivot(frame, column):
        return frame.pivot(index="date", columns="instrument", values=column).reindex(
            index=dates, columns=instruments
        )

    raw = pivot(scores, "factor").to_numpy(dtype=float)
    if np.isinf(raw).any():
        raise ValueError("Infinite model scores")
    op, cl = pivot(prices, "adjusted_open"), pivot(prices, "adjusted_close")
    for values in (op, cl):
        if (values.where(np.isfinite(values)) <= 0).any().any():
            raise ValueError("Nonpositive finite price")
    tradable = np.isfinite(op.to_numpy()) & (op.to_numpy() > 0)
    marks = op.where(tradable, cl.shift(1)).ffill().fillna(1.0).to_numpy()
    missing_closes = ~np.isfinite(cl.to_numpy())
    closing = np.where(missing_closes, marks, cl.to_numpy())
    return Panel(dates, instruments, raw, marks, closing, tradable, missing_closes)


def smooth_ranks(scores: np.ndarray, window: int = 5) -> np.ndarray:
    ranks = pd.DataFrame(scores).rank(axis=1, method="average", pct=True)
    result = ranks.rolling(window, min_periods=1).mean().to_numpy(copy=True)
    result[~np.isfinite(scores)] = np.nan
    return result


def targets(scores: np.ndarray, current: np.ndarray, buffer: bool = False) -> np.ndarray:
    eligible = np.flatnonzero(np.isfinite(scores))
    if len(eligible) < 10:
        raise ValueError("Too few signal-date eligible stocks")
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
    count = len(ranked) // 5
    long_names, short_names = ranked[:count], ranked[-count:]
    if buffer:
        band = int(np.ceil(len(ranked) * 0.30))

        def select(order, held):
            retained = [int(i) for i in order[:band] if held[i]][:count]
            chosen = set(retained)
            fills = [int(i) for i in order if int(i) not in chosen]
            return np.array(retained + fills[:count - len(retained)], dtype=int)

        long_names = select(ranked, current > 1e-12)
        short_names = select(ranked[::-1], current < -1e-12)
    result = np.zeros(len(scores), dtype=float)
    result[long_names], result[short_names] = 1.0 / count, -1.0 / count
    return result


def execute(qty, cash, prices, tradable, weights, costs):
    before = qty * prices
    nav = float(cash + before.sum())
    if nav <= 0:
        raise ValueError("Portfolio insolvent")
    post = nav
    for _ in range(100):
        desired = weights * post
        desired[~tradable] = before[~tradable]
        updated = nav - costs.charge(desired - before)
        if abs(updated - post) < 1e-13 * max(nav, 1):
            post = updated
            break
        post = updated
    else:
        raise RuntimeError("Post-fee target sizing did not converge")
    desired = weights * post
    desired[~tradable] = before[~tradable]
    trades = desired - before
    fee = costs.charge(trades)
    new_qty = desired / prices
    new_cash = float(cash - trades.sum() - fee)
    np.testing.assert_allclose(new_cash + desired.sum(), nav - fee, atol=1e-11, rtol=0)
    return new_qty, new_cash, trades, fee


def run(data: Panel, strategy: str, costs: Costs, capture: bool = False):
    if strategy not in NAMES:
        raise ValueError(strategy)
    signals = smooth_ranks(data.scores) if strategy == "rank_mean_5d" else data.scores
    qty = np.zeros(len(data.instruments))
    cash, previous_nav = 1.0, 1.0
    records, positions = [], {}
    for t, date in enumerate(data.dates):
        old_qty = qty.copy()
        opening_nav = float(cash + np.dot(qty, data.opens[t]))
        trades = np.zeros_like(qty)
        fee, blocked = 0.0, 0
        terminal = t == len(data.dates) - 1
        if t > 0:
            weights = (np.zeros_like(qty) if terminal else targets(
                signals[t - 1], qty * data.opens[t] / opening_nav,
                buffer=strategy == "buffer_20_30",
            ))
            blocked = int(((np.abs(weights * opening_nav - qty * data.opens[t]) > 1e-10)
                           & ~data.tradable[t]).sum())
            qty, cash, trades, fee = execute(
                qty, cash, data.opens[t], data.tradable[t], weights, costs
            )
        closing_nav = float(cash + np.dot(qty, data.closes[t]))
        overnight = float(np.dot(old_qty, data.opens[t] - data.closes[t - 1])) if t else 0.0
        intraday = float(np.dot(qty, data.closes[t] - data.opens[t]))
        np.testing.assert_allclose(
            closing_nav - previous_nav, overnight + intraday - fee, atol=1e-10, rtol=0
        )
        buys, sells = float(np.maximum(trades, 0).sum()), float(np.maximum(-trades, 0).sum())
        records.append({
            "strategy": strategy, "scenario": costs.scenario, "date": str(date.date()),
            "signal_date": str(data.dates[t - 1].date()) if t > 0 and not terminal else "",
            "rank_window_start": str(data.dates[max(0, t - 5)].date())
            if strategy == "rank_mean_5d" and t > 0 and not terminal else "",
            "previous_close_nav": previous_nav, "opening_nav": opening_nav,
            "closing_nav": closing_nav, "cash": cash,
            "position_value": float(np.dot(qty, data.closes[t])),
            "long_value": float(np.maximum(qty * data.closes[t], 0).sum()),
            "short_value": float(np.maximum(-qty * data.closes[t], 0).sum()),
            "daily_return": closing_nav / previous_nav - 1,
            "overnight_pnl": overnight, "intraday_pnl": intraday,
            "buy_notional": buys, "sell_notional": sells, "cost_amount": fee,
            "one_way_turnover": (buys + sells) / (2 * opening_nav),
            "blocked_trade_names": blocked,
            "missing_close_held_names": int(((np.abs(qty) > 1e-12)
                                               & data.missing_closes[t]).sum()),
            "terminal_residual_value": float(np.abs(qty * data.closes[t]).sum())
            if terminal else 0.0,
        })
        if capture:
            positions[str(date.date())] = qty.copy()
        previous_nav = closing_nav
    return pd.DataFrame(records), positions


def summarize(daily: pd.DataFrame) -> dict:
    returns = daily.daily_return.to_numpy()
    nav = np.r_[1.0, daily.closing_nav.to_numpy()]
    sd = returns.std(ddof=1)
    return {
        "strategy": daily.strategy.iloc[0], "scenario": daily.scenario.iloc[0],
        "start": daily.date.iloc[0], "end": daily.date.iloc[-1], "days": len(daily),
        "average_one_way_turnover": float(daily.one_way_turnover.mean()),
        "annualized_return": float(returns.mean() * 252),
        "sharpe": float(returns.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "cagr": float(nav[-1] ** (252 / len(daily)) - 1),
        "cumulative_return": float(nav[-1] - 1),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "total_fees": float(daily.cost_amount.sum()),
        "terminal_residual_value": float(daily.terminal_residual_value.iloc[-1]),
    }
