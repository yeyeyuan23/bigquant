"""Retrospective turnover policies; accounting reuses the audited execute function."""

import numpy as np
import pandas as pd
from raw_engine import Costs, Panel, execute, targets


def policy_targets(scores, current, rule):
    eligible = np.flatnonzero(np.isfinite(scores))
    if len(eligible) < 10:
        raise ValueError("Too few eligible scores")
    order = eligible[np.argsort(scores[eligible], kind="stable")][::-1]
    count = len(order) // 10
    band = int(np.ceil(len(order) * rule["buffer_fraction"]))
    weights = targets(scores, current, groups=10)
    if rule["buffer_fraction"] > 0.10:
        weights[:] = 0
        for ranked, sign in ((order, 1), (order[::-1], -1)):
            keep = [int(i) for i in ranked[:band] if sign * current[i] > 1e-12][:count]
            held = set(keep)
            fill = [int(i) for i in ranked if int(i) not in held][: count - len(keep)]
            weights[keep + fill] = sign / count
    alpha = rule["adjustment"]
    if alpha < 1 and (current > 1e-12).any() and (current < -1e-12).any():
        old = current.copy()
        old[~np.isfinite(scores)] = 0
        for sign in (1, -1):
            mask = old * sign > 0
            denom = np.abs(old[mask]).sum()
            if denom:
                old[mask] /= denom
        mix = (1 - alpha) * old + alpha * weights
        if (mix > 1e-12).any() and (mix < -1e-12).any():
            for sign in (1, -1):
                mask = mix * sign > 0
                mix[mask] /= np.abs(mix[mask]).sum()
            weights = mix
    assert not np.any(weights[~np.isfinite(scores)])
    np.testing.assert_allclose(weights[weights > 0].sum(), 1, atol=1e-12)
    np.testing.assert_allclose(weights[weights < 0].sum(), -1, atol=1e-12)
    return weights


def run_rule(data: Panel, rule: dict, costs: Costs, capture: bool = False):
    strategy = rule["name"]
    signals = data.scores
    long_only = False
    qty = np.zeros(len(data.instruments))
    cash, previous_nav = 1.0, 1.0
    records, positions = [], {}
    for t, date in enumerate(data.dates):
        old_qty = qty.copy()
        opening_nav = float(cash + np.dot(qty, data.opens[t]))
        trades = np.zeros_like(qty)
        fee, blocked = 0.0, 0
        terminal = t == len(data.dates) - 1
        trade_day = (
            t == 1 or terminal or (t > 1 and (t - 1 - rule["phase"]) % rule["rebalance_every"] == 0)
        )
        if trade_day:
            weights = (
                np.zeros_like(qty)
                if terminal
                else policy_targets(signals[t - 1], qty * data.opens[t] / opening_nav, rule)
            )
            blocked = int(
                (
                    (np.abs(weights * opening_nav - qty * data.opens[t]) > 1e-10)
                    & ~data.tradable[t]
                ).sum()
            )
            qty, cash, trades, fee = execute(
                qty, cash, data.opens[t], data.tradable[t], weights, costs, long_only=long_only
            )
        closing_nav = float(cash + np.dot(qty, data.closes[t]))
        overnight = float(np.dot(old_qty, data.opens[t] - data.closes[t - 1])) if t else 0.0
        intraday = float(np.dot(qty, data.closes[t] - data.opens[t]))
        np.testing.assert_allclose(
            closing_nav - previous_nav, overnight + intraday - fee, atol=1e-10, rtol=0
        )
        buys, sells = float(np.maximum(trades, 0).sum()), float(np.maximum(-trades, 0).sum())
        records.append(
            {
                "strategy": strategy,
                "scenario": costs.scenario,
                "date": str(date.date()),
                "signal_date": str(data.dates[t - 1].date()) if t > 0 and not terminal else "",
                "rank_window_start": str(data.dates[max(0, t - 5)].date())
                if strategy == "rank_mean_5d" and t > 0 and not terminal
                else "",
                "previous_close_nav": previous_nav,
                "opening_nav": opening_nav,
                "closing_nav": closing_nav,
                "cash": cash,
                "position_value": float(np.dot(qty, data.closes[t])),
                "long_value": float(np.maximum(qty * data.closes[t], 0).sum()),
                "short_value": float(np.maximum(-qty * data.closes[t], 0).sum()),
                "daily_return": closing_nav / previous_nav - 1,
                "overnight_pnl": overnight,
                "intraday_pnl": intraday,
                "buy_notional": buys,
                "sell_notional": sells,
                "cost_amount": fee,
                "one_way_turnover": (buys + sells) / (2 * opening_nav),
                "blocked_trade_names": blocked,
                "missing_close_held_names": int(
                    ((np.abs(qty) > 1e-12) & data.missing_closes[t]).sum()
                ),
                "terminal_residual_value": float(np.abs(qty * data.closes[t]).sum())
                if terminal
                else 0.0,
            }
        )
        if capture:
            positions[str(date.date())] = qty.copy()
        previous_nav = closing_nav
    return pd.DataFrame(records), positions
