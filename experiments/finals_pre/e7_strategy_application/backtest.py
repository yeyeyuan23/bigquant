"""Fixed-signal, self-financing open-to-open strategy comparison.

All portfolios use the same idealized opening-price execution assumption. Missing
opens block trades. Borrow availability, borrow fees and market impact are not
estimated, so results are research scenarios, not executable P&L promises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ("baseline_daily", "buffer_20_30", "rebalance_3d", "partial_50")
NAMES = dict(zip(STRATEGIES, ("每日五分位", "排名缓冲区", "每三日调仓", "每日调整50%")))


@dataclass(frozen=True)
class Costs:
    scenario: str
    buy_bps: float
    sell_bps: float
    slippage_bps: float = 0.0

    def __post_init__(self):
        if min(self.buy_bps, self.sell_bps, self.slippage_bps) < 0:
            raise ValueError("Costs must be nonnegative")

    def charge(self, trades: np.ndarray) -> float:
        buys = float(np.maximum(trades, 0).sum())
        sells = float(np.maximum(-trades, 0).sum())
        return (
            buys * self.buy_bps + sells * self.sell_bps + (buys + sells) * self.slippage_bps
        ) / 10000


SCENARIOS = (
    Costs("gross", 0, 0),
    Costs("fees_slip0", 3, 8, 0),
    Costs("fees_slip2", 3, 8, 2),
    Costs("fees_slip5", 3, 8, 5),
)


@dataclass
class Inputs:
    dates: pd.DatetimeIndex
    instruments: np.ndarray
    scores: np.ndarray
    prices: np.ndarray
    tradable: np.ndarray


def load_inputs(root: Path) -> Inputs:
    signals = pd.read_parquet(root / "signals.parquet")
    prices = pd.read_parquet(root / "daily_prices.parquet")
    for frame in (signals, prices):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["instrument"] = frame["instrument"].astype(str)
        if frame.duplicated(["date", "instrument"]).any():
            raise ValueError("Duplicate input keys")
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    instruments = np.array(sorted(set(prices["instrument"]) | set(signals["instrument"])))

    def panel(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        return frame.pivot(index="date", columns="instrument", values=column).reindex(
            index=dates, columns=instruments
        )

    opens = panel(prices, "adjusted_open")
    closes = panel(prices, "adjusted_close")
    tradable = np.isfinite(opens.to_numpy()) & (opens.to_numpy() > 0)
    # At a missing opening quote, yesterday's close is already known. Persist
    # that mark until a new observation; never backfill using a future price.
    marks = opens.where(tradable, closes.shift(1)).ffill().fillna(1.0)
    return Inputs(
        dates, instruments, panel(signals, "factor").to_numpy(), marks.to_numpy(), tradable
    )


def normalize_sides(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    pos, neg = result > 0, result < 0
    if pos.any():
        result[pos] /= result[pos].sum()
    if neg.any():
        result[neg] /= -result[neg].sum()
    return result


def targets(scores: np.ndarray, current: np.ndarray, strategy: str) -> np.ndarray:
    eligible = np.flatnonzero(np.isfinite(scores))
    if len(eligible) < 10:
        raise ValueError("Too few signal-date eligible stocks")
    # Input instruments are sorted. Stable sorting breaks ties by instrument.
    ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
    count = len(ranked) // 5
    long_names, short_names = ranked[:count], ranked[-count:]
    if strategy == "buffer_20_30":
        band = int(np.ceil(len(ranked) * 0.30))

        def select(order: np.ndarray, held: np.ndarray) -> np.ndarray:
            retained = [int(i) for i in order[:band] if held[i]][:count]
            chosen = set(retained)
            fills = [int(i) for i in order if int(i) not in chosen]
            return np.array(retained + fills[: count - len(retained)], dtype=int)

        long_names = select(ranked, current > 1e-12)
        short_names = select(ranked[::-1], current < -1e-12)
    result = np.zeros(len(scores), dtype=float)
    result[long_names] = 1.0 / count
    result[short_names] = -1.0 / count
    if strategy == "partial_50" and np.any(np.abs(current) > 1e-12):
        result = 0.5 * current + 0.5 * result
        result[~np.isfinite(scores)] = 0.0
        result = normalize_sides(result)
    return result


def execute(
    qty: np.ndarray,
    cash: float,
    price: np.ndarray,
    tradable: np.ndarray,
    weights: np.ndarray,
    cost: float | Costs,
) -> tuple[np.ndarray, float, float, float]:
    """Solve post-cost NAV so desired weights don't create free leverage/cash."""
    before = qty * price
    nav = cash + before.sum()
    if nav <= 0:
        raise ValueError("Portfolio insolvent")
    model = cost if isinstance(cost, Costs) else Costs("symmetric", cost * 1e4, cost * 1e4)
    after_nav = nav
    for _ in range(100):
        desired = weights * after_nav
        desired[~tradable] = before[~tradable]
        new_nav = nav - model.charge(desired - before)
        if abs(new_nav - after_nav) < 1e-13 * max(nav, 1):
            after_nav = new_nav
            break
        after_nav = new_nav
    else:
        raise RuntimeError("Post-cost sizing failed to converge")
    desired = weights * after_nav
    desired[~tradable] = before[~tradable]
    trades = desired - before
    notional = float(np.abs(trades).sum())
    fee = model.charge(trades)
    new_qty = desired / price
    new_cash = float(cash - trades.sum() - fee)
    assert np.isclose(new_cash + (new_qty * price).sum(), nav - fee, atol=1e-11)
    return new_qty, new_cash, float(fee), notional


def run(
    data: Inputs,
    strategy: str,
    cost_bps: float | Costs,
    phase: int = 0,
    capture_positions: bool = False,
    rebalance_every: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if not isinstance(rebalance_every, int) or rebalance_every < 1:
        raise ValueError("rebalance_every must be a positive integer")
    if strategy == "rebalance_3d" and not 0 <= phase < rebalance_every:
        raise ValueError("phase must be within the rebalance interval")
    q = np.zeros(len(data.instruments))
    cash = 1.0
    cost = cost_bps if isinstance(cost_bps, Costs) else Costs("symmetric", cost_bps, cost_bps)
    rows, positions = [], []
    # Signal date d -> entry at d+1 open -> mark at d+2 open. The last
    # available opening date is reserved for liquidation, with its cost.
    for t in range(1, len(data.dates) - 1):
        signal = data.scores[t - 1]
        p0, p1 = data.prices[t], data.prices[t + 1]
        nav_start = float(cash + (q * p0).sum())
        current = q * p0 / nav_start
        rebalance = t == 1 or strategy != "rebalance_3d" or (t - 1) % rebalance_every == phase
        fee, traded = 0.0, 0.0
        buys, sells = 0.0, 0.0
        blocked_names = 0
        if rebalance:
            w = targets(signal, current, strategy)
            blocked_names = int(((np.abs(w - current) > 1e-10) & ~data.tradable[t]).sum())
            before_qty = q.copy()
            q, cash, fee, traded = execute(q, cash, p0, data.tradable[t], w, cost)
            changes = (q - before_qty) * p0
            buys = float(np.maximum(changes, 0).sum())
            sells = float(np.maximum(-changes, 0).sum())
        nav_after_trade = float(cash + (q * p0).sum())
        gross_exposure = float(np.abs(q * p0).sum() / nav_after_trade)
        net_exposure = float((q * p0).sum() / nav_after_trade)
        missing_held = int(((np.abs(q) > 1e-12) & ~data.tradable[t + 1]).sum())
        price_pnl = float((q * (p1 - p0)).sum())
        residual_terminal = 0.0
        if capture_positions:
            active = np.flatnonzero(np.abs(q) > 1e-12)
            for i in active:
                positions.append(
                    {
                        "entry_date": str(data.dates[t].date()),
                        "instrument": str(data.instruments[i]),
                        "adjusted_units": float(q[i]),
                        "weight": float(q[i] * p0[i] / nav_after_trade),
                    }
                )
        if t == len(data.dates) - 2:
            before_qty = q.copy()
            q, cash, close_fee, close_traded = execute(
                q, cash, p1, data.tradable[t + 1], np.zeros_like(q), cost
            )
            fee += close_fee
            traded += close_traded
            changes = (q - before_qty) * p1
            buys += float(np.maximum(changes, 0).sum())
            sells += float(np.maximum(-changes, 0).sum())
            residual_terminal = float(np.abs(q * p1).sum())
        nav_end = float(cash + (q * p1).sum())
        net_return = nav_end / nav_start - 1
        assert np.isclose(nav_end - nav_start, price_pnl - fee, atol=1e-10)
        rows.append(
            {
                "strategy": strategy,
                "scenario": cost.scenario,
                "buy_cost_bps": cost.buy_bps,
                "sell_cost_bps": cost.sell_bps,
                "slippage_bps": cost.slippage_bps,
                "phase": phase,
                "signal_date": str(data.dates[t - 1].date()),
                "entry_date": str(data.dates[t].date()),
                "exit_date": str(data.dates[t + 1].date()),
                "rebalance": rebalance,
                "nav_start": nav_start,
                "nav_end": nav_end,
                "price_pnl": price_pnl,
                "cost_amount": fee,
                "buy_notional": buys,
                "sell_notional": sells,
                "fee_amount": (buys * cost.buy_bps + sells * cost.sell_bps) / 10000,
                "slippage_amount": traded * cost.slippage_bps / 10000,
                "traded_notional": traded,
                "traded_nav_ratio": traded / nav_start,
                "one_way_turnover": 0.5 * traded / nav_start,
                "gross_component_return": price_pnl / nav_start,
                "cost_return": fee / nav_start,
                "net_return": net_return,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "blocked_trade_names": blocked_names,
                "missing_price_held_names": missing_held,
                "terminal_residual_value": residual_terminal,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(positions)


def summarize(daily: pd.DataFrame) -> dict:
    returns = daily["net_return"].to_numpy()
    wealth = np.r_[1.0, daily["nav_end"].to_numpy()]
    dd = wealth / np.maximum.accumulate(wealth) - 1
    std = returns.std(ddof=1)
    assert np.allclose(np.cumprod(1 + returns), wealth[1:], atol=1e-10)
    return {
        "strategy": daily["strategy"].iloc[0],
        "strategy_name": NAMES[daily["strategy"].iloc[0]],
        "scenario": daily["scenario"].iloc[0],
        "buy_cost_bps": float(daily["buy_cost_bps"].iloc[0]),
        "sell_cost_bps": float(daily["sell_cost_bps"].iloc[0]),
        "slippage_bps": float(daily["slippage_bps"].iloc[0]),
        "phase": int(daily["phase"].iloc[0]),
        "days": len(daily),
        "signal_start": daily["signal_date"].iloc[0],
        "signal_end": daily["signal_date"].iloc[-1],
        "entry_start": daily["entry_date"].iloc[0],
        "exit_end": daily["exit_date"].iloc[-1],
        "annualized_return": float(returns.mean() * 252),
        "cagr": float(wealth[-1] ** (252 / len(daily)) - 1),
        "sharpe": float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "max_drawdown": float(dd.min()),
        "total_return": float(wealth[-1] - 1),
        "average_one_way_turnover": float(daily["one_way_turnover"].mean()),
        "average_traded_nav_ratio": float(daily["traded_nav_ratio"].mean()),
        "annualized_cost_drag": float(daily["cost_return"].mean() * 252),
        "mean_gross_exposure": float(daily["gross_exposure"].mean()),
        "max_abs_net_exposure": float(daily["net_exposure"].abs().max()),
        "blocked_trade_name_events": int(daily["blocked_trade_names"].sum()),
        "missing_price_held_events": int(daily["missing_price_held_names"].sum()),
        "terminal_residual_value": float(daily["terminal_residual_value"].iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.inputs)
    summaries, days = [], []
    for strategy in STRATEGIES:
        for cost in SCENARIOS:
            daily, _ = run(data, strategy, cost)
            summary = summarize(daily)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            summaries.append(summary)
            days.append(daily)
    for phase in (1, 2):
        daily, _ = run(data, "rebalance_3d", SCENARIOS[1], phase)
        summaries.append(summarize(daily))
        days.append(daily)
    summary = pd.DataFrame(summaries)
    daily = pd.concat(days, ignore_index=True)
    summary.to_csv(args.out / "strategy_summary.csv", index=False)
    daily.to_csv(args.out / "strategy_daily.csv", index=False)
    base = summary[summary["phase"] == 0]
    primary = base[base["scenario"] == "fees_slip0"].copy()
    gross = base[base["scenario"] == "gross"].set_index("strategy")["annualized_return"]
    primary["gross_annualized_return_0bp"] = primary["strategy"].map(gross)
    primary.to_csv(args.out / "strategy_primary.csv", index=False)
    input_audit = json.loads((args.inputs / "input_audit.json").read_text())
    audit = {
        "status": "complete",
        "input_audit": input_audit,
        "strategy_order": list(STRATEGIES),
        "main_scenario": "fees_slip0",
        "scenarios": [vars(c) for c in SCENARIOS],
        "fee_basis": "BigTrader public example: commission 3bp each side, sale stamp duty 5bp; proportional rate scenario, not actual account fee calibration",
        "fee_source": "https://fund.bigquant.com/wiki/doc/3gG2rg4jBd",
        "stamp_duty_source": "https://one.sse.com.cn/onething/gptz/",
        "cost_model": "buy/sell direction from signed change in holdings, including short opening/covering; fees and symmetric linear slippage debited against opening-price traded notional; no minimum commission or separately added transfer fee; no assumed capital base",
        "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.inputs / "signals.parquet", args.inputs / "daily_prices.parquet"]
        },
        "holding": "signal at d close, rebalance at d+1 open; all intervening overnight returns included; last available open is terminal liquidation",
        "accounting": "self-financing adjusted-unit holdings and cash; costs on gross buys plus sells; target weights relative to post-cost NAV; initial entry and terminal exit included",
        "exposure": "long +1 / short -1 at rebalance where tradable; no free weight resets on non-rebalance days",
        "selection": "signal-date pool only, no future-label or future-price filter; stable instrument tie-break",
        "missing_quotes": "no trade at missing open; mark at last known price, using previous close when available; no future backfill",
        "execution_assumptions": [
            "observed opening-price reference plus linear per-side slippage scenarios 0/2/5bp",
            "no minimum commission floor or separately added transfer fee; proportional fee assumption only",
            "unlimited short availability assumed",
            "no borrow/financing fee or nonlinear market impact",
            "no price-limit/queue or opening-volume feasibility model",
            "adjusted prices approximate corporate actions",
        ],
        "research_status": "same previously inspected private period; strategy rules fixed before this run; comparative retrospective analysis, not independent strategy-selection OOS",
        "daily_rows": len(daily),
        "main_strategy_count": len(primary),
    }
    assert audit["input_hashes"] == {
        k: input_audit["input_hashes"][k] for k in audit["input_hashes"]
    }
    (args.out / "strategy_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    artifacts = [
        "strategy_summary.csv",
        "strategy_daily.csv",
        "strategy_primary.csv",
        "strategy_audit.json",
    ]
    (args.out / "checksums.json").write_text(
        json.dumps(
            {f: hashlib.sha256((args.out / f).read_bytes()).hexdigest() for f in artifacts},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
