"""价格水平是否已经隐含在 17 个通道里。

model.md 1.1 原先断言「价格的水平被除掉了……模型拿不到这只股票是 5 块还是 500 块」。
本审计推翻它：价格水平有两条独立路径进了模型。

  A 路  a = log_amount - log_volume = log(成交额/成交量) = log(均价)
        两个已有通道之差，一个线性层即可得到。
  B 路  b = -log(relative_spread 的 1% 分位)
        A 股最小报价单位 0.01 元在相对价差上压出的地板 relative_spread >= 0.01/P。
        不需要模型做任何事，是通道自身分布的下沿。

判据不是相关系数而是一个恒等式。若两条路都是 log(价格)：

    b - a = log(1 / 0.01) = 4.60517          （与股票、与日期无关的常数）

逐股票算这个残差，看它是否落在常数上。残差偏离常数的股票即盘口数据退化
（价差 1% 分位接近 0），一并计数上报，不做剔除。

A 路算两个版本：
  incl_zero  含零成交量分钟（log1p(0)-log1p(0)=0，模型的日级均值就是这么算的）
  excl_zero  剔除零成交量分钟（干净口径）
两版都报——只报干净那版会高估模型实际拿到的信噪比。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STORE = Path("/root/autodl-tmp/unified_microstructure_store_v2_2019_2024/data")
OUT = Path(__file__).resolve().parents[3] / "reports/dependencies/finals_pre/price_level_audit"
TICK = 0.01                      # A 股最小报价单位（元）
TICK_CONST = math.log(1.0 / TICK)  # 4.60517…，b - a 的理论值
BAND = 0.5                       # 残差偏离 TICK_CONST 超过这个数即判为盘口退化


def day_stats(path: Path) -> dict[str, float] | None:
    df = pd.read_parquet(path, columns=["instrument", "log_amount", "log_volume", "relative_spread"])
    zero_vol = df["log_volume"].eq(0.0)

    def route_a(frame: pd.DataFrame) -> pd.Series:
        g = frame.groupby("instrument")
        return g["log_amount"].mean() - g["log_volume"].mean()

    routes = {"incl_zero": route_a(df), "excl_zero": route_a(df.loc[~zero_vol])}
    floor = df.groupby("instrument")["relative_spread"].quantile(0.01)
    b = -np.log(floor.replace(0.0, np.nan))

    out: dict[str, float] = {
        "instruments": len(routes["incl_zero"]),
        "minutes": len(df),
        "zero_volume_rate": float(zero_vol.mean()),
        "spread_floor_zero": int((floor == 0).sum()),
    }
    for tag, a in routes.items():
        ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        x, y = a[ok], b[ok]
        if len(x) < 50:
            return None
        residual = y - x
        out[f"resid_p5_{tag}"] = float(np.percentile(residual, 5))
        out[f"resid_median_{tag}"] = float(np.median(residual))
        out[f"resid_p95_{tag}"] = float(np.percentile(residual, 95))
        out[f"degenerate_rate_{tag}"] = float((residual - TICK_CONST).abs().gt(BAND).mean())
        out[f"spearman_{tag}"] = float(x.corr(y, method="spearman"))
    price = np.exp(routes["excl_zero"].dropna())
    for q in (5, 50, 95):
        out[f"implied_price_p{q}"] = float(np.percentile(price, q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=24, help="每隔多少个交易日采样一天")
    args = ap.parse_args()

    days = sorted(STORE.glob("trade_date=*"))
    rows = []
    for d in days[:: args.every]:
        files = glob.glob(f"{d}/*.parquet")
        if not files:
            continue
        stats = day_stats(Path(files[0]))
        if stats is not None:
            rows.append({"trade_date": d.name.split("=", 1)[1], **stats})

    frame = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_day.csv", index=False)

    summary: dict[str, object] = {
        "claim": "price level is recoverable from the shipped 17 channels via two independent routes",
        "test": "b - a should equal log(1/tick) = 4.60517 for every stock on every day",
        "store": str(STORE.parent),
        "tick_size_cny": TICK,
        "tick_constant": TICK_CONST,
        "degenerate_band": BAND,
        "days_total": len(days),
        "days_sampled": len(frame),
        "sample_every_n_days": args.every,
        "date_range": [frame["trade_date"].iloc[0], frame["trade_date"].iloc[-1]],
        "instruments_per_day_median": float(frame["instruments"].median()),
        "zero_volume_rate_median": float(frame["zero_volume_rate"].median()),
    }
    for tag in ("incl_zero", "excl_zero"):
        summary[f"resid_median_min_{tag}"] = float(frame[f"resid_median_{tag}"].min())
        summary[f"resid_median_max_{tag}"] = float(frame[f"resid_median_{tag}"].max())
        summary[f"resid_p5_min_{tag}"] = float(frame[f"resid_p5_{tag}"].min())
        summary[f"resid_p95_max_{tag}"] = float(frame[f"resid_p95_{tag}"].max())
        summary[f"degenerate_rate_median_{tag}"] = float(frame[f"degenerate_rate_{tag}"].median())
        summary[f"degenerate_rate_max_{tag}"] = float(frame[f"degenerate_rate_{tag}"].max())
        summary[f"spearman_min_{tag}"] = float(frame[f"spearman_{tag}"].min())
        summary[f"spearman_median_{tag}"] = float(frame[f"spearman_{tag}"].median())
    for q in (5, 50, 95):
        summary[f"implied_price_p{q}_first"] = float(frame[f"implied_price_p{q}"].iloc[0])
        summary[f"implied_price_p{q}_last"] = float(frame[f"implied_price_p{q}"].iloc[-1])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
