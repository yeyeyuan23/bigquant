"""E11：从 AIStudio 的 bar1m 取 4 个新盘口通道，落成 sidecar。

sidecar 只含新通道，键 (instrument, timestamp) 与现有 store 对齐，
回 AutoDL 后 join 即可——不重建 store（那边磁盘只剩 8.8 G）。

通道（见 experiment_log.md 的 E11 立项）：
  1 log_size_per_order_l1   = log1p((B1+A1) / (nB1+nA1))
  2 order_count_imbalance_l1= (nB1-nA1) / (nB1+nA1)
  3 deep_depth_ratio        = (B4+B5+A4+A5) / (sum B1..5 + sum A1..5)
  4 log_book_gap_ticks      = log1p(买方空档 + 卖方空档)，空档 = 五档跨度 tick 数 - 4

num_orders 在 2019-06 之前恒为 0（不是 NULL），那一段通道 1、2 写 NaN，
由下游 observed_mask 标成未观测——绝不能把 0 当真值喂进模型。
四五档价量 2019-01 就有值，通道 3、4 全程可用。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import dai
import numpy as np
import pandas as pd

TICK = 0.01
CONTIGUOUS_TICKS = 4          # 五档连续时的跨度
ORDERS_VALID_FROM = pd.Timestamp("2019-06-01")
CHANNELS = ["log_size_per_order_l1", "order_count_imbalance_l1",
            "deep_depth_ratio", "log_book_gap_ticks"]

LEVELS = (1, 2, 3, 4, 5)
COLS = (["date", "instrument"]
        + [f"{s}_price{i}" for s in ("bid", "ask") for i in LEVELS]
        + [f"{s}_volume{i}" for s in ("bid", "ask") for i in LEVELS]
        + ["bid_num_orders1", "ask_num_orders1"])


def _side_volume(df: pd.DataFrame, side: str, level: int) -> pd.Series:
    """照抄 _positive_book_side：价格非正或量非正时记 0。"""
    price = df[f"{side}_price{level}"]
    volume = df[f"{side}_volume{level}"]
    return volume.where(price.gt(0) & volume.gt(0), 0.0)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """照抄 _safe_ratio：分母非有限或为 0 时给 NaN。"""
    ok = den.notna() & np.isfinite(den) & den.ne(0)
    return num.div(den.where(ok))


def build_channels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"instrument": df["instrument"], "timestamp": df["date"]})

    # 无报价的一侧记 0 笔，跟 _positive_book_side 的惯例一致：跌停锁死时买盘为空，
    # 已有的 depth_imbalance_l1 给的是 -1（极度卖方压倒），不是缺失。若这里改判缺失，
    # 就会恰好在压力日把信号扔掉 —— 而压力日 IC/IR 是 A 类四项之一。
    # 实测 2020-02-03（新冠暴跌复市首日）只有 22.2% 的分钟买方有报价。
    quoted = {}
    for side in ("bid", "ask"):
        n = df[f"{side}_num_orders1"]
        price = df[f"{side}_price1"]
        quoted[side] = n.where(n.gt(0) & price.gt(0), 0.0)
    nb, na = quoted["bid"], quoted["ask"]
    b1 = _side_volume(df, "bid", 1)
    a1 = _side_volume(df, "ask", 1)
    out["log_size_per_order_l1"] = np.log1p(_ratio(b1 + a1, nb + na))
    out["order_count_imbalance_l1"] = _ratio(nb - na, nb + na)

    bid = {i: _side_volume(df, "bid", i) for i in LEVELS}
    ask = {i: _side_volume(df, "ask", i) for i in LEVELS}
    deep = bid[4] + bid[5] + ask[4] + ask[5]
    total = sum(bid.values()) + sum(ask.values())
    out["deep_depth_ratio"] = _ratio(deep, total)

    bid_span = _ratio(df["bid_price1"] - df["bid_price5"], pd.Series(TICK, index=df.index))
    ask_span = _ratio(df["ask_price5"] - df["ask_price1"], pd.Series(TICK, index=df.index))
    # 这一个反过来：单边书的「稀疏程度」没有定义，硬凑会让通道在不同日子含义不同。
    # 保持缺失，由 observed_fraction 告诉模型「今天盘口是单边的」。
    both_quoted = df["bid_price5"].gt(0) & df["ask_price5"].gt(0) & df["bid_price1"].gt(0)
    gap = ((bid_span.round() - CONTIGUOUS_TICKS).clip(lower=0)
           + (ask_span.round() - CONTIGUOUS_TICKS).clip(lower=0))
    out["log_book_gap_ticks"] = np.log1p(gap.where(both_quoted))

    # num_orders 生效之前，通道 1、2 一律 NaN（下游按未观测处理）
    early = pd.to_datetime(out["timestamp"]) < ORDERS_VALID_FROM
    out.loc[early, ["log_size_per_order_l1", "order_count_imbalance_l1"]] = np.nan
    return out


def run_month(month: pd.Timestamp, keys: pd.DataFrame, outdir: Path) -> dict:
    start = month.strftime("%Y-%m-%d")
    end = (month + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    t0 = time.time()
    df = dai.query(
        f"SELECT {', '.join(COLS)} FROM bigalpha_2026_stock_bar1m "
        f"WHERE date >= '{start}' AND date < '{end}'"
    ).df()
    t_query = time.time() - t0

    t0 = time.time()
    df["trade_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.merge(keys, on=["trade_date", "instrument"], how="inner")
    if df.empty:
        return {"month": start, "rows": 0, "days": 0, "query_s": t_query, "build_s": 0.0}

    chan = build_channels(df)
    chan["trade_date"] = df["trade_date"].to_numpy()
    days = 0
    for day, part in chan.groupby("trade_date", sort=True):
        d = outdir / f"trade_date={day}"
        d.mkdir(parents=True, exist_ok=True)
        (part.drop(columns=["trade_date"])
             .sort_values(["instrument", "timestamp"])
             .to_parquet(d / "part-000.parquet", index=False, compression="zstd"))
        days += 1
    return {"month": start, "rows": len(chan), "days": days,
            "query_s": t_query, "build_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="不含")
    ap.add_argument("--out", default=str(Path.home() / "e11/sidecar"))
    args = ap.parse_args()

    keys = pd.read_parquet(Path.home() / "e11/store_keys.parquet")
    keys["trade_date"] = keys["trade_date"].astype(str)
    keys["instrument"] = keys["instrument"].astype(str)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    months = pd.date_range(args.start, args.end, freq="MS", inclusive="left")
    for m in months:
        done = sorted((outdir).glob(f"trade_date={m.strftime('%Y-%m')}-*"))
        if done:
            print(f"{m:%Y-%m} 已存在 {len(done)} 天，跳过", flush=True)
            continue
        r = run_month(m, keys, outdir)
        print(f"{r['month']}  {r['days']:2d} 天  {r['rows']:>9,} 行  "
              f"query {r['query_s']:6.1f}s  build {r['build_s']:6.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
