"""Build exact-next-trading-day C2C labels inside AIStudio."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from bigquant import dai

ROOT = Path("/home/aiuser/work/e17_raw23_direct")
STORE = Path("/home/aiuser/work/e17_raw23_store_2023_2024")
OUTPUT = ROOT / "c2c_labels.parquet"
START = "2023-01-01"
END = "2025-01-10"


def load_daily_returns() -> pd.DataFrame:
    frame = dai.query(
        f"""
        SELECT
          CAST(date AS DATE) AS trade_date,
          instrument,
          arg_max(close, date) AS close,
          arg_min(pre_close, date) AS pre_close
        FROM bigalpha_2026_stock_bar1m
        WHERE date >= TIMESTAMP '{START}' AND date < TIMESTAMP '{END}'
        GROUP BY CAST(date AS DATE), instrument
        """,
        filters={"date": [START, END]},
        compression=True,
    ).df()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    close = pd.to_numeric(frame["close"], errors="coerce")
    pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
    frame["next_day_c2c"] = (close / pre_close.where(pre_close > 0) - 1.0).replace(
        [np.inf, -np.inf], np.nan
    )
    if frame.duplicated(["trade_date", "instrument"]).any():
        raise RuntimeError("duplicate daily-return keys")
    return frame[["trade_date", "instrument", "next_day_c2c"]]


def main() -> int:
    pool = pd.read_parquet(STORE / "pool.parquet", columns=["date", "instrument"])
    pool["date"] = pd.to_datetime(pool["date"], errors="raise").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.drop_duplicates(["date", "instrument"])
    daily = load_daily_returns()
    calendar = sorted(daily["trade_date"].unique())
    next_day = dict(pairwise(calendar))
    labels = pool.copy()
    labels["label_date"] = labels["date"].map(next_day)
    labels = labels.merge(
        daily.rename(columns={"trade_date": "label_date"}),
        on=["label_date", "instrument"],
        how="left",
        validate="many_to_one",
    ).rename(columns={"next_day_c2c": "ret_close_to_close"})
    labels = labels[["date", "instrument", "ret_close_to_close"]].sort_values(
        ["date", "instrument"], kind="stable"
    )
    labels.to_parquet(OUTPUT, index=False, compression="zstd")
    audit = {
        "label": "ret_close_to_close",
        "definition": "exact next market day close / next market day pre_close - 1",
        "factor_start": labels["date"].min().date().isoformat(),
        "factor_end": labels["date"].max().date().isoformat(),
        "factor_days": int(labels["date"].nunique()),
        "rows": len(labels),
        "non_null_rows": int(labels["ret_close_to_close"].notna().sum()),
        "non_null_2024_days": int(
            labels.loc[
                (labels["date"].dt.year == 2024)
                & labels["ret_close_to_close"].notna(),
                "date",
            ].nunique()
        ),
    }
    (ROOT / "c2c_labels_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
