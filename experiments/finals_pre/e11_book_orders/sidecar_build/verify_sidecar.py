"""sidecar 必须逐日、逐 (股票,分钟) 完整覆盖 store —— 缺一天训练就会抛错，
缺几行则会静默变成未观测，后者更危险。"""
import glob
from pathlib import Path

import pandas as pd

SIDE = Path("/home/aiuser/e11/sidecar")
keys = pd.read_parquet("/home/aiuser/e11/store_keys.parquet")
keys["trade_date"] = keys["trade_date"].astype(str)

store_days = set(keys["trade_date"])
side_days = {p.name.split("=", 1)[1] for p in SIDE.glob("trade_date=*")}
print(f"store 交易日 {len(store_days)}   sidecar 分区 {len(side_days)}")
missing = sorted(store_days - side_days)
extra = sorted(side_days - store_days)
print(f"缺失的日子: {len(missing)}  {missing[:5]}")
print(f"多出的日子: {len(extra)}  {extra[:5]}")

rows = 0
bad = []
for day in sorted(side_days)[::120]:
    f = glob.glob(f"{SIDE}/trade_date={day}/*.parquet")[0]
    d = pd.read_parquet(f, columns=["instrument", "timestamp"])
    want = set(keys.loc[keys["trade_date"] == day, "instrument"])
    got = set(d["instrument"].astype(str))
    rows += len(d)
    if want - got:
        bad.append((day, len(want - got)))
print(f"抽查 {len(sorted(side_days)[::120])} 天，共 {rows:,} 行")
print("股票覆盖不全的日子:", bad if bad else "无")
