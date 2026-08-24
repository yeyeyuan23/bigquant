"""Add open-to-open to the label decomposition for every variant."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/root/autodl-tmp/projects/bigquant-default")
for e in (ROOT/"src", ROOT/"scripts"):
    sys.path.insert(0, str(e))
from bigalpha2026.competition_score_proxy import preprocess_factor

DATA = Path("/root/autodl-tmp/data")
o2o = pd.read_parquet(ROOT / "reports/dependencies/finals_pre/o2o_labels.parquet")
o2o["date"] = pd.to_datetime(o2o["date"]).dt.normalize()
o2o["instrument"] = o2o["instrument"].astype(str)
o2o = o2o[o2o["date"].dt.year == 2024]
exposures = pd.read_parquet(DATA/"exposures"/"year=2024"/"part-2024.parquet")
exposures["date"] = pd.to_datetime(exposures["date"]).dt.normalize()
exposures["instrument"] = exposures["instrument"].astype(str)

rows = []
for name, path in json.loads(sys.argv[1]).items():
    f = pd.read_parquet(path)
    col = "factor" if "factor" in f.columns else "value"
    f = f.rename(columns={col: "factor"})[["date","instrument","factor"]]
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["instrument"] = f["instrument"].astype(str)
    f = f[f["date"].dt.year == 2024]
    neut = preprocess_factor(f, exposures).rename(columns={"factor":"neut"})
    m = f.merge(neut[["date","instrument","neut"]], on=["date","instrument"]) \
         .merge(o2o, on=["date","instrument"]).dropna(subset=["ret_open_to_open"])
    r = {"variant": name}
    for fc, tag in (("factor","raw"), ("neut","neut")):
        ics = m.dropna(subset=[fc]).groupby("date").apply(
            lambda g: g[fc].corr(g["ret_open_to_open"], method="spearman")
            if len(g) >= 50 else np.nan, include_groups=False).dropna()
        r[f"{tag}_open_to_open"] = float(ics.mean())
        if tag == "neut":
            r["neut_open_to_open_ir"] = float(ics.mean()/ics.std())
            r["neut_open_to_open_t"] = float(ics.mean()/ics.std()*np.sqrt(len(ics)))
            r["days"] = int(len(ics))
    rows.append(r); print(name, "ok", flush=True)

out = pd.DataFrame(rows).set_index("variant")
out.to_csv(ROOT/"reports/dependencies/finals_pre/o2o_decomposition.csv")
pd.set_option("display.width", 200)
print(out.round(4).to_string())
