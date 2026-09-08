"""Canonical O2C diagnostics and predeclared paired, synchronous block inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(__file__).resolve().parent / "_runtime"
sys.path[:0] = [
    str(ROOT),
    str(RUNTIME / "src"),
    str(RUNTIME / "scripts"),
    str(RUNTIME / "experiments/finals_pre/common"),
]

import importlib.util

import numpy as np
import pandas as pd

from competition_score_proxy import prepare_full_barra_exposures, preprocess_factor

_canonical_spec = importlib.util.spec_from_file_location(
    "tcn_canonical_score", RUNTIME / "experiments/finals_pre/e3_progressive_add/score.py"
)
_canonical_score = importlib.util.module_from_spec(_canonical_spec)
_canonical_spec.loader.exec_module(_canonical_score)
daily_ic = _canonical_score.daily_ic
long_short_sharpe = _canonical_score.long_short_sharpe
from experiments.finals_pre.tcn_architecture_o2c.protocol import (
    ARMS,
    BASELINE,
    BLOCK_LENGTH,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    LABEL,
    METRICS,
    SEEDS,
    file_hash,
    now,
    object_hash,
    seed_tag,
    write_json,
)
from experiments.finals_pre.tcn_architecture_o2c.train import actual_key_hash


def evaluation_inputs(meta):
    labels = pd.read_parquet(Path(meta["data_root"]) / "labels/year=2024/part-2024.parquet")
    labels["date"] = pd.to_datetime(labels.date).dt.normalize()
    labels["instrument"] = labels.instrument.astype(str)
    labels = labels.loc[labels.date.dt.year == 2024].dropna(subset=[LABEL])
    exposures = prepare_full_barra_exposures(pd.read_parquet(meta["exposure"]))
    return labels, exposures


def expected_scoring_keys(meta, prediction_keys):
    """Freeze exposure-complete score rows, independently of model predictions."""
    labels, exposures = evaluation_inputs(meta)
    dummy = prediction_keys.copy()
    dummy["factor"] = dummy.groupby("date").cumcount().astype(float)
    neutral = preprocess_factor(dummy, exposures)
    merged = (
        neutral.merge(labels, on=["date", "instrument"], validate="one_to_one")
        .dropna(subset=["factor", LABEL])
        .sort_values(["date", "instrument"])
    )
    if not np.isfinite(merged[["factor", LABEL]]).all().all():
        raise RuntimeError("nonfinite evaluation inputs")
    return actual_key_hash(merged), len(merged)


def verify_run(prepared, run):
    prepared, run = Path(prepared), Path(run)
    meta = json.loads((prepared / "prepared.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "complete" or manifest["prepared_sha256"] != file_hash(
        prepared / "prepared.json"
    ):
        raise RuntimeError(f"unfinished or foreign run: {run}")
    for artifact in ("checkpoint", "factor"):
        suffix = "pt" if artifact == "checkpoint" else "parquet"
        if file_hash(run / f"{artifact}.{suffix}") != manifest[f"{artifact}_sha256"]:
            raise RuntimeError(f"changed {artifact}: {run}")
    if manifest["actual_sample_hash"] != meta["actual_sample_hashes"][str(manifest["seed"])]:
        raise RuntimeError("training sample mismatch")
    return meta, manifest


def score_run(prepared, run):
    run = Path(run)
    meta, manifest = verify_run(prepared, run)
    factor = pd.read_parquet(run / "factor.parquet").sort_values(["date", "instrument"])
    if factor.duplicated(["date", "instrument"]).any() or not np.isfinite(factor.factor).all():
        raise RuntimeError("invalid factor rows")
    if (
        actual_key_hash(factor) != meta["prediction_keys_hash"]
        or object_hash(factor.observed.tolist()) != meta["prediction_availability_hash"]
    ):
        raise RuntimeError("factor keys/missing-data mask mismatch")
    labels, exposures = evaluation_inputs(meta)
    neutral = preprocess_factor(factor[["date", "instrument", "factor"]], exposures)
    merged = (
        neutral.merge(labels, on=["date", "instrument"], validate="one_to_one")
        .dropna(subset=["factor", LABEL])
        .sort_values(["date", "instrument"])
    )
    if actual_key_hash(merged) != meta["scoring_keys_hash"]:
        raise RuntimeError("effective scoring stock/date keys differ from preflight")
    ic = daily_ic(merged, "factor").sort_index()
    if [str(d.date()) for d in ic.index] != meta["scoring_dates"]:
        raise RuntimeError("daily score calendar differs from preflight")
    dispersion = labels.groupby("date")[LABEL].std()
    stress_days = set(dispersion[dispersion >= dispersion.quantile(0.75)].index)
    stress = ic.loc[ic.index.isin(stress_days)]
    metrics = {
        "arm": manifest["arm"],
        "seed": manifest["seed"],
        "rank_ic": float(ic.mean()),
        "rank_ic_ir": float(ic.mean() / ic.std()),
        "long_short_sharpe": long_short_sharpe(merged, "factor"),
        "stress_ic_ir": float(stress.mean() / stress.std()),
        "parameter_count": manifest["parameter_count"],
        "receptive_field": manifest["receptive_field"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "peak_cuda_allocated_bytes": manifest["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": manifest["peak_cuda_reserved_bytes"],
        "scored_days": len(ic),
        "scored_rows": len(merged),
        "stress_days": len(stress),
        "scoring_keys_hash": actual_key_hash(merged),
        "factor_sha256": manifest["factor_sha256"],
        "prepared_sha256": manifest["prepared_sha256"],
    }
    if not np.isfinite([metrics[m] for m in METRICS]).all():
        raise FloatingPointError("nonfinite aggregate score")
    daily = ic.rename("rank_ic").rename_axis("date").to_frame()
    daily["stress"] = daily.index.isin(stress_days)
    work = merged.copy()
    work["bucket"] = work.groupby("date").factor.transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    returns = work.groupby(["date", "bucket"])[LABEL].mean().unstack()
    daily["long_short_return"] = returns[4] - returns[0]
    if not np.isfinite(daily[["rank_ic", "long_short_return"]]).all().all():
        raise FloatingPointError("nonfinite daily score")
    daily.to_csv(run / "daily_metrics.csv")
    metrics["daily_metrics_sha256"] = file_hash(run / "daily_metrics.csv")
    write_json(run / "metrics.json", metrics)
    return metrics


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("invalid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return adjusted


def paired_bootstrap(
    differences, *, length=BLOCK_LENGTH, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
):
    """Input [candidate, seed, date]; synchronize each draw over all arms/seeds.

    Seeds are held fixed, not independently resampled. The two-sided null test
    uses bootstrap means of the centered daily difference process, with a +1
    Monte Carlo correction. CI is the percentile interval for the raw process.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all() or not 1 <= length <= values.shape[2]:
        raise ValueError("expected finite [candidate, seed, date] and a valid block length")
    daily = values.mean(axis=1)
    point = daily.mean(axis=1)
    draws = np.empty((len(values), replicates))
    rng = np.random.default_rng(seed)
    n = daily.shape[1]
    for offset in range(0, replicates, 256):
        count = min(256, replicates - offset)
        starts = rng.integers(0, n - length + 1, size=(count, (n + length - 1) // length))
        indices = (starts[..., None] + np.arange(length)).reshape(count, -1)[:, :n]
        draws[:, offset : offset + count] = daily[:, indices].mean(axis=2)
    low, high = np.quantile(draws, [0.025, 0.975], axis=1)
    p = (1 + (np.abs(draws - point[:, None]) >= np.abs(point[:, None])).sum(axis=1)) / (
        replicates + 1
    )
    return point, low, high, p


def aggregate(output):
    output = Path(output)
    prepared = output / "prepared"
    meta = json.loads((prepared / "prepared.json").read_text())
    rows, daily_arrays = [], []
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            run = output / "runs" / seed_tag(seed) / arm.name
            verify_run(prepared, run)
            metrics = json.loads((run / "metrics.json").read_text())
            if (metrics["arm"], metrics["seed"]) != (arm.name, seed):
                raise RuntimeError("grid identity mismatch")
            if file_hash(run / "daily_metrics.csv") != metrics["daily_metrics_sha256"]:
                raise RuntimeError("daily metrics changed")
            daily = pd.read_csv(run / "daily_metrics.csv")
            if daily.date.tolist() != meta["scoring_dates"]:
                raise RuntimeError("daily metrics not aligned")
            per_seed.append(daily.rank_ic.to_numpy())
            rows.append(metrics)
        daily_arrays.append(per_seed)
    cube = np.asarray(daily_arrays)
    if cube.shape != (15, 3, 241):
        raise RuntimeError("incomplete experiment grid")
    differences = cube[1:] - cube[0:1]
    point, low, high, p = paired_bootstrap(differences)
    adjusted = holm_adjust(p)
    summary, pairs = [], []
    frame = pd.DataFrame(rows)
    for index, arm in enumerate(ARMS):
        arm_rows = frame[frame.arm == arm.name].set_index("seed").loc[list(SEEDS)]
        baseline = frame[frame.arm == BASELINE].set_index("seed").loc[list(SEEDS)]
        item = {
            "arm": arm.name,
            "group": arm.group,
            "depth": arm.depth,
            "kernels": "/".join(map(str, arm.kernels)),
            "parameters": arm.parameter_count,
            "receptive_field": arm.receptive_field,
            "mean_minutes": float(arm_rows.elapsed_seconds.mean() / 60),
            "peak_gpu_gib": float(arm_rows.peak_cuda_allocated_bytes.max() / 2**30),
        }
        for metric in METRICS:
            item[metric] = float(arm_rows[metric].mean())
            item[metric + "_seed_sd"] = float(arm_rows[metric].std())
        for seed in SEEDS:
            pairs.append(
                {
                    "arm": arm.name,
                    "seed": seed,
                    **{
                        f"delta_{m}": float(arm_rows.loc[seed, m] - baseline.loc[seed, m])
                        for m in METRICS
                    },
                }
            )
        if index:
            j = index - 1
            positive = int((differences[j].mean(axis=1) > 0).sum())
            negative = int((differences[j].mean(axis=1) < 0).sum())
            if point[j] > 0 and low[j] > 0 and positive >= 2 and adjusted[j] < 0.05:
                conclusion = "在本协议下支持该配置改善 RankIC"
            elif point[j] < 0 and high[j] < 0 and negative >= 2 and adjusted[j] < 0.05:
                conclusion = "在本协议下支持该配置降低 RankIC"
            else:
                conclusion = "现有实验不足以区分"
            item.update(
                delta_rank_ic=float(point[j]),
                ci_low=float(low[j]),
                ci_high=float(high[j]),
                p_raw=float(p[j]),
                p_holm=float(adjusted[j]),
                positive_seeds=positive,
                conclusion=conclusion,
            )
        else:
            item.update(
                delta_rank_ic=0.0,
                ci_low=None,
                ci_high=None,
                p_raw=None,
                p_holm=None,
                positive_seeds=None,
                conclusion="重新训练的配对基线",
            )
        summary.append(item)
    result = output / "results"
    result.mkdir(exist_ok=True)
    frame.to_csv(result / "per_run.csv", index=False)
    pd.DataFrame(pairs).to_csv(result / "paired_deltas.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(result / "summary.csv", index=False)
    write_json(result / "summary.json", summary)
    np.savez_compressed(
        result / "daily_rankic_cube.npz",
        rank_ic=cube,
        arms=[a.name for a in ARMS],
        seeds=SEEDS,
        dates=meta["scoring_dates"],
    )
    lines = [
        "# TCN 结构比较：2024 历史验证期",
        "",
        "45 次训练全部完成；每次三轮，使用全新权重。2024 已参与此前研究，不称为全新未见测试。",
        "",
        ("结构计算：R=1+L(max(k)−1)。这只描述卷积链的直接覆盖；"
        "统计通路、全日标准化与时间汇总让整网依赖全天信息。"),
        "",
        ("日期区间：同步 10 日移动块 bootstrap，10,000 次，95% percentile 区间；"
        "双侧中心化 bootstrap 检验，14 个基线对比作 Holm 校正。"
        "固定三个种子，区间不覆盖全部初始化不确定性。"),
        "",
    ]
    for group, title in (
        ("depth", "深度"),
        ("branches", "分支数"),
        ("kernels", "核宽"),
        ("controls", "解释对照"),
    ):
        subset = table[table.group.isin(["baseline", group])]
        subset.to_csv(result / f"{group}.csv", index=False)
        lines += [
            f"## {title}",
            "",
            "| 配置 | RankIC | 配对差值 | 95% 区间 | Holm p | 判断 |",
            "|---|---:|---:|---|---:|---|",
        ]
        for item in summary:
            if item["group"] not in ("baseline", group):
                continue
            interval = (
                "—"
                if item["ci_low"] is None
                else (f"[{item['ci_low']:+.6f}, {item['ci_high']:+.6f}]")
            )
            ptext = "—" if item["p_holm"] is None else f"{item['p_holm']:.4f}"
            lines.append(
                f"| {item['arm']} | {item['rank_ic']:.6f} | "
                f"{item['delta_rank_ic']:+.6f} | {interval} | {ptext} | {item['conclusion']} |"
            )
        lines += [""]
    lines += [
        "## 解释边界",
        "",
        ("全部结论仅适用于三轮训练预算；更深配置的结果不是充分训练上限。"
        "差异不显著不代表效果等价。多空 Sharpe 未扣交易成本，仅作因子诊断。"),
        "",
        ("金融解释是待检验假设：三块是让收盘卷积路径连接上午后段的最浅深度。"
        "若两块或四块更好，应调整选型结论；若近似覆盖对照削弱三块优势，"
        "优先解释为覆盖收益。核宽、参数量仍有差异，不能声称完全隔离的机制证明。"),
        "",
        ("DeepLOB 支持局部卷积、多尺度提取和时间关系组合的研究思路，"
        "不提供本模型的三块或 3/15/60 数值依据。"),
        "",
    ]
    (result / "README.md").write_text("\n".join(lines))
    write_json(
        result / "audit.json",
        {
            "completed_at": now(),
            "runs": len(frame),
            "prepared_sha256": file_hash(prepared / "prepared.json"),
            "files": {
                p.name: file_hash(p)
                for p in result.iterdir()
                if p.is_file() and p.name != "audit.json"
            },
        },
    )
    return summary
