"""One-command AIStudio validation for BigAlpha factor submissions."""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import json
import re
import symtable
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]
FACTOR_TABLES = {
    "bigalpha_2026_stock_bar1m",
    "bigalpha_2026_financial",
}
AUXILIARY_TABLES = {"bigalpha_2026_instruments"}
FORBIDDEN_TABLES = {
    "bigalpha_2026_stock_bar5m",
    "bigalpha_2026_stock_bar15m",
    "bigalpha_2026_stock_bar30m",
    "bigalpha_2026_factorlib",
    "bigalpha_2026_exposure",
}
TABLE_RE = re.compile(r"\bbigalpha_2026_[A-Za-z0-9_]+\b")


def candidate_sources(path: Path, tree: ast.Module) -> dict[str, str]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "sources"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            return value
    package = path.parent / "bigalpha2026"
    if not package.is_dir():
        return {}
    return {
        ".".join(module.relative_to(path.parent).with_suffix("").parts):
        module.read_text(encoding="utf-8")
        for module in sorted(package.rglob("*.py"))
        if module.name != "__init__.py"
    }


def undefined_globals(source: str, filename: str) -> list[str]:
    table = symtable.symtable(source, filename, "exec")
    bound = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    allowed = set(dir(builtins)) | bound | {
        "__builtins__", "__cached__", "__doc__", "__file__", "__loader__",
        "__name__", "__package__", "__spec__",
    }
    missing: set[str] = set()

    def visit(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            if (
                symbol.is_referenced()
                and symbol.is_global()
                and symbol.get_name() not in allowed
            ):
                missing.add(symbol.get_name())
        for child in scope.get_children():
            visit(child)

    visit(table)
    return sorted(missing)


def source_check(path: Path, max_months: int) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    compile(source, str(path), "exec")
    main_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if not main_nodes:
        raise ValueError("submission has no main()")
    signature = [item.arg for item in main_nodes[0].args.args]
    if signature != ["datasources", "start_date", "end_date"]:
        raise ValueError(f"invalid main signature: {signature}")

    modules = candidate_sources(path, tree)
    undefined = {}
    for name, module_source in modules.items():
        compile(module_source, name, "exec")
        missing = undefined_globals(module_source, name)
        if missing:
            undefined[name] = missing
    combined_source = "\n".join([source, *modules.values()])
    tables = sorted(set(TABLE_RE.findall(combined_source)))
    forbidden = sorted(set(tables) & FORBIDDEN_TABLES)
    unknown = sorted(
        set(tables) - FACTOR_TABLES - AUXILIARY_TABLES - FORBIDDEN_TABLES
    )
    excessive = []
    for match in re.finditer(
        r"(?:DateOffset|Timedelta)\s*\([^)]*"
        r"(?:(months)\s*=\s*(\d+)|(days)\s*=\s*(\d+))",
        combined_source,
    ):
        unit = "months" if match.group(1) else "days"
        value = int(match.group(2) or match.group(4))
        if (unit == "months" and value > max_months) or (
            unit == "days" and value > 184
        ):
            excessive.append(match.group(0))
    notebook = path.with_suffix(".ipynb")
    notebook_status: dict[str, Any] = {
        "status": "skipped",
        "reason": "adjacent notebook not found",
    }
    if notebook.exists():
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        cells = [
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        if len(cells) != 1 or cells[0] != source:
            raise ValueError("notebook must contain exactly the same single code cell")
        notebook_status = {"status": "ok", "notebook": notebook.name}
    return {
        "status": (
            "ok"
            if not forbidden and not unknown and not excessive and not undefined
            else "error"
        ),
        "candidate_modules": len(modules),
        "undefined_module_globals": undefined,
        "detected_tables": tables,
        "forbidden_tables": forbidden,
        "unknown_tables": unknown,
        "excessive_lookback": excessive,
        "max_lookback_months": max_months,
        "notebook": notebook_status,
    }


def platform_schema_check(schema_date: str) -> dict[str, Any]:
    import dai

    specs = {
        "bigalpha_2026_stock_bar1m": (
            "date", "instrument", "pre_close", "open", "high", "low",
            "close", "amount", "volume",
        ),
        "bigalpha_2026_financial": (
            "date", "instrument", "category", "shift", "report_date",
        ),
        "bigalpha_2026_instruments": ("date", "instrument"),
    }
    results = []
    for table, columns in specs.items():
        try:
            frame = dai.query(
                f"SELECT {', '.join(columns)} FROM {table} LIMIT 1",
                filters={"date": [schema_date, schema_date]},
                compression=True,
            ).df()
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise ValueError(f"missing columns: {missing}")
            results.append({"status": "ok", "table": table, "rows": len(frame)})
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "status": "error",
                    "table": table,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": (
            "ok" if all(item["status"] == "ok" for item in results) else "error"
        ),
        "tables": results,
    }


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("submission_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load submission: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError("submission has no main()")
    return module


@contextmanager
def query_audit(start: pd.Timestamp, max_months: int):
    import dai

    original = dai.query
    calls: list[dict[str, Any]] = []

    def query(sql, *args, **kwargs):
        calls.append(
            {
                "tables": sorted(set(TABLE_RE.findall(str(sql)))),
                "date_filter": (kwargs.get("filters") or {}).get("date"),
            }
        )
        return original(sql, *args, **kwargs)

    dai.query = query
    try:
        yield calls
    finally:
        dai.query = original
        earliest = start - pd.DateOffset(months=max_months)
        for call in calls:
            call["status"] = "ok"
            if set(call["tables"]) & FORBIDDEN_TABLES:
                call["status"] = "forbidden_table"
            date_filter = call["date_filter"]
            if date_filter and pd.Timestamp(date_filter[0]) < earliest:
                call["status"] = "excessive_lookback"


def normalize(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    if set(frame.columns) != {"date", "instrument", "factor"}:
        raise ValueError(
            "output columns must be exactly date, instrument, factor; "
            f"got {list(frame.columns)}"
        )
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str)
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")
    if result[KEYS].isna().any().any():
        raise ValueError("output contains invalid keys")
    if result.duplicated(KEYS).any():
        raise ValueError("output contains duplicate keys")
    return result.loc[result["date"].le(end)].sort_values(KEYS).reset_index(drop=True)


def expected_universe(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import dai

    return dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={
            "date": [
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            ]
        },
        compression=True,
    ).df()


def output_check(
    frame: pd.DataFrame,
    universe: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    output = normalize(frame, end)
    output = output.loc[output["date"].between(start, end)]
    expected = universe[KEYS].copy()
    expected["date"] = pd.to_datetime(expected["date"]).dt.normalize()
    expected["instrument"] = expected["instrument"].astype(str)
    expected = expected.loc[expected["date"].between(start, end)].drop_duplicates()
    if expected.empty:
        raise ValueError("expected universe is empty")
    checked = expected.merge(output, on=KEYS, how="left", validate="one_to_one")
    checked["finite"] = np.isfinite(checked["factor"].to_numpy(dtype=float))
    daily = checked.groupby("date")["finite"].agg(["count", "sum"])
    daily["missing_rate"] = 1.0 - daily["sum"] / daily["count"]
    missing = daily.index[daily["sum"].eq(0)]
    excessive = daily.index[daily["missing_rate"].gt(0.40)]
    return {
        "status": (
            "ok"
            if len(missing) == 0 and len(excessive) == 0
            else "invalid_output"
        ),
        "expected_rows": len(expected),
        "returned_rows": len(output),
        "missing_date_sample": [
            pd.Timestamp(value).strftime("%Y-%m-%d") for value in missing[:20]
        ],
        "excessive_missing_sample": [
            {
                "date": pd.Timestamp(value).strftime("%Y-%m-%d"),
                "missing_rate": float(daily.loc[value, "missing_rate"]),
            }
            for value in excessive[:20]
        ],
    }


def prefix_check(
    full: pd.DataFrame,
    cut: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    left = normalize(full, cutoff)
    right = normalize(cut, cutoff)
    merged = left.merge(
        right,
        on=KEYS,
        how="outer",
        suffixes=("_full", "_cut"),
        indicator=True,
    )
    a = merged["factor_full"].to_numpy(dtype=float)
    b = merged["factor_cut"].to_numpy(dtype=float)
    equal = (np.isnan(a) & np.isnan(b)) | (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isclose(a, b, rtol=1e-5, atol=1e-8)
    )
    equal &= merged["_merge"].eq("both").to_numpy()
    differences = merged.loc[~equal]
    return {
        "status": "ok" if differences.empty else "lookahead_suspected",
        "compared_rows": len(merged),
        "difference_rows": len(differences),
        "first_difference_date": (
            None
            if differences.empty
            else pd.Timestamp(differences["date"].min()).strftime("%Y-%m-%d")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--max-lookback-months", type=int, default=6)
    parser.add_argument("--skip-platform-schema", action="store_true")
    parser.add_argument("--bar1m", default="bigalpha_2026_stock_bar1m")
    parser.add_argument("--financial", default="bigalpha_2026_financial")
    args = parser.parse_args()
    report: dict[str, Any] = {"status": "ok", "submission": args.submission.name}
    try:
        start = pd.Timestamp(args.start).normalize()
        cutoff = pd.Timestamp(args.cutoff).normalize()
        end = pd.Timestamp(args.end).normalize()
        if not start <= cutoff < end:
            raise ValueError("require start <= cutoff < end")
        report["source"] = source_check(
            args.submission,
            args.max_lookback_months,
        )
        if report["source"]["status"] != "ok":
            report["status"] = "source_policy_error"
        if report["status"] == "ok" and not args.skip_platform_schema:
            report["platform_schema"] = platform_schema_check(args.start)
            if report["platform_schema"]["status"] != "ok":
                report["status"] = "platform_schema_error"
        if report["status"] == "ok":
            submission = load_submission(args.submission)
            universe = expected_universe(start, end)
            datasources = {
                "bar1m": args.bar1m,
                "financial": args.financial,
            }
            with query_audit(start, args.max_lookback_months) as full_queries:
                began = time.perf_counter()
                full = submission.main(datasources, args.start, args.end)
                full_seconds = time.perf_counter() - began
            with query_audit(start, args.max_lookback_months) as cut_queries:
                began = time.perf_counter()
                cut = submission.main(datasources, args.start, args.cutoff)
                cut_seconds = time.perf_counter() - began
            checks = {
                "full_output": output_check(full, universe, start, end),
                "cut_output": output_check(cut, universe, start, cutoff),
                "prefix": prefix_check(full, cut, cutoff),
            }
            query_failure = any(
                call["status"] != "ok"
                for call in [*full_queries, *cut_queries]
            )
            check_failure = any(
                result["status"] != "ok" for result in checks.values()
            )
            report["execution"] = {
                "status": (
                    "query_policy_error"
                    if query_failure
                    else ("validation_error" if check_failure else "ok")
                ),
                "full_seconds": round(full_seconds, 3),
                "cut_seconds": round(cut_seconds, 3),
                **checks,
                "queries": {"full": full_queries, "cut": cut_queries},
            }
            if report["execution"]["status"] != "ok":
                report["status"] = report["execution"]["status"]
    except Exception as exc:  # noqa: BLE001
        report["status"] = "runtime_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
