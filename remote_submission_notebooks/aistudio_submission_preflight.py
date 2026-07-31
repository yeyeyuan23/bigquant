"""Fast preflight checks for self-contained BigAlpha submission sources.

The default checks are local and do not execute ``main``.  ``--platform-schema``
adds tiny ``LIMIT 1`` queries and therefore must run inside AIStudio.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import inspect
import json
import symtable
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUNTIME_GLOBALS = {
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}
SUPPORTED_BUILDER_INPUTS = {
    "bars",
    "daily_bars",
    "daily_features",
    "exposure",
    "exposures",
    "factorlib",
    "financial",
    "financial_panel",
    "micro",
    "micro_daily",
    "pool",
    "pv",
}


def _main_signature(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ):
            return [argument.arg for argument in node.args.args]
    raise ValueError("submission has no main()")


def _installer_node(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_install_bigalpha_candidate_modules"
        ):
            return node
    raise ValueError("submission has no embedded candidate-module installer")


def _embedded_sources(
    installer: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    for node in ast.walk(installer):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "sources" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(source, str)
            for key, source in value.items()
        ):
            raise TypeError("embedded sources must be a string-to-string dictionary")
        return value
    raise ValueError("installer has no literal sources dictionary")


def _bound_module_names(table: symtable.SymbolTable) -> set[str]:
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }


def _undefined_globals(source: str, filename: str) -> list[str]:
    table = symtable.symtable(source, filename, "exec")
    allowed = set(dir(builtins)) | RUNTIME_GLOBALS | _bound_module_names(table)
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


def _literal_assignment(tree: ast.Module, name: str) -> tuple[str, ...]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
            return tuple(value)
    return ()


def _check_notebook_pair(source_path: Path, source: str) -> dict[str, Any]:
    notebook_path = source_path.with_suffix(".ipynb")
    if not notebook_path.exists():
        return {"status": "skipped", "reason": "adjacent notebook not found"}
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    if len(code_cells) != 1:
        raise ValueError(f"{notebook_path.name} must contain exactly one code cell")
    if code_cells[0] != source:
        raise ValueError(f"{notebook_path.name} code differs from {source_path.name}")
    return {"status": "ok", "notebook": notebook_path.name}


def _install_embedded_modules(
    installer: ast.FunctionDef | ast.AsyncFunctionDef,
    source_path: Path,
) -> None:
    for name in tuple(sys.modules):
        if name == "bigalpha2026" or name.startswith("bigalpha2026."):
            del sys.modules[name]
    namespace: dict[str, Any] = {}
    isolated = ast.Module(body=[installer], type_ignores=[])
    exec(compile(isolated, str(source_path), "exec"), namespace)  # noqa: S102
    namespace["_install_bigalpha_candidate_modules"]()


def _check_embedded_runtime(sources: dict[str, str]) -> dict[str, Any]:
    transforms = importlib.import_module("bigalpha2026.candidate_transforms")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03"] * 3),
            "factor_raw": [1.0, np.nan, 3.0],
        }
    )
    result = transforms.daily_median_centered_rank(frame)
    expected = np.asarray([-2.0 / 3.0, 0.0, 2.0 / 3.0])
    if not np.allclose(result.to_numpy(dtype=float), expected):
        raise ValueError(f"embedded rank smoke returned {result.tolist()}")

    checked_builders = 0
    for module_name in sources:
        if ".candidates." not in module_name or module_name.endswith("._common"):
            continue
        module = importlib.import_module(module_name)
        stem = module_name.rsplit(".", maxsplit=1)[-1]
        function = None
        function_name = ""
        for suffix in ("factor_from_daily", "factor_from_panel", "factor"):
            function_name = f"build_{stem}_{suffix}"
            function = getattr(module, function_name, None)
            if function is not None:
                break
        if function is None:
            raise ValueError(f"{module_name} has no submission builder")
        required = {
            parameter.name
            for parameter in inspect.signature(function).parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and parameter.default is inspect.Parameter.empty
        }
        unsupported = sorted(required - SUPPORTED_BUILDER_INPUTS)
        if unsupported:
            raise ValueError(
                f"{module_name}.{function_name} has unsupported required "
                f"parameters: {unsupported}"
            )
        checked_builders += 1
    return {"status": "ok", "checked_builders": checked_builders}


def static_preflight(source_path: Path) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    signature = _main_signature(tree)
    if signature != ["datasources", "start_date", "end_date"]:
        raise ValueError(f"unexpected main signature: {signature}")
    installer = _installer_node(tree)
    sources = _embedded_sources(installer)
    undefined: dict[str, list[str]] = {}
    for module_name, module_source in sources.items():
        compile(module_source, module_name, "exec")
        missing = _undefined_globals(module_source, module_name)
        if missing:
            undefined[module_name] = missing
    if undefined:
        raise NameError(f"undefined embedded globals: {undefined}")
    notebook = _check_notebook_pair(source_path, source)
    _install_embedded_modules(installer, source_path)
    runtime = _check_embedded_runtime(sources)
    return {
        "status": "ok",
        "submission": source_path.name,
        "embedded_modules": len(sources),
        "notebook": notebook,
        "runtime": runtime,
    }


def _schema_specs(tree: ast.Module, financial_table: str) -> list[tuple[str, tuple[str, ...]]]:
    public_columns = _literal_assignment(tree, "public_columns")
    specs: list[tuple[str, tuple[str, ...]]] = [
        ("bigalpha_2026_instruments", ("date", "instrument")),
        (
            "bigalpha_2026_factorlib",
            ("date", "instrument", "daily_return", *public_columns),
        ),
        (
            "bigalpha_2026_exposure",
            ("date", "instrument", "float_market_cap", "industry_level1_code"),
        ),
        (
            financial_table,
            (
                "date",
                "instrument",
                "category",
                "shift",
                "report_date",
                "net_cffoa",
                "net_profit",
                "operating_revenue",
                "total_assets",
            ),
        ),
    ]
    source = ast.unparse(tree)
    if "bigalpha_2026_stock_bar5m" in source:
        specs.append(
            (
                "bigalpha_2026_stock_bar5m",
                (
                    "date",
                    "instrument",
                    "pre_close",
                    "open",
                    "high",
                    "low",
                    "close",
                    "amount",
                    "volume",
                    "deal_number",
                    *(f"ask_price{level}" for level in range(1, 6)),
                    *(f"bid_price{level}" for level in range(1, 6)),
                    *(f"ask_volume{level}" for level in range(1, 6)),
                    *(f"bid_volume{level}" for level in range(1, 6)),
                ),
            )
        )
    return specs


def platform_schema_preflight(
    source_path: Path,
    financial_table: str,
    schema_date: str,
) -> dict[str, Any]:
    try:
        import dai
    except ImportError as exc:
        raise RuntimeError("--platform-schema must run inside AIStudio") from exc

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    results: list[dict[str, Any]] = []
    failures = 0
    for table, columns in _schema_specs(tree, financial_table):
        sql = f"SELECT {', '.join(dict.fromkeys(columns))} FROM {table} LIMIT 1"
        try:
            frame = dai.query(
                sql,
                filters={"date": [schema_date, schema_date]},
                compression=True,
            ).df()
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                raise ValueError(f"query result missing columns: {missing}")
            results.append(
                {
                    "status": "ok",
                    "table": table,
                    "columns": list(columns),
                    "rows": len(frame),
                }
            )
        except Exception as exc:  # noqa: BLE001 - platform errors vary by release
            failures += 1
            results.append(
                {
                    "status": "error",
                    "table": table,
                    "columns": list(columns),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "ok" if failures == 0 else "error",
        "submission": source_path.name,
        "tables": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--platform-schema", action="store_true")
    parser.add_argument(
        "--financial",
        default="bigalpha_2026_financial",
        help="financial table supplied to submission datasources",
    )
    parser.add_argument(
        "--schema-date",
        default="2023-01-04",
        help="single partition date used by --platform-schema",
    )
    args = parser.parse_args()
    report: dict[str, Any] = {"status": "ok"}
    try:
        report["static"] = static_preflight(args.submission)
        if args.platform_schema:
            report["platform_schema"] = platform_schema_preflight(
                args.submission,
                args.financial,
                args.schema_date,
            )
            if report["platform_schema"]["status"] != "ok":
                report["status"] = "error"
    except Exception as exc:  # noqa: BLE001 - CLI must emit structured failures
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
