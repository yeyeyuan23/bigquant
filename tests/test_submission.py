import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "smoke_v01.py"
NOTEBOOK = ROOT / "submissions" / "smoke_v01.ipynb"
SELF_PV_SOURCE = ROOT / "submissions" / "rule_v01.py"
SELF_PV_NOTEBOOK = ROOT / "submissions" / "rule_v01.ipynb"
FINAL_SUBMISSIONS = (
    (
        ROOT / "submissions" / "rule_v02.py",
        ROOT / "submissions" / "rule_v02.ipynb",
    ),
    (
        ROOT / "submissions" / "rule_v03.py",
        ROOT / "submissions" / "rule_v03.ipynb",
    ),
    (
        ROOT / "submissions" / "rule_v04.py",
        ROOT / "submissions" / "rule_v04.ipynb",
    ),
)
CURRENT_LEARNED_SUBMISSIONS = (
    (
        ROOT / "submissions" / "enet_v01.py",
        ROOT / "submissions" / "enet_v01.ipynb",
        ("PV-014", "FR-002"),
    ),
    (
        ROOT / "submissions" / "lgbm_v01.py",
        ROOT / "submissions" / "lgbm_v01.ipynb",
        (
            "FR-002",
            "FR-005",
            "FR-015",
            "HF-003",
            "HF-004",
            "OB-001",
            "OB-003",
            "PV-001",
            "PV-009",
            "PV-014",
            "PV-020",
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v02.py",
        ROOT / "submissions" / "lgbm_v02.ipynb",
        (
            "FR-002",
            "FR-005",
            "FR-015",
            "HF-003",
            "HF-004",
            "OB-001",
            "OB-003",
            "PV-001",
            "PV-009",
            "PV-014",
            "PV-020",
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v03.py",
        ROOT / "submissions" / "lgbm_v03.ipynb",
        (
            "PV-008",
            "OB-003",
            "PV-020",
            "PV-013",
        ),
    ),
    (
        ROOT / "submissions" / "enet_v03.py",
        ROOT / "submissions" / "enet_v03.ipynb",
        (
            "FR-002",
            "FR-004",
            "FR-005",
            "FR-006",
            "FR-010",
            "FR-011",
            "FR-012",
            "FR-013",
            "FR-014",
            "FR-015",
            "HF-001",
            "HF-003",
            "INT-002",
            "INT-003",
            "OB-001",
            "OB-003",
            "OB-004",
            "PV-003",
            "PV-004",
            "PV-005",
            "PV-008",
            "PV-009",
            "PV-010",
            "PV-011",
            "PV-013",
            "PV-014",
            "PV-015",
            "PV-016",
            "PV-017",
            "PV-019",
            "PV-020",
            "PV-021",
            "PV-022",
            "HF-004",
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_v04.py",
        ROOT / "submissions" / "lgbm_v04.ipynb",
        (
            "FR-013",
            "INT-002",
            "OB-004",
            "PV-009",
            "OB-003",
            "FR-012",
            "PV-017",
            "PV-019",
            "HF-001",
            "PV-010",
            "FR-002",
            "PV-016",
            "FR-005",
            "PV-004",
            "PV-015",
            "FR-004",
            "PV-011",
            "PV-021",
            "INT-003",
            "OB-005",
            "PV-023",
            "FR-001",
            "HF-004",
            "PV-002",
            "OB-002",
            "OB-001",
            "FR-010",
        ),
    ),
    (
        ROOT / "submissions" / "lgbm_platform_top_v01.py",
        ROOT / "submissions" / "lgbm_platform_top_v01.ipynb",
        (
            "FR-002",
            "FR-004",
            "FR-011",
            "HF-002",
            "PV-001",
            "PV-002",
            "PV-003",
            "PV-006",
            "PV-014",
            "HF-003",
            "OB-005",
        ),
    ),
)
PLATFORM_SAFE_SUBMISSIONS = (
    ROOT / "submissions" / "rule_v03.py",
    ROOT / "submissions" / "enet_v01.py",
    ROOT / "submissions" / "enet_v02.py",
    ROOT / "submissions" / "lgbm_v02.py",
    ROOT / "submissions" / "lgbm_v03.py",
)
ALLOWED_COMPETITION_TABLES = {
    "bigalpha_2026_exposure",
    "bigalpha_2026_factorlib",
    "bigalpha_2026_financial",
    "bigalpha_2026_instruments",
    "bigalpha_2026_stock_bar1m",
}
FORBIDDEN_PLATFORM_TABLES = {
    "all_trading_days",
    "cn_stock_bar1d",
    "cn_stock_bar1m",
}
SQL_TABLE_PATTERN = re.compile(
    r"^\s*(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    flags=re.IGNORECASE | re.MULTILINE,
)
SQL_CTE_PATTERN = re.compile(
    r"(?:\bwith\b|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(",
    flags=re.IGNORECASE,
)
SQL_FIRST_LAST_PATTERN = re.compile(
    r"\b(first|last)\s*\(([^)]*)\)",
    flags=re.IGNORECASE | re.DOTALL,
)


class SubmissionTest(unittest.TestCase):
    def test_source_is_self_contained_and_has_main(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("main", functions)
        self.assertEqual(
            [argument.arg for argument in functions["main"].args.args],
            ["datasources", "start_date", "end_date"],
        )

    def test_notebook_has_one_code_cell_and_exact_source(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_cells), 1)
        self.assertEqual(
            "".join(code_cells[0]["source"]),
            SOURCE.read_text(encoding="utf-8"),
        )

    def test_self_pv_source_is_self_contained_and_has_main(self):
        tree = ast.parse(SELF_PV_SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("main", functions)
        self.assertEqual(
            [argument.arg for argument in functions["main"].args.args],
            ["datasources", "start_date", "end_date"],
        )

    def test_self_pv_notebook_has_one_code_cell_and_exact_source(self):
        notebook = json.loads(SELF_PV_NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_cells), 1)
        self.assertEqual(
            "".join(code_cells[0]["source"]),
            SELF_PV_SOURCE.read_text(encoding="utf-8"),
        )

    def test_final_submission_sources_and_notebooks_match(self):
        for source, notebook_path in FINAL_SUBMISSIONS:
            with self.subTest(source=source.name):
                tree = ast.parse(source.read_text(encoding="utf-8"))
                functions = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                }
                self.assertIn("main", functions)
                self.assertEqual(
                    [argument.arg for argument in functions["main"].args.args],
                    ["datasources", "start_date", "end_date"],
                )
                notebook = json.loads(
                    notebook_path.read_text(encoding="utf-8")
                )
                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertEqual(len(code_cells), 1)
                self.assertEqual(
                    "".join(code_cells[0]["source"]),
                    source.read_text(encoding="utf-8"),
                )

    def test_current_learned_submissions_match_frozen_pools(self):
        for source_path, notebook_path, expected_members in (
            CURRENT_LEARNED_SUBMISSIONS
        ):
            with self.subTest(source=source_path.name):
                source = source_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                main = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main"
                )
                self_columns = next(
                    ast.literal_eval(node.value)
                    for node in ast.walk(main)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "self_columns"
                        for target in node.targets
                    )
                )
                self.assertEqual(tuple(self_columns), expected_members)
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertEqual(len(code_cells), 1)
                self.assertEqual("".join(code_cells[0]["source"]), source)
                if source_path.name in {"enet_v03.py", "lgbm_v04.py"}:
                    self.assertIn(
                        "PARTITION BY instrument, trading_day, session_id",
                        source,
                    )
                    self.assertIn("shift(-recovery_minutes)", source)
                    self.assertNotIn("target.shift(-", source.lower())
                    self.assertNotIn("daily_return.shift(-", source.lower())
                else:
                    self.assertNotIn(".shift(-", source)
                    self.assertNotIn(" lead(", source.lower())
        elastic_source = CURRENT_LEARNED_SUBMISSIONS[0][0].read_text(
            encoding="utf-8"
        )
        lightgbm_source = CURRENT_LEARNED_SUBMISSIONS[2][0].read_text(
            encoding="utf-8"
        )
        self.assertIn("positive=True", elastic_source)
        self.assertIn("monotone_constraints=[1] * len(feature_columns)", lightgbm_source)
        for source in (elastic_source, lightgbm_source):
            self.assertNotIn("cn_stock_bar1d", source)
            self.assertNotIn("cn_stock_bar1m", source)

        self.assertIn("bigalpha_2026_factorlib", elastic_source)
        self.assertIn("bigalpha_2026_stock_bar1m", lightgbm_source)
        self.assertIn(
            "first(open ORDER BY timestamp) AS open",
            lightgbm_source,
        )
        self.assertIn(
            "last(close ORDER BY timestamp) AS close",
            lightgbm_source,
        )
        self.assertIn(
            "first(pre_close ORDER BY timestamp) AS pre_close",
            lightgbm_source,
        )
        self.assertIn(
            "if cursor == final_period",
            lightgbm_source,
        )
        self.assertIn(
            'end_ts.strftime("%Y-%m-%d 23:59:59")',
            lightgbm_source,
        )
        self.assertNotIn("intraday_end_ts", lightgbm_source)
        self.assertNotIn("end_ts + pd.Timedelta(days=1)", lightgbm_source)
        self.assertIn('filters={"date": [financial_start, end_date]}', lightgbm_source)

    def test_platform_safe_submissions_do_not_query_after_end_date(self):
        for source_path in PLATFORM_SAFE_SUBMISSIONS:
            with self.subTest(source=source_path.name):
                source = source_path.read_text(encoding="utf-8")
                self.assertNotRegex(source, r"\bend_ts\s*\+")
                self.assertNotRegex(source, r"\bend_date\s*\+")
                self.assertNotIn("pd.Timedelta(days=1)", source)

    def test_platform_safe_first_last_aggregations_are_deterministic(self):
        for source_path in PLATFORM_SAFE_SUBMISSIONS:
            source = source_path.read_text(encoding="utf-8")
            for match in SQL_FIRST_LAST_PATTERN.finditer(source):
                function_name, arguments = match.groups()
                with self.subTest(
                    source=source_path.name,
                    aggregation=function_name,
                    arguments=arguments.strip(),
                ):
                    self.assertIn("ORDER BY", arguments.upper())

    def test_every_submission_uses_only_competition_tables(self):
        sources = sorted((ROOT / "submissions").glob("*.py"))
        self.assertTrue(sources)
        for source_path in sources:
            with self.subTest(source=source_path.name):
                source = source_path.read_text(encoding="utf-8")
                notebook_path = source_path.with_suffix(".ipynb")
                self.assertTrue(notebook_path.exists())
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                code_cells = [
                    cell
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
                self.assertEqual(len(code_cells), 1)
                self.assertEqual("".join(code_cells[0]["source"]), source)
                lowered = source.lower()
                for forbidden in FORBIDDEN_PLATFORM_TABLES:
                    self.assertNotIn(forbidden, lowered)
                sql_strings = [
                    node.value
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "SELECT" in node.value
                    and "FROM" in node.value
                ]
                cte_names = {
                    match.group(1).lower()
                    for sql in sql_strings
                    for match in SQL_CTE_PATTERN.finditer(sql)
                }
                literal_tables = {
                    match.group(1).lower()
                    for sql in sql_strings
                    for match in SQL_TABLE_PATTERN.finditer(sql)
                }
                self.assertEqual(
                    literal_tables
                    - cte_names
                    - ALLOWED_COMPETITION_TABLES,
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
