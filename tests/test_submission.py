import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "factor_int_001_equal_rank.py"
NOTEBOOK = ROOT / "submissions" / "factor_int_001_equal_rank.ipynb"
SELF_PV_SOURCE = ROOT / "submissions" / "factor_self_pv_rank.py"
SELF_PV_NOTEBOOK = ROOT / "submissions" / "factor_self_pv_rank.ipynb"
FINAL_SUBMISSIONS = (
    (
        ROOT / "submissions" / "factor_self_family_rank.py",
        ROOT / "submissions" / "factor_self_family_rank.ipynb",
    ),
)
CURRENT_LEARNED_SUBMISSIONS = (
    (
        ROOT / "submissions" / "factor_joint_elastic_net_current.py",
        ROOT / "submissions" / "factor_joint_elastic_net_current.ipynb",
        ("PV-014", "FR-002"),
    ),
    (
        ROOT / "submissions" / "factor_joint_lightgbm_current.py",
        ROOT / "submissions" / "factor_joint_lightgbm_current.ipynb",
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
                self.assertNotIn(".shift(-", source)
                self.assertNotIn(" lead(", source.lower())
        elastic_source = CURRENT_LEARNED_SUBMISSIONS[0][0].read_text(
            encoding="utf-8"
        )
        lightgbm_source = CURRENT_LEARNED_SUBMISSIONS[1][0].read_text(
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
            "intraday_end_ts = (",
            lightgbm_source,
        )
        self.assertIn(
            "month_end = min(intraday_end_ts, cursor.end_time)",
            lightgbm_source,
        )
        self.assertNotIn(
            "month_end = min(end_ts, cursor.end_time.normalize())",
            lightgbm_source,
        )

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
