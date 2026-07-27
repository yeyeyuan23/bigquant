import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "factor_int_001_equal_rank.py"
NOTEBOOK = ROOT / "submissions" / "factor_int_001_equal_rank.ipynb"
JOINT_SOURCE = ROOT / "submissions" / "factor_joint_elastic_net.py"
JOINT_NOTEBOOK = ROOT / "submissions" / "factor_joint_elastic_net.ipynb"
SELF_PV_SOURCE = ROOT / "submissions" / "factor_self_pv_rank.py"
SELF_PV_NOTEBOOK = ROOT / "submissions" / "factor_self_pv_rank.ipynb"
FINAL_SUBMISSIONS = (
    (
        ROOT / "submissions" / "factor_self_family_rank.py",
        ROOT / "submissions" / "factor_self_family_rank.ipynb",
    ),
    (
        ROOT / "submissions" / "factor_joint_lightgbm.py",
        ROOT / "submissions" / "factor_joint_lightgbm.ipynb",
    ),
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

    def test_joint_source_is_self_contained_and_has_main(self):
        tree = ast.parse(JOINT_SOURCE.read_text(encoding="utf-8"))
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

    def test_joint_notebook_has_one_code_cell_and_exact_source(self):
        notebook = json.loads(JOINT_NOTEBOOK.read_text(encoding="utf-8"))
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_cells), 1)
        self.assertEqual(
            "".join(code_cells[0]["source"]),
            JOINT_SOURCE.read_text(encoding="utf-8"),
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

    def test_lightgbm_submission_matches_frozen_tree_pool(self):
        source_path = ROOT / "submissions" / "factor_joint_lightgbm.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        self_columns = None
        for node in ast.walk(main):
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "self_columns"
                for target in node.targets
            ):
                self_columns = tuple(ast.literal_eval(node.value))
                break
        self.assertIsNotNone(self_columns)

        frozen_state = json.loads(
            (
                ROOT
                / "data"
                / "cache"
                / "tree_v2"
                / "frozen"
                / "frozen_state.json"
            ).read_text(encoding="utf-8")
        )
        expected = tuple(
            candidate.removeprefix("self__")
            for candidate in frozen_state["frozen_candidates"]
        )
        self.assertEqual(self_columns, expected)

        source = source_path.read_text(encoding="utf-8")
        for required in (
            'panel["HF-003"]',
            'panel["OB-005"]',
            "downside_realized_volatility",
            "tail_60_microprice_gap_median",
            "tail_60_microprice_gap_sign_consistency",
            '"label_observed_date": all_dates[1:]',
            '"date": all_dates[:-1]',
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn(".shift(-", source)
        self.assertNotIn(" lead(", source.lower())


if __name__ == "__main__":
    unittest.main()
