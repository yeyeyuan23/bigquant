import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "factor_int_001_equal_rank.py"
NOTEBOOK = ROOT / "submissions" / "factor_int_001_equal_rank.ipynb"
JOINT_SOURCE = ROOT / "submissions" / "factor_joint_elastic_net.py"
JOINT_NOTEBOOK = ROOT / "submissions" / "factor_joint_elastic_net.ipynb"


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


if __name__ == "__main__":
    unittest.main()
