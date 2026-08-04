from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
EXPECTED_FILES = {
    "README.md",
    "smoke_v01.py",
    "smoke_v01.ipynb",
    "rule_v03.py",
    "rule_v03.ipynb",
    "lgbm_platform_top_v01.py",
    "lgbm_platform_top_v01.ipynb",
    "enet_i_51_candidate.py",
    "enet_i_51_candidate.ipynb",
    "enet_i_51_candidate_deps.py",
    "lgbm_t_orthogonal_26_candidate.py",
    "lgbm_t_orthogonal_26_candidate.ipynb",
    "lgbm_t_orthogonal_26_candidate_deps.py",
    "lgbm_t_orthogonal_26_no15_candidate.py",
    "lgbm_t_orthogonal_26_no15_candidate.ipynb",
    "lgbm_t_orthogonal_26_no15_candidate_deps.py",
    "lgbm_t_orthogonal_26_add15_candidate.py",
    "lgbm_t_orthogonal_26_add15_candidate.ipynb",
    "lgbm_t_orthogonal_26_add15_candidate_deps.py",
}


class SubmissionTest(unittest.TestCase):
    def test_directory_contains_only_retained_submissions(self):
        actual = {path.name for path in SUBMISSIONS.iterdir() if path.is_file()}
        self.assertEqual(actual, EXPECTED_FILES)

    def test_every_source_has_competition_main(self):
        for source_path in sorted(SUBMISSIONS.glob("*.py")):
            if source_path.name.endswith("_deps.py"):
                continue
            with self.subTest(source=source_path.name):
                tree = ast.parse(source_path.read_text(encoding="utf-8"))
                main = next(
                    (
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "main"
                    ),
                    None,
                )
                self.assertIsNotNone(main)
                self.assertEqual(
                    [argument.arg for argument in main.args.args],
                    ["datasources", "start_date", "end_date"],
                )

    def test_every_notebook_has_one_exact_code_cell(self):
        for source_path in sorted(SUBMISSIONS.glob("*.py")):
            if source_path.name.endswith("_deps.py"):
                continue
            with self.subTest(source=source_path.name):
                notebook_path = source_path.with_suffix(".ipynb")
                self.assertTrue(notebook_path.exists())
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
                    source_path.read_text(encoding="utf-8"),
                )

    def test_candidate_submissions_use_flat_python_dependency(self):
        sources = sorted(SUBMISSIONS.glob("*_candidate.py"))
        for source_path in sources:
            with self.subTest(source=source_path.name):
                source = source_path.read_text(encoding="utf-8")
                self.assertNotIn("_install_bigalpha_candidate_modules", source)
                self.assertNotIn("exec(compile(", source)
                dependency = source_path.with_name(
                    source_path.stem + "_deps.py"
                )
                self.assertTrue(dependency.is_file())
                dependency_text = dependency.read_text(encoding="utf-8")
                self.assertNotIn("exec(", dependency_text)
                dependency_tree = ast.parse(dependency_text)
                package_imports = [
                    node
                    for node in ast.walk(dependency_tree)
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        and node.module.startswith("bigalpha2026")
                    )
                    or (
                        isinstance(node, ast.Import)
                        and any(
                            alias.name.startswith("bigalpha2026")
                            for alias in node.names
                        )
                    )
                ]
                self.assertEqual(package_imports, [])
                self.assertIn(
                    f"from {dependency.stem} import get_candidate_spec",
                    source,
                )

                smoke = (
                    "import importlib,sys;"
                    f"sys.path.insert(0,{str(source_path.parent)!r});"
                    f"m=importlib.import_module({dependency.stem!r});"
                    "assert m.CANDIDATE_SPECS;"
                    "assert all(callable(v[0]) for v in m.CANDIDATE_SPECS.values())"
                )
                subprocess.run(
                    [sys.executable, "-c", smoke],
                    check=True,
                    cwd=ROOT,
                )

    def test_current_generated_learned_models_keep_frozen_training_contracts(self):
        enet = (SUBMISSIONS / "enet_i_51_candidate.py").read_text(encoding="utf-8")
        lgbm = (
            SUBMISSIONS / "lgbm_t_orthogonal_26_add15_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from sklearn.linear_model import ElasticNet", enet)
        self.assertIn("positive=True", enet)
        self.assertNotIn("LGBMRegressor", enet)
        self.assertIn("from lightgbm import LGBMRegressor", lgbm)
        for source in (enet, lgbm):
            self.assertIn("eligible_history[-60:]", source)
            self.assertIn(
                "for offset in range(0, len(prediction_dates), 20)",
                source,
            )
            self.assertIn("feature_columns = tuple(self_columns)", source)
            self.assertIn(
                "residual_baseline_columns = tuple(public_columns)",
                source,
            )
            self.assertIn('train["target_residual"]', source)
            self.assertNotIn(
                "feature_columns = (*public_columns, *self_columns)",
                source,
            )
        self.assertIn("screened15_lambda = 0.0", enet)
        self.assertIn("screened15_lambda = 1.0", lgbm)
        for source in (enet, lgbm):
            self.assertIn("screened15_lambda * baseline_prediction", source)
        self.assertIn("def _iter_bar5m_parts(", lgbm)
        self.assertIn("yield part", lgbm)
        self.assertIn("gc.collect()", lgbm)
        self.assertNotIn("return pd.concat(parts, ignore_index=True)", lgbm)
        self.assertIn('"exposures": exposure', lgbm)
        self.assertIn("inspect.signature(builder)", lgbm)
        self.assertIn("return builder(*arguments)", lgbm)


if __name__ == "__main__":
    unittest.main()
