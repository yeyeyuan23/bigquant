from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
REMOTE_SUBMISSIONS = ROOT / "remote_submission_notebooks"
EXPECTED_FILES = {
    "README.md",
    "smoke_v01.py",
    "smoke_v01.ipynb",
    "rule_v03.py",
    "rule_v03.ipynb",
    "lgbm_platform_top_v01.py",
    "lgbm_platform_top_v01.ipynb",
    "enet_i_54_candidate.py",
    "enet_i_54_candidate.ipynb",
    "lgbm_t_orthogonal_28_candidate.py",
    "lgbm_t_orthogonal_28_candidate.ipynb",
    "lgbm_t_orthogonal_28_no15_candidate.py",
    "lgbm_t_orthogonal_28_no15_candidate.ipynb",
    "lgbm_t_orthogonal_28_add15_candidate.py",
    "lgbm_t_orthogonal_28_add15_candidate.ipynb",
}


class SubmissionTest(unittest.TestCase):
    def test_directory_contains_only_retained_submissions(self):
        actual = {path.name for path in SUBMISSIONS.iterdir() if path.is_file()}
        self.assertEqual(actual, EXPECTED_FILES)

    def test_every_source_has_competition_main(self):
        for source_path in sorted(SUBMISSIONS.glob("*.py")):
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

    def test_embedded_candidate_transforms_execute_in_isolation(self):
        sources = [
            *sorted(SUBMISSIONS.glob("*_candidate.py")),
            *sorted(REMOTE_SUBMISSIONS.glob("*_candidate.py")),
        ]
        smoke = r"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
installer = next(
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "_install_bigalpha_candidate_modules"
)
namespace = {}
exec(
    compile(ast.Module(body=[installer], type_ignores=[]), str(path), "exec"),
    namespace,
)
namespace["_install_bigalpha_candidate_modules"]()
from bigalpha2026.candidate_transforms import daily_median_centered_rank

frame = pd.DataFrame(
    {
        "date": pd.to_datetime(["2023-01-03"] * 3),
        "factor_raw": [1.0, np.nan, 3.0],
    }
)
result = daily_median_centered_rank(frame)
assert result.tolist() == [-2.0 / 3.0, 0.0, 2.0 / 3.0], result.tolist()
"""
        for source_path in sources:
            if "_install_bigalpha_candidate_modules" not in source_path.read_text(
                encoding="utf-8"
            ):
                continue
            with self.subTest(source=source_path.name):
                subprocess.run(
                    [sys.executable, "-c", smoke, str(source_path)],
                    check=True,
                    cwd=ROOT,
                )

    def test_current_learned_models_keep_frozen_training_contracts(self):
        enet = (
            SUBMISSIONS / "enet_i_54_candidate.py"
        ).read_text(encoding="utf-8")
        lgbm = (
            SUBMISSIONS / "lgbm_t_orthogonal_28_candidate.py"
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
        self.assertIn("screened15_lambda = 0.0", lgbm)
        self.assertIn(
            "screened15_lambda * baseline_prediction",
            lgbm,
        )
        self.assertIn("def _iter_bar5m_parts(", lgbm)
        self.assertIn("yield part", lgbm)
        self.assertIn("gc.collect()", lgbm)
        self.assertNotIn("return pd.concat(parts, ignore_index=True)", lgbm)
        self.assertIn('"exposures": exposure', lgbm)
        self.assertIn("inspect.signature(builder)", lgbm)
        self.assertIn("return builder(*arguments)", lgbm)


if __name__ == "__main__":
    unittest.main()
