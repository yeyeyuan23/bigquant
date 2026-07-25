from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from bigalpha2026.synthetic import make_synthetic_datasources


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_standalone_notebooks_define_working_main(self) -> None:
        datasources = make_synthetic_datasources(
            days=5,
            instruments=6,
            minutes_per_day=5,
        )
        universe = datasources["bigalpha_2026_instruments"]
        for path in sorted((ROOT / "submissions").glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join(
                "".join(cell["source"])
                if isinstance(cell["source"], list)
                else cell["source"]
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            namespace: dict[str, object] = {}
            exec(compile(code, str(path), "exec"), namespace)
            self.assertIn("main", namespace)
            if path.stem.endswith("_v2"):
                # These two notebooks intentionally mirror the official template
                # and require BigQuant's injected logical tables plus the DAI
                # runtime. Their real-data contract is exercised in AIStudio.
                self.assertIn('datasources["bar1m"]', code)
                self.assertIn("bigalpha_2026_instruments", code)
                continue
            frame = namespace["main"](  # type: ignore[operator]
                datasources,
                universe["date"].min(),
                universe["date"].max(),
            )
            self.assertEqual(list(frame.columns), ["date", "instrument", "factor"])
            self.assertTrue(np.isfinite(frame["factor"]).all())


if __name__ == "__main__":
    unittest.main()
