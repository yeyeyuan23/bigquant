"""Exercise relocated source assembly without loading ML or platform runtimes."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class MultiAxisBundleSourceTests(unittest.TestCase):
    def test_relocated_builder_assembles_standalone_model_and_trainer(self):
        path = ROOT / 'scripts' / 'build_m_multiaxis_submission.py'
        spec = importlib.util.spec_from_file_location('multiaxis_builder_under_test', path)
        module = importlib.util.module_from_spec(spec)
        # These imports are used by the private CLI, not by source assembly.
        with patch.dict(sys.modules, {'torch': ModuleType('torch'),
                                      'nbformat': ModuleType('nbformat')}):
            spec.loader.exec_module(module)
        source = module._submission_train_source(ROOT)
        tree = ast.parse(source)
        definitions = [node.name for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        for name in ('MultiAxisNetwork', 'MultiAxisConfig', 'train', 'load_model'):
            self.assertEqual(1, definitions.count(name), name)
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and node.module in
                             {'alpha_models.multiaxis', 'unified_m_multiaxis_model'}
                             for node in ast.walk(tree)),
                         'generated upload source must contain its own model')


if __name__ == '__main__':
    unittest.main()
