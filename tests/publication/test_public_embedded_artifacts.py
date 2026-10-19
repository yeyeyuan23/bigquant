"""Safe release tests: parse existing sources; execute only the new stdlib loader."""
import ast
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SUBMISSIONS = ROOT / 'submissions'
RUNTIME_ROOTS = (SUBMISSIONS, ROOT / 'experiments' / 'legacy_submissions')


def runtime_paths(pattern):
    return [path for directory in RUNTIME_ROOTS for path in directory.rglob(pattern)]


class PublicEmbeddedArtifactTests(unittest.TestCase):
    def loader(self):
        path = SUBMISSIONS / 'm_raw' / 'public_artifacts.py'
        self.assertTrue(path.is_file(), 'source-only artifact loader is missing')
        spec = importlib.util.spec_from_file_location('public_artifacts_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_no_embedded_trained_payload_modules(self):
        forbidden = [p for p in runtime_paths('*') if p.name.endswith(('_weights.py', '_checkpoint.py')) or p.name == 'weights.json']
        self.assertEqual([], forbidden)

    def test_unset_path_fails_with_external_artifact_contract(self):
        loader = self.loader()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'PUBLIC_TEST_ARTIFACT'):
            loader.artifact_path('PUBLIC_TEST_ARTIFACT')

    def test_missing_or_directory_path_fails(self):
        loader = self.loader()
        with tempfile.TemporaryDirectory() as temporary:
            for value in (temporary, str(Path(temporary) / 'absent.json')):
                with patch.dict(os.environ, {'PUBLIC_TEST_ARTIFACT': value}), self.assertRaises(FileNotFoundError):
                    loader.artifact_path('PUBLIC_TEST_ARTIFACT')

    def test_explicit_environment_json_fixture_and_required_keys(self):
        loader = self.loader()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.json'
            path.write_text('{"config": {}, "state_dict": {}}')
            with patch.dict(os.environ, {'PUBLIC_TEST_ARTIFACT': str(path)}):
                self.assertEqual(path.resolve(), loader.artifact_path('PUBLIC_TEST_ARTIFACT'))
                self.assertEqual({'config': {}, 'state_dict': {}}, loader.load_json_artifact('PUBLIC_TEST_ARTIFACT', required_keys=('config', 'state_dict')))
                with self.assertRaisesRegex(ValueError, 'missing'):
                    loader.load_json_artifact('PUBLIC_TEST_ARTIFACT', required_keys=('preprocessing',))
            self.assertEqual(path.resolve(), loader.artifact_path('PUBLIC_TEST_ARTIFACT', explicit_path=path))

    def test_invalid_json_container_fails(self):
        loader = self.loader()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'invalid.json'
            for content in ('[]', '{broken'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    loader.load_json_artifact('PUBLIC_TEST_ARTIFACT', explicit_path=path)

    def test_torch_loader_requires_safe_mode_and_never_retries_unsafe(self):
        loader = self.loader()
        calls = []
        class SafeTorch:
            @staticmethod
            def load(path, **kwargs):
                calls.append((path, kwargs))
                return {'config': {}, 'state_dict': {}}
        class OldTorch:
            @staticmethod
            def load(path, **kwargs):
                calls.append((path, kwargs))
                raise TypeError('unsupported weights_only')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'synthetic.pt'
            path.write_bytes(b'not a checkpoint: never deserialized')
            result = loader.load_torch_checkpoint(SafeTorch, 'PUBLIC_TEST_ARTIFACT', explicit_path=path)
            self.assertEqual({'config': {}, 'state_dict': {}}, result)
            self.assertEqual({'map_location': 'cpu', 'weights_only': True}, calls[-1][1])
            with self.assertRaisesRegex(RuntimeError, 'weights_only'):
                loader.load_torch_checkpoint(OldTorch, 'PUBLIC_TEST_ARTIFACT', explicit_path=path)
            self.assertEqual(2, len(calls), 'must not retry without weights_only')

    def test_loader_copies_identical(self):
        expected = (SUBMISSIONS / 'm_raw' / 'public_artifacts.py')
        self.assertTrue(expected.exists())
        copies = list(SUBMISSIONS.rglob('public_artifacts.py'))
        self.assertEqual(1, len(copies))
        for copy in copies:
            self.assertEqual(expected.read_bytes(), copy.read_bytes(), str(copy))

    def test_all_torch_load_calls_require_weights_only(self):
        for path in runtime_paths('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load':
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                    self.assertIn('weights_only', keywords, str(path))
                    self.assertIsInstance(keywords['weights_only'], ast.Constant, str(path))
                    self.assertIs(keywords['weights_only'].value, True, str(path))

    def test_source_manifest_exposes_each_external_artifact_contract(self):
        manifest = json.loads((SUBMISSIONS / 'source_manifest.json').read_text())
        self.assertEqual({'m_raw'}, set(manifest['routes']))
        self.assertEqual({'m_raw'}, {p.name for p in SUBMISSIONS.iterdir() if p.is_dir() and not p.name.startswith('.') and p.name != '__pycache__'})
        self.assertEqual('source_only', manifest['distribution'])
        self.assertEqual('not_included', manifest['trained_artifacts'])
        self.assertEqual('not_run', manifest['runtime_validation'])
        contract = (SUBMISSIONS / manifest['artifact_contract']).read_text()
        for directory, route in manifest['routes'].items():
            self.assertTrue((SUBMISSIONS / directory / route['entrypoint']).is_file())
            variable = route['external_artifact_env']
            if variable:
                source = '\n'.join(p.read_text() for p in (SUBMISSIONS / directory).glob('*.py'))
                self.assertIn(variable, source, directory)
                self.assertIn(variable, contract, directory)

    def test_no_encoded_checkpoints_or_numeric_payload_arrays(self):
        def inspect_source(source, label):
            self.assertNotIn('b85decode', source, label)
            self.assertNotIn('b64decode', source, label)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 25:
                    numeric = all(isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) for value in node.elts)
                    self.assertFalse(numeric, f'{label}:{node.lineno}: numeric payload array')
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and len(node.value.value) > 30000:
                    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                    self.assertTrue(names and all(name.endswith('_SOURCE') for name in names), label)
                    inspect_source(node.value.value, label + ':embedded_source')
        for path in runtime_paths('*.py'):
            inspect_source(path.read_text(), str(path))

    def test_sources_and_notebook_cells_parse_without_execution(self):
        for path in runtime_paths('*.py'):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id.endswith('_SOURCE') for target in node.targets) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    ast.parse(node.value.value, filename=str(path) + ':embedded_source')
        for path in runtime_paths('*.json'):
            json.loads(path.read_text())
        for path in runtime_paths('*.ipynb'):
            notebook = json.loads(path.read_text())
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    self.assertEqual([], cell.get('outputs', []), str(path))
                    self.assertIsNone(cell.get('execution_count'), str(path))
                    ast.parse(''.join(cell['source']), filename=str(path))


if __name__ == '__main__':
    unittest.main()
