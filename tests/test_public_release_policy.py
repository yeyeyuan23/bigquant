"""Public snapshot checks: standard library only; never import research code."""
from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import tempfile
import unittest
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_public_release.py'
REPOSITORY = SCRIPT.parents[1]


class PublicReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self):
        spec = importlib.util.spec_from_file_location('public_release_check', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.scan_tree(self.root)

    def put(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_rejects_checkpoint_and_market_binary(self):
        self.put('model.pt', 'private checkpoint')
        self.put('market.parquet', 'private data')
        self.assertEqual(2, len(self.scan()))

    def test_rejects_raw_market_and_daily_derived_csv(self):
        self.put('prices.csv', 'date,instrument,close\n2020-01-01,AAA,10\n')
        self.put('daily.csv', 'date,rank_ic\n2020-01-01,0.1\n')
        self.assertEqual(2, len(self.scan()))

    def test_rejects_learned_csv_and_text_weights(self):
        self.put('joint_elastic_net_weights.csv', 'train_start,feature_a\n2020-01-01,0.123\n')
        self.put('model.txt', 'tree\nversion=v4\nnum_class=1\n')
        self.assertEqual(2, len(self.scan()))

    def test_rejects_embedded_python_and_json_weights(self):
        self.put('weights.py', 'CHECKPOINT_B85 = ' + repr('a' * 2000) + '\n')
        self.put('weights.json', '{"state_dict": {"layer.weight": [0.4, 0.2]}}')
        self.assertEqual(2, len(self.scan()))

    def test_rejects_private_paths_and_remote_logins(self):
        self.put('private.md', '/' + 'Users/example/Desktop/project\n' + 'root' + '@example.test\n')
        self.assertGreaterEqual(len(self.scan()), 1)

    def test_rejects_escaped_private_json_path(self):
        self.put('audit.json', '{"checkpoint":"\\/r' + 'oot\\/private\\/model.pt"}')
        self.assertEqual(1, len(self.scan()))

    def test_rejects_saved_notebook_output(self):
        self.put('demo.ipynb', '{"cells":[{"cell_type":"code","source":[],"outputs":[{"text":"private"}],"execution_count":1}]}')
        self.assertGreaterEqual(len(self.scan()), 1)

    def test_allows_algorithm_source_and_aggregate_results(self):
        self.put('model.py', 'def average(values):\n    return sum(values) / len(values)\n')
        self.put('summary.csv', 'arm,rank_ic,days\nbaseline,0.1,242\n')
        self.put('manifest.json', '{"columns":["date","instrument"],"rows":1000}')
        self.assertEqual([], self.scan())

    def test_scans_tracked_directory_alias_without_crashing(self):
        self.put('current/model.py', 'def predict():\n    return 0\n')
        (self.root / 'legacy').symlink_to('current', target_is_directory=True)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        self.assertEqual([], self.scan())

    def test_rejects_symlink_outside_public_tree(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'private.txt'
            target.write_text('not part of the public source')
            (self.root / 'external.txt').symlink_to(target)
            self.assertEqual(1, len(self.scan()))

    def test_rejects_dangling_symlink_in_export(self):
        (self.root / 'missing').symlink_to('absent')
        self.assertEqual(1, len(self.scan()))

    def test_rejects_symlink_cycle(self):
        (self.root / 'loop').symlink_to('loop')
        self.assertEqual(1, len(self.scan()))

    def test_rejects_tracked_alias_relying_on_untracked_symlink(self):
        self.put('model.py', 'def predict():\n    return 0\n')
        (self.root / 'helper').symlink_to('model.py')
        (self.root / 'public_alias').symlink_to('helper')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', 'model.py', 'public_alias'], check=True)
        self.assertEqual(1, len(self.scan()))

    def test_rejects_parent_traversal_through_directory_symlink(self):
        self.put('model.py', 'def predict():\n    return 0\n')
        self.put('file.py', 'def score():\n    return 0\n')
        self.put('real/deep/source.py', 'def fit():\n    return 0\n')
        (self.root / 'real/file.py').symlink_to('../model.py')
        (self.root / 'alias').symlink_to('real/deep', target_is_directory=True)
        (self.root / 'public_alias').symlink_to('alias/../file.py')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', 'model.py', 'file.py',
                        'real/deep/source.py', 'alias', 'public_alias'], check=True)
        self.assertEqual(1, len(self.scan()))

    def test_allows_embedded_algorithm_source_but_checks_nested_payloads(self):
        source = 'def average(values):\n    return sum(values) / len(values)\n' + '#' * 110000
        self.put('runtime.py', 'SOURCE = ' + repr(source))
        self.assertEqual([], self.scan())
        source += '\nCHECKPOINT_B85 = ' + repr('a' * 2000)
        self.put('runtime.py', 'SOURCE = ' + repr(source))
        self.assertEqual(1, len(self.scan()))

    def test_sanitized_runtime_root_paths_resolve_to_repository(self):
        sources = (
            'scripts/build_unified_t_submission.py',
            'experiments/finals_pre/e6_channel_groups/score.py',
            'experiments/finals_pre/e4_private_fixed_oos/infer_fixed.py',
            'experiments/finals_pre/e4_private_fixed_oos/score_fixed.py',
            'experiments/finals_pre/e5_raw23_direct/build_o2c_labels.py',
            'experiments/finals_pre/e9_tcn_architecture/analysis/build_readme.py',
        )
        for name in sources:
            path = REPOSITORY / name
            tree = ast.parse(path.read_text())
            assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id in {'ROOT', 'root'}
                                   for target in node.targets)]
            self.assertEqual(1, len(assignments), name)
            value = assignments[0].value
            self.assertIsInstance(value, ast.Subscript, name)
            self.assertEqual(REPOSITORY, path.resolve().parents[ast.literal_eval(value.slice)], name)

    def test_replay_matches_external_loader_and_unknown_provenance(self):
        path = REPOSITORY / 'scripts/replay_frozen_candidate454_models.py'
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == '_load_frozen_model']
        self.assertEqual(2, len(calls))
        self.assertTrue(all(len(node.args) == 2 for node in calls))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                self.assertNotEqual('checkpoint_training_window', node.slice.value)
        self.assertIn('"includes_checkpoint_training_period": None', path.read_text())


if __name__ == '__main__':
    unittest.main()
