import tempfile
import unittest
from pathlib import Path

from bigalpha2026.incremental_cache import (
    INCREMENTAL_CACHE_SCHEMA_VERSION,
    IncrementalSummaryCache,
)


class IncrementalSummaryCacheTest(unittest.TestCase):
    def test_official_J_contract_uses_v3_schema(self):
        self.assertEqual(
            INCREMENTAL_CACHE_SCHEMA_VERSION,
            "elastic-net-incremental-cache-v3-J",
        )

    def test_exact_inputs_reuse_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            cache = IncrementalSummaryCache(Path(directory))
            first, hit, key = cache.get_or_compute(
                {"candidate": "A", "fingerprint": "one"},
                lambda: calls.append("fit") or {"increment": 0.1},
            )
            self.assertFalse(hit)
            self.assertEqual(first["increment"], 0.1)

            second_cache = IncrementalSummaryCache(Path(directory))
            second, hit, second_key = second_cache.get_or_compute(
                {"candidate": "A", "fingerprint": "one"},
                lambda: calls.append("unexpected") or {"increment": -1.0},
            )
            self.assertTrue(hit)
            self.assertEqual(key, second_key)
            self.assertEqual(second["increment"], 0.1)
            self.assertEqual(calls, ["fit"])

    def test_changed_candidate_content_cannot_hit_old_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = IncrementalSummaryCache(Path(directory))
            _, _, old_key = cache.get_or_compute(
                {"candidate": "A", "fingerprint": "old"},
                lambda: {"increment": 0.1},
            )
            calls = []
            changed_cache = IncrementalSummaryCache(Path(directory))
            result, hit, new_key = changed_cache.get_or_compute(
                {"candidate": "A", "fingerprint": "new"},
                lambda: calls.append("refit") or {"increment": 0.2},
            )
            self.assertFalse(hit)
            self.assertNotEqual(old_key, new_key)
            self.assertEqual(result["increment"], 0.2)
            self.assertEqual(calls, ["refit"])


if __name__ == "__main__":
    unittest.main()
