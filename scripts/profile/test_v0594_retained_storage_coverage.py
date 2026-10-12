from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v0594_retained_storage_coverage as coverage


class FirstShardTests(unittest.TestCase):
    def test_only_first_split_shard_is_a_model_entrypoint(self) -> None:
        self.assertTrue(coverage.is_first_shard("model.gguf"))
        self.assertTrue(coverage.is_first_shard("model-00001-of-00003.gguf"))
        self.assertFalse(coverage.is_first_shard("model-00002-of-00003.gguf"))


if __name__ == "__main__":
    unittest.main()
