import tempfile
import unittest
from pathlib import Path

import stage


class StageTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "rewritten.gguf_file"
        source.write_bytes(b"fixture")
        return {
            "shards": [
                {
                    "name": "model-00001-of-00001.gguf",
                    "source_path": source.name,
                    "size": 7,
                }
            ]
        }

    def test_canonical_name_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            entry = stage.stage(root, manifest)
            self.assertEqual(entry.name, "model-00001-of-00001.gguf")
            self.assertEqual(entry.read_bytes(), b"fixture")
            self.assertEqual(stage.stage(root, manifest), entry)

    def test_existing_file_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            target = root / "glm5-next"
            target.mkdir()
            existing = target / manifest["shards"][0]["name"]
            existing.write_bytes(b"unrelated")
            with self.assertRaises(ValueError):
                stage.stage(root, manifest)
            self.assertEqual(existing.read_bytes(), b"unrelated")

    def test_partial_and_escaping_sources_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            manifest["shards"][0]["size"] = 8
            with self.assertRaises(ValueError):
                stage.stage(root, manifest)
            manifest["shards"][0]["source_path"] = "../outside"
            with self.assertRaises(ValueError):
                stage.stage(root, manifest)


if __name__ == "__main__":
    unittest.main()
