"""CPU-only negative controls for compact diagnostic provenance checks."""

import copy
from pathlib import Path
import unittest

from inspect_compact import digest, read_json, validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent
        policy = read_json(root / "holdout-256-v2.json")
        self.manifest = {
            "schema": "k2.compact_cache_diagnostic.v1",
            "claim": "native_q8_vs_native_f16_not_independent_oracle_or_new_holdout",
            "policy": policy,
            "policy_sha256": digest(root / "holdout-256-v2.json"),
            "compact_policy_sha256": digest(root / "COMPACT-KV-POLICY.md"),
            "model_sha256": policy["model_sha256"],
            "tokenizer_metadata_id": policy["tokenizer_metadata_id"],
            "inputs": [
                [corpus["name"], base, [], True]
                for corpus, base in zip(
                    policy["corpora"],
                    policy["continuation"]["bases_by_corpus"],
                    strict=True,
                )
            ],
        }

    def test_pinned_identity(self):
        validate_manifest(self.manifest)

    def test_altered_claims_fail(self):
        for key in [
            "schema",
            "claim",
            "policy_sha256",
            "compact_policy_sha256",
            "model_sha256",
            "tokenizer_metadata_id",
        ]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.manifest)
                bad[key] = "altered"
                with self.assertRaises(ValueError):
                    validate_manifest(bad)

    def test_altered_base_selection_and_policy_fail(self):
        for field in ["base", "selected", "policy", "duplicate"]:
            with self.subTest(field=field):
                bad = copy.deepcopy(self.manifest)
                if field == "base":
                    bad["inputs"][0][1] = 37
                elif field == "selected":
                    bad["inputs"][0][3] = False
                elif field == "policy":
                    bad["policy"]["capacity"] = 257
                else:
                    bad["inputs"].append(bad["inputs"][0])
                with self.assertRaises(ValueError):
                    validate_manifest(bad)


if __name__ == "__main__":
    unittest.main()
