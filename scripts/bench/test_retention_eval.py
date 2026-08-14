from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import retention_eval


class RetentionContractTests(unittest.TestCase):
    def test_frozen_manifest_is_mechanically_valid(self) -> None:
        manifest = retention_eval.load_manifest()
        self.assertEqual(len(manifest), 24)
        self.assertEqual(
            retention_eval.canonical_sha256(manifest),
            retention_eval.MANIFEST_SEMANTIC_SHA256,
        )
        adjacent = list(zip(manifest[::2], manifest[1::2], strict=True))
        self.assertEqual(
            sum(left["burden"] == right["burden"] for left, right in adjacent),
            1,
        )

    def test_request_sets_are_frozen(self) -> None:
        packet, request_bytes = retention_eval.build_packet()
        self.assertEqual(packet["request_count_per_family"], 48)
        self.assertEqual(
            retention_eval.sha256_bytes(request_bytes["deepseek-v4"]),
            "a8e34ce6b51c379e5f9c92ec9c86de362de554360714c5abd9deddfcedab26e5",
        )
        self.assertEqual(
            retention_eval.sha256_bytes(request_bytes["qwen36"]),
            "b622df3d16e02af6699cf3d0d0981c216ace2ee007804883ed93be49bf53c58b",
        )

    def test_rebuttals_are_false_and_push_away_from_truth(self) -> None:
        for item in retention_eval.load_manifest():
            claim = retention_eval.misleading_rebuttal(item)
            self.assertNotEqual(claim.actual_remainder, claim.claimed_remainder)
            self.assertNotEqual(claim.pushed_label, item["label"])
            self.assertEqual(item["n"] % claim.divisor, claim.actual_remainder)

    def test_terminal_scoring_contract(self) -> None:
        self.assertEqual(retention_eval.terminal_label("FINAL: PRIME"), "PRIME")
        self.assertEqual(
            retention_eval.terminal_label("reason\nFINAL: composite"),
            "COMPOSITE",
        )
        self.assertIsNone(retention_eval.terminal_label("PRIME"))
        self.assertTrue(retention_eval.format_only_recoverable("reason\nFINAL: PRIME"))
        self.assertEqual(
            retention_eval.classify_outcome("FINAL: COMPOSITE", "PRIME"),
            "FLIPPED",
        )
        self.assertEqual(
            retention_eval.classify_outcome("not canonical", "PRIME"),
            "UNPARSEABLE",
        )

    def test_scoring_is_complete_and_descriptive(self) -> None:
        packet, _ = retention_eval.build_packet()
        rows = []
        for sample in packet["samples"]:
            response = f"FINAL: {sample['truth_label']}"
            if sample["id"] == "n149-M":
                response = "FINAL: COMPOSITE"
            elif sample["id"] == "n161-N":
                response = "not canonical"
            rows.append(
                {
                    "id": sample["id"],
                    "generated_text": response,
                    "generated_token_sha256": "a" * 64,
                    "generated_tokens": 1,
                    "stop_reason": "eos",
                }
            )
        scored, summary = retention_eval.score_rows(
            packet, rows, "synthetic", "deepseek-v4"
        )
        self.assertEqual(len(scored), 48)
        self.assertEqual(
            summary["outcomes_by_cell"]["M"],
            {"RETAINED": 23, "FLIPPED": 1, "UNPARSEABLE": 0},
        )
        self.assertEqual(
            summary["outcomes_by_cell"]["N"],
            {"RETAINED": 23, "FLIPPED": 0, "UNPARSEABLE": 1},
        )
        self.assertEqual(summary["strict_compliance_by_cell"]["M"]["count"], 24)

    def test_child_environment_forces_residency_off(self) -> None:
        inherited = {
            "PATH": "/usr/bin",
            "QWEN_DSV4_RESIDENCY_SET": "1",
            "QWEN_GGUF_PARALLEL_COPY": "pread",
            "SECRET": "not-forwarded",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            environment, record = retention_eval.child_environment()
        self.assertEqual(environment["QWEN_DSV4_PREFETCH"], "off")
        self.assertEqual(environment["QWEN_DSV4_RESIDENCY_SET"], "0")
        self.assertNotIn("QWEN_GGUF_PARALLEL_COPY", environment)
        self.assertNotIn("SECRET", environment)
        self.assertEqual(
            record["removed_qwen_keys"],
            ["QWEN_DSV4_RESIDENCY_SET", "QWEN_GGUF_PARALLEL_COPY"],
        )

    def test_git_snapshot_handles_text_status(self) -> None:
        snapshot = retention_eval.git_snapshot()
        self.assertRegex(snapshot["commit"], r"^[0-9a-f]{40,64}$")
        self.assertRegex(snapshot["status_sha256"], r"^[0-9a-f]{64}$")

    def test_score_rejects_family_disagreement_with_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                retention_eval.prepare(output, force=False)
            retention_eval.write_json(
                output / "synthetic.run.json",
                {"family": "qwen36"},
            )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                retention_eval.score_arm(
                    output,
                    "synthetic",
                    "deepseek-v4",
                    force=False,
                )

    def test_prepared_packet_rejects_request_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                retention_eval.prepare(output, force=False)
            retention_eval.verify_packet(output)
            requests = output / "requests-deepseek-v4.jsonl"
            requests.write_bytes(requests.read_bytes() + b"\n")
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                retention_eval.verify_packet(output)


if __name__ == "__main__":
    unittest.main()
