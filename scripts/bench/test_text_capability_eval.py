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

import text_capability_eval as capability


class TextCapabilityContractTests(unittest.TestCase):
    def test_packet_and_request_sets_are_frozen(self) -> None:
        packet, request_bytes = capability.build_packet()
        self.assertEqual(
            capability.semantic_sha256(packet),
            capability.PACKET_SEMANTIC_SHA256,
        )
        self.assertEqual(len(packet["tasks"]), 22)
        self.assertEqual(packet["request_sets"]["capability"]["count"], 22)
        self.assertEqual(packet["request_sets"]["effort"]["count"], 32)
        self.assertEqual(
            capability.sha256_bytes(request_bytes["capability"]),
            "f1c6dc54033d3df96fd1bf9d2550c41df9346c7214c4227104a00675afca82af",
        )
        self.assertEqual(
            capability.sha256_bytes(request_bytes["effort"]),
            "888f564f3efd95751e7099efe181d2104533f4000394bba55b546f93aa3f69b4",
        )

    def test_renderers_match_exact_qwen_contracts(self) -> None:
        self.assertEqual(
            capability.render_no_thinking(" hello "),
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        medium = capability.render_qwen38_effort("hello", "medium")
        self.assertEqual(
            medium,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
        )
        low = capability.render_qwen38_effort("hello", "low")
        self.assertIn(capability.QWEN38_REASONING_EFFORT_LOW, low)
        self.assertTrue(low.endswith("<|im_start|>assistant\n<think>\n"))
        xhigh = capability.render_qwen38_effort("hello", "xhigh")
        self.assertIn(capability.QWEN38_REASONING_EFFORT_XHIGH, xhigh)

    def test_effort_order_rotates_all_modes_per_task(self) -> None:
        packet, _ = capability.build_packet()
        orders = packet["effort_request_order"]
        self.assertEqual(
            [row["task_id"] for row in orders], list(capability.EFFORT_TASK_IDS)
        )
        for row in orders:
            self.assertEqual(set(row["modes"]), set(capability.MODES))
        first_modes = [row["modes"][0] for row in orders]
        self.assertEqual(first_modes[:4], list(capability.MODES))

    def test_reference_answers_cover_edge_contracts(self) -> None:
        data = {"a": {"b": [9, {"q.r": [4]}]}, "none": None}
        self.assertEqual(capability.ref_deep_get(data, "a.b[1].q\\.r[0]"), (True, 4))
        self.assertEqual(capability.ref_deep_get(data, "none"), (True, None))
        self.assertEqual(capability.ref_deep_get(data, "a.b[-1]"), (False, None))
        self.assertEqual(
            capability.ref_apply_patch(
                {"a": [1, 2]},
                [("del", ["a", 0]), ("set", ["a", 0], 9)],
            ),
            {"a": [9]},
        )
        self.assertIsNone(
            capability.ref_apply_patch({"a": [1]}, [("set", ["a", True], 2)])
        )
        self.assertEqual(capability.ref_minimal_cover("aaabcbc", "aabc"), (1, 5))
        self.assertEqual(capability.ref_natural_compare("a01", "a1"), 1)

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            capability.parse_json('{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            capability.parse_json('{"a":NaN}')

    def test_code_grader_executes_only_one_validated_function(self) -> None:
        source = """\
def merge_spans(spans):
    rows = sorted((min(a, b), max(a, b)) for a, b in spans if a != b)
    out = []
    for start, end in rows:
        if not out or start > out[-1][1]:
            out.append((start, end))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], end))
    return out
"""
        report = capability.grade_code("c1", source)
        self.assertTrue(report["format_compliant"])
        self.assertTrue(report["semantic_correct"])
        self.assertGreater(report["tests_run"], 40)

        rejected = capability.grade_code(
            "c1", "import os\ndef merge_spans(spans):\n    return []"
        )
        self.assertFalse(rejected["format_compliant"])
        self.assertFalse(rejected["semantic_correct"])

    def test_response_partition_requires_aligned_thinking_stats(self) -> None:
        partition = {
            "delimiter": "</think>",
            "delimiter_start_token_index": 2,
            "delimiter_end_token_index_exclusive": 3,
            "delimiter_token_aligned": True,
            "reasoning_tokens": 2,
            "delimiter_tokens": 1,
            "visible_tokens": 2,
        }
        split = capability.split_generated_response(
            "reasoning</think>\n\n42", "low", "eos", 5, partition
        )
        self.assertTrue(split["protocol_complete"])
        self.assertEqual(split["visible_text"], "42")
        self.assertEqual(split["reasoning_tokens"], 2)

        missing = capability.split_generated_response(
            "reasoning only", "xhigh", "token_limit", 2, None
        )
        self.assertFalse(missing["protocol_complete"])
        self.assertFalse(missing["stop_complete"])

    def test_exact_and_json_graders_preserve_visible_whitespace(self) -> None:
        tasks = {task.task_id: task for task in capability.build_tasks()}
        self.assertFalse(
            capability.grade_visible(tasks["r1"], "654\n")["format_compliant"]
        )
        json_grade = capability.grade_visible(
            tasks["j3"], capability.canonical_json(tasks["j3"].expected) + "\n"
        )
        self.assertFalse(json_grade["format_compliant"])
        self.assertTrue(json_grade["semantic_correct"])

    def test_engine_validation_rejects_reordered_serial_rows(self) -> None:
        requests = [{"id": "a"}, {"id": "b"}]
        outputs = [{"id": "b"}, {"id": "a"}]
        stats = [{"id": "a"}, {"id": "b"}]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            capability.validate_engine_artifacts(
                requests,
                outputs,
                stats,
                {"locator": {"path": "/tmp/model.gguf"}},
                "a" * 40,
                True,
                "git-source-sha256-v2:" + "b" * 64,
            )

    def test_scored_rows_index_by_request_id(self) -> None:
        rows = [{"request_id": "a"}, {"request_id": "b"}]
        self.assertEqual(
            set(capability.index_unique(rows, "scored", "request_id")),
            {"a", "b"},
        )

    def test_force_prepare_invalidates_dependent_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "packet"
            with contextlib.redirect_stdout(io.StringIO()):
                capability.prepare(output, force=False)
            stale = output / "stale.outputs.jsonl"
            stale.write_text("stale")
            with contextlib.redirect_stdout(io.StringIO()):
                capability.prepare(output, force=True)
            self.assertFalse(stale.exists())
            capability.verify_prepared(output)

    def test_child_environment_removes_inherited_qwen_controls(self) -> None:
        inherited = {
            "PATH": "/usr/bin",
            "QWEN_DSV4_RESIDENCY_SET": "1",
            "QWEN_GGUF_PARALLEL_COPY": "pread",
            "SECRET": "not-forwarded",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            environment, record = capability.child_environment()
        self.assertEqual(environment["QWEN_DSV4_RESIDENCY_SET"], "0")
        self.assertEqual(environment["QWEN_DSV4_PREFETCH"], "off")
        self.assertNotIn("QWEN_GGUF_PARALLEL_COPY", environment)
        self.assertNotIn("SECRET", environment)
        self.assertEqual(
            record["removed_qwen_keys"],
            ["QWEN_DSV4_RESIDENCY_SET", "QWEN_GGUF_PARALLEL_COPY"],
        )

    def test_run_command_forces_serial_uncached_unprefetched_execution(self) -> None:
        command = capability.run_command(
            Path("qwen"),
            Path("model.gguf"),
            Path("requests.jsonl"),
            Path("stats.jsonl"),
        )
        self.assertIn("--execution-mode", command)
        self.assertEqual(command[command.index("--execution-mode") + 1], "serial")
        self.assertEqual(command[command.index("--model-prefetch") + 1], "off")
        self.assertEqual(command[command.index("--prefix-cache-max-mib") + 1], "0")
        self.assertEqual(
            command[command.index("--cache-prefix-auto-min-tokens") + 1], "0"
        )
        self.assertIn("--no-special-tokens", command)

    def test_terminate_and_reap_never_escalates_to_kill(self) -> None:
        process = mock.Mock()
        process.poll.side_effect = [None, None, 0]
        process.returncode = -15
        self.assertEqual(capability.terminate_and_reap(process), -15)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with()
        self.assertFalse(process.kill.called)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
