"""CPU-only adapter tests. All endpoint calls are mocked; no pilot is executed."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "eval_v41_pilot", Path(__file__).with_name("eval-v41-pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def terminal(maximum, *, count=3):
    return {
        "status": "completed",
        "model": "test",
        "incomplete_details": None,
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "plan"}],
            },
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": count,
            "total_tokens": 20 + count,
            "input_tokens_details": {"cached_tokens": 0},
        },
        "x_qwen": {
            "version": "serve_stats_v1",
            "matched_tokens": 0,
            "restore_ms": 0.0,
            "prompt_tokens": 20,
            "phases": {
                "family": "deepseek_v41",
                "prefill_mode": "single_token_forward",
                "prefill_rows": 1,
                "prefill_ms": 10.0,
                "decode_ms": 20.0,
                "transition_ms": 7.0 * (count - 1),
                "prefill_tokens": 20,
                "generated_tokens": count,
                "transitions": count - 1,
                "required_forwards": 20 + maximum - 1,
                "capacity": 32768,
                "reused_tokens": 0,
                "ced": False,
            },
        },
    }


def result(maximum, answer="answer"):
    value = pilot.frozen.validate_response(terminal(maximum))
    value.update(
        answer=answer,
        client_ttft_s=0.01,
        first_visible_answer_s=0.02,
        request_elapsed_s=0.04,
    )
    value["native"] = pilot.native_stats(value, maximum)
    return value


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.frozen = (pilot.FROZEN / "cases.json").read_bytes()
        self.cases = json.loads(self.frozen)

    def test_exact_request_inputs_only_remove_glm_lineage(self):
        for case in self.cases:
            expected = pilot.frozen.request_body("test", case)
            del expected["x_qwen"]["prefill_lineage"]
            self.assertEqual(pilot.request_body("test", case), expected)
            self.assertEqual(expected["reasoning"], {"effort": "low"})
            self.assertEqual(expected["max_output_tokens"], case["max_tokens"])

    def test_native_stats_counts_zero_transitions_and_rejects_old_binary(self):
        value = pilot.frozen.validate_response(terminal(1, count=1))
        native = pilot.native_stats(value, 1)
        self.assertEqual(native["transitions"], 0)
        self.assertEqual(native["phases"]["reused_tokens"], "0")
        self.assertEqual(native["stats"]["generated_tokens"], "1")
        limited = terminal(1, count=1)
        limited.update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        self.assertEqual(
            pilot.native_stats(pilot.frozen.validate_response(limited), 1)["stats"][
                "stop_reason"
            ],
            "token_limit",
        )
        for key, bad in [
            ("family", "glm5_next"),
            ("transitions", 1),
            ("ced", True),
            ("prefill_ms", float("nan")),
            ("transition_ms", 30.0),
            ("required_forwards", 32768),
        ]:
            changed = copy.deepcopy(value)
            changed["x_qwen"]["phases"][key] = bad
            with self.assertRaises(pilot.frozen.InvalidRun):
                pilot.native_stats(changed, 1)
        del value["x_qwen"]["phases"]
        with self.assertRaises(pilot.frozen.InvalidRun):
            pilot.native_stats(value, 1)

    def test_existing_sse_parser_and_artifact_format(self):
        body = pilot.request_body("test", self.cases[0])
        response = terminal(body["max_output_tokens"])

        def frame(kind, data):
            return f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()

        data = frame(
            "response.reasoning.delta",
            {"type": "response.reasoning.delta", "delta": "plan"},
        )
        data += (
            frame(
                "response.completed",
                {"type": "response.completed", "response": response},
            )
            + b"data: [DONE]\n\n"
        )
        wire = Mock(status=200)
        wire.readline = io.BytesIO(data).readline
        wire.getheaders.return_value = [("Content-Type", "text/event-stream")]
        wire.getheader.return_value = "text/event-stream"
        conn = Mock()
        conn.getresponse.return_value = wire
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(pilot, "connection", return_value=conn),
        ):
            actual = pilot.evaluate("http://127.0.0.1:8737", body, Path(directory), 900)
            self.assertEqual(actual["answer"], "answer")
            self.assertEqual(actual["reasoning"], "plan")
            self.assertEqual(actual["native"]["transitions"], 2)
            self.assertEqual((Path(directory) / "response.sse").read_bytes(), data)
            self.assertEqual(
                json.loads((Path(directory) / "request.json").read_text()), body
            )
            conn.close.assert_called_once()

    def test_mocked_eight_case_summary_scoring_rates_and_frozen_blinding(self):
        def evaluate(endpoint, body, directory, timeout):
            case = self.cases[int(directory.name[-2:]) - 1]
            answer = json.dumps(
                {
                    "facts": case["reference"]["facts"],
                    "code": "pass",
                    "explanation": "test fixture only",
                }
            )
            return result(body["max_output_tokens"], answer)

        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "run"
            manifest = Path(root) / "owned-manifest.json"
            declared = {
                "revision": "f792019bcb369768220f01088f0f7af230786982",
                "shards": [
                    {
                        "path": "/DO_NOT_OPEN/weights.safetensors",
                        "sha256": "declared-not-rehashed",
                    }
                ],
            }
            manifest.write_text(json.dumps(declared))
            with (
                patch.object(pilot, "connection", return_value=Mock()),
                patch.object(pilot, "evaluate", side_effect=evaluate) as calls,
                patch.object(
                    pilot.frozen,
                    "integrity",
                    side_effect=AssertionError("must not hash weights"),
                ),
                patch.object(
                    pilot.frozen,
                    "start_server",
                    side_effect=AssertionError("must not reload"),
                ),
                patch.object(
                    pilot.frozen,
                    "cleanup",
                    side_effect=AssertionError("not our process"),
                ),
                patch.object(
                    pilot.frozen,
                    "environment",
                    side_effect=AssertionError("no environment whitelist"),
                ),
            ):
                self.assertEqual(
                    pilot.main(
                        [
                            "--endpoint",
                            "http://127.0.0.1:8737",
                            "--model-id",
                            "test",
                            "--out",
                            str(out),
                            "--download-manifest",
                            str(manifest),
                        ]
                    ),
                    0,
                )
                self.assertEqual(calls.call_count, 8)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual((out / "cases.json").read_bytes(), self.frozen)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["checkpoint"]["owned_download_manifest"], declared)
            self.assertFalse(summary["checkpoint"]["artifact_hashes_verified"])
            self.assertFalse(summary["reasoning"]["same_internal_budget_as_glm"])
            self.assertEqual(
                summary["comparison_to_maya"]["v41"]["weighted_prefill_tps"], 2000.0
            )
            self.assertEqual(
                summary["comparison_to_maya"]["v41"]["weighted_generation_tps"], 150.0
            )
            for case, row in zip(self.cases, summary["results"]):
                self.assertEqual(
                    row["checks"], pilot.frozen.check_answer(case, row["answer"])
                )
                self.assertIsNone(row["server_request_ordinal"])
            blind = pilot.load_frozen("blind")
            maya = pilot.read_metadata(pilot.FROZEN / "results/maya.json")
            iq3 = pilot.read_metadata(pilot.FROZEN / "results/stock-iq3.json")
            packet, mapping = blind.build_packet(
                self.frozen, [summary, maya, iq3], rng=random.Random(7)
            )
            self.assertEqual(len(packet["cases"]), 8)
            for case, item, row in zip(self.cases, packet["cases"], summary["results"]):
                self.assertEqual(
                    (
                        item["system"],
                        item["task"],
                        item["rubric"],
                        item["reference_facts"],
                    ),
                    (
                        case["system"],
                        case["user"],
                        case["rubric"],
                        case["reference"]["facts"],
                    ),
                )
                label = next(
                    k
                    for k, v in mapping["cases"][case["id"]].items()
                    if v == summary["arm"]
                )
                self.assertEqual(
                    next(
                        c["answer"]
                        for c in item["candidates"]
                        if c["candidate"] == label
                    ),
                    row["answer"],
                )
            with self.assertRaises(ValueError):
                blind.build_packet(self.frozen, [summary, maya])

    def test_failure_does_not_restart_or_overwrite(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(pilot, "connection", return_value=Mock()),
            patch.object(
                pilot, "evaluate", side_effect=pilot.frozen.InvalidRun("bad response")
            ) as calls,
        ):
            out = Path(root) / "run"
            args = [
                "--endpoint",
                "http://127.0.0.1:8737",
                "--model-id",
                "test",
                "--out",
                str(out),
            ]
            self.assertEqual(pilot.main(args), 1)
            self.assertEqual(calls.call_count, 1)
            before = (out / "summary.json").read_bytes()
            self.assertEqual(json.loads(before)["status"], "aborted")
            with self.assertRaises(FileExistsError):
                pilot.main(args)
            self.assertEqual((out / "summary.json").read_bytes(), before)

    def test_endpoint_validation_is_not_an_environment_gate(self):
        for endpoint in (
            "file:///weights",
            "http://user:secret@localhost",
            "http://localhost/other",
            "http://localhost?token=secret",
        ):
            with self.assertRaises(pilot.frozen.InvalidRun):
                pilot.connection(endpoint, 1)
        pilot.connection(
            "http://127.0.0.1:8737/v1", 1
        ).close()  # Constructor does not connect.


if __name__ == "__main__":
    unittest.main()
