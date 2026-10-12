"""CPU-only stdlib tests; all HTTP and server lifecycle operations are mocked."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call as mock_call, patch

SPEC = importlib.util.spec_from_file_location("screen", Path(__file__).with_name("serve_agent_reuse_screen.py"))
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)
SHAPE = {"instructions": "same instructions", "tools": [{"type": "function", "name": "read"}]}


def call(path="README.md", name="read"):
    return {"type": "function_call", "call_id": "call_1", "name": name,
            "arguments": json.dumps({"filePath": path})}


def response(output, status="completed"):
    return {"status": status, "output": output, "usage": {"input_tokens_details": {"cached_tokens": 100}}}


def message(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


class ScreenTests(unittest.TestCase):
    def test_canned_paths_errors_and_no_host_reads(self):
        for path, expected in [("README.md", "counts words"), ("/work/project/Cargo.toml", "[dependencies]"),
                               ("src/error.rs", "From<io::Error>")]:
            result = json.loads(screen.canned_result(call(path))["output"])
            self.assertTrue(result["synthetic"])
            self.assertIsNone(result["error"])
            self.assertIn(expected, result["content"])
        for tool in [call("/etc/passwd"), call(name="write"), {**call(), "arguments": "{"},
                     {**call(), "arguments": "[]"}, {**call(), "arguments": '{"n":NaN}'}]:
            self.assertTrue(json.loads(screen.canned_result(tool)["output"])["error"])

    def test_sse_timing_terminal_evidence_and_connection_close(self):
        terminal = response([message("answer"), call()])
        events = [
            {"type": "response.reasoning_text.delta", "delta": "thought"},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.output_item.done", "item": call()},
            {"type": "response.completed", "response": terminal},
        ]
        chunks = [("data: " + json.dumps(event) + "\n\n").encode() for event in events]
        http = Mock(status=200)
        http.getheader.return_value = None
        http.read1.side_effect = [chunks[0][:9], chunks[0][9:], *chunks[1:], b""]
        connection = Mock()
        connection.getresponse.return_value = http
        evidence = []
        with patch.object(screen.http.client, "HTTPConnection", return_value=connection), \
             patch.object(screen.time, "monotonic", side_effect=[0, 1, 2, 3, 4, 5]):
            actual, times = screen.post_stream("127.0.0.1:8797", {"input": []}, evidence.append)
        self.assertEqual(actual, terminal)
        self.assertEqual(times, {"first_delta_s": 1, "first_answer_delta_s": 2,
                                 "first_tool_available_s": 3, "wall_s": 5})
        self.assertEqual(evidence[-1]["event"], "terminal_response")
        self.assertEqual(evidence[-1]["terminal_s"], 4)
        self.assertEqual(evidence[-1]["response"], terminal)
        self.assertEqual([event["event"] for event in evidence], ["http", "terminal_response"])
        connection.close.assert_called_once()

    def test_partial_deltas_and_items_are_flushed_only_after_read_failure(self):
        observations = [
            {"type": "response.output_item.added", "item": {"type": "message", "id": "m1"}},
            {"type": "response.reasoning_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": "unfinished answer"},
        ]
        data = b"".join(("data: " + json.dumps(event) + "\n\n").encode() for event in observations)
        http = Mock(status=200)
        http.getheader.return_value = None
        evidence = []
        reads = iter([data, OSError("stream disconnected")])

        def read(_size):
            # No flush for any successfully parsed delta or item.
            self.assertEqual([event["event"] for event in evidence], ["http"])
            value = next(reads)
            if isinstance(value, Exception):
                raise value
            return value

        http.read1.side_effect = read
        connection = Mock()
        connection.getresponse.return_value = http
        with patch.object(screen.http.client, "HTTPConnection", return_value=connection), \
             patch.object(screen.time, "monotonic", side_effect=[0, 1, 2, 3, 4]):
            with self.assertRaisesRegex(OSError, "stream disconnected"):
                screen.post_stream("127.0.0.1:8797", {}, evidence.append)
        failure = evidence[-1]
        self.assertEqual(failure["event"], "stream_failure")
        self.assertFalse(failure["terminal_received"])
        self.assertEqual([item["sse"] for item in failure["observations"]], observations)
        self.assertEqual(failure["timings"], {"first_delta_s": 2, "first_answer_delta_s": 3,
                                             "first_tool_available_s": None, "wall_s": 4})
        connection.close.assert_called_once()

    def test_terminal_evidence_survives_later_read_failure(self):
        terminal = response([message("partial evidence")], "incomplete")
        terminal["incomplete_details"] = {"reason": "max_output_tokens"}
        data = ("data: " + json.dumps({"type": "response.incomplete", "response": terminal}) + "\n\n").encode()
        http = Mock(status=200)
        http.getheader.return_value = None
        http.read1.side_effect = [data, TimeoutError("EOF wait")]
        connection = Mock()
        connection.getresponse.return_value = http
        evidence = []
        with patch.object(screen.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(TimeoutError, "EOF wait"):
                screen.post_stream("127.0.0.1:8797", {}, evidence.append)
        self.assertEqual(evidence[-1]["response"], terminal)
        connection.close.assert_called_once()

    def test_http_503_is_recorded_without_retry(self):
        http = Mock(status=503)
        http.getheader.return_value = "1"
        http.read.return_value = b'{"error":{"code":"server_busy"}}'
        connection = Mock()
        connection.getresponse.return_value = http
        evidence = []
        with patch.object(screen.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                screen.post_stream("127.0.0.1:8797", {}, evidence.append)
        self.assertEqual(evidence, [{"event": "http", "http_status": 503, "retry_after": "1",
                                     "body": '{"error":{"code":"server_busy"}}'}])
        connection.request.assert_called_once()
        connection.close.assert_called_once()

    def test_steps_preserve_shape_and_record_workspace_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "serve.log"
            log.write_text("")
            evidence = []
            client = screen.Screen("127.0.0.1:8797", "model", SHAPE, "low", 2048, log, evidence.append)
            bodies = []

            def send(addr, body, emit):
                bodies.append(body)
                with log.open("a") as output:
                    if len(bodies) == 1:
                        output.write("serve: glm5_next prefill plan rows=1024 session_bytes=123\n")
                    output.write("serve phases: reused_tokens=100 prefill_tokens=10 prefill_ms=2.0 decode_ms=3.0 "
                                 "reuse_source=snapshot restore_ms=1.1 boundary_ms=0.1 lineage=fast "
                                 "snapshot_schedule=fast_shared_split_v1 cuts=[100] captures=[100:present] "
                                 "snapshot_capture_ms=0.0 snapshot_cache_bytes=500 snapshot_cache_entries=1\n")
                return response([call("Cargo.toml")]), {"first_delta_s": 0.1, "wall_s": 1}

            with patch.object(screen, "post_stream", side_effect=send), patch.object(screen.time, "sleep") as settle:
                client.steps("B", [{"role": "user", "content": "edited"}], "B", 1, "branch")
            self.assertEqual(len(bodies), 2)
            for body in bodies:
                self.assertEqual(body["instructions"], SHAPE["instructions"])
                self.assertEqual(body["tools"], SHAPE["tools"])
                self.assertEqual(body["reasoning"], {"effort": "low"})
                self.assertNotIn("x_qwen", body)
            self.assertIn("[dependencies]", bodies[1]["input"][-1]["output"])
            self.assertEqual([r["workspace_rows"] for r in client.records], [1024, 1024])
            self.assertEqual(client.records[-1]["captures"], "100:present")
            self.assertEqual(client.records[-1]["cached_tokens"], 100)
            self.assertTrue(evidence[-1]["step_budget_exhausted"])
            self.assertEqual(settle.call_count, 2)
            settle.assert_called_with(0.5)

    def test_incomplete_turn_is_not_replayed(self):
        client = screen.Screen("127.0.0.1:1", "m", SHAPE, "low", 10, Path("unused"), Mock())
        incomplete = response([call()], "incomplete")
        incomplete["incomplete_details"] = {"reason": "max_output_tokens"}
        with patch.object(client, "send", return_value=incomplete) as send:
            _, result = client.steps("A", [], "A1", 3, "cold")
        send.assert_called_once()
        self.assertEqual(result, incomplete)
        event = client.evidence.call_args.args[0]
        self.assertFalse(event["step_budget_exhausted"])
        self.assertEqual(event["incomplete_details"], {"reason": "max_output_tokens"})

    def test_main_failure_preserves_partial_evidence_and_only_owned_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, fixture, out = root / "fake-qwen", root / "shape.json", root / "out"
            binary.write_bytes(b"not an executable")
            fixture.write_text(json.dumps(SHAPE))
            owned = Mock(pid=12345)
            owned.poll.side_effect = [None, 0]
            env = {"MTL_DEBUG_LAYER": "1", "QWEN_GLM_FAST_PRECISION": "half", "RUST_LOG": "info",
                   "QWEN_SERVE_IDLE_RESIDENCY_SECS": "30"}
            argv = ["screen", "--qwen", str(binary), "--model", "maya.gguf", "--shape", str(fixture), "--out", str(out)]
            with patch("sys.argv", argv), patch.dict(screen.os.environ, env, clear=True), \
                 patch.object(screen.subprocess, "check_output", side_effect=["revision\n", ""]), \
                 patch.object(screen.subprocess, "Popen", return_value=owned) as start, \
                 patch.object(screen, "wait_listening"), patch.object(screen.time, "sleep"), \
                 patch.object(screen, "post_stream", side_effect=OSError("connection failed")):
                with self.assertRaisesRegex(SystemExit, "connection failed"):
                    screen.main()
            self.assertEqual(start.call_args.kwargs["env"], env)
            owned.send_signal.assert_called_once_with(screen.signal.SIGINT)
            owned.terminate.assert_not_called()
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["timing_mode"], "validation-on")
            self.assertEqual(report["inherited_overrides"], env)
            self.assertEqual(report["owned_pid"], 12345)
            self.assertEqual(report["binary_sha256"], screen.file_sha(binary))
            self.assertIn("connection failed", report["requests"][0]["error"])
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertIn("request", [event["event"] for event in events])
            self.assertEqual(events[-1]["event"], "end")

    def test_owned_shutdown_timeout_uses_only_same_process(self):
        owned = Mock()
        owned.poll.return_value = None
        owned.wait.side_effect = [subprocess.TimeoutExpired("owned", 180), 0]
        screen.stop_owned(owned)
        owned.send_signal.assert_called_once_with(screen.signal.SIGINT)
        owned.terminate.assert_called_once()
        owned.kill.assert_not_called()

    def test_owned_shutdown_third_escalation_is_same_child_and_bounded(self):
        for final_timeout in (False, True):
            with self.subTest(final_timeout=final_timeout):
                owned = Mock()
                owned.poll.return_value = None
                owned.wait.side_effect = [subprocess.TimeoutExpired("owned", 180),
                                         subprocess.TimeoutExpired("owned", 60),
                                         subprocess.TimeoutExpired("owned", 60) if final_timeout else 0]
                with patch.object(screen.os, "kill") as unrelated_kill:
                    if final_timeout:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            screen.stop_owned(owned)
                    else:
                        screen.stop_owned(owned)
                self.assertEqual(owned.mock_calls, [mock_call.poll(), mock_call.send_signal(screen.signal.SIGINT),
                                                    mock_call.wait(timeout=180), mock_call.terminate(),
                                                    mock_call.wait(timeout=60), mock_call.kill(), mock_call.wait(timeout=60)])
                unrelated_kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
