"""CPU-only stdlib tests; no real HTTP, server, tokenizer, or model load."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("repository_context", Path(__file__).with_name("serve_repository_context.py"))
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def response(text, status="completed"):
    return {"status": status, "output": [
        {"type": "reasoning", "id": "server-reasoning", "content": [{"type": "reasoning_text", "text": "actual reasoning"}]},
        {"type": "message", "id": "server-message", "content": [{"type": "output_text", "text": text}]},
    ], "usage": {"input_tokens": 42001, "input_tokens_details": {"cached_tokens": 41000}},
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None}


def make_sources(root):
    for name in driver.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "// actual source " + name + "\nfn test_fixture() {}\n"
        if name == "docs/SERVE.md":
            text = "# Excluded\n" + driver.DOC_HEADING + "\nRelevant section\n## Next\nExcluded\n"
        path.write_text(text)


class RepositoryContextTests(unittest.TestCase):
    def test_packet_exact_ranges_hashes_and_no_prior_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sources(root)
            packet, checkpoint = driver.source_packet(root)
            self.assertEqual(checkpoint["source_count"], 6)
            self.assertEqual(checkpoint["packet_sha256"], driver.sha(packet.encode()))
            self.assertEqual(checkpoint["packet_bytes"], len(packet.encode()))
            self.assertEqual(checkpoint["packet_lines"], len(packet.splitlines()))
            self.assertIsNone(checkpoint["token_count"])
            for item in checkpoint["sources"]:
                raw = (root / item["path"]).read_bytes()
                selected = b"".join(raw.splitlines(keepends=True)[item["start_line"] - 1:item["end_line_inclusive"]])
                self.assertEqual(item["file_sha256"], driver.sha(raw))
                self.assertEqual(item["selected_sha256"], driver.sha(selected))
                self.assertEqual(packet.count(selected.decode()), 1)
            self.assertEqual(checkpoint["sources"][-1]["start_line"], 2)
            self.assertEqual(checkpoint["sources"][-1]["end_line_inclusive"], 3)
            self.assertNotIn("Excluded", packet)
            self.assertNotIn("11104", packet)
            self.assertIn(driver.QUESTION, packet)

    def test_followup_uses_report_observation_not_report_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"schema": "prior", "requests": [{"label": "A'", "reuse_source": "snapshot",
                            "reused_tokens": 11104, "prefill_tokens": 303, "answer_text": "Do not copy this answer"}]}))
            question, metadata = driver.followup_from_report(path)
            self.assertIn("reused_tokens=11104", question)
            self.assertIn("prefill_tokens=303", question)
            self.assertNotIn("Do not copy", question)
            self.assertEqual(metadata["sha256"], driver.screen.file_sha(path))

    def test_second_http_request_replays_actual_first_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "serve.log"
            log.write_text("")
            events, bodies = [], []
            client = driver.screen.Screen("127.0.0.1:1", "model", {"instructions": driver.INSTRUCTIONS, "tools": []},
                                          "low", 4096, log, events.append)
            first = response("actual operator table, not a canned answer")

            def send(addr, body, evidence):
                bodies.append(body)
                with log.open("a") as stream:
                    stream.write("serve: glm5_next prefill plan rows=2048\n")
                    stream.write("serve phases: reuse_source=live reused_tokens=41000 prefill_tokens=1000\n")
                return first if len(bodies) == 1 else response("actual followup"), {"wall_s": 1.0}

            with patch.object(driver.screen, "post_stream", side_effect=send), patch.object(driver.screen.time, "sleep"), \
                 patch.object(driver.screen, "canned_result", side_effect=AssertionError("No synthetic results")), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(driver.two_requests(client, "real source packet", "followup question", events.append), {"sent": True})
            self.assertEqual(len(bodies), 2)
            self.assertEqual(bodies[1]["input"], [
                {"role": "user", "content": "real source packet"},
                driver.screen.client_shape(first["output"][0]), driver.screen.client_shape(first["output"][1]),
                {"role": "user", "content": "followup question"}])
            self.assertEqual(bodies[0]["tools"], [])
            self.assertEqual(bodies[0]["instructions"], bodies[1]["instructions"])
            self.assertEqual(client.records[0]["usage"]["input_tokens"], 42001)
            self.assertEqual(client.records[1]["workspace_rows"], 2048)

    def test_incomplete_or_pending_response_never_gets_followup(self):
        pending = response("some answer")
        pending["output"].append({"type": "function_call", "name": "read", "call_id": "x", "arguments": "{}"})
        for first in (response("partial", "incomplete"), response("", "completed"), pending):
            with self.subTest(first=first):
                client, events = Mock(), []
                client.send.return_value = first
                result = driver.two_requests(client, "packet", "followup", events.append)
                client.send.assert_called_once()
                self.assertFalse(result["sent"])
                self.assertEqual(events[-1]["event"], "followup_skipped")

    def test_incomplete_http_output_is_preserved_in_records_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "serve.log"
            log.write_text("")
            events = []
            client = driver.screen.Screen("127.0.0.1:1", "model", {"instructions": driver.INSTRUCTIONS, "tools": []},
                                          "low", 4096, log, events.append)
            partial = response("unfinished operator table", "incomplete")
            http = Mock(status=200)
            http.getheader.return_value = None
            http.read1.side_effect = [
                ("data: " + json.dumps({"type": "response.incomplete", "response": partial}) + "\n\n").encode(), b""]
            connection = Mock()
            connection.getresponse.return_value = http
            with patch.object(driver.screen.http.client, "HTTPConnection", return_value=connection), \
                 patch.object(driver.screen.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                result = driver.two_requests(client, "packet", "followup", events.append)
            self.assertFalse(result["sent"])
            connection.request.assert_called_once()
            connection.close.assert_called_once()
            self.assertEqual(client.records[0]["response"], partial)
            self.assertEqual(client.records[0]["answer_text"], "unfinished operator table")
            self.assertEqual(client.records[0]["incomplete_details"], {"reason": "max_output_tokens"})
            self.assertEqual(next(e["response"] for e in events if e["event"] == "terminal_response"), partial)

    def test_invalid_limits_rejected_before_spawning(self):
        for context, output in ((0, 4096), (-1, 4096), (65536, 0), (65536, -1), (4096, 4096),
                                (1024, 4096), (2**32, 4096), (65536, 2**32)):
            with self.subTest(context=context, output=output), patch.object(driver.subprocess, "Popen") as start, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    driver.main(["--qwen", "fake", "--model", "fake", "--out", "unused",
                                 "--max-context-tokens", str(context), "--max-tokens", str(output)])
                start.assert_not_called()

    def test_failed_server_preserves_checkpoint_and_cleans_up_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sources(root)
            binary, model, out = root / "fake-qwen", root / "fake.gguf", root / "out"
            binary.write_bytes(b"not executable")
            model.write_bytes(b"not a model")
            prior = root / "prior.json"
            prior.write_text(json.dumps({"requests": [{"label": "A'", "reuse_source": "snapshot", "reused_tokens": 11104, "prefill_tokens": 303}]}))
            owned = Mock(pid=12345)
            owned.poll.side_effect = [None, 130]
            env = {"QWEN_SERVE_IDLE_RESIDENCY_SECS": "30", "QWEN_METAL_LEASE_WAIT": "1", "MTL_DEBUG_LAYER": "1"}
            with patch.object(driver, "ROOT", root), patch.object(driver, "PRIOR_REPORT", prior), \
                 patch.dict(driver.os.environ, env, clear=True), \
                 patch.object(driver.subprocess, "check_output", side_effect=["revision\n", ""]), \
                 patch.object(driver.subprocess, "Popen", return_value=owned) as start, \
                 patch.object(driver.screen, "wait_listening", side_effect=SystemExit("server exited early")):
                with self.assertRaisesRegex(SystemExit, "server exited early"):
                    driver.main(["--qwen", str(binary), "--model", str(model), "--out", str(out)])
            owned.send_signal.assert_called_once_with(driver.screen.signal.SIGINT)
            owned.wait.assert_called_once_with(timeout=180)
            owned.terminate.assert_not_called()
            self.assertEqual(start.call_args.kwargs["env"], env)
            command = start.call_args.args[0]
            self.assertNotIn("--snapshot-cache-mib", command)
            self.assertIn("65536", command)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["inherited_overrides"], env)
            self.assertEqual(report["timing_mode"], "validation-on")
            self.assertEqual(report["exit_code"], 130)
            self.assertEqual(report["requests"], [])
            self.assertEqual(report["packet_sha256"], driver.screen.file_sha(out / "packet.txt"))
            self.assertTrue((out / "packet.json").exists())
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["event"], "end")


if __name__ == "__main__":
    unittest.main()
