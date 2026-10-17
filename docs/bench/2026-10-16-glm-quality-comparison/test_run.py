"""CPU-only tests; never starts the native model server."""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import run


def terminal(incomplete=False):
    return {
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "think"}],
            },
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        ],
        "usage": {
            "input_tokens": 9,
            "output_tokens": 3,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "x_qwen": {"matched_tokens": 0},
    }


def frame(kind, data):
    return f"event: {kind}\r\ndata: {json.dumps(data)}\r\n\r\n".encode()


DONE = b"data: [DONE]\n\n"


def failed_terminal():
    response = terminal()
    response.update(
        status="failed",
        usage=None,
        incomplete_details=None,
        error={
            "type": "server_error",
            "code": "",
            "param": "",
            "message": "invalid GLM-5.3-Flash model output: stop before reasoning close",
        },
    )
    del response["x_qwen"]
    response["output"] = response["output"][:1]
    return response


def native_stream(incomplete=False, failed=False):
    # events.rs lifecycle order and SseWriter framing; http.rs adds ping and DONE.
    response = failed_terminal() if failed else terminal(incomplete)
    status = response["status"]
    reasoning = response["output"][0]
    reasoning.update(
        id="rs_fixture", status="incomplete" if failed else "completed", summary=[]
    )
    if not failed:
        message = response["output"][1]
        message.update(id="msg_fixture", status=status, role="assistant")
        message["content"][0]["annotations"] = []
    opening = {
        **response,
        "status": "in_progress",
        "output": [],
        "usage": None,
        "incomplete_details": None,
        "error": None,
    }
    lifecycle = [
        ("response.created", {"response": opening}),
        ("response.in_progress", {"response": opening}),
    ]
    for index, item in enumerate(response["output"]):
        content = item["content"][0]
        common = {"item_id": item["id"], "output_index": index, "content_index": 0}
        lifecycle.extend(
            [
                (
                    "response.output_item.added",
                    {
                        "output_index": index,
                        "item": {**item, "status": "in_progress", "content": []},
                    },
                ),
                (
                    "response.content_part.added",
                    {**common, "part": {**content, "text": ""}},
                ),
            ]
        )
        kind = "reasoning" if index == 0 else "output_text"
        lifecycle.append(
            (f"response.{kind}.delta", {**common, "delta": content["text"]})
        )
        if index == 0:
            lifecycle.append(
                ("response.reasoning_text.delta", {**common, "delta": content["text"]})
            )
        lifecycle.extend(
            [
                (f"response.{kind}.done", {**common, "text": content["text"]}),
                ("response.content_part.done", {**common, "part": content}),
                ("response.output_item.done", {"output_index": index, "item": item}),
            ]
        )
    lifecycle.append((f"response.{status}", {"response": response}))
    wire = b": ping\n\n"
    for sequence, (kind, payload) in enumerate(lifecycle, 1):
        wire += frame(
            kind, {**payload, "type": kind, "sequence_number": sequence}
        ).replace(b"\r\n", b"\n")
    return wire + DONE


class StreamTests(unittest.TestCase):
    def test_failed_outcome_still_requires_complete_transport(self):
        with self.assertRaisesRegex(run.InvalidRun, "missing DONE"):
            self.parse(native_stream(failed=True).removesuffix(DONE))

    def test_failed_visible_text_is_kept_separate_from_reasoning(self):
        response = failed_terminal()
        response["error"]["message"] = (
            "invalid GLM-5.3-Flash model output: unsupported stop token"
        )
        response["output"].append(terminal()["output"][1])
        result = run.validate_response(response)
        self.assertEqual(result["answer"], "answer")
        self.assertEqual(result["reasoning"], "think")

    def test_actual_failed_wire_is_model_outcome(self):
        (response, timing), events = self.parse(native_stream(failed=True))
        result = run.validate_response(response)
        self.assertEqual(result["status"], "model_output_error")
        self.assertEqual(result["reasoning"], "think")
        self.assertEqual(result["answer"], "")
        self.assertIsNone(result["usage"])
        self.assertIsNone(result["x_qwen"])
        self.assertIsNone(result["native_stop_reason"])
        self.assertIsNone(timing["first_visible_answer_s"])
        self.assertGreaterEqual(timing["request_elapsed_s"], 0)
        self.assertEqual(
            json.loads(events.splitlines()[-2])["event"], "response.failed"
        )
        self.assertEqual(json.loads(events.splitlines()[-1])["data"], "[DONE]")

    def test_failed_classification_exact_whitelist(self):
        for key, value in (
            ("message", "out of memory"),
            ("message", "invalid K2 model output: stop before reasoning close"),
            (
                "message",
                "invalid GLM-5.3-Flash model output: stop before reasoning close extra",
            ),
            ("code", "engine_error"),
            ("param", "model"),
            ("type", "invalid_request"),
        ):
            response = failed_terminal()
            response["error"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(run.InvalidRun):
                run.validate_response(response)
        for message in run.MODEL_OUTPUT_ERRORS:
            response = failed_terminal()
            response["error"]["message"] = message
            self.assertEqual(
                run.validate_response(response)["status"], "model_output_error"
            )

    def parse(self, wire):
        raw, events = io.BytesIO(), io.StringIO()
        result = run.parse_sse(io.BytesIO(wire), raw, events, time.monotonic())
        self.assertEqual(raw.getvalue(), wire)
        return result, events.getvalue()

    def test_duplicate_reasoning_alias_not_counted(self):
        wire = frame("response.reasoning_text.delta", {"delta": "duplicate"})
        wire += frame("response.reasoning.delta", {"delta": ""})
        wire += frame("response.reasoning.delta", {"delta": "think"})
        wire += frame("response.output_text.delta", {"delta": "answer"})
        wire += frame("response.completed", {"response": terminal()})
        wire += DONE
        with patch.object(run.time, "monotonic", side_effect=range(8)):
            (response, timing), events = self.parse(wire)
        self.assertEqual(timing["client_ttft_s"], 3)
        self.assertEqual(timing["first_visible_answer_s"], 4)
        self.assertEqual(len(events.splitlines()), 6)
        result = run.validate_response(response)
        self.assertEqual(result["answer"], "answer")
        self.assertEqual(result["reasoning"], "think")

    def test_incomplete_is_outcome(self):
        (response, _), _ = self.parse(
            frame("response.incomplete", {"response": terminal(True)}) + DONE
        )
        self.assertEqual(
            run.validate_response(response)["native_stop_reason"], "token_limit"
        )

    def test_missing_and_duplicate_terminal(self):
        end = frame("response.completed", {"response": terminal()})
        for wire in (
            b"",
            frame("response.output_text.delta", {"delta": "a"}),
            end + end,
            frame("error", {"message": "bad"}),
            end.rstrip(),
            end,
            DONE,
            DONE + end,
            end + DONE + DONE,
            end + DONE + frame("response.output_text.delta", {"delta": "late"}),
            frame("response.completed", {"response": terminal(True)}) + DONE,
        ):
            with self.subTest(wire=wire), self.assertRaises(run.InvalidRun):
                self.parse(wire)

    def test_multiline_data_and_comments(self):
        wire = b': keepalive\n\nevent: response.completed\ndata: {\ndata: "response": '
        wire += json.dumps(terminal()).encode() + b"}\n\n" + DONE
        self.parse(wire)

    def test_native_completed_and_incomplete_wire_fixtures(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                wire = native_stream(incomplete)
                with patch.object(run.time, "monotonic", side_effect=range(100)):
                    (response, timing), captured = self.parse(wire)
                records = [json.loads(line) for line in captured.splitlines()]
                self.assertEqual(
                    records[-1],
                    {"event": "[DONE]", "data": "[DONE]", "seconds": len(records)},
                )
                self.assertEqual(
                    records[-2]["event"],
                    "response.incomplete" if incomplete else "response.completed",
                )
                self.assertEqual(timing["client_ttft_s"], 5)
                self.assertEqual(timing["first_visible_answer_s"], 12)
                self.assertEqual(timing["request_elapsed_s"], len(records) + 1)
                result = run.validate_response(response)
                self.assertEqual(result["answer"], "answer")
                self.assertEqual(result["reasoning"], "think")
                self.assertEqual(
                    result["native_stop_reason"], "token_limit" if incomplete else "eos"
                )

    def test_premature_done_preserves_raw_and_timestamp(self):
        raw, events = io.BytesIO(), io.StringIO()
        with patch.object(run.time, "monotonic", return_value=12):
            with self.assertRaisesRegex(run.InvalidRun, "premature DONE"):
                run.parse_sse(io.BytesIO(DONE), raw, events, 10)
        self.assertEqual(raw.getvalue(), DONE)
        self.assertEqual(
            json.loads(events.getvalue()),
            {"seconds": 2, "event": "[DONE]", "data": "[DONE]"},
        )

    def test_usage_cache_validation(self):
        for field, value in (
            ("input_tokens", -1),
            ("total_tokens", 99),
            ("output_tokens", True),
        ):
            response = terminal()
            response["usage"][field] = value
            with self.assertRaises(run.InvalidRun):
                run.validate_response(response)
        for missing in (False, True):
            for container, key in (
                ("x_qwen", "matched_tokens"),
                ("usage", "input_tokens_details"),
            ):
                response = terminal()
                if missing:
                    del response[container][key]
                else:
                    response[container][key] = (
                        1 if container == "x_qwen" else {"cached_tokens": 1}
                    )
                with self.assertRaises(run.InvalidRun):
                    run.validate_response(response)


PHASE = (
    "INFO qwen_diag: serve phases: family=glm5_next prefill_ms=123.4 reused_tokens=0 "
    "reuse_source=none restore_origin=none restore_ms=0.0 boundary_ms=0.0 "
    "prefill_tokens=9 decode_ms=56.7 required_forwards=3 capacity=32768 transitions=2 "
    "lineage=fast snapshot_schedule=off cuts=[] captures=[] snapshot_capture_ms=0.0 "
    "snapshot_cache_bytes=0 snapshot_cache_entries=0 checkpoint_target=0 "
    "checkpoint_captures=[] checkpoint_capture_ms=0.0\n"
)
STATS = (
    "INFO qwen_diag: serve stats: version=serve_stats_v1 prompt_tokens=9 generated_tokens=3 "
    "stop_reason=eos matched_tokens=0 restore_ms=0.0 decode_tps=52.91\n"
)


PLAN = "serve: glm5_next prefill plan rows=512 session_bytes=123456 prefix_budget=0 prefix_preference_held=true\n"


class NativeTests(unittest.TestCase):
    def test_post_allocation_plan_rows_not_startup_max(self):
        startup = "serve limits: family=glm5_next capacity=32768 prefill_rows_max=2048 fast_precision=half\n"
        for rows in (512, 0):
            with self.subTest(rows=rows):
                text = (
                    startup + PLAN.replace("rows=512", f"rows={rows}") + PHASE + STATS
                )
                self.assertEqual(run.startup_descriptor(text)["prefill_rows_max"], 2048)
                plan = run.prefill_plan(text)
                self.assertEqual(plan["rows"], rows)
                self.assertEqual(plan["session_bytes"], 123456)
                self.assertEqual(plan["prefix_budget"], 0)
                self.assertTrue(plan["prefix_preference_held"])
                self.assertEqual(plan["stderr_line"], 2)
                self.assertEqual(run.prefill_plan(text + PHASE + STATS), plan)
        self.assertIsNone(run.prefill_plan(startup + PHASE + STATS))
        self.assertIsNone(run.prefill_plan(""))
        for bad in (
            PLAN.replace("rows=512", "rows=-1"),
            PLAN + PLAN,
            PLAN.replace(
                "prefix_preference_held=true", "prefix_preference_held=unknown"
            ),
        ):
            with self.assertRaises(run.InvalidRun):
                run.prefill_plan(bad)

    def test_startup_descriptor_is_max_not_selected(self):
        descriptor = run.startup_descriptor(
            "serve limits: family=glm5_next input=verified_chat_and_tools renderer=glm5_next "
            "capacity=32768 default_max_tokens=6144 prefill_rows_max=128 fast_precision=half snapshots=off\n"
        )
        self.assertEqual(descriptor["capacity"], 32768)
        self.assertEqual(descriptor["prefill_rows_max"], 128)
        self.assertEqual(descriptor["fast_precision"], "half")
        self.assertIsNone(descriptor["selected_prefill_rows"])

    def test_failed_accounting_unavailable_is_null(self):
        result = run.validate_response(failed_terminal())
        native = run.parse_stats("", 1, result)
        for field in ("prefill_ms", "decode_ms", "transitions", "phases", "stats"):
            self.assertIsNone(native[field])
        observed = run.parse_stats(PHASE + STATS, 1, result)
        self.assertEqual(observed["prefill_ms"], 123.4)
        with self.assertRaises(run.InvalidRun):
            run.parse_stats(
                (PHASE + STATS).replace("reused_tokens=0", "reused_tokens=1"), 1, result
            )

    def test_model_failure_restart_no_retry_or_dropped_case(self):
        for unknown in (False, True):
            with (
                self.subTest(unknown=unknown),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cases = [
                    {"id": str(i), "system": "s", "user": str(i), "max_tokens": 10}
                    for i in range(8)
                ]
                (root / "cases.json").write_text(json.dumps(cases))
                (root / "manifest.json").write_text("{}")
                children = [Mock(), Mock()]
                for child in children:
                    child.poll.return_value = None
                logs = [
                    root / "out" / f"server-{i:02d}" / "server-1.stderr.log"
                    for i in (1, 2)
                ]
                connection = Mock()
                requests = []
                real_evaluate = run.evaluate

                def evaluate(port, body, output):
                    index = int(body["input"][0]["content"])
                    requests.append(index)
                    wire = native_stream(failed=index in (1, 7))
                    if unknown and index == 1:
                        wire = wire.replace(
                            b"invalid GLM-5.3-Flash model output: stop before reasoning close",
                            b"device lost",
                        )
                    response = io.BytesIO(wire)
                    response.status = 200
                    response.getheader = lambda *args: "text/event-stream"
                    response.getheaders = lambda: [
                        ("Content-Type", "text/event-stream")
                    ]
                    connection.getresponse.return_value = response
                    with logs[0 if index < 2 else 1].open("a") as log:
                        if index in (0, 2):
                            log.write(
                                PLAN
                                if index == 0
                                else PLAN.replace("rows=512", "rows=0")
                            )
                        log.write(PHASE + STATS)
                    return real_evaluate(port, body, output)

                def launch(*args):
                    index = launch.calls
                    if index:
                        children[0].terminate.assert_called_once_with()
                        children[0].wait.assert_called_once_with(timeout=10)
                    launch.calls += 1
                    logs[index].write_text(
                        "serve limits: family=glm5_next capacity=32768 "
                        "prefill_rows_max=2048 fast_precision=half\n"
                    )
                    return children[index], 12345, "native-id", logs[index]

                launch.calls = 0
                with (
                    patch.object(run, "environment", return_value={}),
                    patch.object(run, "integrity", return_value=({}, [])),
                    patch.object(run, "memory", return_value={}),
                    patch.object(run, "start_server", side_effect=launch) as start,
                    patch.object(run, "evaluate", side_effect=evaluate),
                    patch.object(
                        run.http.client, "HTTPConnection", return_value=connection
                    ),
                    patch.object(run, "check_answer", return_value={"partial": True}),
                ):
                    code = run.main(
                        [
                            "--binary",
                            str(root / "binary"),
                            "--model",
                            str(root / "model"),
                            "--out",
                            str(root / "out"),
                            "--cases",
                            str(root / "cases.json"),
                            "--manifest",
                            str(root / "manifest.json"),
                            "--arm",
                            "opaque",
                        ]
                    )
                summary = json.loads((root / "out" / "summary.json").read_text())
                self.assertEqual(code, int(unknown))
                self.assertEqual(requests, list(range(2 if unknown else 8)))
                self.assertEqual(start.call_count, 1 if unknown else 2)
                self.assertEqual(
                    summary["status"], "aborted" if unknown else "completed"
                )
                children[0].terminate.assert_called_once_with()
                children[0].kill.assert_not_called()
                if unknown:
                    children[1].terminate.assert_not_called()
                    continue
                children[1].terminate.assert_called_once_with()
                children[1].kill.assert_not_called()
                self.assertEqual(
                    [r["server_request_ordinal"] for r in summary["results"]],
                    [1, 2, 1, 2, 3, 4, 5, 6],
                )
                self.assertEqual(
                    [r["selected_prefill_rows"] for r in summary["results"]],
                    [512, 512, 0, 0, 0, 0, 0, 0],
                )
                self.assertEqual(
                    summary["runner_sha256"], run.digest(Path(run.__file__))
                )
                self.assertEqual(
                    summary["checker_sha256"],
                    run.digest(Path(run.__file__).with_name("cases.py")),
                )
                for result in summary["results"]:
                    plan = result["prefill_plan"]
                    self.assertEqual(plan["server_number"], result["server_number"])
                    self.assertEqual(plan["observed_at_server_request_ordinal"], 1)
                    self.assertEqual(
                        plan["stderr_artifact"],
                        f"server-{result['server_number']:02d}/server-1.stderr.log",
                    )
                for index in (1, 7):
                    result = summary["results"][index]
                    self.assertEqual(result["status"], "model_output_error")
                    self.assertIsNone(result["usage"])
                    self.assertGreaterEqual(result["request_elapsed_s"], 0)
                    self.assertTrue(
                        (
                            root / "out" / f"case-{index + 1:02d}" / "response.json"
                        ).exists()
                    )
                for server in summary["servers"]:
                    self.assertGreaterEqual(server["startup_elapsed_s"], 0)
                    self.assertGreaterEqual(server["cleanup_elapsed_s"], 0)
                    self.assertIsNone(server["descriptor"]["selected_prefill_rows"])
                    self.assertEqual(server["descriptor"]["prefill_rows_max"], 2048)
                first, second = [call.args for call in start.call_args_list]
                self.assertEqual(first[:3], second[:3])
                self.assertEqual(first[4], second[4])
                self.assertNotEqual(first[3], second[3])

    def test_actual_native_schema(self):
        result = run.validate_response(terminal())
        measured = run.parse_stats(PHASE + STATS, 1, result)
        self.assertEqual(measured["prefill_ms"], 123.4)
        self.assertEqual(measured["decode_ms"], 56.7)
        self.assertEqual(measured["transitions"], 2)
        for text, ordinal in (
            (STATS, 1),
            (PHASE + STATS, 2),
            ((PHASE + STATS).replace("reused_tokens=0", "reused_tokens=1"), 1),
            ((PHASE + STATS).replace("stop_reason=eos", "stop_reason=token_limit"), 1),
        ):
            with self.assertRaises(run.InvalidRun):
                run.parse_stats(text, ordinal, result)
        run.parse_stats(
            PHASE + STATS.replace("eos", "token_limit"),
            1,
            run.validate_response(terminal(True)),
        )

    def test_cleanup_only_owned_child(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("owned", 10), 0]
        run.cleanup(child)
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)
        exited = Mock()
        exited.poll.return_value = 0
        run.cleanup(exited)
        exited.terminate.assert_not_called()

    def test_wall_timeout_and_restore(self):
        previous = run.signal.getsignal(run.signal.SIGALRM)
        with self.assertRaises(TimeoutError), run.deadline(0.01):
            time.sleep(0.1)
        self.assertEqual(run.signal.getsignal(run.signal.SIGALRM), previous)

    def test_startup_timeout_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Mock()
            child.poll.return_value = None
            with (
                patch.object(run, "free_port", return_value=12345),
                patch.object(run.subprocess, "Popen", return_value=child),
                patch.object(run.time, "monotonic", side_effect=[0, 601]),
            ):
                with self.assertRaises(TimeoutError):
                    run.start_server(
                        Path("binary"), Path("model"), None, Path(directory), {}
                    )
            child.terminate.assert_called_once_with()

    def test_request_timeout_closes_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            conn = Mock()
            conn.getresponse.side_effect = TimeoutError()
            with patch.object(run.http.client, "HTTPConnection", return_value=conn):
                with self.assertRaises(TimeoutError):
                    run.evaluate(12345, {}, Path(directory))
            conn.close.assert_called_once_with()
            self.assertTrue((Path(directory) / "request.json").exists())

    def test_no_health_probe_without_owned_listening_log(self):
        with tempfile.TemporaryDirectory() as directory:
            child = Mock()
            child.poll.side_effect = [None, 1, 1]
            with (
                patch.object(run, "free_port", return_value=12345),
                patch.object(run.subprocess, "Popen", return_value=child),
                patch.object(run.time, "sleep"),
                patch.object(run.http.client, "HTTPConnection") as http,
            ):
                with self.assertRaises(run.InvalidRun):
                    run.start_server(
                        Path("binary"), Path("model"), None, Path(directory), {}
                    )
                http.assert_not_called()

    def test_incomplete_continues_but_protocol_failure_aborts(self):
        for failure in (False, True):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cases = [
                    {"id": str(i), "system": "s", "user": "u", "max_tokens": 10}
                    for i in range(8)
                ]
                (root / "cases.json").write_text(json.dumps(cases))
                (root / "manifest.json").write_text("{}")
                log = root / "log"
                log.write_text("")
                child = Mock()
                child.poll.return_value = None
                outcomes = [run.validate_response(terminal(True)) for _ in range(8)]
                if failure:
                    outcomes[1] = run.InvalidRun("missing terminal response")
                with (
                    patch.object(run, "environment", return_value={}),
                    patch.object(run, "integrity", return_value=({}, [])),
                    patch.object(run, "memory", return_value={}),
                    patch.object(
                        run,
                        "start_server",
                        return_value=(child, 12345, "native-id", log),
                    ),
                    patch.object(run, "evaluate", side_effect=outcomes) as evaluate,
                    patch.object(run, "check_answer", return_value={"partial": True}),
                    patch.object(run, "parse_stats", return_value={}),
                ):
                    code = run.main(
                        [
                            "--binary",
                            str(root / "binary"),
                            "--model",
                            str(root / "model"),
                            "--out",
                            str(root / "out"),
                            "--cases",
                            str(root / "cases.json"),
                            "--manifest",
                            str(root / "manifest.json"),
                            "--arm",
                            "opaque",
                        ]
                    )
                summary = json.loads((root / "out" / "summary.json").read_text())
                self.assertEqual(code, int(failure))
                self.assertEqual(evaluate.call_count, 2 if failure else 8)
                self.assertEqual(summary["results"][0]["status"], "incomplete")
                self.assertIsNone(summary["results"][0]["selected_prefill_rows"])
                self.assertIsNone(summary["results"][0]["prefill_plan"])
                self.assertEqual(
                    summary["status"], "aborted" if failure else "completed"
                )
                child.terminate.assert_called_once_with()


class IntegrityTests(unittest.TestCase):
    def test_split_symlink_entrypoint_reaches_loader_and_is_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "qwen"
            binary.write_bytes(b"binary")
            shards = []
            for i in (1, 2):
                target = root / f"stock-{i}.gguf_file"
                target.write_bytes(f"weights-{i}".encode())
                entry = root / f"model-{i:05d}-of-00002.gguf"
                entry.symlink_to(target.name)
                shards.append(
                    {
                        "name": entry.name,
                        "path": str(target),
                        "size": target.stat().st_size,
                        "sha256": run.digest(target),
                    }
                )
            model = root / shards[0]["name"]
            manifest = {"binary_sha256": run.digest(binary), "shards": shards}
            stamps, public = run.integrity(binary, model, manifest)
            self.assertEqual(
                [p["name"] for p in public[1:]], [s["name"] for s in shards]
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "cases.json").write_text(
                json.dumps(
                    [
                        {
                            "id": f"../opaque/{i}",
                            "system": "s",
                            "user": " u ",
                            "max_tokens": 10,
                        }
                        for i in range(8)
                    ]
                )
            )
            with (
                patch.object(run, "environment", return_value={}),
                patch.object(run, "memory", return_value={}),
                patch.object(
                    run, "start_server", side_effect=run.InvalidRun("test stop")
                ) as start,
            ):
                code = run.main(
                    [
                        "--binary",
                        str(binary),
                        "--model",
                        str(model),
                        "--out",
                        str(root / "out"),
                        "--cases",
                        str(root / "cases.json"),
                        "--manifest",
                        str(root / "manifest.json"),
                        "--arm",
                        "opaque",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertEqual(start.call_args.args[1], model.absolute())
            self.assertNotEqual(start.call_args.args[1], model.resolve())
            model.unlink()
            model.symlink_to("stock-2.gguf_file")
            with self.assertRaises(run.InvalidRun):
                run.guard(stamps)

    def test_invalid_frozen_cases_rejected_before_output_or_server(self):
        for key, value in (
            ("user", ""),
            ("user", " \n"),
            ("user", 1),
            ("id", ""),
            ("id", " \t"),
            ("id", []),
        ):
            with (
                self.subTest(key=key, value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cases = [
                    {"id": str(i), "system": "s", "user": "u", "max_tokens": 10}
                    for i in range(8)
                ]
                cases[0][key] = value
                (root / "cases.json").write_text(json.dumps(cases))
                with (
                    patch.object(run, "environment", return_value={}),
                    patch.object(run, "start_server") as start,
                ):
                    with self.assertRaisesRegex(run.InvalidRun, "invalid case"):
                        run.main(
                            [
                                "--binary",
                                "qwen",
                                "--model",
                                "model.gguf",
                                "--arm",
                                "opaque",
                                "--out",
                                str(root / "out"),
                                "--cases",
                                str(root / "cases.json"),
                                "--manifest",
                                str(root / "manifest.json"),
                            ]
                        )
                start.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_manifest_and_after_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model = root / "qwen", root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"weights")
            manifest = {
                "binary_sha256": run.digest(binary),
                "shards": [
                    {
                        "name": model.name,
                        "path": str(model),
                        "size": 7,
                        "sha256": run.digest(model),
                    }
                ],
            }
            stamps, public = run.integrity(binary, model, manifest)
            self.assertNotIn(str(root), json.dumps(public))
            for key, value in (
                ("sha256", "0" * 64),
                ("size", 8),
                ("name", "../model.gguf"),
                ("path", str(root / "elsewhere.gguf")),
            ):
                bad = copy.deepcopy(manifest)
                bad["shards"][0][key] = value
                with self.assertRaises(run.InvalidRun):
                    run.integrity(binary, model, bad)
            bad = copy.deepcopy(manifest)
            bad["binary_sha256"] = "0" * 64
            with self.assertRaises(run.InvalidRun):
                run.integrity(binary, model, bad)
            model.write_bytes(b"changed")
            with self.assertRaises(run.InvalidRun):
                run.guard(stamps)

    def test_omitted_split_shards_rejected(self):
        manifest = {
            "shards": [
                {"name": "model-00001-of-00002.gguf", "size": 1, "sha256": "0" * 64}
            ]
        }
        with self.assertRaisesRegex(run.InvalidRun, "incomplete split"):
            run.integrity(Path("qwen"), Path("model-00001-of-00002.gguf"), manifest)

    def test_environment_refuses_unknown_without_disclosing_values(self):
        with patch.dict(os.environ, {"QWEN_UNKNOWN": "secret"}, clear=True):
            with self.assertRaisesRegex(run.InvalidRun, "unknown inherited") as error:
                run.environment()
            self.assertNotIn("secret", str(error.exception))
        with patch.dict(
            os.environ,
            {"MTL_DEBUG_LAYER": "1", "QWEN_GLM_PREFIX_REUSE": "1"},
            clear=True,
        ):
            env = run.environment()
            self.assertNotIn("MTL_DEBUG_LAYER", env)
            self.assertEqual(env["QWEN_GLM_PREFIX_REUSE"], "0")

    def test_request_contract(self):
        body = run.request_body(
            "native-id", {"system": "s", "user": "u", "max_tokens": 4096}
        )
        self.assertEqual(body["model"], "native-id")
        self.assertEqual(body["x_qwen"]["prefill_lineage"], "fast")
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])


if __name__ == "__main__":
    unittest.main()
