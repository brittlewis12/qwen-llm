"""CPU-only: uv run --no-project python -B -m unittest discover -s scripts/bench -p test_dflash_native_replay.py"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_native_replay as replay


COMMIT = "a" * 40
LOGS = {
    "flash-next": "qwen4exp dflash: emitted_tokens=16 packets=2 draft_calls=2 proposed_tokens=14 "
    "accepted_drafts=7 replay_inputs=1 draft_ms=1.234 target_command_timings=unavailable\n",
    "glm-flash": "glm5_next dflash: verification=configured_packed initial_selection_ms=0.100 "
    "scratch_bytes=1024 counters=Glm5NextDFlashCounters { prefill_tokens: 20, "
    "attempts: 2, packets: 2, draft_calls: 2, drafted_tokens: 14, verified_rows: 16, "
    "accepted_drafts: 7, replayed_inputs: 1, failures: 0 }\n",
}


def jsonl_row(row):
    return (json.dumps(row, allow_nan=False) + "\n").encode()


def stats(family, cap, spec=False):
    label, template = replay.FAMILIES[family]
    return {
        "schema": "qwen-llm.request-stats",
        "schema_version": 2,
        "record_type": "request_stats",
        "invocation_id": "fixture",
        "request_index": 0,
        "status": "ok",
        "model": {"family": label},
        "input": {"kind": "messages", "template": template},
        "usage": {"input_tokens": 20, "output_tokens": cap},
        "finish": {"reason": "token_limit"},
        "timing_ms": {"total": 200, "tokenization": 0, "prefill": 100, "decode": 100},
        "throughput_tps": {"prefill": 200, "decode": 40 if spec else 20},
        "output_fingerprint": {
            "algorithm": "sha256-qwen-generated-token-ids-v1",
            "value": "b" * 64,
        },
        "build": {"commit": COMMIT, "dirty": False},
    }


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source.json"
        self.messages = [{"role": "system", "content": "A public toy conversation."}]
        self.messages += [
            {"role": "user" if i % 2 else "assistant", "content": f"Toy turn {i}"}
            for i in range(1, 16)
        ]
        self.source_bytes = replay.json_bytes(
            {
                "messages": self.messages,
                "temperature": 0,
                "tools": [{"ignored": True}],
                "private": "not copied",
            }
        )
        self.source.write_bytes(self.source_bytes)
        for name in ("binary", "model", "drafter"):
            (self.root / name).write_bytes(b"inert test fixture; never executed\n")
        (self.root / "binary").chmod(0o700)
        self.argv = []
        for key, value in {
            "binary": self.root / "binary",
            "model": self.root / "model",
            "drafter": self.root / "drafter",
            "source": self.source,
            "output-dir": self.root / "output",
            "expected-commit": COMMIT,
            "family": "flash-next",
        }.items():
            self.argv += [f"--{key}", str(value)]
        self.args = replay.parser().parse_args(self.argv)
        self.calls = []

    def manifest(self):
        return json.loads((self.args.output_dir / "manifest.json").read_text())

    def fake_popen(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        snapshot = self.manifest()
        self.assertEqual(snapshot["runs"][-1]["status"], "running")
        self.assertEqual(snapshot["runs"][-1]["command"], cmd)
        cap = int(cmd[cmd.index("--max-tokens") + 1])
        spec = "--drafter" in cmd
        path = Path(cmd[cmd.index("--request-stats-jsonl") + 1])
        path.write_bytes(jsonl_row(stats(self.args.family, cap, spec)))
        kwargs["stdout"].write(b"fixture model output\n")
        kwargs["stderr"].write(
            LOGS[self.args.family].encode() if spec else b"serial fixture\n"
        )
        child = Mock(pid=4242, returncode=0)
        child.wait.return_value = 0
        child.poll.return_value = 0
        return child

    def test_success_schedule_hashes_private_artifacts_and_aggregates(self):
        with (
            patch.object(replay.subprocess, "Popen", side_effect=self.fake_popen),
            patch.dict(
                os.environ,
                {
                    "HOME": str(self.root),
                    "PATH": "/usr/bin",
                    "QWEN_DFLASH_SHADOW_PROBE": "1",
                    "DYLD_INSERT_LIBRARIES": "bad",
                    "MTL_DEBUG_LAYER": "0",
                    "RUST_LOG": "trace",
                },
                clear=True,
            ),
        ):
            result = replay.run(self.args)
        self.assertEqual(result, self.manifest())
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(
            [(r["phase"], r["mode"], r["output_cap"]) for r in result["runs"]],
            [
                ("warmup", "serial", 16),
                ("warmup", "spec", 16),
                ("measured", "serial", 256),
                ("measured", "spec", 256),
                ("measured", "spec", 256),
                ("measured", "serial", 256),
            ],
        )
        frozen = self.args.output_dir / "messages.json"
        self.assertEqual(
            json.loads(frozen.read_text()), {"messages": self.messages[:14]}
        )
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        for path, digest in result["input_sha256"].items():
            self.assertEqual(
                hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest
            )
        self.assertIn("QWEN_DFLASH_SHADOW_PROBE", result["ignored_ambient_names"])
        for cmd, kwargs in self.calls:
            self.assertEqual(cmd[:2], [str(self.args.binary), "run"])
            self.assertEqual(cmd[cmd.index("--temp") + 1], "1")
            self.assertEqual(cmd[cmd.index("--seed") + 1], "42")
            for flag in (
                "--top-k",
                "--top-p",
                "--min-p",
                "--reasoning-effort",
                "--no-thinking",
            ):
                self.assertNotIn(flag, cmd)
            self.assertEqual(
                kwargs["env"],
                {"HOME": str(self.root), "PATH": "/usr/bin", **replay.FIXED_ENV},
            )
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(stat.S_IMODE(self.args.output_dir.stat().st_mode), 0o700)
        for file in self.args.output_dir.iterdir():
            self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o600, file)
        for record in result["runs"]:
            for key, digest in record["artifact_sha256"].items():
                self.assertEqual(
                    replay.sha256(self.args.output_dir / record[key]), digest
                )
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["arithmetic_means"]["serial"]["output_tokens"], 256)
        self.assertEqual(aggregate["spec_over_serial"]["decode_tps"], 2)
        self.assertIsNone(aggregate["spec_over_serial"]["tokenization_ms"])

    def test_glm_native_schema_and_counter_names(self):
        self.args.family = "glm-flash"
        with patch.object(replay.subprocess, "Popen", side_effect=self.fake_popen):
            result = replay.run(self.args)
        record = result["runs"][1]
        self.assertEqual(record["input"]["template"], "glm53_flash_chat_v2")
        self.assertEqual(record["counters"]["drafted_tokens"], 14)
        self.assertNotIn("proposed_tokens", record["counters"])

    def test_nonzero_exit_preserved_and_stops(self):
        def fail(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            child.returncode = child.wait.return_value = child.poll.return_value = 7
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=fail),
            self.assertRaisesRegex(ValueError, "exited 7"),
        ):
            replay.run(self.args)
        result = self.manifest()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["status"], "failed")
        record = result["runs"][0]
        self.assertEqual(record["returncode"], 7)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["artifact_sha256"]), 3)
        self.assertNotIn("aggregate", result)
        self.assertGreaterEqual(record["elapsed_seconds"], 0)

    def test_spawn_error_still_records_attempt(self):
        with (
            patch.object(
                replay.subprocess, "Popen", side_effect=OSError("cannot spawn")
            ),
            self.assertRaises(OSError),
        ):
            replay.run(self.args)
        record = self.manifest()["runs"][0]
        self.assertEqual(record["status"], "failed")
        self.assertIn("cannot spawn", record["error"])
        self.assertEqual(len(record["artifact_sha256"]), 3)

    def test_interruption_reaps_only_owned_child_and_records_partial_files(self):
        child = None

        def interrupted(cmd, **kwargs):
            nonlocal child
            child = self.fake_popen(cmd, **kwargs)
            child.returncode = -15
            child.poll.return_value = None
            child.wait.side_effect = [KeyboardInterrupt("test interrupt"), -15]
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=interrupted),
            self.assertRaises(KeyboardInterrupt),
        ):
            replay.run(self.args)
        child.terminate.assert_called_once_with()
        child.kill.assert_not_called()
        result = self.manifest()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["runs"][0]["status"], "interrupted")
        self.assertEqual(result["runs"][0]["returncode"], -15)
        self.assertEqual(len(result["runs"][0]["artifact_sha256"]), 3)
        self.assertNotIn("aggregate", result)

    def test_termination_waits_for_owned_child_without_forced_gpu_exit(self):
        child = Mock()
        child.poll.return_value = None
        replay.stop_child(child)
        child.terminate.assert_called_once_with()
        child.kill.assert_not_called()
        child.wait.assert_called_once_with()

    def test_sigterm_handler_records_interruption_and_restores_handlers(self):
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            child.wait.side_effect = lambda: signal.getsignal(signal.SIGTERM)(
                signal.SIGTERM, None
            )
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=interrupted),
            patch("sys.stderr"),
        ):
            self.assertEqual(replay.main(self.argv), 130)
        self.assertEqual(self.manifest()["status"], "interrupted")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_stats_contract_rejects_guessed_or_invalid_fields(self):
        path = self.root / "stats.jsonl"
        base = stats("flash-next", 16)
        changes = [
            ("build", {"commit": "b" * 40, "dirty": False}),
            ("build", {"commit": COMMIT, "dirty": 0}),
            ("build", {"commit": COMMIT, "dirty": True}),
            ("build", {"commit": COMMIT}),
            ("build", None),
            ("schema_version", 1),
            ("status", "error"),
            ("input", {"kind": "messages", "template": "generic"}),
            ("input", {"kind": "raw", "template": "qwen38"}),
            ("model", {"family": "glm5_next"}),
            ("usage", {"input_tokens": 20, "output_tokens": 17}),
            ("usage", {"input_tokens": 20, "output_tokens": True}),
            ("usage", {"input_tokens": 20, "output_tokens": 15}),
            ("timing_ms", {"total": -1}),
            ("throughput_tps", {"prefill": 0, "decode": 5}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(base)
                row[key] = value
                path.write_bytes(jsonl_row(row))
                with self.assertRaises(ValueError):
                    replay.validate_stats(path, self.args, 16)
        for contents in (
            b"",
            b"not json",
            jsonl_row(base) * 2,
            b'{"schema_version":2,"schema_version":2}',
            b'{"x":NaN}',
        ):
            path.write_bytes(contents)
            with self.assertRaises(ValueError):
                replay.validate_stats(path, self.args, 16)
        base["usage"]["output_tokens"] = 3
        base["finish"]["reason"] = "eos"
        path.write_bytes(jsonl_row(base))
        replay.validate_stats(path, self.args, 16)

    def test_validation_failure_persisted_before_next_run(self):
        def wrong_template(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            path = Path(cmd[cmd.index("--request-stats-jsonl") + 1])
            row = json.loads(path.read_text())
            row["input"]["template"] = "wrong"
            path.write_bytes(jsonl_row(row))
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=wrong_template),
            self.assertRaisesRegex(ValueError, "template"),
        ):
            replay.run(self.args)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.manifest()["runs"][0]["status"], "failed")

    def test_late_failure_keeps_successful_runs_without_aggregate(self):
        def fail_fourth(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            if len(self.calls) == 4:
                child.returncode = child.poll.return_value = child.wait.return_value = 9
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=fail_fourth),
            self.assertRaises(ValueError),
        ):
            replay.run(self.args)
        result = self.manifest()
        self.assertEqual(
            [row["status"] for row in result["runs"]],
            ["succeeded", "succeeded", "succeeded", "failed"],
        )
        self.assertNotIn("aggregate", result)

    def test_missing_spec_summary_fails_and_preserves_empty_stderr(self):
        def missing(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            if "--drafter" in cmd:
                kwargs["stderr"].flush()
                Path(kwargs["stderr"].name).write_bytes(b"")
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=missing),
            self.assertRaisesRegex(ValueError, "draft_calls"),
        ):
            replay.run(self.args)
        result = self.manifest()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result["runs"][-1]["counters"], {})
        self.assertEqual(
            (self.args.output_dir / result["runs"][-1]["stderr"]).read_bytes(), b""
        )

    def test_invalid_arguments_never_launch(self):
        for key, value in (
            ("expected_commit", "abc123"),
            ("output_cap", 0),
            ("messages_count", -1),
        ):
            with (
                self.subTest(key=key),
                patch.object(replay.subprocess, "Popen") as popen,
            ):
                args = copy.copy(self.args)
                setattr(args, key, value)
                with self.assertRaises(ValueError):
                    replay.run(args)
                popen.assert_not_called()
                self.assertFalse(args.output_dir.exists())

    def test_missing_or_zero_draft_calls_fails_spec_warmup(self):
        def no_draft(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            if "--drafter" in cmd:
                kwargs["stderr"].flush()
                Path(kwargs["stderr"].name).write_text(
                    "qwen4exp dflash: draft_calls=0\n"
                )
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=no_draft),
            self.assertRaisesRegex(ValueError, "draft_calls"),
        ):
            replay.run(self.args)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.manifest()["runs"][-1]["counters"], {"draft_calls": 0})
        self.assertEqual(
            replay.extract_counters("unrelated draft_calls=90\n", "flash-next"), {}
        )

    def test_counter_extraction_is_conservative(self):
        for family, text in LOGS.items():
            with self.subTest(family=family):
                counters = replay.extract_counters(text, family)
                self.assertEqual(counters["draft_calls"], 2)
                self.assertNotIn("draft_ms", counters)
                self.assertNotIn("scratch_bytes", counters)
                with self.assertRaises(ValueError):
                    replay.extract_counters(text * 2, family)
        self.assertEqual(
            replay.extract_counters("qwen4exp dflash: draft_calls=2.5\n", "flash-next"),
            {},
        )
        with self.assertRaises(ValueError):
            replay.extract_counters("glm5_next dflash: counters=unknown\n", "glm-flash")

    def test_input_mutation_fails_before_next_process(self):
        def mutation(cmd, **kwargs):
            child = self.fake_popen(cmd, **kwargs)
            Path(cmd[cmd.index("--messages") + 1]).write_text("changed")
            return child

        with (
            patch.object(replay.subprocess, "Popen", side_effect=mutation),
            self.assertRaisesRegex(ValueError, "input changed"),
        ):
            replay.run(self.args)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_source_and_output_collisions_and_existing_symlinks_rejected(self):
        paths = [self.source, self.root]
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        alias = self.root / "alias"
        alias.symlink_to(self.source)
        paths += [dangling, alias]
        for path in paths:
            with (
                self.subTest(path=path),
                patch.object(replay.subprocess, "Popen") as popen,
            ):
                self.args.output_dir = path
                with self.assertRaises(ValueError):
                    replay.run(self.args)
                popen.assert_not_called()
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_no_overwrite_after_success(self):
        with patch.object(replay.subprocess, "Popen", side_effect=self.fake_popen):
            replay.run(self.args)
        before = (self.args.output_dir / "manifest.json").read_bytes()
        with (
            patch.object(replay.subprocess, "Popen") as popen,
            self.assertRaises(ValueError),
        ):
            replay.run(self.args)
        popen.assert_not_called()
        self.assertEqual((self.args.output_dir / "manifest.json").read_bytes(), before)

    def test_freeze_rejects_bad_prefix_without_creating_output(self):
        for count in (0, -1, 13, 100):
            with (
                self.subTest(count=count),
                patch.object(replay.subprocess, "Popen") as popen,
            ):
                self.args.messages_count = count
                with self.assertRaises(ValueError):
                    replay.run(self.args)
                self.assertFalse(self.args.output_dir.exists())
                popen.assert_not_called()
        self.assertEqual(
            json.loads(replay.freeze_messages(replay.json_bytes(self.messages), 14)),
            {"messages": self.messages[:14]},
        )


if __name__ == "__main__":
    unittest.main()
