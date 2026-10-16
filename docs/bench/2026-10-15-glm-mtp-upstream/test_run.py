# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""CPU-only tests; no server, GPU, memory probes, or network access."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location(
    "bench_run", Path(__file__).with_name("run.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class HarnessTests(unittest.TestCase):
    def test_fixed_work_and_short_eos(self):
        response = {
            "tokens": list(range(128)),
            "content": "test",
            "tokens_predicted": 128,
            "tokens_evaluated": 32768,
        }
        self.assertTrue(bench.assess(response, "A1", 155136)["fixed_work"])
        response.update(tokens=[1], tokens_predicted=1, stop_type="eos")
        self.assertFalse(bench.assess(response, "A1", 155136)["fixed_work"])
        self.assertTrue(bench.assess(response, "A1", 155136)["short_output"])

    def test_invalid_measurement_evidence(self):
        response = {
            "tokens": list(range(128)),
            "content": "ok",
            "tokens_predicted": 128,
            "tokens_evaluated": 32768,
            "timings": {
                "prompt_ms": 1.0,
                "predicted_ms": 2.0,
                "prompt_n": 32768,
                "predicted_n": 128,
                "cache_n": 0,
                "draft_n": 10,
                "draft_n_accepted": 5,
            },
        }
        self.assertTrue(bench.assess(response, "B1", 155136)["measurement_valid"])
        for field, value in (
            ("cache_n", -1),
            ("cache_n", 1),
            ("prompt_ms", float("nan")),
            ("predicted_ms", float("inf")),
            ("prompt_ms", 0),
            ("draft_n", None),
            ("draft_n", 0),
            ("draft_n_accepted", 11),
            ("draft_n_accepted", 1.5),
            ("prompt_n", 32767),
        ):
            with self.subTest(field=field, value=value):
                modified = dict(
                    response, timings=dict(response["timings"], **{field: value})
                )
                self.assertFalse(
                    bench.assess(modified, "B1", 155136)["measurement_valid"]
                )
        response["timings"].update(draft_n=0, draft_n_accepted=0)
        self.assertTrue(bench.assess(response, "A1", 155136)["measurement_valid"])
        response["timings"].pop("draft_n")
        self.assertFalse(bench.assess(response, "A1", 155136)["measurement_valid"])
        response["timings"].pop("draft_n_accepted")
        self.assertTrue(bench.assess(response, "A1", 155136)["measurement_valid"])
        self.assertFalse(bench.assess(response, "B1", 155136)["measurement_valid"])

    def test_token_bounds(self):
        for token in (-1, 155136, True, 1.5):
            self.assertFalse(bench.token_ids([token] * 32768, 155136, 32768))
        self.assertTrue(bench.token_ids([155135] * 32768, 155136, 32768))

    def test_probe_requires_live_cache_and_full_output(self):
        response = {
            "tokens": [1, 2, 3],
            "content": "ok",
            "tokens_predicted": 3,
            "timings": {"cache_n": 32768},
        }
        self.assertTrue(bench.assess_probe(response, 155136)["valid"])
        for cache in (-1, 0, 32767, None):
            self.assertFalse(
                bench.assess_probe(dict(response, timings={"cache_n": cache}), 155136)[
                    "valid"
                ]
            )
        for changes in (
            {"tokens": [1, 2]},
            {"truncated": True},
            {"error": "bad"},
            {"tokens": [1, 2, 155136]},
        ):
            self.assertFalse(
                bench.assess_probe(dict(response, **changes), 155136)["valid"]
            )

    def test_shard_identity(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "shard.gguf"
            model.write_bytes(b"headerweights")
            stat = model.stat()
            manifest = {
                "vocab_size": 155136,
                "header_bytes": 6,
                "modelshards": [
                    {
                        "path": str(model),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "header_sha256": bench.hashlib.sha256(b"header").hexdigest(),
                    }
                ],
            }
            self.assertTrue(
                bench.shards_match(manifest, bench.shard_snapshot(manifest, model))
            )
            model.write_bytes(b"changedweights")
            self.assertFalse(
                bench.shards_match(manifest, bench.shard_snapshot(manifest, model))
            )

    def test_refuse_overrides_without_values(self):
        with patch.dict(os.environ, {"GGML_SECRET": "credential"}, clear=True):
            with self.assertRaises(RuntimeError) as error:
                bench.environment()
        self.assertIn("GGML_SECRET", str(error.exception))
        self.assertNotIn("credential", str(error.exception))

    def test_only_owned_child_escalation(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("owned", 20), 0]
        bench.stop(proc)
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_count, 2)

    def test_arm_flags(self):
        args = argparse.Namespace(server=Path("/bin/server"), model=Path("/model"))
        for arm, mode in (("A1", "none"), ("B1", "draft-mtp")):
            cmd = bench.command(args, arm, 12345, "nonce")
            self.assertEqual(cmd[cmd.index("--spec-type") + 1], mode)
            self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "1")
            self.assertIn("--warmup", cmd)
            self.assertNotIn("--ignore-eos", cmd)
            self.assertIn("--no-context-shift", cmd)
            for flag, value in (
                ("-fa", "on"),
                ("--spec-draft-type-k", "f16"),
                ("--spec-draft-type-v", "f16"),
                ("--spec-draft-n-ubatch", "512"),
                ("--spec-draft-sampling", "greedy"),
            ):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_nonce_mismatch_never_sends_completion_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(
                out=Path(root), server=Path("/server"), model=Path("/model")
            )
            proc = MagicMock()
            proc.poll.return_value = None
            with (
                patch.object(bench.subprocess, "Popen", return_value=proc),
                patch.object(bench, "monitor"),
                patch.object(bench, "memory", return_value={}),
                patch.object(
                    bench, "http", return_value=b'{"model_alias":"foreign"}'
                ) as http,
            ):
                with self.assertRaisesRegex(RuntimeError, "nonce mismatch"):
                    bench.run_arm(args, "B1", b"{}", {})
            self.assertEqual(http.call_count, 1)
            self.assertEqual(http.call_args.args[1], "props")
            proc.terminate.assert_called_once()
            self.assertIn(
                "error", json.loads((Path(root) / "B1/result.json").read_text())
            )

    def test_existing_output_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prompt = root / "prompt.json"
            prompt.write_text(json.dumps([1] * 32768))
            argv = [
                "run.py",
                "--server",
                str(prompt),
                "--model",
                str(prompt),
                "--prompt",
                str(prompt),
                "--out",
                str(root),
                "--artifact-manifest",
                str(prompt),
            ]
            with (
                patch("sys.argv", argv),
                patch.object(bench, "environment", return_value={}),
                patch.object(bench, "shard_snapshot", return_value=[]),
                patch.object(bench, "shards_match", return_value=True),
                patch.object(
                    bench.json,
                    "loads",
                    side_effect=[[1] * 32768, {"vocab_size": 155136}],
                ),
                patch.object(bench.subprocess, "Popen") as popen,
            ):
                with self.assertRaises(FileExistsError):
                    bench.main()
                popen.assert_not_called()

    def test_success_preserves_raw_response(self):
        with tempfile.TemporaryDirectory() as root:
            args = argparse.Namespace(
                out=Path(root),
                server=Path("/server"),
                model=Path("/model"),
                vocab_size=155136,
            )
            proc = MagicMock()
            proc.poll.return_value = None
            response = json.dumps(
                {
                    "tokens": list(range(128)),
                    "content": "ok",
                    "tokens_predicted": 128,
                    "tokens_evaluated": 32768,
                    "timings": {
                        "draft_n": 10,
                        "draft_n_accepted": 5,
                        "prompt_ms": 1,
                        "predicted_ms": 2,
                        "prompt_n": 32768,
                        "predicted_n": 128,
                        "cache_n": 0,
                    },
                }
            ).encode()
            replies = [
                b'{"model_alias":"glm-bench-test","model_path":"/model"}',
                b'{"status":"ok"}',
                response,
                b'{"tokens":[1,2,3],"content":"next","tokens_predicted":3,"timings":{"cache_n":32895}}',
            ]
            with (
                patch.object(bench.subprocess, "Popen", return_value=proc),
                patch.object(bench, "monitor"),
                patch.object(bench, "memory", return_value={}),
                patch.object(
                    bench.uuid, "uuid4", return_value=argparse.Namespace(hex="test")
                ),
                patch.object(bench, "http", side_effect=replies),
            ):
                result = bench.run_arm(
                    args,
                    "B1",
                    json.dumps(
                        {"prompt": [9] * 32768, "temperature": 0, "return_tokens": True}
                    ).encode(),
                    {},
                )
            self.assertTrue(result["fixed_work"])
            self.assertTrue(result["measurement_valid"])
            self.assertTrue(result["probe"]["valid"])
            probe = json.loads((Path(root) / "B1/probe-request.json").read_bytes())
            self.assertEqual(probe["prompt"], [9] * 32768 + list(range(128)))
            self.assertEqual(probe["n_predict"], 3)
            self.assertTrue(probe["cache_prompt"])
            self.assertEqual(probe["id_slot"], 0)
            self.assertIn("probe-request_seconds", result)
            self.assertEqual((Path(root) / "B1/response.json").read_bytes(), response)
            proc.terminate.assert_called_once()

    def test_invalid_measurement_stops_schedule(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prompt, manifest, out = (
                root / "prompt.json",
                root / "artifact.json",
                root / "out",
            )
            prompt.write_text(json.dumps([1] * 32768))
            manifest.write_text(json.dumps({"vocab_size": 155136}))
            argv = [
                "run.py",
                "--server",
                str(prompt),
                "--model",
                str(prompt),
                "--prompt",
                str(prompt),
                "--out",
                str(out),
                "--artifact-manifest",
                str(manifest),
            ]

            def invalid_arm(args, arm, payload, env):
                directory = args.out / arm
                directory.mkdir()
                bench.save(directory / "response.json", {"tokens": [], "content": ""})
                result = {"arm": arm, "fixed_work": False, "measurement_valid": False}
                bench.save(directory / "result.json", result)
                return result

            with (
                patch("sys.argv", argv),
                patch.object(bench, "environment", return_value={}),
                patch.object(bench, "shard_snapshot", return_value=[]),
                patch.object(bench, "shards_match", return_value=True),
                patch.object(bench, "snapshot", return_value={}),
                patch.object(bench, "run_arm", side_effect=invalid_arm) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "stopping schedule"):
                    bench.main()
            self.assertEqual(run.call_count, 1)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertFalse(summary["all_measurements_valid"])
            self.assertFalse(summary["complete_schedule"])
            self.assertTrue((out / "A1-shards-after.json").is_file())


if __name__ == "__main__":
    unittest.main()
