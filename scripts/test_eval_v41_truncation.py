"""CPU-only diagnostic tests with mocked HTTP; no worker calls."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_eval_v41_pilot import terminal

spec = importlib.util.spec_from_file_location(
    "truncation", Path(__file__).with_name("eval-v41-truncation.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = (runner.pilot.FROZEN / "cases.json").read_bytes()
        self.cases = json.loads(data)
        self.baseline = {
            "status": "completed",
            "cases_sha256": hashlib.sha256(data).hexdigest(),
            "checkpoint": {"retained": True},
            "results": [
                {
                    "id": c["id"],
                    "ordinal": i,
                    "status": "incomplete" if i in (1, 3, 5, 8) else "completed",
                    "native_stop_reason": "token_limit" if i in (1, 3, 5, 8) else "eos",
                }
                for i, c in enumerate(self.cases, 1)
            ],
        }
        self.path = self.root / "baseline.json"
        self.path.write_text(json.dumps(self.baseline))

    def args(self, condition="low-double"):
        return [
            "--endpoint",
            "http://localhost:9999",
            "--model-id",
            "test",
            "--baseline",
            str(self.path),
            "--condition",
            condition,
            "--out",
            str(self.root / "out"),
        ]

    def connection(self, endpoint, timeout):
        conn = Mock()

        def request(method, path, body, headers):
            body = json.loads(body)
            self.sent.append(body)
            value = terminal(body["max_output_tokens"])
            if len(self.sent) % 2 == 0:
                value.update(
                    status="incomplete",
                    incomplete_details={"reason": "max_output_tokens"},
                )
            kind = "response." + value["status"]
            raw = (
                f"event: {kind}\ndata: "
                + json.dumps({"type": kind, "response": value})
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            self.raw.append(raw)
            wire = Mock(status=200)
            wire.getheaders.return_value = [("Content-Type", "text/event-stream")]
            wire.getheader.return_value = "text/event-stream"
            wire.readline = io.BytesIO(raw).readline
            conn.getresponse.return_value = wire

        conn.request.side_effect = request
        return conn

    def test_four_requests_exact_changes_and_native_artifacts(self):
        original = self.path.read_bytes()
        for condition in ("none-frozen", "low-double"):
            with self.subTest(condition=condition):
                self.sent, self.raw = [], []
                args = self.args(condition)
                out = self.root / condition
                args[-1] = str(out)
                with patch.object(
                    runner.pilot, "connection", side_effect=self.connection
                ):
                    self.assertEqual(runner.main(args), 0)
                self.assertEqual(len(self.sent), 4)
                summary = json.loads((out / "summary.json").read_text())
                self.assertEqual(summary["selected_case_ordinals"], [1, 3, 5, 8])
                self.assertEqual(
                    summary["baseline"]["sha256"], hashlib.sha256(original).hexdigest()
                )
                self.assertEqual(out.stat().st_mode & 0o777, 0o700)
                for index, ordinal in enumerate((1, 3, 5, 8)):
                    expected = runner.pilot.request_body(
                        "test", self.cases[ordinal - 1]
                    )
                    if condition == "none-frozen":
                        expected["reasoning"]["effort"] = "none"
                    else:
                        expected["max_output_tokens"] *= 2
                    self.assertEqual(self.sent[index], expected)
                    row = summary["results"][index]
                    self.assertEqual(
                        row["status"], "completed" if index % 2 == 0 else "incomplete"
                    )
                    self.assertEqual(row["reasoning"], "plan")
                    self.assertEqual(row["answer"], "answer")
                    self.assertIn("x_qwen", row)
                    self.assertEqual(
                        (out / f"case-{ordinal:02d}" / "response.sse").read_bytes(),
                        self.raw[index],
                    )
                self.assertEqual(self.path.read_bytes(), original)
                with patch.object(runner.pilot, "evaluate") as evaluate:
                    with self.assertRaises(FileExistsError):
                        runner.main(args)
                    evaluate.assert_not_called()
        self.assertEqual({c["max_tokens"] * 2 for c in self.cases}, {8192, 12288})

    def test_malformed_baselines_refused_before_http(self):
        variants = [
            None,
            {},
            {**self.baseline, "status": "running"},
            {**self.baseline, "cases_sha256": "wrong"},
        ]
        for field, value in [
            ("id", "foreign"),
            ("ordinal", 8),
            ("status", "error"),
            ("native_stop_reason", "eos"),
        ]:
            changed = copy.deepcopy(self.baseline)
            changed["results"][0][field] = value
            variants.append(changed)
        changed = copy.deepcopy(self.baseline)
        changed["results"][0].update(status="completed", native_stop_reason="eos")
        variants.append(changed)
        for value in variants:
            self.path.write_text(json.dumps(value))
            with patch.object(runner.pilot, "connection") as connection:
                with self.assertRaises((ValueError, TypeError)):
                    runner.main(self.args())
                connection.assert_not_called()
            self.assertFalse((self.root / "out").exists())

    def test_technical_error_aborts_without_retry(self):
        self.sent, self.raw = [], []
        with (
            patch.object(runner.pilot, "connection", side_effect=self.connection),
            patch.object(
                runner.pilot, "evaluate", side_effect=OSError("transport failed")
            ) as evaluate,
        ):
            self.assertEqual(runner.main(self.args()), 1)
        evaluate.assert_called_once()
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertEqual(summary["status"], "aborted")
        self.assertEqual(summary["results"][0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
