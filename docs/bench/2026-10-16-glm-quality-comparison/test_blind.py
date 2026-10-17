import copy
import hashlib
import json
from pathlib import Path
import random
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import blind


HERE = Path(__file__).parent


class BlindTests(unittest.TestCase):
    def setUp(self):
        self.frozen = (HERE / "cases.json").read_bytes()
        self.cases = json.loads(self.frozen)
        self.summaries = [
            {
                "arm": f"SECRET_ARM_{arm}",
                "status": "completed",
                "cases_sha256": hashlib.sha256(self.frozen).hexdigest(),
                "integrity": "SECRET_BINARY",
                "memory_before": "SECRET_MACHINE",
                "results": [
                    {
                        "id": case["id"],
                        "ordinal": i,
                        "status": "completed",
                        "answer": f"  answer {arm}/{i}\r\n\t\u2603  ",
                        "reasoning": "SECRET_REASONING",
                        "usage": "SECRET_TOKENS",
                        "native": "SECRET_LATENCY",
                        "path": "SECRET_PATH",
                    }
                    for i, case in enumerate(self.cases, 1)
                ],
            }
            for arm in range(3)
        ]

    def build(self):
        return blind.build_packet(self.frozen, self.summaries, rng=random.Random(7))

    def test_whitelist_exact_answers_and_mapping(self):
        self.summaries[0]["results"][0]["answer"] = "I am SECRET_ARM_0\n"
        self.summaries[0]["results"][0]["status"] = "incomplete"
        packet, mapping = self.build()
        for i, case in enumerate(packet["cases"]):
            self.assertEqual(
                set(case),
                {"id", "task", "system", "rubric", "reference_facts", "candidates"},
            )
            self.assertEqual(case["task"], self.cases[i]["user"])
            self.assertEqual(
                case["reference_facts"], self.cases[i]["reference"]["facts"]
            )
            for candidate in case["candidates"]:
                self.assertEqual(set(candidate), {"candidate", "answer", "status"})
                arm = mapping["cases"][case["id"]][candidate["candidate"]]
                result = next(s for s in self.summaries if s["arm"] == arm)["results"][
                    i
                ]
                self.assertEqual(candidate["answer"], result["answer"])
                self.assertEqual(candidate["status"], result["status"])
        metadata = copy.deepcopy(packet)
        for case in metadata["cases"]:
            for candidate in case["candidates"]:
                candidate.pop("answer")
        self.assertNotIn("SECRET_", json.dumps(metadata))
        for needle in (
            "SECRET_REASONING",
            "SECRET_TOKENS",
            "SECRET_LATENCY",
            "SECRET_PATH",
            "SECRET_BINARY",
            "SECRET_MACHINE",
        ):
            self.assertNotIn(needle, json.dumps(packet))

    def test_model_output_error_retained_without_private_metadata(self):
        for answer in ("", "  partial final answer\r\n\t"):
            with self.subTest(answer=answer):
                self.summaries[0]["results"][0].update(
                    status="model_output_error",
                    answer=answer,
                    wire_status="failed",
                    error={"type": "server_error", "message": "SECRET_ERROR_MESSAGE"},
                    model="SECRET_MODEL",
                    binary="SECRET_BINARY",
                    text_scope="SECRET_TEXT_SCOPE",
                    cache_validation="SECRET_CACHE_VALIDATION",
                    usage=None,
                    x_qwen=None,
                    native_stop_reason=None,
                )
                packet, mapping = self.build()
                self.assertEqual(len(packet["cases"]), 8)
                self.assertTrue(all(len(c["candidates"]) == 3 for c in packet["cases"]))
                case = packet["cases"][0]
                candidate = next(
                    c
                    for c in case["candidates"]
                    if mapping["cases"][case["id"]][c["candidate"]] == "SECRET_ARM_0"
                )
                self.assertEqual(
                    candidate,
                    {
                        "candidate": candidate["candidate"],
                        "answer": answer,
                        "status": "model_output_error",
                    },
                )
                self.assertNotIn("SECRET_", json.dumps(packet))
                self.assertIn(
                    "0: no usable final answer or unrelated response.", blind.PROMPT
                )

    def test_independent_shuffle_calls(self):
        class Shuffle:
            def __init__(self):
                self.calls = []

            def shuffle(self, values):
                self.calls.append(copy.deepcopy(values))
                values.reverse()

        rng = Shuffle()
        packet, mapping = blind.build_packet(self.frozen, self.summaries, rng=rng)
        self.assertEqual(len(rng.calls), 16)
        for i in range(8):
            self.assertEqual(rng.calls[2 * i], [0, 1, 2])
            self.assertEqual(
                [c["candidate"] for c in rng.calls[2 * i + 1]], list(blind.LABELS)
            )
            self.assertEqual(packet["cases"][i]["candidates"][0]["candidate"], "C3")
            self.assertEqual(
                mapping["cases"][self.cases[i]["id"]]["C1"], "SECRET_ARM_2"
            )
        self.assertEqual(self.build(), self.build())

    def test_reject_invalid_runs(self):
        mutations = [
            lambda s: s.pop(),
            lambda s: s[0].update(status="running"),
            lambda s: s[0].update(status="aborted"),
            lambda s: s[0].update(cases_sha256="wrong"),
            lambda s: s[0].update(arm=s[1]["arm"]),
            lambda s: s[0]["results"].pop(),
            lambda s: s[0]["results"].reverse(),
            lambda s: s[0]["results"][0].update(id=s[0]["results"][1]["id"]),
            lambda s: s[0]["results"][0].update(ordinal=True),
            lambda s: s[0]["results"][0].update(status="error"),
            lambda s: s[0]["results"][0].update(answer=None),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                summaries = copy.deepcopy(self.summaries)
                mutation(summaries)
                with self.assertRaises(ValueError):
                    blind.build_packet(self.frozen, summaries)
        with self.assertRaises(ValueError):
            blind.build_packet(self.frozen + b" ", self.summaries)

    def test_default_rng_and_invalid_cases(self):
        with mock.patch.object(
            blind.secrets, "SystemRandom", return_value=random.Random(9)
        ) as factory:
            blind.build_packet(self.frozen, self.summaries)
            factory.assert_called_once_with()
        for mutation in (
            lambda c: c.pop(),
            lambda c: c[0].update(id=c[1]["id"]),
            lambda c: c[0].update(system=None),
            lambda c: c[0].update(user=""),
            lambda c: c[0].update(rubric=[]),
            lambda c: c[0].update(reference={}),
        ):
            cases = copy.deepcopy(self.cases)
            mutation(cases)
            with self.assertRaises(ValueError):
                blind.build_packet(json.dumps(cases).encode(), self.summaries)

    def test_grades(self):
        packet, _ = self.build()
        grades = {
            "grades": [
                {
                    "case_id": c["id"],
                    "candidate": label,
                    "score": 4,
                    "definite_bugs": [],
                    "uncertain_concerns": ["not executed"],
                    "rationale": "Evidence from answer",
                }
                for c in self.cases
                for label in blind.LABELS
            ]
        }
        self.assertIs(blind.validate_grades(self.cases, packet, grades), grades)
        mutations = [
            lambda g: g["grades"].pop(),
            lambda g: g["grades"].append(g["grades"][0]),
            lambda g: g["grades"][0].update(candidate="C4"),
            lambda g: g["grades"][0].update(case_id=[]),
            lambda g: g["grades"][0].update(rationale=" "),
            lambda g: g["grades"][0].update(definite_bugs="bug"),
            lambda g: g["grades"][0].update(uncertain_concerns=[1]),
        ]
        for score in (-1, 5, True, 1.0, "3", None):
            bad = copy.deepcopy(grades)
            bad["grades"][0]["score"] = score
            with self.assertRaises(ValueError):
                blind.validate_grades(self.cases, packet, bad)
        for mutation in mutations:
            bad = copy.deepcopy(grades)
            mutation(bad)
            with self.assertRaises(ValueError):
                blind.validate_grades(self.cases, packet, bad)
        for bad in (None, [], {"grades": None}):
            with self.assertRaises(ValueError):
                blind.validate_grades(self.cases, packet, bad)

    def test_cli_privacy_reverse_and_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for i, summary in enumerate(self.summaries):
                path = root / f"summary-{i}.json"
                path.write_text(json.dumps(summary))
                paths.append(str(path))
            out = root / "output"
            command = [
                sys.executable,
                "-B",
                str(HERE / "blind.py"),
                "--cases",
                str(HERE / "cases.json"),
                "--results",
                *paths,
                "--out",
                str(out),
            ]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("may mention identity", run.stderr)
            self.assertNotIn(tmp, run.stdout + run.stderr)
            self.assertEqual(
                stat.S_IMODE((out / "private/map.json").stat().st_mode), 0o600
            )
            for directory in (out, out / "blind", out / "private"):
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            packet = json.loads((out / "blind/packet.json").read_text())
            reverse = json.loads((out / "blind/packet-b.json").read_text())
            for case in reverse["cases"]:
                case["candidates"].reverse()
            self.assertEqual(packet, reverse)
            self.assertFalse((out / "blind/map.json").exists())
            for path in (out / "blind").iterdir():
                self.assertNotIn("SECRET_", path.read_text())
            before = (out / "private/map.json").read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertNotIn(tmp, rerun.stderr)
            self.assertEqual(before, (out / "private/map.json").read_bytes())
            self.summaries[0]["status"] = "running"
            Path(paths[0]).write_text(json.dumps(self.summaries[0]))
            command[-1] = str(root / "rejected")
            self.assertNotEqual(
                subprocess.run(command, capture_output=True).returncode, 0
            )
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
