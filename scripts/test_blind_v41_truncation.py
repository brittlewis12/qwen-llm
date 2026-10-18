"""CPU-only contract tests; no model execution or assets."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "blind_truncation", Path(__file__).with_name("blind-v41-truncation.py")
)
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


class TruncationTests(unittest.TestCase):
    def setUp(self):
        self.raw = (blind.FROZEN / "cases.json").read_bytes()
        self.cases = json.loads(self.raw)
        self.maya = json.loads((blind.FROZEN / "results/maya.json").read_bytes())
        self.diag = {
            "status": "completed",
            "cases_sha256": hashlib.sha256(self.raw).hexdigest(),
            "condition": "none-frozen",
            "arm": "SECRET_MODEL",
            "max_tokens": 987654,
            "performance": "SECRET_PERFORMANCE",
            "results": [
                {
                    "id": c["id"],
                    "ordinal": i,
                    "status": "incomplete",
                    "answer": f"diagnostic answer {i}",
                    "cap": 987654,
                }
                for i, c in enumerate(self.cases, 1)
                if c["id"] in blind.SELECTED_IDS
            ],
        }

    def build(self, seed=42):
        return blind.build_packet(self.raw, [self.diag, self.maya], seed=seed)

    def test_mapping_and_no_mutation_or_contamination(self):
        before = copy.deepcopy([self.diag, self.maya])
        mappings, presentations = set(), set()
        for seed in range(32):
            packet, key = self.build(seed)
            self.assertEqual((packet, key), self.build(seed))
            mappings.add(tuple(key["cases"][cid]["A"] for cid in blind.SELECTED_IDS))
            presentations.add(
                tuple(c["candidates"][0]["candidate"] for c in packet["cases"])
            )
            self.assertEqual(
                [c["id"] for c in packet["cases"]], list(blind.SELECTED_IDS)
            )
            for case in packet["cases"]:
                original = next(c for c in self.cases if c["id"] == case["id"])
                self.assertEqual(case["rubric"], original["rubric"])
                self.assertEqual(
                    case["reference_facts"], original["reference"]["facts"]
                )
                for candidate in case["candidates"]:
                    source = key["cases"][case["id"]][candidate["candidate"]]
                    summary = self.diag if source == "v41" else self.maya
                    row = next(r for r in summary["results"] if r["id"] == case["id"])
                    self.assertEqual(candidate["answer"], row["answer"])
            text = json.dumps(packet) + blind.PROMPT
            for secret in (
                "SECRET_MODEL",
                "SECRET_PERFORMANCE",
                "987654",
                "none-frozen",
                "low-double",
            ):
                self.assertNotIn(secret, text)
        self.assertGreater(len(mappings), 4)
        self.assertGreater(len(presentations), 4)
        self.assertEqual([self.diag, self.maya], before)
        packet["cases"][0]["rubric"].clear()
        self.assertEqual(json.loads(self.raw), self.cases)

    def test_strict_inputs(self):
        mutations = [
            lambda s: s.update(status="failed"),
            lambda s: s.update(cases_sha256="wrong"),
            lambda s: s["results"].reverse(),
            lambda s: s["results"].pop(),
            lambda s: s["results"][0].update(ordinal=6),
            lambda s: s["results"][0].update(id="sql-null-bag-semantics"),
            lambda s: s["results"][0].update(status="technical_error"),
            lambda s: s["results"][0].update(answer=None),
        ]
        for source in (0, 1):
            for mutate in mutations:
                with self.subTest(source=source, mutation=mutate):
                    summaries = copy.deepcopy([self.diag, self.maya])
                    mutate(summaries[source])
                    with self.assertRaises(ValueError):
                        blind.build_packet(self.raw, summaries, seed=0)
        # Even an unselected Maya row must be valid.
        self.maya["results"][7]["status"] = "failed"
        with self.assertRaises(ValueError):
            self.build()

    def test_conditions_never_pooled(self):
        packet, _ = self.build()
        self.diag["condition"] = "low-double"
        other, key = self.build()
        self.assertEqual(packet, other)
        self.assertEqual(key["condition"], "low-double")
        for condition in (["none-frozen", "low-double"], "pooled", None):
            self.diag["condition"] = condition
            with self.assertRaises(ValueError):
                self.build()
        self.diag["condition"] = "low-double"
        self.diag["results"] *= 2
        with self.assertRaises(ValueError):
            self.build()

    def test_grades(self):
        packet, _ = self.build()
        grades = {
            "grades": [
                dict(
                    case_id=cid,
                    candidate=label,
                    score=4,
                    definite_bugs=[],
                    uncertain_concerns=[],
                    rationale="evidence",
                )
                for cid in blind.SELECTED_IDS
                for label in blind.LABELS
            ]
        }
        self.assertEqual(blind.validate_grades(self.cases, packet, grades), grades)
        for mutate in (
            lambda g: g["grades"].pop(),
            lambda g: g["grades"].append(g["grades"][0]),
            lambda g: g["grades"][0].update(candidate="C"),
            lambda g: g["grades"][0].update(case_id="sql-null-bag-semantics"),
            lambda g: g["grades"][0].update(score=True),
            lambda g: g["grades"][0].update(score=5),
            lambda g: g["grades"][0].update(rationale=""),
            lambda g: g["grades"][0].update(definite_bugs="bug"),
        ):
            invalid = copy.deepcopy(grades)
            mutate(invalid)
            with self.assertRaises(ValueError):
                blind.validate_grades(self.cases, packet, invalid)

    def test_cli_private_fresh_and_reversed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "diagnostic.json"
            source.write_text(json.dumps(self.diag))
            before = source.read_bytes()
            out = root / "output"
            args = [
                "--diagnostic-summary",
                str(source),
                "--out",
                str(out),
                "--seed",
                "7",
            ]
            self.assertEqual(blind.main(args), 0)
            a = json.loads((out / "blind/packet.json").read_bytes())
            b = json.loads((out / "blind/packet-b.json").read_bytes())
            for case in b["cases"]:
                case["candidates"].reverse()
            self.assertEqual(a, b)
            self.assertEqual(out.stat().st_mode & 0o777, 0o700)
            self.assertEqual((out / "private/map.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                blind.main(args)
            self.diag["status"] = "failed"
            source.write_text(json.dumps(self.diag))
            with self.assertRaises(ValueError):
                blind.main(
                    ["--diagnostic-summary", str(source), "--out", str(root / "failed")]
                )
            self.assertFalse((root / "failed").exists())


if __name__ == "__main__":
    unittest.main()
