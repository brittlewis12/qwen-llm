"""Two-arm packet/shape tests only: no models, judges, or quality evaluation."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "blind_v41_pilot", Path(__file__).with_name("blind-v41-pilot.py")
)
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


class BlindTests(unittest.TestCase):
    def setUp(self):
        self.frozen = (blind.FROZEN / "cases.json").read_bytes()
        self.cases = json.loads(self.frozen)
        self.summaries = [
            {
                "arm": f"PRIVATE_MODEL_{source}",
                "status": "completed",
                "cases_sha256": hashlib.sha256(self.frozen).hexdigest(),
                "performance": "PRIVATE_TIMINGS",
                "manifest": "PRIVATE_PATH",
                "results": [
                    {
                        "id": case["id"],
                        "ordinal": index,
                        "status": "completed",
                        "answer": f"  A B C1 C2 / {source}-{index}\r\n\t\u2603  ",
                        "candidate": "PRIVATE_LABEL",
                        "A": "PRIVATE_METADATA",
                        "reasoning": "PRIVATE_REASONING",
                    }
                    for index, case in enumerate(self.cases, 1)
                ],
            }
            for source in range(2)
        ]

    def test_seeded_two_arm_key_is_invertible_and_packet_has_only_whitelisted_fields(
        self,
    ):
        packet, key = blind.build_packet(self.frozen, self.summaries, seed=19)
        self.assertEqual(
            (packet, key), blind.build_packet(self.frozen, self.summaries, seed=19)
        )
        self.assertNotEqual(
            key["cases"],
            blind.build_packet(self.frozen, self.summaries, seed=20)[1]["cases"],
        )
        self.assertEqual(len(packet["cases"]), 8)
        self.assertEqual(key["seed"], 19)
        public = json.dumps(packet)
        self.assertNotIn("PRIVATE_", public)
        self.assertNotIn('"seed"', public)
        for index, (case, item) in enumerate(zip(self.cases, packet["cases"])):
            self.assertEqual(
                set(item),
                {"id", "task", "system", "rubric", "reference_facts", "candidates"},
            )
            self.assertEqual(
                (
                    item["id"],
                    item["task"],
                    item["system"],
                    item["rubric"],
                    item["reference_facts"],
                ),
                (
                    case["id"],
                    case["user"],
                    case["system"],
                    case["rubric"],
                    case["reference"]["facts"],
                ),
            )
            self.assertEqual({c["candidate"] for c in item["candidates"]}, {"A", "B"})
            self.assertEqual(
                set(key["cases"][case["id"]].values()),
                {s["arm"] for s in self.summaries},
            )
            for candidate in item["candidates"]:
                self.assertEqual(set(candidate), {"candidate", "answer", "status"})
                arm = key["cases"][case["id"]][candidate["candidate"]]
                source = next(s for s in self.summaries if s["arm"] == arm)
                self.assertEqual(
                    candidate["answer"], source["results"][index]["answer"]
                )

    def test_labels_never_replace_answer_text_or_depend_on_metadata(self):
        original, _ = blind.build_packet(self.frozen, self.summaries, seed=7)
        renamed = copy.deepcopy(self.summaries)
        for summary, label in zip(renamed, ("A", "B")):
            summary["arm"] = label
            summary["candidate"] = label
            summary["performance"] = {label: "do not expose"}
        self.assertEqual(blind.build_packet(self.frozen, renamed, seed=7)[0], original)
        for summary in renamed:
            summary["results"][0]["answer"] = "A"
        packet, key = blind.build_packet(self.frozen, renamed, seed=7)
        self.assertEqual(
            [c["answer"] for c in packet["cases"][0]["candidates"]], ["A", "A"]
        )
        self.assertEqual(set(key["cases"][self.cases[0]["id"]].values()), {"A", "B"})

    def test_requires_two_distinct_complete_eight_case_summaries(self):
        variants = [self.summaries[:1], self.summaries + [self.summaries[0]]]
        for change in (
            "same_arm",
            "missing",
            "duplicate_case",
            "wrong_order",
            "wrong_hash",
            "aborted",
            "wrong_ordinal",
        ):
            summaries = copy.deepcopy(self.summaries)
            row = summaries[1]
            if change == "same_arm":
                row["arm"] = summaries[0]["arm"]
            if change == "missing":
                row["results"].pop()
            if change == "duplicate_case":
                row["results"][1]["id"] = row["results"][0]["id"]
            if change == "wrong_order":
                row["results"].reverse()
            if change == "wrong_hash":
                row["cases_sha256"] = "0" * 64
            if change == "aborted":
                row["status"] = "aborted"
            if change == "wrong_ordinal":
                row["results"][0]["ordinal"] = True
            variants.append(summaries)
        for summaries in variants:
            with self.assertRaises(ValueError):
                blind.build_packet(self.frozen, summaries, seed=19)
        duplicate = copy.deepcopy(self.cases)
        duplicate[1]["id"] = duplicate[0]["id"]
        for cases in (self.cases[:-1], duplicate):
            with self.assertRaises(ValueError):
                blind.build_packet(json.dumps(cases).encode(), self.summaries, seed=19)

    def test_judge_contract_is_sixteen_records_with_unchanged_rubric(self):
        packet, _ = blind.build_packet(self.frozen, self.summaries, seed=19)
        self.assertIn("exactly 16 records", blind.PROMPT)
        self.assertIn("maximum 32 per source", blind.PROMPT)
        self.assertNotIn("24 records", blind.PROMPT)
        self.assertNotIn("C1", blind.PROMPT)
        self.assertNotIn("Maya", blind.PROMPT)
        self.assertNotIn("V4.1", blind.PROMPT)
        for line in blind.frozen.PROMPT.splitlines():
            if line.startswith(tuple(f"{n}:" for n in range(5))):
                self.assertIn(line, blind.PROMPT)
        grades = {
            "grades": [
                {
                    "case_id": c["id"],
                    "candidate": label,
                    "score": 2,
                    "definite_bugs": [],
                    "uncertain_concerns": [],
                    "rationale": "fixture only",
                }
                for c in self.cases
                for label in ("A", "B")
            ]
        }
        self.assertIs(blind.validate_grades(self.cases, packet, grades), grades)
        for change in ("missing", "duplicate", "label", "score"):
            wrong = copy.deepcopy(grades)
            if change == "missing":
                wrong["grades"].pop()
            if change == "duplicate":
                wrong["grades"][-1] = wrong["grades"][0]
            if change == "label":
                wrong["grades"][0]["candidate"] = "C1"
            if change == "score":
                wrong["grades"][0]["score"] = True
            with self.assertRaises(ValueError):
                blind.validate_grades(self.cases, packet, wrong)

    def test_private_files_reverse_packet_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            v41, maya, out = root / "v41.json", root / "maya.json", root / "packet"
            v41.write_text(json.dumps(self.summaries[0]))
            maya.write_text(json.dumps(self.summaries[1]))
            args = [
                "--v41-summary",
                str(v41),
                "--maya-summary",
                str(maya),
                "--out",
                str(out),
                "--seed",
                "19",
            ]
            self.assertEqual(blind.main(args), 0)
            self.assertEqual(
                stat.S_IMODE((out / "private/map.json").stat().st_mode), 0o600
            )
            self.assertEqual(stat.S_IMODE((out / "private").stat().st_mode), 0o700)
            packet = json.loads((out / "blind/packet.json").read_bytes())
            reverse = json.loads((out / "blind/packet-b.json").read_bytes())
            for a, b in zip(packet["cases"], reverse["cases"]):
                self.assertEqual(a["candidates"], list(reversed(b["candidates"])))
            for path in (out / "blind").iterdir():
                self.assertNotIn("PRIVATE_", path.read_text())
                self.assertNotIn(str(root), path.read_text())
            before = (out / "private/map.json").read_bytes()
            with self.assertRaises(FileExistsError):
                blind.main(args)
            self.assertEqual((out / "private/map.json").read_bytes(), before)
            with self.assertRaises(ValueError):
                blind.main(
                    [
                        "--v41-summary",
                        str(v41),
                        "--maya-summary",
                        str(v41),
                        "--out",
                        str(root / "bad"),
                    ]
                )
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
