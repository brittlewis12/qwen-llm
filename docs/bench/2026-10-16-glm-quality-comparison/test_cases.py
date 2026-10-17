"""Local stdlib tests; fixtures do not execute model-generated code."""

import copy
import json
import re
import unittest

from cases import build_cases, check_answer


def fixture(case):
    return {
        "facts": copy.deepcopy(case["reference"]["facts"]),
        "code": "# Human review still required"
        if case["reference"]["code_required"]
        else "",
        "explanation": "Fixture only; exact facts are not a quality grade.",
    }


class CaseTests(unittest.TestCase):
    def test_determinism_and_contract(self):
        a, b = build_cases(), build_cases()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8)
        self.assertEqual(len({c["id"] for c in a}), 8)
        for case in a:
            self.assertEqual(
                set(case),
                {
                    "id",
                    "category",
                    "system",
                    "user",
                    "max_tokens",
                    "reference",
                    "rubric",
                },
            )
            self.assertIn(case["max_tokens"], (4096, 6144))
            self.assertIn('"facts"', case["system"])
            self.assertIn("Required facts", case["user"])
            case["user"].encode("ascii")
        a[0]["reference"]["facts"]["source_calls"] = 999
        self.assertEqual(build_cases(), b)

    def test_correct_and_wrong_fixtures(self):
        for case in build_cases():
            with self.subTest(case=case["id"]):
                good = fixture(case)
                result = check_answer(case, json.dumps(good))
                self.assertTrue(result["parsed"])
                self.assertEqual(result["schema_errors"], [])
                self.assertEqual(result["facts_matched"], result["facts_total"])
                self.assertNotIn("pass", result)
                for key in good["facts"]:
                    bad = copy.deepcopy(good)
                    bad["facts"][key] = {"wrong": True}
                    result = check_answer(case, json.dumps(bad))
                    self.assertEqual(result["facts_matched"], result["facts_total"] - 1)

    def test_reasoning_is_not_graded(self):
        case = build_cases()[0]
        good = json.dumps(fixture(case))
        for prefix in (
            "<think>not JSON</think>",
            "private reasoning</think>",
            "<think>" + good + "</think>",
        ):
            raw = prefix + good
            result = check_answer(case, raw)
            self.assertEqual(result["raw_answer"], raw)
            self.assertTrue(result["reasoning_delimiter_seen"])
            self.assertEqual(result["facts_matched"], result["facts_total"])
        for raw in (
            "<think>" + good,
            good + "</think>",
            "<think>" + good + "</think>no answer",
        ):
            self.assertFalse(check_answer(case, raw)["parsed"])

    def test_no_truncation_or_prefix_salvage(self):
        case = build_cases()[0]
        good = json.dumps(fixture(case))
        for raw in (
            "",
            "not JSON",
            good[:-1],
            good + " TRUNCATED",
            good + good,
            "```json\n" + good,
            good + "\n[output truncated]",
            "Here is the answer: " + good,
        ):
            result = check_answer(case, raw)
            self.assertFalse(result["parsed"])
            self.assertEqual(result["raw_answer"], raw)
            self.assertIsNotNone(result["parse_error"])
        raw = "<think>ignored</think>\n```json\n" + good + "\n```"
        result = check_answer(case, raw)
        self.assertTrue(result["parsed"])
        self.assertTrue(result["outer_fence_removed"])
        self.assertEqual(result["raw_answer"], raw)

    def test_strict_json_and_schema(self):
        case = build_cases()[0]
        for raw in ('{"facts":{},"facts":{}}', '{"n":NaN}', '{"n":Infinity}'):
            self.assertFalse(check_answer(case, raw)["parsed"])
        for raw in ("null", "[]", "{}", '{"facts":[]}'):
            result = check_answer(case, raw)
            self.assertTrue(result["parsed"])
            self.assertTrue(result["schema_errors"])
        answer = fixture(case)
        answer["facts"]["prefetch_in_init"] = 0
        answer["facts"]["source_calls"] = 4.0
        answer["code"] = ""
        result = check_answer(case, json.dumps(answer))
        self.assertEqual(result["facts_matched"], 1)
        self.assertTrue(result["schema_errors"])

    def test_long_context_record_placement(self):
        for case, base, winner, loser in (
            (build_cases()[-2], "R9-C6", "R9-C8", "R9-C7"),
            (build_cases()[-1], "L40", "L42", "L41"),
        ):
            with self.subTest(case=case["id"]):
                text = case["user"]
                notes = list(re.finditer(r"ARCHIVE NOTE (\d{2});", text))
                self.assertEqual(
                    [m.group(1) for m in notes], [f"{i:02d}" for i in range(48)]
                )
                position = text.index("id=" + winner)
                self.assertGreater(position / len(text), 0.4)
                self.assertLess(position / len(text), 0.6)
                self.assertLess(text.index("id=" + base), notes[0].start())
                self.assertEqual(sum(m.start() < position for m in notes), 24)
                for tail in ("id=" + loser, "TERMINAL DRAFT"):
                    tail_position = text.index(tail)
                    self.assertEqual(
                        sum(position < m.start() < tail_position for m in notes), 24
                    )

    def test_rollout_requires_early_base(self):
        text = build_cases()[-1]["user"]
        base = text[text.index("id=L40") : text.index("ARCHIVE NOTE 00;")]
        winner = text[text.index("id=L42") : text.index("ARCHIVE NOTE 24;")]
        self.assertIn("base=L40", winner)
        self.assertIn("defaults: enabled=false", base)
        self.assertIn("east={enabled:true,ttl:15}", base)
        self.assertNotIn("defaults.enabled", winner)
        self.assertNotIn("east", winner)
        self.assertNotIn(
            "east={enabled:true,ttl:15}", text[text.index("ARCHIVE NOTE 00;") :]
        )
        self.assertIn("regions.west=null", winner)

    def test_long_context_conflicts_and_authority(self):
        recovery, rollout = build_cases()[-2:]
        for case in (recovery, rollout):
            # Word/character proxies, deliberately not asserted tokenizer counts.
            self.assertGreater(len(case["user"].split()), 6500)
            self.assertLess(len(case["user"].split()), 11000)
            self.assertEqual(case["user"].count("ARCHIVE NOTE"), 48)
            self.assertIn("TERMINAL DRAFT", case["user"])
            self.assertIn("Highest numeric", case["user"])
        self.assertLess(
            recovery["user"].index("id=R9-C8"), recovery["user"].index("id=R9-C7")
        )
        self.assertLess(
            rollout["user"].index("id=L42"), rollout["user"].index("id=L41")
        )
        wrong_recovery = fixture(recovery)
        wrong_recovery["facts"].update(
            authority="R9-C7",
            visible_seq=304,
            truncate_offset=6350,
            x="red",
            y="absent",
        )
        wrong_rollout = fixture(rollout)
        wrong_rollout["facts"].update(
            authority="L41", east_ttl=5, west_enabled=True, old_published=True
        )
        for case, wrong in ((recovery, wrong_recovery), (rollout, wrong_rollout)):
            result = check_answer(case, json.dumps(wrong))
            self.assertLess(result["facts_matched"], result["facts_total"])
        self.assertEqual(recovery["reference"]["facts"]["discarded_bytes"], 6437 - 6400)
        self.assertEqual(rollout["reference"]["facts"]["east_ttl"], 15)
        self.assertEqual(rollout["reference"]["facts"]["west_ttl"], 45)


if __name__ == "__main__":
    unittest.main()
