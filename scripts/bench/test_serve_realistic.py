"""CPU checks for frozen-work replay and timeline accounting."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_realistic as bench


class WorkloadTests(unittest.TestCase):
    def population(self):
        return [dict(model="glm", purpose="primary", reused="48000", reuse_source="live_session",
                     call=f"glm/{i}", new=str(i + 5), generated=str(i + 20),
                     tool_union_ms=str(i * 10), server_pre_ms="100", decode_ms="1000")
                for i in range(6)]

    def test_seed_preserves_joint_observations_and_excludes_nonwarm_work(self):
        rows = self.population()
        excluded = [dict(rows[0], purpose="auxiliary/title inferred"),
                    dict(rows[0], reused="0"), dict(rows[0], reuse_source="snapshot_ram"),
                    dict(rows[0], model="fn")]
        a = bench.make_plan(rows + excluded, "glm", 42, 15)
        self.assertEqual(a, bench.make_plan(rows, "glm", 42, 15))
        self.assertNotEqual(a["agents"], bench.make_plan(rows, "glm", 43, 15)["agents"])
        allowed = {(int(r["new"]), int(r["generated"]), float(r["tool_union_ms"]) / 1000) for r in rows}
        for agent in a["agents"]:
            self.assertTrue(agent)
            for t in agent:
                self.assertIn((t["append_tokens"], t["reply_tokens"], t["tool_gap_s"]), allowed)

    def test_stream_intervals_are_exclusive_and_conserve_cohort_wall(self):
        rows = [dict(sent=0, first=1, end=5), dict(sent=1, first=3, end=6),
                dict(sent=2, first=4, end=7)]
        result = bench.interval_buckets(rows, 0, 8)
        self.assertEqual(result, dict(decode_stream_width_1_s=3,
                                     decode_stream_width_2_s=2,
                                     decode_stream_width_3_s=1,
                                     predecode_only_s=1, idle_s=1))
        self.assertEqual(sum(result.values()), 8)
        clipped = bench.interval_buckets(rows, 3, 6)
        self.assertEqual(clipped, dict(decode_stream_width_1_s=0,
                                      decode_stream_width_2_s=2,
                                      decode_stream_width_3_s=1,
                                      predecode_only_s=0, idle_s=0))

    def test_native_width_census_counts_sequence_transitions(self):
        # 2 singleton sequence steps, 3 width-two and 4 width-three steps.
        self.assertEqual(bench.native_widths(dict(stepped_transitions=9, batched_transitions=7,
                                                width_mean=20 / 9, width_max=3)), [2, 3, 4])
        with self.assertRaises(ValueError):
            bench.native_widths(dict(stepped_transitions=9, batched_transitions=7,
                                     width_mean=3, width_max=4))

    def test_compare_refuses_even_one_different_generated_token(self):
        a = dict(rows=[dict(label="a0-t0", request_sha256="same", prompt=48000,
                           generated=128)], wall_s=10, generated=128, new=40,
                 output_shortfalls=["a0-t0"])
        b = copy.deepcopy(a)
        b["wall_s"] = 8
        with patch.object(bench, "analyze", side_effect=[a, b]):
            self.assertEqual(bench.compare(Path("a"), Path("b"))["speed_ratio"], 1.25)
        b["rows"][0]["generated"] = 127
        with patch.object(bench, "analyze", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "unequal generated"):
                bench.compare(Path("a"), Path("b"))

    def test_partial_reasoning_has_replayable_assistant_and_utf8_is_preserved(self):
        self.assertEqual(bench.history_items([dict(type="reasoning", content=[dict(text="λ")])]),
                         [dict(type="reasoning", content="λ"), dict(role="assistant", content="")])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "body.json"
            bench.dump(p, {"input": "λ"})
            self.assertEqual(json.loads(p.read_text()), {"input": "λ"})

    def test_own_history_comparison_allows_generated_history_but_not_new_work_changes(self):
        a = dict(rows=[dict(label="a0-t0", prompt=48000, generated=128)],
                 wall_s=10, generated=128, new=40)
        b = copy.deepcopy(a)
        b["rows"][0]["prompt"] = 48002
        b["wall_s"] = 8
        b["history"] = "own"
        with tempfile.TemporaryDirectory() as d:
            ref, candidate = Path(d) / "ref", Path(d) / "candidate"
            ref.mkdir()
            candidate.mkdir()
            for agent in range(3):
                for directory in (ref, candidate):
                    bench.dump(directory / f"a{agent}-warm.request.json", {"input": f"root{agent}"})
            body = dict(model="m", max_output_tokens=128,
                        input=[dict(role="assistant", content="old history"),
                               dict(type="function_call", name="read_file"),
                               dict(type="function_call_output", output="fixed fresh source")])
            bench.dump(ref / "a0-t0.request.json", body)
            body["input"][0]["content"] = "different generated history"
            bench.dump(candidate / "a0-t0.request.json", body)
            with patch.object(bench, "analyze", side_effect=[a, b]):
                result = bench.compare(ref, candidate, own_history=True)
                self.assertEqual(result["speed_ratio"], 1.25)
                self.assertEqual(result["full_prompt_count_deltas"], [dict(label="a0-t0", tokens=2)])
            body["input"][-1]["output"] = "different fresh source"
            bench.dump(candidate / "a0-t0.request.json", body)
            with patch.object(bench, "analyze", side_effect=[a, b]):
                with self.assertRaisesRegex(ValueError, "unequal new input"):
                    bench.compare(ref, candidate, own_history=True)

    def test_changed_tool_gaps_are_rejected_even_with_identical_token_work(self):
        a = dict(rows=[], wall_s=10, generated=0, new=0, plan_sha256="first")
        b = dict(a, plan_sha256="changed")
        with patch.object(bench, "analyze", side_effect=[a, b]):
            with self.assertRaisesRegex(ValueError, "tool gaps"):
                bench.compare(Path("a"), Path("b"))


if __name__ == "__main__":
    unittest.main()
