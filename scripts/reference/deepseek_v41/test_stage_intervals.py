"""Offline stdlib-only interval diagnostics; no packet or latch is read/written."""

import copy
import json
import unittest

from stage_intervals import (
    CROSS_API_ALLOWANCE_NS,
    FFN_STAGES,
    U64_MAX,
    analyze_intervals,
)


# Normalized offsets + 1 ns only; no host-clock or session metadata.
REAL_PAIRS_NS = (
    (1, 2381459),
    (2382126, 3198709),
    (3199459, 13236334),
    (3199459, 7203584),
    (176450126, 209824876),
    (209825376, 210818167),
    (210817292, 222302001),
    (222302542, 223638751),
)
REAL_SAMPLES_NS = tuple(value for pair in REAL_PAIRS_NS for value in pair)
REAL_ENVELOPE_NS = (1, 223638751)


def analyze_pairs(names, pairs, envelope):
    return analyze_intervals(
        names, [value for pair in pairs for value in pair], envelope
    )


class StageIntervalsTests(unittest.TestCase):
    def test_normalized_real_fixture_exact_nanoseconds(self):
        report = analyze_intervals(FFN_STAGES, REAL_SAMPLES_NS, REAL_ENVELOPE_NS)
        expected = {
            "stage_sum_ns": 64427500,
            "union_ns": 60422500,
            "overlap_excess_ns": 4005000,
            "covered_ge2_ns": 4005000,
            "counter_hull_duration_ns": 223638750,
            "command_envelope_duration_ns": 223638750,
            "uncovered_hull_ns": 163216250,
            "uncovered_command_envelope_ns": 163216250,
            "largest_uncovered_hull_ns": 163213792,
        }
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertIs(type(report[key]), int)
                self.assertEqual(report[key], value)
        self.assertEqual(report["units"], "ns")
        self.assertEqual(report["counter_hull_ns"], list(REAL_ENVELOPE_NS))
        self.assertEqual(
            [stage["duration_ns"] for stage in report["stages"]],
            [2381458, 816583, 10036875, 4004125, 33374750, 992791, 11484709, 1336209],
        )
        self.assertEqual(
            report["raw_adjacent_encoded_gaps_ns"],
            [667, 750, -10036875, 169246542, 500, -875, 541],
        )
        self.assertEqual(
            report["uncovered_hull_intervals_ns"],
            [
                [2381459, 2382126],
                [3198709, 3199459],
                [13236334, 176450126],
                [209824876, 209825376],
                [222302001, 222302542],
            ],
        )
        reported_pairs = tuple((s["start_ns"], s["end_ns"]) for s in report["stages"])
        self.assertEqual(reported_pairs, REAL_PAIRS_NS)
        shared, route = reported_pairs[2:4]
        mid, down = reported_pairs[5:7]
        self.assertEqual(min(shared[1], route[1]) - max(shared[0], route[0]), 4004125)
        self.assertEqual(min(mid[1], down[1]) - max(mid[0], down[0]), 875)

    def test_real_dependency_anomaly_is_not_excused_or_acceptance(self):
        report = analyze_intervals(FFN_STAGES, REAL_SAMPLES_NS, REAL_ENVELOPE_NS)
        diagnostics = report["dependency_diagnostics"]
        edges = diagnostics["edges"]
        self.assertEqual(len(edges), 9)
        self.assertEqual(diagnostics["omitted_edges"], [])
        self.assertFalse(diagnostics["cross_api_allowance_applied"])
        self.assertEqual(diagnostics["anomaly_count"], 1)
        (anomaly,) = [edge for edge in edges if edge["anomaly"]]
        self.assertEqual((anomaly["from_stage"], anomaly["to_stage"]), ("mid", "down"))
        self.assertEqual(anomaly["gap_ns"], -875)
        self.assertLess(-anomaly["gap_ns"], CROSS_API_ALLOWANCE_NS)
        self.assertTrue(report["records_well_formed"])
        self.assertEqual(report["attribution"], "nonexclusive_attribution")
        self.assertTrue(all(value is False for value in report["acceptance"].values()))
        self.assertIn("FAILED", " ".join(report["limits"]))
        self.assertIn("uncalibrated", " ".join(report["limits"]))

    def test_every_source_dependency_boundary_and_signed_gap(self):
        report = analyze_intervals(FFN_STAGES, REAL_SAMPLES_NS, REAL_ENVELOPE_NS)
        actual = [
            (
                e["from_stage"],
                e["from_boundary"],
                e["to_stage"],
                e["to_boundary"],
                e["gap_ns"],
            )
            for e in report["dependency_diagnostics"]["edges"]
        ]
        self.assertEqual(
            actual,
            [
                ("hc", "end", "router", "start", 667),
                ("router", "end", "shared", "start", 750),
                ("router", "end", "route", "start", 750),
                ("route", "end", "gate_up", "start", 169246542),
                ("gate_up", "end", "mid", "start", 500),
                ("shared", "end", "mid", "end", 197581833),
                ("mid", "end", "down", "start", -875),
                ("down", "end", "sum", "start", 541),
                ("shared", "end", "sum", "start", 209066208),
            ],
        )

    def test_independent_overlap_shared_starts_and_reverse_encoded_order(self):
        for pairs in (
            [(10, 30), (20, 40)],
            [(10, 30), (10, 20)],
            [(30, 40), (10, 20)],
        ):
            with self.subTest(pairs=pairs):
                report = analyze_pairs(
                    ["independent_b", "independent_a"], pairs, (1, 50)
                )
                self.assertEqual(
                    [s["name"] for s in report["stages"]],
                    ["independent_b", "independent_a"],
                )
                self.assertEqual(
                    [(s["start_ns"], s["end_ns"]) for s in report["stages"]], pairs
                )
                self.assertEqual(report["dependency_diagnostics"]["edges"], [])
                self.assertEqual(
                    len(report["dependency_diagnostics"]["omitted_edges"]), 9
                )

    def test_duplicate_whole_windows_can_sum_above_envelope(self):
        report = analyze_pairs(["a", "b"], [(1, 101), (1, 101)], (1, 101))
        self.assertEqual(report["stage_sum_ns"], 200)
        self.assertEqual(report["command_envelope_duration_ns"], 100)
        self.assertEqual(report["union_ns"], 100)
        self.assertEqual(report["covered_ge2_ns"], 100)
        self.assertEqual(report["overlap_excess_ns"], 100)

    def test_triple_overlap_excess_is_not_time_covered_ge2(self):
        report = analyze_pairs(["a", "b", "c"], [(1, 11), (3, 9), (5, 7)], (1, 11))
        self.assertEqual(report["stage_sum_ns"], 18)
        self.assertEqual(report["union_ns"], 10)
        self.assertEqual(report["covered_ge2_ns"], 6)
        self.assertEqual(report["overlap_excess_ns"], 8)

    def test_touching_half_open_boundaries_have_zero_overlap(self):
        report = analyze_pairs(["b", "a"], [(11, 21), (1, 11)], (1, 21))
        self.assertEqual(report["union_intervals_ns"], [[1, 21]])
        self.assertEqual(report["union_ns"], 20)
        self.assertEqual(report["covered_ge2_ns"], 0)
        self.assertEqual(report["overlap_excess_ns"], 0)
        self.assertEqual(report["uncovered_hull_ns"], 0)
        self.assertEqual(report["raw_adjacent_encoded_gaps_ns"], [-20])
        forward = analyze_pairs(["a", "b"], [(1, 11), (11, 21)], (1, 21))
        self.assertEqual(forward["raw_adjacent_encoded_gaps_ns"], [0])

    def test_absolute_envelope_rejects_shifted_same_width(self):
        for envelope in ((10001, 10011), (20001, 20011)):
            with (
                self.subTest(envelope=envelope),
                self.assertRaisesRegex(ValueError, "absolute"),
            ):
                analyze_pairs(["a"], [(1, 11)], envelope)

    def test_allowance_is_fixed_absolute_and_does_not_normalize(self):
        report = analyze_pairs(["a"], [(5000, 25000)], (10000, 20000))
        self.assertEqual(report["cross_api_envelope_allowance_ns"], 5000)
        self.assertEqual(report["stages"][0]["start_ns"], 5000)
        self.assertEqual(report["stages"][0]["end_ns"], 25000)
        self.assertEqual(report["stages"][0]["command_start_margin_ns"], -5000)
        self.assertEqual(report["stages"][0]["command_end_margin_ns"], -5000)
        self.assertEqual(report["union_ns"], 20000)
        self.assertEqual(report["uncovered_command_envelope_ns"], 0)
        for pair in ((4999, 20000), (10000, 25001)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                analyze_pairs(["a"], [pair], (10000, 20000))

    def test_uncovered_envelope_is_exact_set_difference(self):
        report = analyze_pairs(
            ["a", "b"], [(5000, 12000), (15000, 25000)], (10000, 20000)
        )
        self.assertEqual(
            report["uncovered_command_envelope_intervals_ns"], [[12000, 15000]]
        )
        self.assertEqual(report["uncovered_command_envelope_ns"], 3000)
        disjoint = analyze_pairs(["a"], [(1, 1000)], (2000, 3000))
        self.assertEqual(disjoint["uncovered_command_envelope_ns"], 1000)
        self.assertEqual(disjoint["uncovered_hull_ns"], 0)
        interior = analyze_pairs(["a"], [(5, 10)], (1, 20))
        self.assertEqual(
            interior["uncovered_command_envelope_intervals_ns"], [[1, 5], [10, 20]]
        )

    def test_near_u64_max_and_sums_beyond_u64_do_not_overflow(self):
        high = U64_MAX - 1
        report = analyze_pairs(["a", "b", "c"], [(1, high)] * 3, (1, high))
        self.assertEqual(report["stage_sum_ns"], 3 * (high - 1))
        self.assertGreater(report["stage_sum_ns"], U64_MAX)
        self.assertEqual(report["overlap_excess_ns"], 2 * (high - 1))
        self.assertEqual(report["covered_ge2_ns"], high - 1)
        near = analyze_pairs(["a"], [(high - 5, high)], (high - 10, high - 1))
        self.assertEqual(near["stage_sum_ns"], 5)
        self.assertEqual(near["stages"][0]["command_end_margin_ns"], -1)
        self.assertEqual(near["uncovered_command_envelope_ns"], 5)

    def test_malformed_sample_counts_including_15_and_17(self):
        for samples in (REAL_SAMPLES_NS[:-1], REAL_SAMPLES_NS + (1,), (), (1,)):
            with self.subTest(count=len(samples)), self.assertRaises(ValueError):
                analyze_intervals(FFN_STAGES, samples, REAL_ENVELOPE_NS)

    def test_invalid_sample_and_envelope_values_are_not_coerced(self):
        for invalid in (0, U64_MAX, -1, U64_MAX + 1, True, False, 1.0, "1", None):
            for position in (0, 1):
                with self.subTest(invalid=invalid, position=position):
                    samples = [1, 10]
                    samples[position] = invalid
                    with self.assertRaises(ValueError):
                        analyze_intervals(["a"], samples, (1, 10))
                    envelope = [1, 10]
                    envelope[position] = invalid
                    with self.assertRaises(ValueError):
                        analyze_intervals(["a"], [1, 10], envelope)

    def test_equal_or_reversed_stage_and_command_pairs_rejected(self):
        for pair in ((10, 10), (11, 10)):
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError):
                    analyze_pairs(["a"], [pair], (1, 20))
                with self.assertRaises(ValueError):
                    analyze_pairs(["a"], [(1, 20)], pair)

    def test_names_and_container_shapes_are_validated(self):
        for names in ([], ["a", "a"], [""], [True], [1], "a", None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                analyze_intervals(names, [1, 10], (1, 10))
        for samples in ("12", {1, 2}, None, [[1, 2]]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                analyze_intervals(["a"], samples, (1, 10))
        for envelope in ([], [1], [1, 2, 3], "12", None):
            with self.subTest(envelope=envelope), self.assertRaises(ValueError):
                analyze_intervals(["a"], [1, 10], envelope)

    def test_explicit_mapping_preserves_name_and_sample_association(self):
        mapping = {"later": [2, 3], "earlier": [0, 1]}
        report = analyze_intervals(
            ["later", "earlier"], [1, 10, 20, 30], (1, 30), source_indices=mapping
        )
        self.assertEqual([s["name"] for s in report["stages"]], ["later", "earlier"])
        self.assertEqual([s["start_ns"] for s in report["stages"]], [20, 1])
        self.assertEqual(
            [s["source_indices"] for s in report["stages"]], [[2, 3], [0, 1]]
        )
        self.assertEqual(report["union_intervals_ns"], [[1, 10], [20, 30]])
        self.assertEqual(report["raw_adjacent_encoded_gaps_ns"], [10])
        split = analyze_intervals(
            ["a", "b"],
            [1, 2, 10, 20],
            (1, 20),
            source_indices={"a": [0, 2], "b": [1, 3]},
        )
        self.assertEqual([s["duration_ns"] for s in split["stages"]], [9, 18])

    def test_bad_mapping_duplicate_out_of_range_omitted_or_extra_names(self):
        for mapping in (
            {},
            {"a": [0, 1]},
            {"a": [0, 1], "other": [2, 3]},
            {"a": [0, 1], "b": [2, 3], "extra": [4, 5]},
            {"a": [0, 1], "b": [1, 3]},
            {"a": [0, 0], "b": [2, 3]},
            {"a": [-1, 1], "b": [2, 3]},
            {"a": [0, 1], "b": [2, 4]},
            {"a": [False, 1], "b": [2, 3]},
            {"a": [0.0, 1], "b": [2, 3]},
            {"a": ["0", 1], "b": [2, 3]},
            {"a": [0], "b": [2, 3]},
            {"a": [0, 1, 2], "b": [2, 3]},
            {"a": None, "b": [2, 3]},
            [("a", [0, 1]), ("b", [2, 3])],
        ):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                analyze_intervals(
                    ["a", "b"], [1, 10, 20, 30], (1, 30), source_indices=mapping
                )

    def test_explicit_mapping_cannot_hide_reversed_pairs(self):
        with self.assertRaises(ValueError):
            analyze_intervals(["a"], [1, 10], (1, 10), source_indices={"a": [1, 0]})

    def test_shared_end_constraint_and_no_canonical_name_inference(self):
        report = analyze_pairs(["shared", "mid"], [(1, 20), (5, 10)], (1, 20))
        (edge,) = report["dependency_diagnostics"]["edges"]
        self.assertEqual(edge["gap_ns"], -10)
        self.assertTrue(edge["anomaly"])
        self.assertIn("SharedEnd <= MidEnd", edge["constraint"])
        generic = analyze_pairs(["Shared", "MID"], [(1, 20), (5, 10)], (1, 20))
        self.assertEqual(generic["dependency_diagnostics"]["edges"], [])

    def test_input_and_returned_records_are_not_mutated_or_aliased(self):
        names, samples, envelope = (
            list(FFN_STAGES),
            list(REAL_SAMPLES_NS),
            list(REAL_ENVELOPE_NS),
        )
        mapping = {name: [2 * i, 2 * i + 1] for i, name in enumerate(names)}
        before = copy.deepcopy((names, samples, envelope, mapping))
        report = analyze_intervals(names, samples, envelope, source_indices=mapping)
        self.assertEqual((names, samples, envelope, mapping), before)
        self.assertEqual(json.loads(json.dumps(report)), report)
        report["stages"][0]["source_indices"][0] = 99
        report["command_envelope_ns"][0] = 99
        self.assertEqual((names, samples, envelope, mapping), before)


if __name__ == "__main__":
    unittest.main()
