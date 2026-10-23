"""CPU-only regression checks; set TENSOR_DOWN_ARCHIVE to the durable archive."""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/reference/glm53"))
import tensor_down_analysis as analysis


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = os.environ.get("TENSOR_DOWN_ARCHIVE")
        if not archive:
            raise unittest.SkipTest("set TENSOR_DOWN_ARCHIVE for archived-report tests")
        raw = (analysis.HERE / "quality-v1.json").read_bytes()
        cls.fixture = json.loads(raw)
        cls.fixture_hash = hashlib.sha256(raw).hexdigest()
        cls.reports = {}
        for kind, name in (
            ("plain", "plain-v2r.jsonl"),
            ("profile", "stock-profile-v2r.jsonl"),
            ("quality", "quality-v2r.jsonl"),
        ):
            cls.reports[kind], _ = analysis.read_report(
                Path(archive) / name, kind, cls.fixture, cls.fixture_hash
            )
        extended = Path(archive) / "extended-v2r.jsonl"
        if extended.exists():
            cls.reports["extended"], _ = analysis.read_report(
                extended, "extended", cls.fixture, cls.fixture_hash
            )

    def setUp(self):
        self.rows = copy.deepcopy(self.reports["quality"])

    def validate(self):
        analysis.validate_common(self.rows, "quality", self.fixture, self.fixture_hash)
        analysis.validate_quality(self.rows, self.fixture, False)

    def first(self, event):
        return next(r for r in self.rows if r["event"] == event)

    def test_complete_reports_and_metrics(self):
        self.validate()
        result, documents, tools, identities = analysis.quality_analysis(
            self.rows, self.fixture, False
        )
        self.assertEqual(
            (result["quality_token_rows"], len(documents), len(tools), len(identities)),
            (576, 18, 24, 24),
        )
        self.assertTrue(all(r["strict_correct"] for r in tools))
        self.assertTrue(all(r["sampled_ids_identical"] for r in identities))
        self.assertNotIn("central90", json.dumps(result))
        for kind in ("quality", "plain"):
            analysis.validate_census(self.reports[kind], kind)
        plain, samples = analysis.plain_analysis(self.reports["plain"])
        self.assertEqual(len(samples), 16)
        self.assertAlmostEqual(plain[0]["candidate_change_percent"], -9.1361268655)
        self.assertAlmostEqual(plain[1]["candidate_change_percent"], -8.67886741226)
        profile, stages = analysis.profile_analysis(self.reports["profile"])
        self.assertEqual(len(stages), 16)
        self.assertAlmostEqual(profile["profiling_overhead_percent"], 6.6626806902)

    def test_incomplete_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "not COMPLETE"):
            self.validate()

    def test_duplicate_completion_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[-1]))
        with self.assertRaisesRegex(ValueError, "multiple completion"):
            self.validate()

    def test_wrong_arm_flags_rejected(self):
        self.rows[0]["arm_flags"][0]["dense"] = True
        with self.assertRaisesRegex(ValueError, "arm flags"):
            self.validate()

    def test_fixture_hash_rejected(self):
        self.rows[0]["fixture_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fixture hash"):
            self.validate()

    def test_frozen_document_hash_rejected(self):
        self.rows[0]["documents"][0]["tokens_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen documents/hash"):
            self.validate()

    def test_duplicate_token_rejected(self):
        self.rows.insert(-1, copy.deepcopy(self.first("document_token")))
        with self.assertRaisesRegex(ValueError, "duplicate document token"):
            self.validate()

    def test_missing_token_rejected(self):
        self.rows.remove(self.first("document_token"))
        with self.assertRaisesRegex(ValueError, "64 unique tokens"):
            self.validate()

    def test_target_alignment_rejected(self):
        self.first("document_token")["target_token"] += 1
        with self.assertRaisesRegex(ValueError, "token alignment"):
            self.validate()

    def test_both_reference_deltas_checked(self):
        self.first("document_token")["references"]["dense_only"]["nll_delta"] += 1
        with self.assertRaisesRegex(ValueError, "paired NLL"):
            self.validate()

    def test_summary_kl_checked(self):
        self.first("document_arm")["references"]["dense_only"][
            "mean_kl_reference_to_arm"
        ] += 1
        with self.assertRaisesRegex(ValueError, "mean_kl_reference_to_arm"):
            self.validate()

    def test_duplicate_tool_generation_rejected(self):
        self.rows.insert(-1, copy.deepcopy(self.first("task_generation")))
        with self.assertRaisesRegex(ValueError, "duplicate task_generation"):
            self.validate()

    def test_tool_expectation_must_match_fixture(self):
        self.first("task_generation")["expected"]["city"] = "wrong"
        with self.assertRaisesRegex(ValueError, "tool expectation"):
            self.validate()

    def test_prompt_hash_checked(self):
        self.rows[0]["tool_tasks"][0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "frozen prompt hash"):
            self.validate()

    def test_generated_token_hash_checked(self):
        self.first("task_generation")["sampled_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "output token hash"):
            self.validate()

    def test_1455_controls_not_marked_eligible(self):
        self.first("task_prefill")["down_eligible_chunks"] = 1
        with self.assertRaisesRegex(ValueError, "actual rows/eligibility"):
            self.validate()

    def test_strict_scorer_ignores_lenient_flag(self):
        row = self.first("task_generation")
        expected = next(t for t in self.fixture["tool_tasks"] if t["id"] == row["id"])[
            "expected"
        ]
        row["lenient_first_call_match"] = True
        row["stop"] = 154820
        self.assertFalse(analysis.tool_correct(row, expected, row["id"]))
        row["stop"] = 154829
        row["calls"] *= 2
        self.assertFalse(analysis.tool_correct(row, expected, row["id"]))

    def test_dispatch_census_checked(self):
        self.first("warm_down_attribution")["iq3_s_tensor_calls"] = 39
        with self.assertRaisesRegex(ValueError, "tensor down calls"):
            analysis.validate_census(self.rows, "quality")

    def test_missing_plain_sample_rejected(self):
        rows = copy.deepcopy(self.reports["plain"])
        rows.remove(next(r for r in rows if r["event"] == "plain_sample"))
        with self.assertRaisesRegex(ValueError, "16 plain samples"):
            analysis.plain_analysis(rows)

    def test_extended_cohort_not_three_doc(self):
        self.rows[0]["schema"] = analysis.SCHEMAS["extended"]
        with self.assertRaisesRegex(ValueError, "selection count"):
            analysis.validate_common(
                self.rows, "extended", self.fixture, self.fixture_hash
            )
        docs = analysis.selected_documents(self.fixture, True)
        self.assertEqual(len(docs), 22)
        self.assertEqual(
            analysis.Counter(d["stratum"] for d in docs.values()),
            {"prose": 8, "code": 8, "long": 6},
        )

    def test_extended_interval_plumbing(self):
        validated = analysis.validate_quality(self.rows, self.fixture, False)
        with (
            patch.object(analysis, "validate_quality", return_value=validated),
            patch.object(
                analysis, "bootstrap", return_value=np.array([-0.25, 0.25])
            ) as boot,
        ):
            result, _, _, _ = analysis.quality_analysis(self.rows, self.fixture, True)
        self.assertEqual(boot.call_count, 3 * 2 * 4 * 6)
        self.assertTrue(all(call.args[2] is True for call in boot.call_args_list))
        interval = result["comparisons"][0]["nll_delta"]["central90"]
        np.testing.assert_allclose(interval, [-0.225, 0.225])

    def test_actual_extended_acceptance_and_completeness(self):
        if "extended" not in self.reports:
            self.skipTest("completed extended report not present in archive")
        rows = self.reports["extended"]
        summaries, tools = analysis.validate_quality(rows, self.fixture, True)
        analysis.validate_census(rows, "extended")
        self.assertEqual(
            (len(summaries), len(tools), len(analysis.events(rows, "document_token"))),
            (66, 24, 4224),
        )
        reproduction = analysis.compare_dense_checkpoint(
            rows, ROOT / "docs/bench/2026-10-22-glm-tensor-prefill"
        )
        self.assertTrue(reproduction["exact_recorded_metrics_match"])
        self.assertEqual(reproduction["checked_token_rows"], 2816)
        strata = {d["path"]: d["stratum"] for d in rows[0]["documents"]}
        for base, expected_mean, expected_interval in (
            (
                "original",
                -0.0006048777973737332,
                [-0.0033749247829012604, 0.002132325020455496],
            ),
            (
                "dense_only",
                -0.004605859057363527,
                [-0.00779616241960812, -0.0016041047575781383],
            ),
        ):
            by_doc = {
                doc: [row["references"][base]["mean_nll_delta"]]
                for (arm, doc), row in summaries.items()
                if arm == "dense_and_down"
            }
            self.assertAlmostEqual(float(np.mean(list(by_doc.values()))), expected_mean)
            samples = analysis.bootstrap(by_doc, strata, True, np.mean)
            np.testing.assert_allclose(
                np.percentile(samples, [5, 95]),
                expected_interval,
                rtol=1e-12,
                atol=1e-15,
            )
        for event in ("document_arm", "document_token", "task_generation"):
            with self.subTest(missing=event):
                corrupt = copy.deepcopy(rows)
                corrupt.remove(next(r for r in corrupt if r["event"] == event))
                with self.assertRaises(ValueError):
                    analysis.validate_quality(corrupt, self.fixture, True)
            with self.subTest(duplicate=event):
                corrupt = copy.deepcopy(rows)
                corrupt.insert(
                    -1, copy.deepcopy(next(r for r in corrupt if r["event"] == event))
                )
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    analysis.validate_quality(corrupt, self.fixture, True)
        with (
            self.subTest(missing_complete=True),
            self.assertRaisesRegex(ValueError, "not COMPLETE"),
        ):
            analysis.validate_common(
                rows[:-1], "extended", self.fixture, self.fixture_hash
            )
        with self.subTest(wrong_completion_count=True):
            corrupt = copy.deepcopy(rows)
            corrupt[-1]["documents_per_arm"] = 21
            with self.assertRaisesRegex(ValueError, "document count"):
                analysis.validate_common(
                    corrupt, "extended", self.fixture, self.fixture_hash
                )


class HelperTests(unittest.TestCase):
    def test_bootstrap_seed_and_stratification(self):
        self.assertEqual((analysis.RESAMPLES, analysis.SEED), (20000, 20261008))
        values = {"a": [0.0], "b": [1.0], "c": [10.0]}
        strata = {"a": "prose", "b": "prose", "c": "code"}
        first = analysis.bootstrap(values, strata, True, np.mean)
        second = analysis.bootstrap(values, strata, True, np.mean)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 20000)
        np.testing.assert_allclose(np.unique(first), [10 / 3, 11 / 3, 12 / 3])

    def test_csv_lf_and_json_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            analysis.write_csv(path, [{"chunks": [2048, 912], "flag": True}])
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
