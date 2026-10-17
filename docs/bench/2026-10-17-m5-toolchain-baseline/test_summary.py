import json
from pathlib import Path
import unittest

from summarize import workspace_totals


class SummaryTests(unittest.TestCase):
    def test_nested_child_results_are_not_extra_workspace_passes(self):
        result = workspace_totals("""
     Running unittests src/lib.rs (target/release/deps/example-abcd1234)
running 14 tests
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored;
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored;
test result: ok. 10 passed; 0 failed; 4 ignored;
   Doc-tests example
running 2 tests
test result: ok. 0 passed; 0 failed; 2 ignored;
""")
        self.assertEqual((result["passed"], result["ignored"]), (10, 6))
        self.assertEqual(result["nested_child_summaries_excluded"], 2)

    def test_incomplete_workspace_logs_are_rejected(self):
        for text in (
            "",
            "Running tests/sample.rs (target/release/deps/sample-abcd)",
            "test result: ok. 1 passed; 0 failed; 0 ignored;",
        ):
            with self.assertRaises(ValueError):
                workspace_totals(text)

    def test_child_summary_cannot_replace_missing_outer_summary(self):
        with self.assertRaisesRegex(ValueError, "missing top-level"):
            workspace_totals("""
     Running unittests src/lib.rs (target/release/deps/example-abcd1234)
running 2013 tests
running 1 test
test result: ok. 1 passed; 0 failed; 0 ignored;
   Doc-tests example
running 2 tests
test result: ok. 0 passed; 0 failed; 2 ignored;
""")

    def test_retained_summary_preserves_failed_and_unexecuted_facets(self):
        data = json.loads(Path(__file__).with_name("results.json").read_text())
        workspace = data["workspace"]
        self.assertEqual(
            (workspace["passed"], workspace["failed"], workspace["ignored"]),
            (2922, 0, 645),
        )
        self.assertEqual(workspace["nested_child_summaries_excluded"], 4)
        self.assertIsNone(workspace["gpu_test_count"])
        qualification = data["qualification"]
        self.assertEqual(qualification["exit_code"], 1)
        self.assertEqual(
            qualification["facet_status_counts"],
            {
                "passed": 12,
                "failed": 3,
                "scope-only": 5,
                "not applicable on this device": 7,
                "none": 7,
            },
        )
        failures = [
            w
            for r in qualification["records"]
            for w in r["witnesses"]
            if w["status"] == "failed"
        ]
        self.assertEqual(
            sum(w["failure_class"] == "missing model fixture" for w in failures), 1
        )
        self.assertEqual(len(data["validated_correctness_replays"]), 9)
        self.assertTrue(
            all(row["passed"] for row in data["validated_correctness_replays"])
        )
        self.assertEqual(len(data["model_witnesses"]["tests"]), 13)
        self.assertEqual(data["glm_runtime"]["prompt_tokens"], 200)
        self.assertLess(max(data["glm_runtime"]["continuation_kl"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
