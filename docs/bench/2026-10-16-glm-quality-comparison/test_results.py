import hashlib
import json
from pathlib import Path
import re
import unittest

import blind
from cases import check_answer


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def load(path):
    return json.loads(path.read_text())


def vm_count(text, label):
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", text, re.MULTILINE)
    if match is None:
        raise ValueError("missing VM counter")
    return int(match[1])


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.cases = load(ROOT / "cases.json")
        self.mapping = load(RESULTS / "unblinded-map.json")
        self.runs = {
            arm: load(RESULTS / (recipe + ".json"))
            for arm, recipe in self.mapping["recipes"].items()
        }
        self.metrics = {m["recipe"]: m for m in load(RESULTS / "metrics.json")}

    def test_runtime_identity_and_request_isolation(self):
        inputs = []
        for run in self.runs.values():
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["integrity_after"], "unchanged")
            self.assertEqual(len(run["servers"]), 1)
            for phase in ("memory_before", "memory_after"):
                self.assertIn("used = 0.00M", run[phase]["swapusage"]["stdout"])
            self.assertEqual(
                vm_count(run["memory_before"]["vm_stat"]["stdout"], "Compressions"),
                vm_count(run["memory_after"]["vm_stat"]["stdout"], "Compressions"),
            )
            for key, filename in (
                ("cases_sha256", "cases.json"),
                ("runner_sha256", "run.py"),
                ("checker_sha256", "cases.py"),
            ):
                self.assertEqual(
                    run[key], hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                )
            self.assertEqual(
                [r["id"] for r in run["results"]], [c["id"] for c in self.cases]
            )
            inputs.append([r["usage"]["input_tokens"] for r in run["results"]])
            for case, result in zip(self.cases, run["results"]):
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["native_stop_reason"], "eos")
                self.assertLessEqual(
                    result["usage"]["output_tokens"], case["max_tokens"]
                )
                self.assertEqual(
                    result["usage"]["input_tokens_details"]["cached_tokens"], 0
                )
                self.assertEqual(result["x_qwen"]["matched_tokens"], 0)
                self.assertEqual(result["native"]["phases"]["reused_tokens"], "0")
                self.assertEqual(result["selected_prefill_rows"], 2048)
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(inputs[0], inputs[2])

    def test_exact_checks_recompute_from_original_answers(self):
        for run in self.runs.values():
            for case, result in zip(self.cases, run["results"]):
                recomputed = check_answer(case, result["answer"])
                for key, value in result["checks"].items():
                    self.assertEqual(value, recomputed[key])

    def test_blind_answers_and_reversed_presentation(self):
        packet = load(RESULTS / "blind/packet.json")
        reverse = load(RESULTS / "blind/packet-b.json")
        for index, (case, item, reversed_item) in enumerate(
            zip(self.cases, packet["cases"], reverse["cases"])
        ):
            self.assertEqual(item["task"], case["user"])
            self.assertEqual(item["system"], case["system"])
            self.assertEqual(
                item["candidates"], list(reversed(reversed_item["candidates"]))
            )
            for candidate in item["candidates"]:
                arm = self.mapping["cases"][case["id"]][candidate["candidate"]]
                result = self.runs[arm]["results"][index]
                self.assertEqual(candidate["answer"], result["answer"])
                self.assertEqual(candidate["status"], result["status"])

    def test_grades_and_totals(self):
        packet = load(RESULTS / "blind/packet.json")
        for judge in ("a", "b"):
            grades = load(RESULTS / f"judge-{judge}.json")
            blind.validate_grades(self.cases, packet, grades)
            for arm, recipe in self.mapping["recipes"].items():
                total = sum(
                    g["score"]
                    for g in grades["grades"]
                    if self.mapping["cases"][g["case_id"]][g["candidate"]] == arm
                )
                self.assertEqual(
                    total, self.metrics[recipe]["reviewer_scores_out_of_32"][judge]
                )

    def test_rates_and_counts(self):
        for run in self.runs.values():
            rows = run["results"]
            metrics = self.metrics[run["recipe"]]
            inputs = sum(r["usage"]["input_tokens"] for r in rows)
            outputs = sum(r["usage"]["output_tokens"] for r in rows)
            self.assertEqual(metrics["input_tokens"], inputs)
            self.assertEqual(metrics["output_tokens"], outputs)
            self.assertEqual(
                metrics["request_seconds_total"],
                sum(r["request_elapsed_s"] for r in rows),
            )
            self.assertEqual(
                metrics["weighted_prefill_tps"],
                inputs / (sum(r["native"]["prefill_ms"] for r in rows) / 1000),
            )
            self.assertEqual(
                metrics["weighted_generation_tps"],
                outputs / (sum(r["native"]["decode_ms"] for r in rows) / 1000),
            )
            self.assertEqual(
                metrics["all_facts_correct_cases"],
                sum(
                    r["checks"]["facts_matched"] == r["checks"]["facts_total"]
                    for r in rows
                ),
            )
            self.assertEqual(
                metrics["strict_json_cases"],
                sum(
                    r["checks"]["parsed"]
                    and not r["checks"]["schema_errors"]
                    and not r["checks"]["outer_fence_removed"]
                    for r in rows
                ),
            )


if __name__ == "__main__":
    unittest.main()
