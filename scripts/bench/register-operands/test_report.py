#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""CPU-only checks for raw-sample accounting, not GPU admission or quality."""

import json
import tempfile
import unittest
from pathlib import Path

from report import summarize


class ReportTests(unittest.TestCase):
    def test_keeps_unstable_samples_and_excludes_only_declared_warmup(self):
        records = []
        for name, values in {
            "kernel_moe_down_iq3_xxs": [100, 100, 500],
            "wxsh_iq3_xxs": [50, 50, 150],
            "staged_resident_iq3_xxs": [40, 40, 80],
            "wx_resident_iq3_xxs": [20, 20, 60],
        }.items():
            records.append({"event": "sample", "name": name, "gpu_ns": 9999, "repeats": 1})
            records += [
                {"event": "sample", "name": name, "gpu_ns": v, "repeats": 3} for v in values
            ]
        records += [
            {"event": "price", "priced_bytes": 123},
            {"event": "operand_parity", "half_values_checked": 42, "bit_mismatches": 0},
            {
                "event": "output_parity",
                "bit_mismatches_vs_baseline": 5,
                "max_abs_vs_baseline": 0.001,
                "cosine": 0.99999,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            result = summarize(path)
        self.assertEqual(result["timing"]["wxsh_iq3_xxs"]["samples_ns"], [50, 50, 150])
        self.assertEqual(result["timing"]["wxsh_iq3_xxs"]["matched_median_speedup"], 2)
        self.assertEqual(result["timing"]["wxsh_iq3_xxs"]["max_over_min"], 3)
        self.assertEqual(
            result["timing"]["wx_resident_iq3_xxs"]["matched_median_speedup"], 2
        )
        self.assertEqual(result["output_bit_mismatches"], 5)
        self.assertEqual(result["operand_half_values_checked"], 42)
        self.assertEqual(result["max_priced_bytes"], 123)


if __name__ == "__main__":
    unittest.main()
