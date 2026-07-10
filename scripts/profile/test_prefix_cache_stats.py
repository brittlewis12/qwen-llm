from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prefix_cache_stats


def write_rows(path: Path, *versions: int) -> None:
    rows = [
        {
            "schema_version": version,
            "id": f"request-{index}",
            "model_ttft_ms": 10.0,
        }
        for index, version in enumerate(versions)
    ]
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))


class PrefixCacheStatsSchemaTests(unittest.TestCase):
    def test_mixed_schema_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.jsonl"
            write_rows(path, 2, 3)

            with self.assertRaisesRegex(SystemExit, "mixed request-stats schemas"):
                prefix_cache_stats.parse_stats(path)

    def test_cross_schema_comparison_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "baseline.jsonl"
            candidate_path = Path(directory) / "candidate.jsonl"
            write_rows(baseline_path, 2)
            write_rows(candidate_path, 3)

            baseline = prefix_cache_stats.parse_stats(baseline_path)
            candidate = prefix_cache_stats.parse_stats(candidate_path)
            with self.assertRaisesRegex(
                SystemExit, "cannot compare request-stats schemas"
            ):
                prefix_cache_stats.compare_rows(baseline, candidate)

    def test_combined_cross_schema_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jsonl"
            second_path = Path(directory) / "second.jsonl"
            write_rows(first_path, 2)
            write_rows(second_path, 3)

            rows = prefix_cache_stats.parse_stats(first_path)
            rows.extend(prefix_cache_stats.parse_stats(second_path))
            with self.assertRaisesRegex(SystemExit, "mixed request-stats schemas"):
                prefix_cache_stats.require_single_schema(rows, "combined stats")


if __name__ == "__main__":
    unittest.main()
