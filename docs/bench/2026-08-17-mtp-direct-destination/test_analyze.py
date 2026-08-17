import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("mtp_direct_analyze", HERE / "analyze.py")
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)
RESULTS = json.loads((HERE / "results.json").read_text())


class AnalyzerTests(unittest.TestCase):
    def write_json(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value))
        return path

    def write_time(self, path: Path, timing: dict) -> None:
        path.write_text(
            f"{timing['real_s']} real\n"
            f"{timing['max_rss_bytes']} maximum resident set size\n"
            f"{timing['peak_footprint_bytes']} peak memory footprint\n"
        )

    def write_chronology(self, root: Path, rows: list[dict]) -> Path:
        path = root / "chronology.tsv"
        fields = [
            "event_sequence",
            "event",
            "pair",
            "order",
            "arm",
            "direct",
            "started_utc",
            "ended_utc",
            "exit_status",
            "artifact",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                encoded = dict(row)
                for field in ["pair", "order", "arm", "direct"]:
                    if encoded[field] is None:
                        encoded[field] = ""
                if isinstance(encoded["direct"], bool):
                    encoded["direct"] = str(int(encoded["direct"]))
                writer.writerow(encoded)
        return path

    def materialize_campaign(self, root: Path) -> dict[str, Path]:
        chronology = self.write_chronology(root, RESULTS["execution_chronology"])
        identity = RESULTS["byte_oracle"]["source_identity"]
        for name in [
            "build-info-before.json",
            "build-info-after-oracle.json",
            "build-info-after-campaign.json",
        ]:
            self.write_json(root, name, identity)
        oracle = self.write_json(root, "byte-oracle.json", RESULTS["byte_oracle"])
        excluded = RESULTS["excluded_setup_samples"][0]
        packed = self.write_json(root, "packed-control.json", excluded["packet"])
        self.write_time(packed.with_suffix(".time"), excluded["time"])
        for arm in RESULTS["arms"]:
            name = f"pair{arm['pair']:02d}-{arm['order']}-{arm['arm']}.json"
            path = self.write_json(root, name, arm["packet"])
            self.write_time(path.with_suffix(".time"), arm["time"])
        return {"manifest": chronology, "oracle": oracle, "packed": packed}

    def test_byte_oracle_mismatch_closes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = copy.deepcopy(RESULTS["byte_oracle"])
            path = self.write_json(root, "oracle.json", packet)
            _, equal, _, _ = ANALYZE.load_byte_oracle(path)
            self.assertTrue(equal)

            packet["banks"][0]["direct_blake3"] = "0" * 64
            packet["staged_equals_direct"] = False
            path.write_text(json.dumps(packet))
            _, equal, _, _ = ANALYZE.load_byte_oracle(path)
            self.assertFalse(equal)

    def test_chronology_rejects_schedule_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = copy.deepcopy(RESULTS["execution_chronology"])
            path = self.write_chronology(root, rows)
            self.assertEqual(len(ANALYZE.load_chronology(path)), 17)

            rows[0], rows[1] = rows[1], rows[0]
            path = self.write_chronology(root, rows)
            with self.assertRaises(ValueError):
                ANALYZE.load_chronology(path)

    def test_arm_rejects_non_treatment_environment_drift(self) -> None:
        packet = copy.deepcopy(RESULTS["arms"][0]["packet"])
        ANALYZE.validate_arm(packet, False)
        packet["qwen_env"]["QWEN_UNREGISTERED_FLAG"] = "1"
        with self.assertRaises(ValueError):
            ANALYZE.validate_arm(packet, False)

    def test_excluded_packet_must_be_staged_failed_n2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            excluded = RESULTS["excluded_setup_samples"][0]
            packet = copy.deepcopy(excluded["packet"])
            path = self.write_json(root, "packed-control.json", packet)
            self.write_time(path.with_suffix(".time"), excluded["time"])
            ANALYZE.validate_excluded_control(path)
            packet["speculative"]["target_state"]["continuation_steps"] = 15
            path.write_text(json.dumps(packet))
            with self.assertRaises(ValueError):
                ANALYZE.validate_excluded_control(path)

    def test_complete_analysis_rejects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.materialize_campaign(root)
            output = root / "output.json"
            command = [
                sys.executable,
                str(HERE / "analyze.py"),
                "--input",
                str(root),
                "--manifest",
                str(paths["manifest"]),
                "--byte-oracle",
                str(paths["oracle"]),
                "--invalid-packed-control",
                str(paths["packed"]),
                "--expected-commit",
                RESULTS["source_identity"]["commit"],
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())["disposition"], "GO")

            drifted = copy.deepcopy(RESULTS["byte_oracle"]["source_identity"])
            drifted["build_source_state"] = "git-source-sha256-v2:" + "0" * 64
            drifted["runtime_source_state"] = drifted["build_source_state"]
            self.write_json(root, "build-info-after-oracle.json", drifted)
            failed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("source identity drifted", failed.stderr)


if __name__ == "__main__":
    unittest.main()
