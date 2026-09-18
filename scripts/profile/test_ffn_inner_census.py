from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import ffn_inner_census as census


class SwigluCensusTests(unittest.TestCase):
    def test_muse_ledger_separates_cached_loads_from_weight_payload(self):
        ledger = census.q8_ffn_ledger(6656, 19968, 52)
        self.assertEqual(ledger["weight_bytes_per_matrix"], 141213696)
        self.assertEqual(ledger["weight_bytes_all_layers"], 22029336576)
        self.assertEqual(ledger["fusion_weight_bytes_removed"], 0)
        self.assertEqual(ledger["fusion_dispatches_removed_all_layers"], 104)
        self.assertEqual(ledger["fusion_intermediate_bytes_removed_per_layer"], 319488)
        self.assertEqual(
            ledger["fusion_shader_x_load_bytes_removed_per_layer"], 265814016
        )
        self.assertEqual(ledger["fusion_x_unique_bytes"], 26624)
        self.assertEqual(ledger["up_bytes_per_removed_block"], 226304)
        self.assertEqual(ledger["down_bytes_per_removed_block"], 226304)
        self.assertEqual(ledger["fusion_threadgroup_bytes_after"], 512)

    def test_ledger_rejects_invalid_geometry(self):
        for dims in [(0, 32, 1), (32, 33, 1), (33, 32, 1), (32, 32, 0), (32.0, 32, 1)]:
            with self.assertRaises(ValueError):
                census.q8_ffn_ledger(*dims)

    def test_small_gate_with_large_up_is_not_a_product_certificate(self):
        gate = np.array([[1e-6] * 32 + [1.0] * 32], dtype=np.float32)
        up = np.array([[1e9] * 32 + [1.0] * 32], dtype=np.float32)
        inner = gate / (1 + np.exp(-gate)) * up
        result = census.analyze_swiglu(gate, up, inner, (1e-3,))
        candidate = result["gate_only_candidates"]["1e-03"]
        self.assertEqual(candidate["selected_blocks"], 1)
        self.assertEqual(candidate["conditional_violation_fraction"], 1)
        self.assertGreater(
            candidate["discarded_observed_inner_energy_fraction"]["min"], 0.99
        )
        self.assertAlmostEqual(
            candidate["all_ffn_weight_fraction_up_down_hypothesis"]["mean"], 1 / 3
        )

    def test_product_oracle_cannot_claim_gate_or_up_removal(self):
        gate = np.ones((1, 64), dtype=np.float32)
        up = np.ones_like(gate)
        inner = np.array([[0.0] * 32 + [1.0] * 32], dtype=np.float32)
        result = census.analyze_swiglu(gate, up, inner, (0.0,))
        oracle = result["post_product_oracle"]["1e-03"]
        self.assertEqual(oracle["block_fraction"]["mean"], 0.5)
        self.assertAlmostEqual(
            oracle["all_ffn_weight_fraction_down_only"]["mean"], 1 / 6
        )
        candidate = result["gate_only_candidates"]["0"]
        self.assertIsNone(candidate["conditional_violation_fraction"])
        self.assertEqual(candidate["missed_product_threshold_blocks"], 1)

    def test_scattered_zeros_do_not_delete_physical_blocks(self):
        inner = np.tile([0.0] * 31 + [1.0], 2).astype(np.float32)[None, :]
        result = census.analyze_swiglu(inner, np.ones_like(inner), inner, (0.0,))
        self.assertEqual(
            result["post_product_oracle"]["1e-03"]["block_fraction"]["mean"], 0
        )

    def test_zero_energy_is_explicit_and_legacy_guard_remains(self):
        zero = np.zeros((1, 32), dtype=np.float32)
        result = census.analyze_swiglu(zero, zero, zero, (0.0,))
        self.assertEqual(result["zero_energy_vectors"], 1)
        self.assertAlmostEqual(
            result["post_product_oracle"]["1e-03"]["all_ffn_weight_fraction_down_only"][
                "mean"
            ],
            1 / 3,
        )
        stats = census.analyze_layer(zero, (0.0,), 32, allow_zero_energy=True)
        self.assertEqual(stats["effective_block_count"]["mean"], 0)
        self.assertEqual(stats["largest_blocks_for_energy_coverage"]["0.90"]["mean"], 0)
        with self.assertRaises(ValueError):
            census.analyze_layer(zero, (0.0,), 32)

    def test_nonfinite_mismatched_and_ambiguous_inputs_fail(self):
        base = np.ones((1, 32), dtype=np.float32)
        for bad in (np.full_like(base, np.nan), base[:, :31], np.ones((2, 32))):
            with self.assertRaises(ValueError):
                census.analyze_swiglu(bad, base, base, (0.0,))
        for thresholds in ((), (-1.0,), (float("inf"),), (1.1e-3, 1.2e-3)):
            with self.assertRaises(ValueError):
                census.analyze_swiglu(base, base, base, thresholds)

    def test_negative_gate_proxy_is_not_raw_gate_magnitude(self):
        gate = np.full((1, 32), -100.0, dtype=np.float32)
        result = census.analyze_swiglu(
            gate, np.ones_like(gate), np.full_like(gate, -3.72e-42), (1e-3,)
        )
        candidate = result["gate_only_candidates"]["1e-03"]
        self.assertEqual(candidate["selected_blocks"], 1)
        self.assertEqual(candidate["raw_gate_blocks_at_same_threshold"], 0)
        self.assertEqual(result["raw_gate_max_abs"]["mean"], 100)

    def test_partial_layers_and_malformed_capture_metadata_fail(self):
        good = {
            "layer_count": 2,
            "layers": [0, 1],
            "weight_dtype": "Q8_0",
            "matvec_variant": "lcpp_nr0_2_nsg_4",
            "source_commit": "synthetic",
            "model_content_identity": "synthetic",
            "prefix_identity": "synthetic",
            "captured_token_ids": [1, 2],
            "captured_positions": [5, 6],
        }
        census.validate_swiglu_provenance(good, 2)
        for change in (
            {"layers": [0]},
            {"layers": [1, 0]},
            {"layer_count": True},
            {"weight_dtype": "BF16"},
            {"matvec_variant": "other"},
            {"captured_positions": [5, 5]},
            {"captured_token_ids": [1]},
            {"captured_positions": [True, 6]},
            {"source_commit": 42},
        ):
            with self.assertRaises(ValueError):
                census.validate_swiglu_provenance(good | change, 2)

    def test_hash_does_not_override_byte_shape_or_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = np.ones((1, 1, 32), dtype="<f4")
            tensor = self.write_tensor(path, "inner", values)
            census.load_tensor(path / "manifest.json", tensor)
            tensor["shape"] = [1, 1, 31]
            with self.assertRaises(ValueError):
                census.load_tensor(path / "manifest.json", tensor)
            tensor = self.write_tensor(path, "inner", np.full_like(values, np.nan))
            with self.assertRaises(ValueError):
                census.load_tensor(path / "manifest.json", tensor)

    @staticmethod
    def write_tensor(directory, name, values):
        data = values.astype("<f4").tobytes()
        (directory / f"{name}.bin").write_bytes(data)
        return {
            "name": f"{name}.bin",
            "dtype": "f32le",
            "shape": list(values.shape),
            "byte_length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_cli_v1_v2_and_invalid_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = np.ones((2, 2, 256), dtype=np.float32)
            tensors = {
                name: self.write_tensor(path, name, values)
                for name in ("gate", "up", "inner")
            }
            common = {
                "layers": [0, 1],
                "hidden_size": 32,
                "layer_count": 2,
                "intermediate_size": 256,
                "weight_dtype": "Q8_0",
                "matvec_variant": "lcpp_nr0_2_nsg_4",
            }
            manifests = [
                {**common, "schema_version": 1, "tensor": tensors["inner"]},
                {
                    **common,
                    "schema_version": 2,
                    "tensors": tensors,
                    "source_commit": "synthetic-test",
                    "model_content_identity": "synthetic-test",
                    "prefix_identity": "synthetic-test",
                    "captured_token_ids": [1, 2],
                    "captured_positions": [7, 8],
                },
            ]
            for version, manifest in enumerate(manifests, 1):
                file = path / "manifest.json"
                file.write_text(json.dumps(manifest))
                subprocess.run(
                    [
                        sys.executable,
                        census.__file__,
                        str(file),
                        "--thresholds",
                        "1e-3",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                report = json.loads((path / "analysis.json").read_text())
                self.assertEqual(report["schema_version"], version)
                self.assertEqual(report["block_size"], 256 if version == 1 else 32)
                self.assertEqual("swiglu" in report, version == 2)
            for key in (
                "source_commit",
                "model_content_identity",
                "prefix_identity",
                "captured_positions",
            ):
                bad = dict(manifests[1])
                del bad[key]
                with self.assertRaises(ValueError):
                    census.validate_swiglu_provenance(bad, 2)


if __name__ == "__main__":
    unittest.main()
