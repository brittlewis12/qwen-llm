import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from compare import TraceError, compare, main


def write_raw(path, tensor):
    path.write_bytes(tensor.float().numpy().astype("<f4").tobytes())


def make_traces(root):
    reference, runtime = root / "reference", root / "runtime"
    reference.mkdir()
    runtime.mkdir()
    for directory in (reference, runtime):
        (directory / "run.json").write_text(
            json.dumps({"mode": "NativeF32", "tokens_completed": 2, "n_layers": 2})
        )
    (reference / "tokens.json").write_text(
        json.dumps({"prompt": [11, 12], "generated": [13]})
    )
    (runtime / "run.json").write_text(
        json.dumps(
            {
                "mode": "NativeF32",
                "tokens_completed": 2,
                "token_ids": [11, 12],
                "n_layers": 2,
            }
        )
    )
    for pos in range(2):
        for layer in range(2):
            stem = f"pos-{pos:05d}-layer-{layer:02d}"
            hidden = (
                torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8 + pos + layer
            )
            pre = torch.tensor([0.25, 0.75])
            torch.save(
                {"hidden": hidden[None, None], "pre_mix": pre[None, None]},
                reference / (stem + ".pt"),
            )
            write_raw(runtime / (stem + ".hidden.f32"), hidden)
            write_raw(runtime / (stem + ".pre.f32"), pre)
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0]) + pos
        torch.save(logits[None], reference / f"pos-{pos:05d}-logits.pt")
        write_raw(runtime / f"pos-{pos:05d}-logits.f32", logits)
    return reference, runtime


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.reference, self.runtime = make_traces(Path(self.tmp.name))

    def compare(self, **kwargs):
        return compare(self.reference, self.runtime, **kwargs)

    def change_meta(self, directory, **values):
        path = directory / "run.json"
        metadata = json.loads(path.read_text())
        metadata.update(values)
        path.write_text(json.dumps(metadata))

    def test_matching_and_cli(self):
        report = self.compare()
        self.assertEqual(report["comparison"], "characterization_only")
        self.assertIsNone(report["first_divergence"])
        self.assertEqual(report["token_check"], "complete")
        self.assertEqual(report["tensors_compared"], 10)
        self.assertEqual(report["worst_maxabs"]["maxabs"], 0)
        self.assertEqual(report["worst_maxabs"]["rmse"], 0)
        self.assertEqual(report["worst_maxabs"]["normrelative"], 0)
        for logits in report["logits"]:
            self.assertEqual(logits["kl_reference_to_runtime"], 0)
            self.assertEqual(logits["reference_top1"], 3)
            self.assertTrue(logits["top1_match"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(
                [
                    "--reference-dir",
                    str(self.reference),
                    "--runtime-dir",
                    str(self.runtime),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue()), report)

    def test_first_different_layer_and_metrics(self):
        path = self.runtime / "pos-00000-layer-01.hidden.f32"
        original = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
        changed = original.copy()
        changed[2] += 0.5
        write_raw(path, torch.from_numpy(changed))
        write_raw(self.runtime / "pos-00001-layer-00.pre.f32", torch.tensor([2.0, 3.0]))
        report = self.compare()
        first = report["first_divergence"]
        self.assertEqual(
            (first["position"], first["layer"], first["tensor"]), (0, 1, "hidden")
        )
        self.assertEqual(first["maxabs"], 0.5)
        self.assertAlmostEqual(first["rmse"], 0.5 / math.sqrt(6))
        self.assertAlmostEqual(
            first["normrelative"], 0.5 / math.sqrt(sum(float(x) ** 2 for x in original))
        )
        self.assertEqual(report["worst_maxabs"]["position"], 1)

    def test_pre_mix_divergence(self):
        write_raw(
            self.runtime / "pos-00000-layer-00.pre.f32", torch.tensor([0.5, 0.75])
        )
        self.assertEqual(self.compare()["first_divergence"]["tensor"], "pre")

    def test_logits_kl_and_top1(self):
        write_raw(
            self.runtime / "pos-00000-logits.f32", torch.tensor([3.0, 2.0, 1.0, 0.0])
        )
        report = self.compare()
        self.assertEqual(report["first_divergence"]["tensor"], "logits")
        self.assertIsNone(report["first_divergence"]["layer"])
        self.assertGreater(report["logits"][0]["kl_reference_to_runtime"], 0)
        self.assertEqual(report["logits"][0]["runtime_top1"], 0)
        self.assertFalse(report["logits"][0]["top1_match"])

    def test_explicit_tolerances_are_not_certification(self):
        write_raw(
            self.runtime / "pos-00000-layer-00.pre.f32", torch.tensor([0.26, 0.75])
        )
        self.assertIsNotNone(self.compare()["first_divergence"])
        report = self.compare(atol=0.011, rtol=0)
        self.assertIsNone(report["first_divergence"])
        self.assertEqual(report["comparison"], "user_supplied_thresholds_not_certified")
        self.assertGreater(report["worst_maxabs"]["maxabs"], 0)
        self.assertIsNone(self.compare(rtol=0.05)["first_divergence"])
        for value in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(TraceError, "finite and nonnegative"):
                self.compare(atol=value)

    def test_missing_file_and_invalid_cli_exit(self):
        (self.runtime / "pos-00000-layer-00.pre.f32").unlink()
        with self.assertRaisesRegex(TraceError, "Missing runtime tensor.*tensor=pre"):
            self.compare()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(
                [
                    "--reference-dir",
                    str(self.reference),
                    "--runtime-dir",
                    str(self.runtime),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("Missing runtime", json.loads(err.getvalue())["error"])

    def test_missing_reference_and_both_missing_tail(self):
        (self.reference / "pos-00001-logits.pt").unlink()
        with self.assertRaisesRegex(TraceError, "Missing reference tensor"):
            self.compare()
        (self.runtime / "pos-00001-logits.f32").unlink()
        with self.assertRaisesRegex(TraceError, "Both traces missing"):
            self.compare()

    def test_truncated_raw_file(self):
        path = self.runtime / "pos-00000-layer-00.hidden.f32"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(TraceError, "size/shape mismatch.*truncated"):
            self.compare()

    def test_shape_mismatch(self):
        self.change_meta(self.runtime, hidden_shape=[2, 4])
        with self.assertRaisesRegex(TraceError, "declared hidden_shape"):
            self.compare()
        self.change_meta(self.runtime, hidden_shape=[2, 3])
        path = self.reference / "pos-00000-layer-00.pt"
        torch.save(
            {"hidden": torch.zeros(1, 1, 2, 3), "pre_mix": torch.zeros(1, 1, 3)}, path
        )
        with self.assertRaisesRegex(TraceError, "pre length.*HC count"):
            self.compare()

    def test_nonfinite_runtime_and_reference(self):
        path = self.runtime / "pos-00000-logits.f32"
        write_raw(path, torch.tensor([0.0, float("nan"), 2.0, 3.0]))
        with self.assertRaisesRegex(TraceError, "Nonfinite runtime"):
            self.compare()
        write_raw(path, torch.tensor([0.0, 1.0, 2.0, 3.0]))
        torch.save(
            torch.tensor([[0.0, float("inf"), 2.0, 3.0]]),
            self.reference / "pos-00000-logits.pt",
        )
        with self.assertRaisesRegex(TraceError, "Nonfinite reference"):
            self.compare()

    def test_mode_mismatch(self):
        self.change_meta(self.runtime, mode="ReferenceFp8")
        with self.assertRaisesRegex(TraceError, "Mode mismatch"):
            self.compare()

    def test_current_rust_manifest_schema(self):
        manifest = {
            "schema_version": 1,
            "status": "success",
            "mode": "NativeF32",
            "dtype": "float32",
            "byte_order": "little",
            "layer_count": 2,
            "shapes": {"hidden": [2, 3], "next_pre_mix": [2], "logits": [4]},
            "bytes_per_file": {"hidden": 24, "pre": 8, "logits": 16},
            "positions": [
                {"position": 0, "token_id": 11},
                {"position": 1, "token_id": 12},
            ],
            "prompt_token_ids": [11, 12],
            "generated_token_ids": [13],
        }
        path = self.runtime / "run.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.compare()["token_check"], "complete")
        self.assertIsNone(self.compare()["first_divergence"])
        manifest["positions"][1]["token_id"] = 99
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TraceError, "Conflicting token metadata"):
            self.compare()
        manifest["positions"][1]["token_id"] = 12
        manifest["shapes"]["next_pre_mix"] = [3]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(TraceError, "declared shapes.pre"):
            self.compare()

    def test_incomplete_trace_and_wrong_encoding(self):
        self.change_meta(self.runtime, status="incomplete")
        with self.assertRaisesRegex(TraceError, "trace status=incomplete"):
            self.compare()
        self.change_meta(self.runtime, status="success", byte_order="big")
        with self.assertRaisesRegex(TraceError, "declared byte_order"):
            self.compare()
        self.change_meta(self.runtime, byte_order="little", layer_count=3)
        with self.assertRaisesRegex(TraceError, "layer_count=3"):
            self.compare()

    def test_zero_reference_norm(self):
        path = self.reference / "pos-00000-layer-00.pt"
        torch.save(
            {
                "hidden": torch.zeros(1, 1, 2, 3),
                "pre_mix": torch.tensor([[[0.25, 0.75]]]),
            },
            path,
        )
        write_raw(self.runtime / "pos-00000-layer-00.hidden.f32", torch.ones(2, 3))
        report = self.compare()
        self.assertEqual(report["first_divergence"]["normrelative"], "inf")
        json.dumps(report, allow_nan=False)

    def test_token_mismatch_counts_and_unavailable(self):
        self.change_meta(self.runtime, token_ids=[11, 99])
        with self.assertRaisesRegex(TraceError, "Token mismatch at position 1"):
            self.compare()
        self.change_meta(self.runtime, token_ids=[11, 12], tokens_completed=3)
        with self.assertRaisesRegex(TraceError, "tokens_completed=3"):
            self.compare()
        self.change_meta(self.runtime, tokens_completed=2)
        metadata = json.loads((self.runtime / "run.json").read_text())
        del metadata["token_ids"]
        (self.runtime / "run.json").write_text(json.dumps(metadata))
        self.assertEqual(self.compare()["token_check"], "unavailable")


if __name__ == "__main__":
    unittest.main()
