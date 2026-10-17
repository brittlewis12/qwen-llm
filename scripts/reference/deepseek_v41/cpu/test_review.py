"""Regression controls for review findings; no fixtures are rewritten."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch
from safetensors.torch import save_file

import kernels
from oracle import Oracle
from test_cpu import synthetic_checkpoint
from weights import Weights
from bindings import validate_bindings
from readiness import descriptor_model


def scalar_reference_scale(amax, maximum):
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]

    product = f32(f32(amax) * f32(1.0 / maximum))
    bits = struct.unpack("<I", struct.pack("<f", product))[0]
    exponent = ((bits >> 23) & 255) + bool(bits & 0x7FFFFF)
    return struct.unpack("<f", struct.pack("<I", exponent << 23))[0]


class ScaleBoundaryTest(unittest.TestCase):
    def test_nextafter_both_sides_fp8_and_fp4(self):
        for maximum, quant, exponents in (
            (448, kernels.act_quant, (-22, -16, -8, 0, 8, 32)),
            (6, kernels.fp4_act_quant, (-126, -100, -16, -8, 0, 8, 32)),
        ):
            for exponent in exponents:
                boundary = torch.tensor(maximum * 2.0**exponent, dtype=torch.float32)
                maxima = torch.stack(
                    (
                        torch.nextafter(boundary, torch.tensor(-torch.inf)),
                        boundary,
                        torch.nextafter(boundary, torch.tensor(torch.inf)),
                    )
                )
                x = maxima[:, None].expand(-1, 32).clone()
                _, scales = quant(x)
                minimum = 1e-4 if maximum == 448 else 6 * 2**-126
                expected = torch.tensor(
                    [
                        scalar_reference_scale(max(float(v), minimum), maximum)
                        for v in maxima
                    ]
                )
                torch.testing.assert_close(
                    scales.float().flatten(), expected, atol=0, rtol=0
                )
                if exponent == -16:
                    self.assertEqual(scales.float()[-1].item(), 2**-15)
                    self.assertEqual(scales.float()[0].item(), 2**-16)
                inplace = x.clone()
                if maximum == 448:
                    kernels.act_quant(inplace, inplace=True)
                else:
                    kernels.fp4_act_quant(inplace, inplace=True)
                self.assertTrue(torch.isfinite(inplace).all())

    def test_unrounded_fp8_uses_fp32_reciprocal_multiply(self):
        x = torch.nextafter(torch.tensor(448 * 2**-16), torch.tensor(torch.inf)).expand(
            1, 32
        )
        _, scale = kernels.act_quant(x, scale_fmt=None, scale_dtype=torch.float32)
        torch.testing.assert_close(
            scale,
            (x.amax(-1) * torch.tensor(1 / 448, dtype=torch.float32)).unsqueeze(-1),
            rtol=0,
            atol=0,
        )


class OwnerNegativeControlTest(unittest.TestCase):
    def test_old_owner_changes_selection_and_attention(self):
        path = Path(__file__).resolve().parent.parent / "operator_fixture.py"
        spec = importlib.util.spec_from_file_location(
            "cpu_review_operator_fixture", path
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        original_build = fixture.build
        traces = {False: {}, True: {}}

        def traced_build(uncorrected=False):
            m, blocks = original_build(uncorrected)
            for layer, block in enumerate(blocks):
                indexer = block.attn.indexer
                if indexer is None or not indexer.owns_k:
                    continue

                def capture(obj, inputs, output, layer_id=layer):
                    traces[uncorrected][inputs[3], layer_id] = (
                        m.shared_attn.index_k is obj.k_cache
                    )

                indexer.register_forward_hook(capture)
            return m, blocks

        fixture.build = traced_build
        correct, old = fixture.generate(), fixture.generate(uncorrected=True)
        self.assertTrue(all(traces[False].values()))
        self.assertTrue(
            traces[True][3, 8], "completed pair must publish even in old source"
        )
        self.assertFalse(
            traces[True][4, 8], "old incomplete pair must retain another owner's keys"
        )
        diffs = {
            key: max(abs(a - b) for a, b in zip(value, old[key]))
            for key, value in correct.items()
            if key.endswith("attention")
        }
        first = next(key for key, delta in diffs.items() if delta > 1e-5)
        self.assertEqual(first, "p.4.l.8.attention")
        prefix = first.removesuffix("attention")
        self.assertEqual(correct[prefix + "attn_input"], old[prefix + "attn_input"])
        self.assertNotEqual(correct[prefix + "selection"], old[prefix + "selection"])
        self.assertGreater(max(diffs.values()), 0.5)
        print(
            f"owner negative control: first={first}, max_attention_error={max(diffs.values()):.9g}"
        )


class BindingReadinessTest(unittest.TestCase):
    def test_missing_scale_wrong_shape_and_unbound_backbone_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root)
            model, module, _ = descriptor_model(
                json.loads((root / "config.json").read_text())
            )
            descriptors = Weights([root], 32).descriptors()
            validate_bindings(model, module, descriptors)
            scale = "layers.0.ffn.experts.0.w1.scale"
            missing = dict(descriptors)
            del missing[scale]
            with self.assertRaisesRegex(KeyError, "Missing checkpoint tensor"):
                validate_bindings(model, module, missing)
            wrong = dict(descriptors)
            wrong[scale] = {"dtype": "F8_E8M0", "shape": [1, 1]}
            with self.assertRaisesRegex(ValueError, "expected shape"):
                validate_bindings(model, module, wrong)
            wrong[scale] = {"dtype": "F32", "shape": descriptors[scale]["shape"]}
            with self.assertRaisesRegex(ValueError, "dtype in"):
                validate_bindings(model, module, wrong)
            extra = dict(descriptors)
            extra["layers.0.ffn.experts.999.w1.weight"] = descriptors[
                "layers.0.ffn.experts.0.w1.weight"
            ]
            with self.assertRaisesRegex(ValueError, "Unbound backbone"):
                validate_bindings(model, module, extra)

    def test_original_base_fallback_only_supplies_engram(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main, base = root / "main", root / "base"
            main.mkdir()
            base.mkdir()
            save_file({"head.weight": torch.ones(32, 32)}, main / "main.safetensors")
            save_file(
                {
                    "norm.weight": torch.ones(32),
                    "head.weight": torch.zeros(32, 32),
                    "layers.1.engram.embed.weight": torch.ones(4, 32),
                },
                base / "base.safetensors",
            )
            for indexed in (False, True):
                if indexed:
                    (base / "model.safetensors.index.json").write_text(
                        json.dumps(
                            {
                                "weight_map": {
                                    key: "base.safetensors"
                                    for key in (
                                        "norm.weight",
                                        "head.weight",
                                        "layers.1.engram.embed.weight",
                                    )
                                }
                            }
                        )
                    )
                store = Weights([main, base], 32)
                self.assertNotIn("norm.weight", store.names)
                self.assertIn("layers.1.engram.embed.weight", store.names)
                self.assertEqual(store.tensor("head.weight")[:].sum().item(), 1024)


class NativeF32Test(unittest.TestCase):
    def test_dense_projection_does_not_round_to_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {"dense.weight": torch.eye(32).bfloat16()}, root / "dense.safetensors"
            )
            weights = Weights([root], 32, "NativeF32")
            x = torch.full((1, 32), 1.0001)
            actual = weights.project(
                "dense.weight", x, output_dtype=torch.bfloat16, bf16_weight=True
            )
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, x, atol=0, rtol=0)
            self.assertFalse(torch.equal(actual, actual.bfloat16().float()))

    def test_full_native_activation_dtypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root, ratio_one=True)
            oracle = Oracle(root, root / "dumps", "NativeF32", 32, 16)
            m = oracle.module
            seen = set()

            def check(obj, args, out):
                self.assertEqual(out.dtype, torch.float32, obj.cpu_name)
                seen.add(obj.cpu_name)

            for obj in oracle.model.modules():
                if isinstance(
                    obj,
                    (
                        m.Linear,
                        m.ParallelEmbedding,
                        m.ParallelEngramEmbedding,
                        m.ParallelHead,
                        m.RMSNorm,
                    ),
                ):
                    obj.register_forward_hook(check)
            logits = oracle.feed([1, 2, 3, 4, 5])
            self.assertEqual(logits.dtype, torch.float32)
            for key in (
                "head",
                "embed",
                "layers.3.attn.compressor.wkv",
                "layers.1.engram.embed",
            ):
                self.assertIn(key, seen)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
