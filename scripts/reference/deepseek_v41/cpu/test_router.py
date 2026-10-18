"""Router tie contract and upstream negative control, without real weights."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from oracle import HERE, MODEL_SHA256, Oracle, source_module
from test_cpu import synthetic_checkpoint


def gate(module, logits, bias, topk=2):
    args = module.ModelArgs(
        dim=1,
        n_routed_experts=len(logits),
        n_activated_experts=topk,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        route_scale=1.5,
        norm_topk_prob=True,
    )
    obj = module.Gate(0, args)
    obj.weight = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32)[:, None])
    obj.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
    return obj


class RouterTest(unittest.TestCase):
    def test_bias_absorption_lower_id_and_unbiased_weights(self):
        m = source_module(stable_router=True)
        logits = [-34.0, -32.0, -30.0, -28.0]
        scores = torch.nn.functional.softplus(torch.tensor(logits)).sqrt()
        self.assertEqual(scores.unique().numel(), 4)
        self.assertEqual((scores + 32.0).unique().numel(), 1)
        weights, ids = gate(m, logits, [32.0] * 4)(torch.ones(1, 1))
        self.assertEqual(ids.tolist(), [[0, 1]])
        expected = scores[:2] / (scores[:2].sum() + 1e-20) * 1.5
        torch.testing.assert_close(weights[0], expected, rtol=0, atol=0)
        self.assertNotEqual(weights[0, 0].item(), weights[0, 1].item())

    def test_no_tie_selection_equivalence_and_id_order(self):
        logits, bias = [-3.0, 2.0, -1.0, 4.0], [0.0] * 4
        stable = gate(source_module(stable_router=True), logits, bias)
        original = gate(source_module(), logits, bias)
        weights, ids = stable(torch.ones(1, 1))
        upstream_weights, upstream_ids = original(torch.ones(1, 1))
        self.assertEqual(ids.tolist(), [[1, 3]])
        order = upstream_ids.argsort(-1)
        self.assertTrue(torch.equal(ids, upstream_ids.gather(1, order)))
        torch.testing.assert_close(
            weights, upstream_weights.gather(1, order), rtol=0, atol=0
        )

    def test_original_method_and_source_hash_unchanged(self):
        text = (HERE / "old_model.py").read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), MODEL_SHA256)
        tree = ast.parse(text)
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gate"
        )
        m = source_module()
        namespace = dict(m.__dict__)
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), "original", "exec"),
            namespace,
        )
        self.assertEqual(
            m.Gate.forward.__code__.co_names,
            namespace["Gate"].forward.__code__.co_names,
        )
        for logits, bias in (
            ([-34.0, -32.0, -30.0, -28.0], [32.0] * 4),
            ([-3.0, 2.0, -1.0, 4.0], [0.0] * 4),
        ):
            obj = gate(m, logits, bias)
            actual = obj(torch.ones(1, 1))
            expected = namespace["Gate"].forward(obj, torch.ones(1, 1))
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(source_module(stable_router=True).source_hash, MODEL_SHA256)

    def test_oracle_modes_negative_control_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root)
            original_code = source_module().Gate.forward.__code__.co_code
            for mode, upstream, stable in (
                ("NativeF32", False, True),
                ("NativeF32", True, False),
                ("ReferenceFp8", False, False),
                ("ReferenceFp8", True, False),
            ):
                with self.subTest(mode=mode, upstream=upstream):
                    dump = root / f"{mode}-{upstream}"
                    oracle = Oracle(
                        root, dump, mode, 32, 16, upstream_router_ties=upstream
                    )
                    manifest = json.loads((dump / "run.json").read_text())
                    self.assertEqual(manifest["source_sha256"], MODEL_SHA256)
                    self.assertEqual(
                        manifest["router_tie_policy"],
                        "lower_expert_id_then_id_sorted"
                        if stable
                        else "upstream_torch_topk",
                    )
                    self.assertEqual(
                        oracle.module.Gate.forward.__code__.co_code == original_code,
                        not stable,
                    )


if __name__ == "__main__":
    unittest.main()
