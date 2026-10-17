"""Model-free control-flow evidence, NOT numerical equivalence or a Rust oracle.

Run offline: uv run --no-project python scripts/reference/deepseek_v41/check.py
Capture: uv run --no-project python scripts/reference/deepseek_v41/capture.py --fetch

Executes the actual Indexer.forward prefix through cache publication, the complete
Attention._compress_topk_idxs dispatch, and linear dispatch. Numerical operations
are identity/recording stubs. Pair completion is supplied as an input, not a claim
to validate compressor arithmetic. Planned native F32 activation arithmetic is
deliberately NOT equivalent to the source FP8 activation policy checked here.
"""

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import capture
from check import require_tests
from capture import verify


def method(tree, cls, name):
    parent = (
        tree
        if cls is None
        else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    )
    return copy.deepcopy(
        next(
            n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )


def compile_function(node, env):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "pinned-model-extract", "exec"), env
    )
    return env[node.name]


class Tensor:
    def __init__(self, length=1, owner=None):
        self.length, self.owner = length, owner
        self.writes = []
        self.device = "stub"

    def size(self, dim=None):
        shape = (1, self.length, 128)
        return shape if dim is None else shape[dim]

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        self.writes.append((key, value))

    def unsqueeze(self, dim):
        return self


class Sources(unittest.TestCase):
    def test_required_suite_rejects_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "selected no tests"):
            require_tests(unittest.TestSuite())
        missing = unittest.defaultTestLoader.loadTestsFromName(
            "missing_v41_source_contract_module"
        )
        result = unittest.TestResult()
        require_tests(missing).run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertFalse(result.wasSuccessful())

    def test_each_fixture_tamper_is_rejected(self):
        contents = verify()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources.json").write_bytes(
                (capture.ROOT / "sources.json").read_bytes()
            )
            for name, data in contents.items():
                path = root / "fixtures" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with patch.object(capture, "ROOT", root):
                for name, data in contents.items():
                    path = root / "fixtures" / name
                    path.write_bytes(data + b"\n")
                    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                        verify()
                    path.write_bytes(data)

    def setUp(self):
        sources = verify()
        self.tree = ast.parse(sources["inference/model.py"])
        self.config = json.loads(sources["config.json"])["text_config"]
        self.shared = NS(index_k=None, topk_idxs=None)
        self.env = dict(
            shared_attn=self.shared,
            fp4_block_size=32,
            apply_rotary_emb=lambda *a: None,
            fp4_act_quant=lambda *a: None,
            torch=NS(empty=lambda *a, **kw: Tensor(0), int32="int32"),
        )

    def prefix(self, broken=False):
        node = method(self.tree, "Indexer", "forward")
        boundary = next(
            i
            for i, n in enumerate(node.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "q"
        )
        node.body = node.body[:boundary]
        append, publish = node.body[-2:]
        self.assertEqual(ast.unparse(append.test), "self.owns_k and latent is not None")
        self.assertEqual(ast.unparse(publish.test), "self.owns_k")
        if broken:
            append.body.extend(publish.body)
            node.body.pop()
        return compile_function(node, self.env)

    def owner(self, layer):
        obj = NS(
            compress_ratio=self.config["compress_ratios"][layer],
            rope_head_dim=64,
            freqs_cis=Tensor(),
            k_cache=Tensor(owner=layer),
            wk=lambda x: x,
            k_norm=lambda x: x,
        )
        init = method(self.tree, "Indexer", "__init__")
        assignment = next(
            n
            for n in init.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.owns_k"
        )
        env = dict(
            self=obj,
            layer_id=layer,
            args=NS(kv_source_layers=self.config["kv_source_layer_ids"]),
        )
        exec(
            compile(
                ast.Module(body=[assignment], type_ignores=[]),
                "ownership-extract",
                "exec",
            ),
            env,
        )
        return obj

    def test_all_owners_pair_phases_and_cache_ranges(self):
        self.assertEqual(self.config["kv_source_layer_ids"], [2, 8, 14, 20])
        run = self.prefix()
        for layer in [2, 8, 14, 20]:
            for start, length in [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
            ]:
                with self.subTest(layer=layer, start=start, length=length):
                    obj = self.owner(layer)
                    ratio = obj.compress_ratio
                    count = (
                        length // ratio if start == 0 else int((start + 1) % ratio == 0)
                    )
                    latent = Tensor(count) if count else None
                    self.shared.index_k = Tensor(owner=20)
                    run(obj, Tensor(length), None, latent, start, 128)
                    self.assertIs(self.shared.index_k, obj.k_cache)
                    self.assertEqual(len(obj.k_cache.writes), int(count > 0))
                    if count:
                        key, value = obj.k_cache.writes[0]
                        self.assertEqual(
                            key[1], slice(start // ratio, start // ratio + count)
                        )
                        self.assertIs(value, latent)

    def test_negative_control_after_decoder20(self):
        for broken in (False, True):
            run = self.prefix(broken)
            decoder = self.owner(20)
            run(decoder, Tensor(), None, Tensor(), 1, 128)
            for layer in (2, 8, 14):
                obj = self.owner(layer)
                run(obj, Tensor(), None, None, 2, 128)
                self.assertEqual(self.shared.index_k.owner, 20 if broken else layer)
                self.assertEqual(obj.k_cache.writes, [])

    def test_reindex_and_reuse_are_separate(self):
        run = self.prefix()
        dispatch = compile_function(
            method(self.tree, "Attention", "_compress_topk_idxs"), self.env
        )
        self.assertEqual(
            self.config["index_source_layer_ids"], [2, 8, 14, 20, 24, 28, 32, 36]
        )
        for layer in range(2, 40):
            obj = self.owner(layer)
            prior = self.shared.index_k = Tensor(owner=20)
            calls = []

            class IndexCall:
                freqs_cis = Tensor()

                def __call__(_, *args):
                    calls.append(layer)
                    run(obj, *args)
                    return object()

            attn = NS(
                is_index_source=layer in self.config["index_source_layer_ids"],
                indexer=IndexCall(),
            )
            previous = self.shared.topk_idxs = object()
            result = dispatch(attn, Tensor(), None, None, 2, 128, 1)
            self.assertEqual(bool(calls), attn.is_index_source)
            self.assertIs(result, self.shared.topk_idxs)
            self.assertEqual(result is previous, not attn.is_index_source)
            self.assertIs(self.shared.index_k, obj.k_cache if obj.owns_k else prior)
            calls.clear()
            result = dispatch(attn, Tensor(), None, None, 0, 128, 0)
            self.assertEqual(calls, [])
            if attn.is_index_source:
                self.assertEqual(result.length, 0)

    def test_linear_activation_policy(self):
        kernel = ast.parse(verify()["inference/kernel.py"])
        quant_source = method(kernel, None, "act_quant")
        output = next(
            n
            for n in quant_source.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "y"
        )
        self.assertEqual(
            ast.unparse(output.value),
            "torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)",
        )
        events = []
        torch = NS(
            float4_e2m1fn_x2="fp4", float8_e4m3fn="fp8", float8_e8m0fnu="ue8m0-dtype"
        )
        env = dict(torch=torch)
        for node in self.tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("fp8_block_size", "scale_fmt", "scale_dtype")
            ):
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        "policy-extract",
                        "exec",
                    ),
                    env,
                )

        def quant(*args):
            events.append(("act_quant", args))
            return "quantized-fp8", "activation-scales"

        def gemm(kind):
            def call(*args, **kwargs):
                events.append((kind, args, kwargs))
                return kind

            return call

        env.update(
            act_quant=quant,
            fp4_gemm=gemm("fp4"),
            fp8_gemm=gemm("fp8"),
            F=NS(linear=gemm("native")),
        )
        linear = compile_function(method(self.tree, None, "linear"), env)
        for dtype in ("fp4", "fp8", "f32"):
            events.clear()
            weight = NS(dtype=dtype, scale="weight-scales")
            linear("input", weight)
            if dtype == "f32":
                self.assertEqual([e[0] for e in events], ["native"])
            else:
                self.assertEqual(
                    events[0], ("act_quant", ("input", 32, "ue8m0", "ue8m0-dtype"))
                )
                self.assertEqual(events[1][0], dtype)
                self.assertEqual(
                    events[1][1],
                    (
                        "quantized-fp8",
                        "activation-scales",
                        weight,
                        "weight-scales",
                        "ue8m0-dtype",
                    ),
                )
                self.assertEqual(
                    events[1][2],
                    {"act_block_size" if dtype == "fp4" else "block_size": 32},
                )


if __name__ == "__main__":
    unittest.main()
