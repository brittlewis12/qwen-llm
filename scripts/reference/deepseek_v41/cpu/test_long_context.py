import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from gguf_weights import ALIASES, GGUFWeights
from long_context import (
    Capture,
    bind_attention,
    configure,
    main,
    metrics,
    replay,
    seed_history,
)
from oracle import source_module
from test_gguf import write_gguf


def tiny_attention_args(m):
    return m.ModelArgs(
        dim=16,
        n_layers=3,
        n_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=3,
        max_seq_len=32,
        max_batch_size=1,
        window_size=4,
        compress_ratios=[2, 2, 2],
        kv_source_layers=[0, 1],
        index_source_layers=[0, 1, 2],
        candidate_source_layer=-1,
    )


class SyntheticWeights:
    def __init__(self, m, args):
        self.values = {}
        with m.set_dtype(torch.float32):
            for layer in range(3):
                attn = m.Attention(layer, args)
                for name, param in attn.named_parameters():
                    self.values[f"layers.{layer}.attn.{name}"] = (
                        torch.randn(param.shape) * 0.2
                    )

    def tensor(self, name):
        return self.values[name]

    def project(self, name, x, start=None, end=None, **kwargs):
        weight = self.values[name]
        if start is not None:
            weight = weight[start:end]
        if kwargs.get("bf16_weight"):
            weight = weight.bfloat16().float()
        return torch.nn.functional.linear(x, weight)


class CaptureTests(unittest.TestCase):
    def test_backend_selection_required_and_exclusive(self):
        for model, gguf in ((None, None), ("hf", "weights.gguf")):
            with patch("long_context.Capture") as capture:
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    replay(model, "capture", gguf=gguf)
                capture.assert_not_called()
        for flags in ([], ["--model", "hf", "--gguf", "weights.gguf"]):
            with (
                patch("sys.argv", ["long_context.py", "--capture", "capture", *flags]),
                patch("sys.stderr", new_callable=io.StringIO),
                patch("long_context.replay") as run,
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        for flag, path in (("--model", "hf"), ("--gguf", "weights.gguf")):
            with (
                patch(
                    "sys.argv", ["long_context.py", "--capture", "capture", flag, path]
                ),
                patch("sys.stdout", new_callable=io.StringIO),
                patch("long_context.replay", return_value=[]) as run,
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 0)
            run.assert_called_once_with(
                path if flag == "--model" else None,
                "capture",
                256,
                0.02,
                0.02,
                gguf=path if flag == "--gguf" else None,
            )

    def test_replay_backend_dispatch_reports_and_cleanup(self):
        for gguf in (None, "weights.gguf"):
            for failure in (None, "metadata", "binding", "history"):
                if gguf is None and failure == "metadata":
                    continue
                with self.subTest(gguf=gguf, failure=failure):
                    cap = Mock(root=Path("capture"))
                    native = {
                        "output": "output.npy",
                        "compressed_ids": [1, 2],
                        "kv_owner": 2,
                        "index_source": 2,
                        "candidate_blocks": [0],
                    }
                    cap.meta = {
                        "checkpoints": [{"position": 7, "layers": {"2": native}}]
                    }
                    args = SimpleNamespace(
                        index_source_layers=[2],
                        window_size=4,
                        candidate_source_layer=2,
                        candidate_block_size=2,
                    )
                    runtime = SimpleNamespace(
                        topk_idxs=torch.tensor([5, 6]),
                        candidates=torch.tensor([True, True, False, False]),
                    )
                    m = Mock()
                    m.SharedAttentionRuntime.return_value = runtime
                    attn = Mock(
                        is_kv_source=True,
                        is_index_source=True,
                        compress_ratio=2,
                        return_value=torch.zeros(3),
                    )
                    store = Mock()
                    if failure == "metadata":
                        store.validate_config.side_effect = ValueError("metadata")
                    diagnostics = {
                        "selection_boundary_gap": 0.0,
                        "selection_boundary_tie": True,
                        "candidate_boundary_tie": True,
                    }
                    with (
                        patch("long_context.Capture", return_value=cap),
                        patch("long_context.source_module", return_value=m),
                        patch("long_context.model_args", return_value=args),
                        patch("long_context.Weights", return_value=store) as hf,
                        patch("long_context.GGUFWeights", return_value=store) as gg,
                        patch("long_context.configure") as setup,
                        patch("long_context.bind_attention", return_value=attn) as bind,
                        patch("long_context.seed_history") as seed,
                        patch(
                            "long_context.np.load",
                            return_value=np.zeros(3, dtype="<f4"),
                        ),
                        patch(
                            "long_context.selection_diagnostics",
                            return_value=diagnostics,
                        ),
                    ):
                        if failure == "binding":
                            bind.side_effect = ValueError("binding")
                        if failure == "history":
                            seed.side_effect = ValueError("history")
                        if failure:
                            with self.assertRaisesRegex(ValueError, failure):
                                replay(None if gguf else "hf", "capture", gguf=gguf)
                        else:
                            rows = replay(None if gguf else "hf", "capture", gguf=gguf)
                            self.assertIsInstance(rows, list)
                            self.assertEqual(len(rows), 1)
                            row = rows[0]
                            self.assertEqual(
                                row["weight_backend"],
                                "independent_gguf" if gguf else "hf_safetensors",
                            )
                            self.assertEqual(
                                row["comparison_scope"],
                                "attention_only_captured_normalized_inputs",
                            )
                            for key in (
                                "selection_exact",
                                "owner_exact",
                                "candidates_exact",
                            ):
                                self.assertTrue(row[key])
                            self.assertTrue(row["output"]["pass"])
                            self.assertEqual(row["selection_discard"], 2)
                            self.assertEqual(row["candidate_discard"], 2)
                            for key, value in diagnostics.items():
                                self.assertEqual(row[key], value)
                        if gguf:
                            gg.assert_called_once_with(gguf, 128, "NativeF32")
                            hf.assert_not_called()
                            store.validate_config.assert_called_once()
                            self.assertIn(
                                "text_config", store.validate_config.call_args.args[0]
                            )
                            store.close.assert_called_once_with()
                        else:
                            hf.assert_called_once_with(["hf"], 128, "NativeF32")
                            gg.assert_not_called()
                            store.validate_config.assert_not_called()
                            store.close.assert_not_called()
                        if failure == "metadata":
                            setup.assert_not_called()
                            bind.assert_not_called()

    def test_capture_completeness_before_weights(self):
        layers = (2, 8, 14, 20, 24, 28, 32, 36)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "inputs.npy", np.zeros((5, 5120), dtype="<f4"))
            good = {
                "format": "dsv41-long-context-v1",
                "mode": "NativeF32",
                "status": "success",
                "requested_checkpoints": [2, 4],
                "inputs": {str(l): {"file": "inputs.npy", "start": 0} for l in layers},
                "checkpoints": [
                    {"position": p, "layers": {str(l): {} for l in layers}}
                    for p in (2, 4)
                ],
            }
            (root / "manifest.json").write_text(json.dumps(good))
            self.assertEqual(Capture(root).rows(36, 4, 5).shape, (1, 1, 5120))
            cases = []
            for status in (None, "running", "failed"):
                bad = copy.deepcopy(good)
                if status is None:
                    del bad["status"]
                else:
                    bad["status"] = status
                cases.append(bad)
            for requested in ([], [4, 2], [2, 2], [2, 4, 10000], [0, 4]):
                bad = copy.deepcopy(good)
                bad["requested_checkpoints"] = requested
                cases.append(bad)
            for checkpoint in (0, 1):
                bad = copy.deepcopy(good)
                del bad["checkpoints"][checkpoint]["layers"]["36"]
                cases.append(bad)
            bad = copy.deepcopy(good)
            bad["checkpoints"].reverse()
            cases.append(bad)
            bad = copy.deepcopy(good)
            del bad["inputs"]["36"]
            cases.append(bad)
            bad = copy.deepcopy(good)
            bad["inputs"]["36"]["start"] = 1
            cases.append(bad)
            np.save(root / "short.npy", np.zeros((4, 5120), dtype="<f4"))
            bad = copy.deepcopy(good)
            bad["inputs"]["36"]["file"] = "short.npy"
            cases.append(bad)
            for i, bad in enumerate(cases):
                with self.subTest(case=i), patch("long_context.Weights") as weights:
                    (root / "manifest.json").write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        replay("unused", root)
                    weights.assert_not_called()


class LongContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.m = source_module(stable_router=True)
        self.m.act_quant = self.m.fp4_act_quant = lambda x, *a, **kw: x

    def test_gguf_attention_validates_before_binding(self):
        m = self.m
        args = tiny_attention_args(m)
        weights = SyntheticWeights(m, args)
        reverse = {v: k for k, v in ALIASES.items()}
        tensors = []
        for name, value in weights.values.items():
            if name.endswith(".scale"):
                continue
            _, layer, tail = name.split(".", 2)
            tensors.append(
                (
                    f"blk.{layer}.{reverse[tail]}",
                    tuple(value.shape),
                    0,
                    value.numpy().tobytes(),
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            write_gguf(path, tensors)
            store = GGUFWeights(path)
            try:
                configure(m, store)
                with patch.object(
                    m, "Transformer", side_effect=AssertionError("full model")
                ):
                    for layer in range(3):
                        attn = bind_attention(m, args, store, layer)
                        for name, obj in attn.named_modules():
                            full = f"layers.{layer}.attn" + ("." + name if name else "")
                            self.assertEqual(obj.cpu_name, full)
                            if isinstance(obj, m.Linear):
                                self.assertTrue(obj.weight.is_meta)
                                self.assertNotIn(full + ".weight", store.parameters)
                                x = torch.ones(1, obj.weight.shape[-1])
                                torch.testing.assert_close(
                                    obj(x),
                                    torch.nn.functional.linear(
                                        x, weights.values[full + ".weight"]
                                    ),
                                    rtol=0,
                                    atol=0,
                                )
                            else:
                                for key, param in obj.named_parameters(recurse=False):
                                    torch.testing.assert_close(
                                        param,
                                        weights.values[full + "." + key],
                                        rtol=0,
                                        atol=0,
                                    )
                name = "layers.0.attn.wo_b.weight"
                shape, kind, offset = store.names[name]
                store.names[name] = ((shape[0] + 1, *shape[1:]), kind, offset)
                with patch.object(store, "tensor") as tensor:
                    with self.assertRaisesRegex(ValueError, name):
                        bind_attention(m, args, store, 0)
                    tensor.assert_not_called()
                self.assertFalse(
                    any(n.startswith("layers.0.attn.") for n in store.parameters)
                )
            finally:
                store.close()

    def test_candidate_pruning_and_future(self):
        m = self.m
        scores = torch.tensor(
            [[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -torch.inf, -torch.inf]]]
        )
        mask = m.select_candidate_blocks(scores, 6, 2, 2)
        self.assertEqual(
            mask.flatten().tolist(),
            [True, True, False, False, True, True, False, False],
        )
        selected = scores.masked_fill(~mask, -torch.inf).topk(3).indices
        self.assertEqual(set(selected.flatten().tolist()), {0, 1, 4})
        bad = scores.clone()
        bad[..., 6] = 100
        self.assertFalse(torch.equal(mask, m.select_candidate_blocks(bad, 6, 2, 2)))

    def test_pinned_candidate_threshold(self):
        for length, discarded in [(10000, 0), (16384, 0), (16385, 8)]:
            scores = torch.arange(length).float().view(1, 1, -1)
            mask = self.m.select_candidate_blocks(scores, length, 2048, 8)
            self.assertEqual(int((~mask).sum()), discarded)

    def test_batched_history_matches_decode_both_phases(self):
        m = self.m
        args = m.ModelArgs(
            dim=16,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            o_groups=1,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=3,
            max_seq_len=32,
            max_batch_size=1,
            window_size=4,
            compress_ratios=[2],
            kv_source_layers=[0],
            index_source_layers=[0],
        )
        with m.set_dtype(torch.float32):
            a = m.Attention(0, args).float()
            b = m.Attention(0, args).float()
        for param in a.parameters():
            torch.nn.init.normal_(param, std=0.1)
        b.load_state_dict(a.state_dict())
        x = torch.randn(1, 20, 16)
        with torch.inference_mode():
            for p in (14, 15):
                m.shared_attn = m.SharedAttentionRuntime()
                for t in range(p):
                    qr = a.q_norm(a.wq_a(x[:, t : t + 1]))
                    a._compress_kv(x[:, t : t + 1], qr, t, 4)
                seed_history(m, b, lambda start, end: x[:, start:end], p, 6)
                count = p // 2
                torch.testing.assert_close(
                    a.compress_kv_cache[:, :count], b.compress_kv_cache[:, :count]
                )
                torch.testing.assert_close(
                    a.indexer.k_cache[:, :count], b.indexer.k_cache[:, :count]
                )
                qr = b.q_norm(b.wq_a(x[:, p : p + 1]))
                m.shared_attn.index_k = b.indexer.k_cache.clone() + 100
                kv, ids = b._compress_kv(x[:, p : p + 1], qr, p, 4)
                self.assertEqual(ids.numel(), 3)
                self.assertGreater(kv.size(1), ids.numel())
                self.assertIs(m.shared_attn.index_k, b.indexer.k_cache)
                stale = b.indexer.k_cache.clone() + 1
                self.assertFalse(torch.equal(stale, m.shared_attn.index_k))

    def test_metrics_reject_corruption(self):
        x = torch.ones(8)
        self.assertTrue(metrics(x, x, 0, 0)["pass"])
        self.assertFalse(metrics(x + 1, x, 0.01, 0.01)["pass"])
        self.assertFalse(metrics(x * float("nan"), x, 1, 1)["pass"])
        self.assertFalse(metrics(x[:0], x[:0], 1e-4, 0)["pass"])
        self.assertFalse(metrics(x[:1], x, 1e-4, 0)["pass"])
        report = metrics(torch.full((8,), 1e-4), torch.zeros(8), 1e-4, 0)
        self.assertTrue(report["pass"])
        self.assertEqual(report["elements"], 8)
        self.assertEqual((report["atol"], report["rtol"]), (1e-4, 0))
        self.assertFalse(metrics(x + 0.000101, x, 1e-4, 0)["pass"])

    @torch.inference_mode()
    def test_full_attention_seeded_replay_both_phases(self):
        m = self.m
        args = tiny_attention_args(m)
        store = SyntheticWeights(m, args)
        configure(m, store)
        inputs = [torch.randn(1, 20, 16) + layer for layer in range(3)]

        def snapshot(attn, output):
            values = {
                "output": output.clone(),
                "window": attn.window_kv_cache.clone(),
                "keys": m.shared_attn.index_k.clone(),
                "compressed": m.shared_attn.compress_kv.clone(),
                "ids": m.shared_attn.topk_idxs.clone(),
            }
            if attn.is_kv_source:
                self.assertIs(m.shared_attn.index_k, attn.indexer.k_cache)
                self.assertIs(m.shared_attn.compress_kv, attn.compress_kv_cache)
            return values

        for position in (12, 13):
            with self.subTest(position=position):
                sequential = [bind_attention(m, args, store, l) for l in range(3)]
                m.shared_attn = m.SharedAttentionRuntime()
                expected = []
                for t in range(position + 1):
                    for layer, attn in enumerate(sequential):
                        output = attn(inputs[layer][:, t : t + 1], t)
                        if t == position:
                            expected.append(snapshot(attn, output))

                seeded = [bind_attention(m, args, store, l) for l in range(3)]
                m.shared_attn = m.SharedAttentionRuntime()
                for layer, attn in enumerate(seeded):
                    seed_history(
                        m, attn, lambda a, b: inputs[layer][:, a:b], position, 6
                    )
                    if attn.is_kv_source:
                        m.shared_attn.index_k = torch.full_like(
                            attn.indexer.k_cache, float("nan")
                        )
                    output = attn(inputs[layer][:, position : position + 1], position)
                    actual = snapshot(attn, output)
                    for name in expected[layer]:
                        torch.testing.assert_close(actual[name], expected[layer][name])
                    if layer == 2:
                        self.assertIs(m.shared_attn.index_k, seeded[1].indexer.k_cache)

                # Suppress owner publication to ensure stale keys are observable.
                broken = bind_attention(m, args, store, 1)
                seed_history(m, broken, lambda a, b: inputs[1][:, a:b], position, 6)
                m.shared_attn = m.SharedAttentionRuntime()
                m.shared_attn.index_k = torch.zeros_like(broken.indexer.k_cache)
                broken.indexer.owns_k = False
                output = broken(inputs[1][:, position : position + 1], position)
                self.assertFalse(
                    torch.equal(m.shared_attn.index_k, expected[1]["keys"])
                )
                self.assertFalse(
                    torch.equal(m.shared_attn.topk_idxs, expected[1]["ids"])
                )
                self.assertFalse(torch.allclose(output, expected[1]["output"]))


if __name__ == "__main__":
    unittest.main()
