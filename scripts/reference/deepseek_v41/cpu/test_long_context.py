import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from long_context import (
    Capture,
    bind_attention,
    configure,
    metrics,
    replay,
    seed_history,
)
from oracle import source_module


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

    @torch.inference_mode()
    def test_full_attention_seeded_replay_both_phases(self):
        m = self.m
        args = m.ModelArgs(
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
