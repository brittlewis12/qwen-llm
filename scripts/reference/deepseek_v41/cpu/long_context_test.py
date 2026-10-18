import unittest

import torch

from long_context import metrics, seed_history
from oracle import source_module


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


if __name__ == "__main__":
    unittest.main()
