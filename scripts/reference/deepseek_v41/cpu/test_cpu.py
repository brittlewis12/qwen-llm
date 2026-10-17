import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

import kernels
from oracle import Oracle, source_module
from weights import Weights


def synthetic_checkpoint(root, ratio_one=False):
    torch.manual_seed(17)
    m = source_module()
    args = m.ModelArgs(
        vocab_size=64,
        dim=32,
        moe_inter_dim=32,
        n_layers=5,
        n_heads=2,
        n_routed_experts=2,
        n_activated_experts=1,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        compress_ratios=(0, 2, 2, 1, 1) if ratio_one else (0, 2, 2, 2, 2),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 2, 3, 4),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=4,
        hc_mult=2,
        window_size=4,
        max_seq_len=16,
        max_batch_size=1,
        candidate_source_layer=3 if ratio_one else -1,
        candidate_topk_blocks=1 if ratio_one else 0,
        candidate_block_size=2 if ratio_one else 0,
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(24, 36),
        engram_max_ngram_size=3,
        engram_vocab_size=11,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    from tokenizers import Tokenizer, models

    class TokenizerAdapter:
        backend_tokenizer = Tokenizer(
            models.WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0")
        )

        def __len__(self):
            return 64

    tokenizer = TokenizerAdapter()
    tokenizer.backend_tokenizer.save(str(root / "tokenizer.json"))
    with torch.device("meta"):
        model = m.Transformer(args, tokenizer)
    tensors = {}
    for name, param in model.named_parameters():
        if param.dtype == torch.float4_e2m1fn_x2:
            value = torch.randint(0, 256, param.shape, dtype=torch.uint8).view(
                torch.int8
            )
        elif name.endswith("wo_a.weight"):
            value = (torch.randn(param.shape) * 0.5).to(torch.float8_e4m3fn)
            tensors[name.removesuffix("weight") + "scale"] = torch.full(
                (param.shape[0] // 32, param.shape[1] // 32), 0.125
            ).to(torch.float8_e8m0fnu)
        elif name.endswith(".scale"):
            value = torch.full(param.shape, 0.03125).to(param.dtype)
        elif "norm.weight" in name:
            value = torch.ones(param.shape, dtype=param.dtype)
        else:
            value = (torch.randn(param.shape) * 0.05).to(param.dtype)
        tensors[name] = value
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(dataclasses.asdict(args)))


class KernelsTest(unittest.TestCase):
    def test_fp8_rounding(self):
        x = torch.tensor([448.0, 1.0625, 1.1875, -1.0625] + [0.0] * 28).reshape(1, 32)
        q, s = kernels.act_quant(x)
        self.assertEqual(s.float().item(), 1.0)
        torch.testing.assert_close(
            q.float()[0, :4], torch.tensor([448.0, 1.0, 1.25, -1.0])
        )
        zero, scale = kernels.act_quant(torch.zeros(1, 32))
        self.assertTrue(torch.isfinite(scale.float()).all())
        self.assertEqual(zero.float().sum(), 0)

    def test_fp4_layout_and_ties(self):
        packed = torch.tensor(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
        )
        torch.testing.assert_close(
            kernels.unpack_fp4(packed), torch.tensor([kernels.FP4])
        )
        x = torch.tensor([[6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4])
        q, s = kernels.fp4_act_quant(x)
        torch.testing.assert_close(
            kernels.unpack_fp4(q)[0, :8],
            torch.tensor([6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]),
        )
        self.assertEqual(s.float().item(), 1.0)

    def test_gemms(self):
        x = torch.randn(2, 32)
        a, s = kernels.act_quant(x)
        w = torch.randn(32, 32).to(torch.float8_e4m3fn)
        ws = torch.ones(1, 1).to(torch.float8_e8m0fnu)
        expected = (
            (a.float() * s.float().repeat_interleave(32, -1)) @ w.float().T
        ).bfloat16()
        torch.testing.assert_close(
            kernels.fp8_gemm(a, s, w, ws), expected, rtol=0, atol=0
        )
        w4, s4 = kernels.fp4_act_quant(torch.randn(32, 32))
        expected = (
            (a.float() * s.float().repeat_interleave(32, -1))
            @ (kernels.unpack_fp4(w4) * s4.float().repeat_interleave(32, -1)).T
        ).bfloat16()
        torch.testing.assert_close(
            kernels.fp4_gemm(a, s, w4, s4), expected, rtol=0, atol=0
        )

    def test_sink_and_sinkhorn(self):
        q = torch.ones(1, 1, 1, 2)
        kv = torch.tensor([[[2.0, 3.0]]])
        out = kernels.sparse_attn(
            q, kv, torch.tensor([5.0]), torch.tensor([[[0, -1]]]), 1.0
        )
        torch.testing.assert_close(out, kv.unsqueeze(2) / 2)
        out = kernels.sparse_attn(q, kv, torch.zeros(1), torch.tensor([[[-1]]]), 1.0)
        self.assertEqual(out.sum(), 0)
        pre, post, comb = kernels.hc_split_sinkhorn(
            torch.zeros(1, 1, 8), torch.ones(3), torch.zeros(8), 2
        )
        torch.testing.assert_close(pre, torch.full_like(pre, 0.500001))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-1), torch.ones(1, 1, 2), atol=2e-6, rtol=0)

    def test_full_model_continuation_and_instance_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root)
            for mode in ("NativeF32", "ReferenceFp8"):
                a = Oracle(root, root / (mode + "-a"), mode, 32, 16)
                owner_calls = []

                def check_owner(obj, inputs, output):
                    self.assertIs(a.module.shared_attn.index_k, obj.k_cache)
                    owner_calls.append(inputs[3])

                for layer in a.model.layers:
                    if layer.attn.indexer is not None and layer.attn.indexer.owns_k:
                        layer.attn.indexer.register_forward_hook(check_owner)
                expected = a.feed([1, 2, 3, 4, 5])
                self.assertIn(2, owner_calls)
                self.assertIn(3, owner_calls)
                b = Oracle(root, root / (mode + "-b"), mode, 64, 16)
                b.feed([1, 2])
                actual = b.feed([3, 4, 5])
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertIsNot(a.module.shared_attn, b.module.shared_attn)
                aa = a.feed([6])
                bb = b.feed([6])
                torch.testing.assert_close(aa, bb, atol=0, rtol=0)
                self.assertEqual(
                    len(list((root / (mode + "-a")).glob("*layer-*.pt"))), 30
                )
                token = aa.argmax(-1).item()
                self.assertTrue(torch.isfinite(a.feed([token])).all())

    def test_ratio_one_candidate_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_checkpoint(root, ratio_one=True)
            for mode in ("NativeF32", "ReferenceFp8"):
                a = Oracle(root, root / (mode + "-a"), mode, 32, 16)
                b = Oracle(root, root / (mode + "-b"), mode, 64, 16)
                expected = a.feed([1, 2, 3, 4, 5, 6, 7, 8, 9])
                b.feed([1, 2, 3, 4])
                actual = b.feed([5, 6, 7, 8, 9])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(a.module.shared_attn.candidates.shape[-1], 9)

    def test_engram_row_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = torch.arange(128).reshape(4, 32).float().to(torch.float8_e4m3fn)
            scale = torch.tensor([[1.0], [2.0], [4.0], [8.0]]).to(torch.float8_e8m0fnu)
            save_file(
                {"table.weight": value, "table.scale": scale}, root / "rows.safetensors"
            )
            store = Weights([root], 32)
            actual = store.embedding("table.weight", torch.tensor([[3, 1, 3]]), True)
            torch.testing.assert_close(
                actual, (value.float() * scale.float())[[3, 1, 3]].unsqueeze(0)
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--make-checkpoint":
        root = Path(sys.argv[2])
        root.mkdir(parents=True, exist_ok=False)
        synthetic_checkpoint(root)
    else:
        unittest.main()
