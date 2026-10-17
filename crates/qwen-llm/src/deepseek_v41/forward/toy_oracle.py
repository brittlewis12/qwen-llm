# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "safetensors>=0.6", "numpy", "sympy"]
# ///
"""Regenerate toy_expected.json using the hash-verified old Python graph.

All hidden channels are equal by construction, so each trace stores four
stream values instead of 128 duplicates. This retains the real forty-layer
attention, six-of-eight MoE, Engram injection, mHC and head execution.
Only table lookup is virtual and activation quantization is disabled.
"""

import dataclasses
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts/reference/deepseek_v41/cpu"))
from oracle import MODEL_SHA256, source_module


def main():
    torch.set_num_threads(1)
    m = source_module()
    m.default_dtype = torch.float32
    m.act_quant = lambda x, *a, **kw: x
    m.fp4_act_quant = lambda x, *a, **kw: x
    m.cpu_grouped = lambda a, x: torch.einsum(
        "bsgd,grd->bsgr", x, a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    )
    args = m.ModelArgs(
        dim=32,
        moe_inter_dim=32,
        vocab_size=8,
        n_layers=40,
        n_mtp_layers=0,
        n_heads=2,
        head_dim=4,
        rope_head_dim=4,
        q_lora_rank=6,
        o_groups=2,
        o_lora_rank=3,
        norm_eps=1e-6,
        expert_dtype=None,
        n_routed_experts=8,
        n_activated_experts=6,
        route_scale=1.5,
        swiglu_limit=0.35,
        max_seq_len=8,
        max_batch_size=1,
        window_size=3,
        compress_ratios=tuple(0 if i < 2 else 2 if i < 20 else 1 for i in range(40)),
        kv_source_layers=(2, 8, 14, 20),
        index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        candidate_source_layer=20,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        original_seq_len=16,
        compress_rope_theta=160000,
        rope_factor=4,
        beta_fast=4,
        engram_layer_ids=(1, 14),
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=99092,
        engram_num_embeddings=(384006168, 384016682),
    )
    layout = m.EngramLayout.from_args(args)
    token_map = [5, 3, 2, 7, 0, 1, 6, 4]
    m.NgramHashState.__init__.__globals__["build_compressed_token_map"] = lambda _: (
        token_map,
        99092,
    )
    hashes = m.NgramHashState(args, layout, None)
    small_layout = dataclasses.replace(layout, num_embeddings=(1, 1))

    class Table(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, ids):
            return (
                (((ids + self.layer) % 17 - 8).float() / 128)
                .unsqueeze(-1)
                .expand(*ids.shape, 256)
            )

    blocks = []
    with torch.no_grad(), m.set_dtype(torch.float32):
        for layer in range(40):
            b = m.Block(layer, args).float()
            if layer in (1, 14):
                b.engram = m.Engram(args, layer, small_layout).float()
                b.engram.embed = Table(layer)
            for name, p in b.named_parameters():
                p.fill_(1 / 128 * (layer % 3 + 1))
                if "norm" in name or name.endswith(("q_weight", "k_weight")):
                    p.fill_(1)
                elif name.endswith("attn_sink"):
                    p.zero_()
                elif name.endswith("_fn"):
                    p.fill_(1 / 1024)
                elif name.endswith("_base"):
                    p.copy_((torch.arange(24) % 7 - 3) / 16)
                elif name.endswith("_scale"):
                    p.copy_(torch.tensor([0.25, 0.5, 0.75]))
                elif name == "ffn.gate.weight":
                    p.copy_(((torch.arange(8) - 3) / 64).unsqueeze(1).expand(8, 32))
                elif name == "ffn.gate.bias":
                    p.copy_(torch.arange(8) / 32)
                elif name.startswith("ffn.experts."):
                    expert = int(name.split(".")[2])
                    part = name.split(".")[3]
                    code = (expert + {"w1": 1, "w3": 3, "w2": 2}[part]) % 8
                    p.fill_([0, 0.5, 1, 1.5, 2, 3, 4, 6][code] / 32)
                elif name.startswith("ffn.shared_experts."):
                    p.fill_(1 / 64)
                elif name == "engram.wkv.weight":
                    p.fill_(1 / 8192)
            blocks.append(b)
        norm = m.RMSNorm(32, args.norm_eps)
        head = m.ParallelHead(8, 32)
        head.weight.copy_(((torch.arange(8) - 3) / 32).unsqueeze(1).expand(8, 32))
        output = {
            "source_sha256": MODEL_SHA256,
            "tokens": [0, 3, 1],
            "traces": [],
            "logits": [],
        }
        for pos, token in enumerate(output["tokens"]):
            addresses = hashes(torch.tensor([[token]]), pos)
            x = torch.full((1, 1, 4, 32), (token - 2) / 8)
            pre = m.make_identity_pre_mix(x, 4)
            for layer, b in enumerate(blocks):
                if b.engram is not None:
                    x = b.engram(x, addresses[:, :, 0 if layer == 1 else 1, :])
                x, pre = b(x, pos, pre, None)
                assert torch.equal(x, x[:, :, :, :1].expand_as(x))
                output["traces"].append(
                    [*x[0, 0, :, 0].tolist(), *pre.flatten().tolist()]
                )
            output["logits"].append(head(norm(b.hc_pre(x, pre))).flatten().tolist())
    Path(__file__).with_name("toy_expected.json").write_text(
        json.dumps(output, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
