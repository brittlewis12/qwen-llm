# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "safetensors>=0.6", "numpy", "sympy"]
# ///
"""Freeze old-source NativeF32 operator outputs; run with uv run this_file.py.

Uses hash-verified old_model.py through cpu/oracle.py, not a reimplementation
of attention or Block.forward. Replaces MoE routing with one real old Expert
to isolate attention/mHC assembly. All projection payloads are already F32,
as after native weight decoding. No checkpoint-byte/dtype qualification claim.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "cpu"))
from oracle import MODEL_SHA256, source_module

OUT = HERE.parents[2] / "crates/qwen-llm/src/deepseek_v41/operators/old_native_f32.json"
OWNERS = (2, 8, 14, 20)
PRODUCERS = (2, 8, 14, 20, 24, 28, 32, 36)
STEPS = 7


def flat(x):
    return x.detach().flatten().tolist()


def build(uncorrected=False):
    m = source_module()
    if uncorrected:
        text = (HERE / "cpu/old_model.py").read_text()
        cls = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.ClassDef) and n.name == "Indexer"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        code = ast.Module(body=[fn], type_ignores=[])
        scope = m.__dict__
        exec(compile(code, "uncorrected_old_indexer", "exec"), scope)
        m.Indexer.forward = scope["forward"]
    m.act_quant = lambda x, *a, **kw: x
    m.fp4_act_quant = lambda x, *a, **kw: x
    m.cpu_grouped = lambda a, x: torch.einsum(
        "bsgd,grd->bsgr", x, a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
    )
    args = m.ModelArgs(
        dim=8,
        n_layers=40,
        n_heads=2,
        q_lora_rank=6,
        head_dim=4,
        rope_head_dim=4,
        o_groups=2,
        o_lora_rank=3,
        norm_eps=1e-6,
        compress_ratios=tuple(0 if i < 2 else 2 if i < 20 else 1 for i in range(40)),
        kv_source_layers=OWNERS,
        index_source_layers=PRODUCERS,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        window_size=3,
        candidate_source_layer=20,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        max_seq_len=16,
        max_batch_size=1,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        original_seq_len=16,
        rope_factor=4.0,
        beta_fast=4,
        beta_slow=1,
    )
    blocks = []
    with m.set_dtype(torch.float32):
        for layer in range(40):
            b = m.Block.__new__(m.Block)
            torch.nn.Module.__init__(b)
            b.layer_id, b.norm_eps = layer, args.norm_eps
            b.hc_mult, b.hc_sinkhorn_iters, b.hc_eps = 4, 20, 1e-6
            b.attn = m.Attention(layer, args)
            b.attn_norm = m.RMSNorm(8, args.norm_eps)
            b.ffn_norm = m.RMSNorm(8, args.norm_eps)
            expert = m.Expert(8, 10, dtype=torch.float32, swiglu_limit=0.35)

            class SingleExpert(torch.nn.Module):
                def __init__(self, expert):
                    super().__init__()
                    self.expert = expert

                def forward(self, x, image_mask=None):
                    return self.expert(x)

            b.ffn = SingleExpert(expert)
            for kind in ("attn", "ffn"):
                for suffix, shape in (
                    ("fn", (24, 32)),
                    ("base", (24,)),
                    ("scale", (3,)),
                ):
                    setattr(
                        b, f"hc_{kind}_{suffix}", torch.nn.Parameter(torch.empty(shape))
                    )
            b.float()
            for name, p in b.named_parameters():
                seed = int.from_bytes(
                    hashlib.sha256(f"{layer}.{name}".encode()).digest()[:4], "little"
                )
                gen = torch.Generator().manual_seed(seed)
                data = torch.randn(p.shape, generator=gen) * 0.25
                if "norm.weight" in name:
                    data = 1.0 + data
                if name.endswith("_scale"):
                    data = torch.tensor([0.6, -0.4, 0.8])
                p.data.copy_(data)
            # Opposite index heads prevent all-ReLU-zero top-k ties. Both heads
            # retain the same learned query weight, so this does not change math.
            if b.attn.indexer is not None:
                iw = b.attn.indexer
                iw.wq_b.weight.data[4:] = -iw.wq_b.weight.data[:4]
                iw.weights_proj.weight.data[1] = iw.weights_proj.weight.data[0]
            blocks.append(b)
    return m, blocks


@torch.inference_mode()
def generate(uncorrected=False):
    m, blocks = build(uncorrected)
    fixture = {}
    expert = m.Expert(6, 6, dtype=torch.float32, swiglu_limit=0.35)
    for projection in (expert.w1, expert.w2, expert.w3):
        projection.weight.data.copy_(torch.eye(6))
    clamp_input = torch.tensor([[-4.0, -1.0, -0.1, 0.1, 1.0, 4.0]])
    fixture["clamp.input"] = flat(clamp_input)
    fixture["clamp.output"] = flat(expert(clamp_input))
    for layer, b in enumerate(blocks):
        for name, p in b.named_parameters():
            fixture[f"w.{layer}.{name}"] = flat(p)
        # Extract frequency (position 1) from the source's actual complex table.
        fixture[f"w.{layer}.frequencies"] = flat(torch.angle(b.attn.freqs_cis[1]))
    for pos in range(STEPS):
        gen = torch.Generator().manual_seed(900 + pos)
        x = torch.randn(1, 1, 4, 8, generator=gen) * 0.7
        pre = m.make_identity_pre_mix(x, 4)
        fixture[f"p.{pos}.initial"] = flat(x)
        for layer, b in enumerate(blocks):
            prefix = f"p.{pos}.l.{layer}."

            def capture_attn(obj, inputs, output):
                fixture[prefix + "attn_input"] = flat(inputs[0])
                fixture[prefix + "attention"] = flat(output)

            hook = b.attn.register_forward_hook(capture_attn)
            am = b.hc_mixes(x, b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base)
            for name, value in zip(("pre", "post", "comb"), am):
                fixture[prefix + "attn_" + name] = flat(value)
            x, pre = b(x, pos, pre, None)
            hook.remove()
            fixture[prefix + "hidden"] = flat(x)
            fixture[prefix + "pre"] = flat(pre)
            a = b.attn
            oldest = max(0, pos + 1 - a.window_size)
            ids = [p % a.window_size for p in range(oldest, pos + 1)]
            fixture[prefix + "window"] = flat(a.window_kv_cache[:, ids])
            if layer in OWNERS:
                count = (pos + 1) // a.compress_ratio
                fixture[prefix + "compressed"] = flat(a.compress_kv_cache[:, :count])
                fixture[prefix + "keys"] = flat(a.indexer.k_cache[:, :count])
                pending = (pos + 1) % a.compress_ratio
                fixture[prefix + "pending_values"] = (
                    flat(a.compressor.kv_state[:, :pending]) if pending else []
                )
                fixture[prefix + "pending_scores"] = (
                    flat(a.compressor.score_state[:, :pending]) if pending else []
                )
            if layer >= 2:
                offset = 1 if pos == 0 else a.window_size
                fixture[prefix + "selection"] = flat(m.shared_attn.topk_idxs - offset)
            if layer >= 20:
                fixture[prefix + "candidates"] = [
                    int(v) for v in flat(m.shared_attn.candidates)
                ]
        fixture[f"p.{pos}.head_input"] = flat(blocks[-1].hc_pre(x, pre))
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    correct = generate()
    stale = generate(uncorrected=True)
    differences = {
        k: max((abs(a - b) for a, b in zip(v, stale[k])), default=0.0)
        for k, v in correct.items()
        if k.endswith("attention")
    }
    delta = max(differences.values())
    assert delta > 1e-5, "negative control must distinguish stale owner keys"
    first = next(k for k, v in differences.items() if v > 1e-5)
    prefix = first.removesuffix("attention")
    assert correct[prefix + "attn_input"] == stale[prefix + "attn_input"]
    assert correct[prefix + "selection"] != stale[prefix + "selection"]
    correct["negative.stale_attention"] = stale[first]
    correct["negative.correct_attention"] = correct[first]
    correct["negative.stale_selection"] = stale[prefix + "selection"]
    correct["negative.correct_selection"] = correct[prefix + "selection"]
    metadata = {
        "source_sha256": MODEL_SHA256,
        "cpu_kernels_sha256": hashlib.sha256(
            (HERE / "cpu/kernels.py").read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
        "arithmetic": "OldReapNativeF32",
        "correction": "publish owner index keys even with no completed pair",
        "positions": STEPS,
        "uncorrected_max_attention_error": delta,
        "uncorrected_first_divergence": first,
    }
    # One numeric array per line keeps the Rust fixture reader dependency-free.
    lines = [
        json.dumps(k) + ": " + json.dumps(v, allow_nan=False)
        for k, v in (metadata | correct).items()
    ]
    args.output.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
