# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "safetensors>=0.6", "numpy", "sympy", "tokenizers"]
# ///
"""Text-only bounded CPU runner for the old REAP-272E checkpoint."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import types

import torch

import kernels
from bindings import validate_bindings
from weights import Weights

HERE = Path(__file__).resolve().parent
MODEL_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
ENGRAM_SHA256 = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"


def source_module():
    # Each execution has its own globals, caches, classes and shared_attn runtime.
    engram = types.ModuleType("cpu_engram")
    sys.modules[engram.__name__] = engram
    engram_source = (HERE.parent / "fixtures/inference/engram.py").read_text()
    if hashlib.sha256(engram_source.encode()).hexdigest() != ENGRAM_SHA256:
        raise ValueError("Pinned Engram source hash mismatch")
    exec(
        compile(
            engram_source,
            "pinned_engram.py",
            "exec",
        ),
        engram.__dict__,
    )
    text = (HERE / "old_model.py").read_text()
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    if source_hash != MODEL_SHA256:
        raise ValueError("Pinned old REAP model source hash mismatch")
    text = text.replace("from engram import EngramLayout, NgramHashState", "")
    text = text.replace(
        "from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START", ""
    )
    text = text.replace("from vision import Aligner, ViT", "")
    begin = text.index("from kernel import (")
    end = text.index("\n)", begin) + 2
    text = text[:begin] + text[end:]
    old = "            shared_attn.index_k = self.k_cache"
    assert text.count(old) == 1
    text = text.replace(old, "        if self.owns_k:\n" + old)
    old = '        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)\n        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)'
    assert text.count(old) == 2
    text = text.replace(old, "        o = cpu_grouped(self, o)")
    module = types.ModuleType("cpu_model_" + str(id(engram)))
    sys.modules[module.__name__] = module
    module.__dict__.update(
        {
            name: getattr(kernels, name)
            for name in (
                "act_quant",
                "fp4_act_quant",
                "fp4_gemm",
                "fp8_gemm",
                "sparse_attn",
                "hc_split_sinkhorn",
            )
        }
    )
    module.EngramLayout, module.NgramHashState = (
        engram.EngramLayout,
        engram.NgramHashState,
    )
    exec(compile(text, str(HERE / "old_model.py"), "exec"), module.__dict__)
    module.source_hash = source_hash
    sys.modules.pop(module.__name__)
    sys.modules.pop(engram.__name__)
    return module


def model_args(module, config, max_seq_len):
    c = config.get("text_config", config)
    aliases = dict(
        hidden_size="dim",
        moe_intermediate_size="moe_inter_dim",
        num_hidden_layers="n_layers",
        num_attention_heads="n_heads",
        qk_rope_head_dim="rope_head_dim",
        rms_norm_eps="norm_eps",
        num_experts_per_tok="n_activated_experts",
        scoring_func="score_func",
        routed_scaling_factor="route_scale",
        sliding_window="window_size",
        kv_source_layer_ids="kv_source_layers",
        index_source_layer_ids="index_source_layers",
        candidate_source_layer_id="candidate_source_layer",
        engram_pad_token_id="engram_pad_id",
    )
    fields = {f.name for f in dataclasses.fields(module.ModelArgs)}
    values = {aliases.get(k, k): v for k, v in c.items() if aliases.get(k, k) in fields}
    rope = c.get("rope_scaling", {})
    for src, dst in [
        ("factor", "rope_factor"),
        ("original_max_position_embeddings", "original_seq_len"),
        ("beta_fast", "beta_fast"),
        ("beta_slow", "beta_slow"),
    ]:
        if src in rope:
            values[dst] = rope[src]
    values.update(
        max_batch_size=1,
        max_seq_len=max_seq_len,
        temperature=0,
        vision_n_layers=0,
        dspark_block_size=0,
        dtype="fp8",
    )
    return module.ModelArgs(**values)


def load_tokenizer(model_dir):
    from tokenizers import Tokenizer

    class TokenizerAdapter:
        backend_tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size()

    return TokenizerAdapter()


class Oracle:
    def __init__(
        self,
        model_dir,
        dump_dir,
        mode="NativeF32",
        projection_rows=128,
        max_seq_len=32,
        engram_dir=None,
    ):
        if mode not in ("NativeF32", "ReferenceFp8") or max_seq_len < 1:
            raise ValueError("invalid arithmetic mode or sequence bound")
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.mode, self.position = mode, 0
        m = self.module = source_module()
        store = self.store = Weights(
            [model_dir] + ([engram_dir] if engram_dir else []), projection_rows, mode
        )
        args = model_args(
            m, json.loads((Path(model_dir) / "config.json").read_text()), max_seq_len
        )
        for cls in (
            m.Linear,
            m.ParallelEmbedding,
            m.ParallelEngramEmbedding,
            m.ParallelHead,
        ):
            original = cls.__init__

            def init(obj, *a, _init=original, **kw):
                with torch.device("meta"):
                    _init(obj, *a, **kw)

            cls.__init__ = init

        def project(obj, x):
            return store.project(
                obj.cpu_name + ".weight",
                x,
                output_dtype=obj.weight.dtype
                if obj.weight.dtype in (torch.float32, torch.bfloat16)
                else torch.bfloat16,
            )

        for cls in (m.Linear, m.ColumnParallelLinear, m.RowParallelLinear):
            cls.forward = project
        m.ParallelEmbedding.forward = lambda obj, ids: store.embedding(
            obj.cpu_name + ".weight", ids
        )
        m.ParallelEngramEmbedding.forward = lambda obj, ids: store.embedding(
            obj.cpu_name + ".weight", ids, True
        )
        m.ParallelHead.forward = lambda obj, x, full_logits=False: store.project(
            obj.cpu_name + ".weight",
            x if full_logits else x[:, -1],
            quantize=False,
            output_dtype=torch.float32,
        )

        def grouped(attn, x):
            return torch.stack(
                [
                    store.project(
                        attn.wo_a.cpu_name + ".weight",
                        x[..., g, :],
                        g * attn.o_lora_rank,
                        (g + 1) * attn.o_lora_rank,
                        quantize=False,
                        bf16_weight=True,
                    )
                    for g in range(attn.n_local_groups)
                ],
                -2,
            )

        m.cpu_grouped = grouped
        if mode == "NativeF32":
            m.act_quant = lambda x, *a, **kw: x
            m.fp4_act_quant = lambda x, *a, **kw: x
        tokenizer = None
        if args.engram_layer_ids:
            tokenizer = load_tokenizer(model_dir)
        dtype = torch.float32 if mode == "NativeF32" else torch.bfloat16
        with m.set_dtype(dtype):
            self.model = m.Transformer(args, tokenizer)
        validate_bindings(self.model, m, store.descriptors())
        lazy = (
            m.Linear,
            m.ParallelEmbedding,
            m.ParallelEngramEmbedding,
            m.ParallelHead,
        )
        for name, obj in self.model.named_modules():
            obj.cpu_name = name
            if isinstance(obj, lazy):
                continue
            for key, param in list(obj.named_parameters(recurse=False)):
                full = name + "." + key if name else key
                value = store.tensor(full)[:].to(
                    torch.float32 if mode == "NativeF32" else param.dtype
                )
                if value.shape != param.shape:
                    raise ValueError((full, value.shape, param.shape))
                setattr(obj, key, torch.nn.Parameter(value, requires_grad=False))
        for i, layer in enumerate(self.model.layers):

            def dump(obj, inputs, output, layer_id=i):
                torch.save(
                    {"hidden": output[0], "pre_mix": output[1]},
                    self.dump_dir / f"pos-{self.position:05d}-layer-{layer_id:02d}.pt",
                )

            layer.register_forward_hook(dump)
        (self.dump_dir / "run.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "source_sha256": m.source_hash,
                    "engram_source_sha256": ENGRAM_SHA256,
                    "owner_correction": "publish index_k whenever owns_k, including incomplete pairs",
                    "projection_rows": projection_rows,
                    "max_seq_len": max_seq_len,
                    "expected_reap_revision": "f792019bcb369768220f01088f0f7af230786982",
                    "expected_engram_base_revision": "314113f8a7dadb388215396ba5fbee2c610a04d6",
                    "checkpoint_revision_verified": False,
                },
                indent=2,
            )
        )

    @torch.inference_mode()
    def feed(self, token_ids):
        if not token_ids or self.position + len(token_ids) > self.model.max_seq_len:
            raise ValueError("empty input or max_seq_len exceeded")
        # Upstream continuation supports one token, not multi-token chunks.
        for token in token_ids:
            _, logits, _ = self.model(torch.tensor([[token]]), self.position)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits at {self.position}")
            torch.save(logits, self.dump_dir / f"pos-{self.position:05d}-logits.pt")
            self.position += 1
        return logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--engram-dir",
        help="original base directory containing Engram shards 47/48 and index",
    )
    p.add_argument("--token-ids", required=True, help="comma-separated integer IDs")
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--mode", choices=["NativeF32", "ReferenceFp8"], default="NativeF32")
    p.add_argument("--projection-rows", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=32)
    p.add_argument("--generate", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    ids = [int(x) for x in a.token_ids.split(",")]
    if a.generate < 1 or len(ids) + a.generate - 1 > a.max_seq_len:
        p.error("generation must fit max-seq-len and generate must be positive")
    torch.set_num_threads(a.threads)
    oracle = Oracle(
        a.model_dir, a.dump_dir, a.mode, a.projection_rows, a.max_seq_len, a.engram_dir
    )
    logits = oracle.feed(ids)
    generated = []
    for i in range(a.generate):
        token = logits.argmax(-1).item()
        generated.append(token)
        print(token, flush=True)
        if i + 1 < a.generate:
            logits = oracle.feed([token])
    (Path(a.dump_dir) / "tokens.json").write_text(
        json.dumps({"prompt": ids, "generated": generated})
    )


if __name__ == "__main__":
    main()
