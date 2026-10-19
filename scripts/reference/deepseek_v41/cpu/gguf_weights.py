"""Independent little-endian GGUF reader for the offline NativeF32 witness.

Only source bytes are mapped. Decoding is row-bounded and never cached.
"""

import math
import mmap
import re
import struct

import numpy as np
import torch

from weights import Weights


TYPES = {0: (1, 4), 8: (32, 34), 30: (1, 2), 39: (32, 17)}


def decode_bytes(data, kind, shape):
    count = math.prod(shape)
    block, size = TYPES[kind]
    if count % block or len(data) != count // block * size:
        raise ValueError("invalid decoded byte extent")
    if kind == 0:
        out = np.frombuffer(data, dtype="<f4").copy()
    elif kind == 30:
        out = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(
            np.float32
        )
    elif kind == 8:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 34)
        scale = raw[:, :2].copy().view("<f2").astype(np.float32)
        out = raw[:, 2:].view(np.int8).astype(np.float32) * scale
    else:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 17)
        codes = np.concatenate((raw[:, 1:] & 15, raw[:, 1:] >> 4), axis=1)
        levels = np.array(
            [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
            dtype=np.float32,
        )
        # GGML half-scales and doubled E2M1 levels preserve subnormals at
        # exponents 0/1 and the finite half-scale at exponent 255.
        exponents = raw[:, :1].astype(np.int32)
        scale = np.exp2(exponents.astype(np.float64) - 128).astype(np.float32)
        out = levels[codes] * scale
    return torch.from_numpy(out.reshape(shape))


ALIASES = {
    "token_embd.weight": "embed.weight",
    "output.weight": "head.weight",
    "output_norm.weight": "norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "engram_embd.weight": "engram.embed.weight",
    "engram_q_norm.weight": "engram.q_weight",
    "engram_q.weight": "engram.q_weight",
    "engram_k_norm.weight": "engram.k_weight",
    "engram_k.weight": "engram.k_weight",
    "engram_wkv.weight": "engram.wkv.weight",
    "engram_kv.weight": "engram.wkv.weight",
    **{
        f"{name}.weight": name
        for name in (
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
        )
    },
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
}


class GGUFWeights(Weights):
    def __init__(self, path, rows=128, mode="NativeF32"):
        if mode != "NativeF32":
            raise ValueError("GGUF witness supports NativeF32 only")
        if rows < 32 or rows % 32:
            raise ValueError("projection rows must be a positive multiple of 32")
        self.rows, self.mode = rows, mode
        with open(path, "rb") as source:
            self.data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        self.cursor = 0
        try:
            self._parse()
        except Exception:
            self.data.close()
            raise

    def close(self):
        self.data.close()

    def _take(self, size):
        if size < 0 or size > len(self.data) - self.cursor:
            raise ValueError("GGUF header outside file")
        start = self.cursor
        self.cursor += size
        return self.data[start : self.cursor]

    def _unpack(self, fmt):
        return struct.unpack("<" + fmt, self._take(struct.calcsize("<" + fmt)))[0]

    def _string(self):
        return self._take(self._unpack("Q")).decode("utf-8")

    def _value(self, kind, depth=0):
        formats = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }
        if kind in formats:
            return self._unpack(formats[kind])
        if kind == 8:
            return self._string()
        if kind == 9 and depth == 0:
            element, count = self._unpack("I"), self._unpack("Q")
            if count > len(self.data) - self.cursor:
                raise ValueError("GGUF array outside file")
            return [self._value(element, 1) for _ in range(count)]
        raise ValueError(f"unsupported GGUF metadata type {kind}")

    def _parse(self):
        if self._take(4) != b"GGUF" or self._unpack("I") != 3:
            raise ValueError("expected little-endian GGUF v3")
        count, kv_count = self._unpack("Q"), self._unpack("Q")
        self.metadata = {}
        for _ in range(kv_count):
            key = self._string()
            if key in self.metadata:
                raise ValueError("duplicate metadata: " + key)
            self.metadata[key] = self._value(self._unpack("I"))
        tensors = {}
        for _ in range(count):
            name, ndim = self._string(), self._unpack("I")
            if not 1 <= ndim <= 4 or name in tensors:
                raise ValueError("invalid tensor descriptor: " + name)
            shape = tuple(reversed([self._unpack("Q") for _ in range(ndim)]))
            kind, offset = self._unpack("I"), self._unpack("Q")
            if kind not in TYPES or not all(shape):
                raise ValueError(f"unsupported tensor {name}: {shape}, type {kind}")
            block, size = TYPES[kind]
            if shape[-1] % block:
                raise ValueError("non-integral quantized row: " + name)
            tensors[name] = (shape, kind, offset, math.prod(shape) // block * size)
        alignment = self.metadata.get("general.alignment", 32)
        if (
            not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise ValueError("invalid GGUF alignment")
        base = (self.cursor + alignment - 1) // alignment * alignment
        self.names = {}
        extents = []
        for name, (shape, kind, offset, size) in tensors.items():
            if offset % alignment or base + offset + size > len(self.data):
                raise ValueError("tensor extent outside file or unaligned: " + name)
            extents.append((offset, offset + size))
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
            prefix, tail = (f"layers.{match[1]}.", match[2]) if match else ("", name)
            bank = re.fullmatch(r"ffn_(gate|up|down)_exps.weight", tail)
            if bank:
                if len(shape) != 3:
                    raise ValueError("expert bank must have three dimensions")
                experts, *matrix = shape
                part = {"gate": "w1", "up": "w3", "down": "w2"}[bank[1]]
                for expert in range(experts):
                    self._add(
                        prefix + f"ffn.experts.{expert}.{part}.weight",
                        tuple(matrix),
                        kind,
                        base + offset + expert * size // experts,
                    )
            elif tail in ALIASES:
                self._add(prefix + ALIASES[tail], shape, kind, base + offset)
            elif prefix:
                # Retain unknown layer names so binding cannot silently drop backbone weights.
                self._add(prefix + tail, shape, kind, base + offset)
        extents.sort()
        if any(a[1] > b[0] for a, b in zip(extents, extents[1:])):
            raise ValueError("overlapping GGUF tensors")

    def _add(self, name, shape, kind, offset):
        if name in self.names:
            raise ValueError("duplicate canonical tensor: " + name)
        self.names[name] = (shape, kind, offset)

    def shape(self, name):
        return self.names[name][0]

    def validate_config(self, config):
        c = config.get("text_config", config)
        if self.metadata.get("general.architecture") != "deepseek41":
            raise ValueError("GGUF architecture must be deepseek41")
        aliases = {
            "block_count": "num_hidden_layers",
            "embedding_length": "hidden_size",
            "expert_feed_forward_length": "moe_intermediate_size",
            "expert_count": "n_routed_experts",
            "expert_used_count": "num_experts_per_tok",
            "expert_shared_count": "n_shared_experts",
            "expert_weights_scale": "routed_scaling_factor",
            "expert_weights_norm": "norm_topk_prob",
            "attention.head_count": "num_attention_heads",
            "attention.head_count_kv": "num_key_value_heads",
            "attention.key_length": "head_dim",
            "attention.value_length": "head_dim",
            "attention.q_lora_rank": "q_lora_rank",
            "rope.dimension_count": "qk_rope_head_dim",
            "attention.output_lora_rank": "o_lora_rank",
            "attention.o_lora_rank": "o_lora_rank",
            "attention.output_group_count": "o_groups",
            "attention.o_groups": "o_groups",
            "attention.indexer.head_count": "index_n_heads",
            "attention.indexer.key_length": "index_head_dim",
            "attention.indexer.top_k": "index_topk",
            "attention.kv_source_layers": "kv_source_layer_ids",
            "attention.indexer.source_layers": "index_source_layer_ids",
            "attention.indexer.candidate_source_layer": "candidate_source_layer_id",
            "attention.indexer.candidate_block_size": "candidate_block_size",
            "attention.indexer.candidate_top_k_blocks": "candidate_topk_blocks",
            "attention.sliding_window": "sliding_window",
            "attention.compress_ratios": "compress_ratios",
            "attention.compress_rope_freq_base": "compress_rope_theta",
            "attention.layer_norm_rms_epsilon": "rms_norm_eps",
            "hyper_connection.count": "hc_mult",
            "hyper_connection.epsilon": "hc_eps",
            "hyper_connection.sinkhorn_iterations": "hc_sinkhorn_iters",
            "rope.freq_base": "rope_theta",
            "context_length": "max_position_embeddings",
            "engram.layer_ids": "engram_layer_ids",
            "engram.layers": "engram_layer_ids",
            "engram.head_count": "engram_n_heads",
            "engram.n_heads": "engram_n_heads",
            "engram.key_length": "engram_head_dim",
            "engram.head_dim": "engram_head_dim",
            "engram.max_ngram_size": "engram_max_ngram_size",
            "engram.compressed_vocab_size": "engram_compressed_vocab_size",
            "engram.pad_token_id": "engram_pad_token_id",
            "engram.num_embeddings": "engram_num_embeddings",
            "engram.rows": "engram_num_embeddings",
            "hidden_activation": "hidden_act",
            "rope.scaling.factor": "rope_scaling.factor",
            "rope.scaling.type": "rope_scaling.rope_type",
            "rope.scaling.original_context_length": "rope_scaling.original_max_position_embeddings",
            "rope.scaling.yarn_beta_fast": "rope_scaling.beta_fast",
            "rope.scaling.yarn_beta_slow": "rope_scaling.beta_slow",
        }
        expected = dict(c)
        expected.update(
            {"rope_scaling." + k: v for k, v in c.get("rope_scaling", {}).items()}
        )
        for key, value in self.metadata.items():
            if not key.startswith("deepseek41."):
                continue
            short = key.removeprefix("deepseek41.")
            target = aliases.get(short, short)
            if short in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
                want = [c["swiglu_limit"]] * c["num_hidden_layers"]
            elif short == "gate_temperature":
                # The pinned HF gate applies softplus directly, without temperature.
                want = 1.0
            elif short == "engram.encoding":
                kinds = {0: "f32", 8: "q8_0", 30: "bf16", 39: "mxfp4"}
                tables = [
                    desc
                    for name, desc in self.names.items()
                    if name.endswith(".engram.embed.weight")
                ]
                if (
                    not isinstance(value, str)
                    or not tables
                    or any(value.lower() != kinds[desc[1]] for desc in tables)
                ):
                    raise ValueError("GGUF Engram encoding differs from tensor storage")
                continue
            elif short == "expert_gating_func":
                want = 4 if c["scoring_func"] == "sqrtsoftplus" else None
            elif target in expected:
                want = expected[target]
            elif short in aliases or short in aliases.values():
                raise ValueError(
                    f"missing independent HF config field for {key}: {target}"
                )
            else:
                continue
            if isinstance(want, float):
                equal = isinstance(value, (float, int)) and math.isclose(
                    value, want, rel_tol=1e-6
                )
            else:
                equal = value == want
            if not equal:
                raise ValueError(
                    f"GGUF/HF metadata mismatch: {key}: {value!r} != {want!r}"
                )

    def decode(self, name, start, end, engram=False):
        shape, kind, offset = self.names[name]
        if not 0 <= start <= end <= shape[0] or end - start > self.rows:
            raise ValueError("row selection outside tensor or decode bound")
        block, size = TYPES[kind]
        stride = math.prod(shape[1:]) * size // block
        if len(shape) == 1 and block != 1:
            raise ValueError("quantized vectors are unsupported")
        return decode_bytes(
            self.data[offset + start * stride : offset + end * stride],
            kind,
            (end - start, *shape[1:]),
        )

    def tensor(self, name):
        # Only non-lazy model parameters use this path; embeddings cannot reach it.
        if name not in getattr(self, "parameters", set()):
            raise ValueError("tensor() is restricted to validated non-lazy parameters")
        return torch.cat(
            [
                self.decode(name, lo, min(lo + self.rows, self.shape(name)[0]))
                for lo in range(0, self.shape(name)[0], self.rows)
            ]
        )

    def _validate_parameters(self, model, lazy, prefix=""):
        used = set()
        parameters = set()
        for name, obj in model.named_modules():
            params = (
                [("weight", obj.weight)]
                if isinstance(obj, lazy)
                else obj.named_parameters(recurse=False)
            )
            for key, param in params:
                full = ".".join(part for part in (prefix, name, key) if part)
                if full not in self.names:
                    raise ValueError("missing GGUF tensor: " + full)
                expected = list(param.shape)
                if param.dtype == torch.float4_e2m1fn_x2:
                    expected[-1] *= 2
                if tuple(expected) != self.shape(full):
                    raise ValueError(
                        f"{full}: GGUF {self.shape(full)} != HF {expected}"
                    )
                used.add(full)
                if not isinstance(obj, lazy):
                    parameters.add(full)
        return used, parameters

    def validate_attention(self, attn, module, layer):
        """Validate one attention subtree without allocating a transformer."""
        prefix = f"layers.{layer}.attn"
        self.parameters = {
            name
            for name in getattr(self, "parameters", set())
            if not name.startswith(prefix + ".")
        }
        used, parameters = self._validate_parameters(attn, (module.Linear,), prefix)
        extra = [
            name
            for name in self.names
            if name.startswith(prefix + ".") and name not in used
        ]
        if extra:
            raise ValueError("unbound GGUF attention tensors: " + str(extra[:10]))
        self.parameters.update(parameters)

    def validate_model(self, model, module):
        lazy = (
            module.Linear,
            module.ParallelEmbedding,
            module.ParallelEngramEmbedding,
            module.ParallelHead,
        )
        self.parameters = set()
        used, parameters = self._validate_parameters(model, lazy)
        layers = len(model.layers)
        extra = [
            n
            for n in self.names
            if n.startswith("layers.")
            and int(n.split(".")[1]) < layers
            and n not in used
            and not n.endswith(".gate.bias_vl")
        ]
        if extra:
            raise ValueError("unbound GGUF backbone tensors: " + str(extra[:10]))
        if model.engram_hash is not None:
            for key in ("primes", "offsets", "multipliers", "token_map", "pad_id"):
                expected = getattr(model.engram_hash, key)
                if isinstance(expected, torch.Tensor):
                    expected = expected.flatten().tolist()
                if self.metadata.get("deepseek41.engram." + key) != expected:
                    raise ValueError("GGUF/HF Engram mismatch: " + key)
        self.parameters = parameters
