"""Mmap packed checkpoints; never construct a dense expert or embedding table."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from kernels import act_quant, unpack_fp4


def canonical(name):
    return (
        name.removeprefix("model.")
        .replace("self_attn", "attn")
        .replace("mlp", "ffn")
        .replace("weight_scale_inv", "scale")
        .replace("e_score_correction_bias", "bias")
    )


class Weights:
    def __init__(self, roots, rows=128, mode="NativeF32"):
        if rows < 32 or rows % 32:
            raise ValueError("projection rows must be a positive multiple of 32")
        self.rows, self.mode = rows, mode
        self.files, self.names = {}, {}
        for root_index, root in enumerate(roots):
            root = Path(root)
            index = root / "model.safetensors.index.json"
            if index.exists():
                pairs = json.loads(index.read_text())["weight_map"].items()
                for name, file in pairs:
                    if root_index and ".engram.embed." not in canonical(name):
                        continue
                    path = root / file
                    if path.exists():
                        self.names.setdefault(canonical(name), (path, name))
            else:
                for path in root.glob("*.safetensors"):
                    with safe_open(path, framework="pt", device="cpu") as f:
                        for name in f.keys():
                            if root_index and ".engram.embed." not in canonical(name):
                                continue
                            self.names.setdefault(canonical(name), (path, name))

    def tensor(self, name):
        path, key = self.names[name]
        if path not in self.files:
            self.files[path] = safe_open(path, framework="pt", device="cpu")
        return self.files[path].get_slice(key)

    def shape(self, name):
        return self.tensor(name).get_shape()

    def descriptors(self):
        result = {}
        for name in self.names:
            if name.startswith(("layers.", "embed.", "head.", "norm.")):
                tensor = self.tensor(name)
                result[name] = {
                    "shape": tensor.get_shape(),
                    "dtype": tensor.get_dtype(),
                }
        return result

    def decode(self, name, start, end, engram=False):
        raw = self.tensor(name)[start:end]
        scale_name = name.removesuffix("weight") + "scale"
        if scale_name not in self.names:
            return raw.float()
        if raw.dtype in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2):
            scale = self.tensor(scale_name)[start:end].float()
            return unpack_fp4(raw) * scale.repeat_interleave(32, -1)
        if engram:
            scale = self.tensor(scale_name)[start:end].float().repeat_interleave(32, -1)
        else:
            scale = self.tensor(scale_name)[start // 32 : (end + 31) // 32].float()
            scale = scale.repeat_interleave(32, 0)[
                start % 32 : start % 32 + end - start
            ].repeat_interleave(32, 1)
        return raw.float() * scale[:, : raw.shape[1]]

    def embedding(self, name, ids, engram=False):
        rows = [self.decode(name, i, i + 1, engram) for i in ids.flatten().tolist()]
        out = torch.cat(rows).reshape(*ids.shape, -1)
        return out if self.mode == "NativeF32" else out.bfloat16()

    def project(
        self,
        name,
        x,
        start=0,
        end=None,
        quantize=True,
        output_dtype=None,
        bf16_weight=False,
    ):
        end = self.shape(name)[0] if end is None else end
        if (
            self.mode == "ReferenceFp8"
            and quantize
            and name.removesuffix("weight") + "scale" in self.names
        ):
            q, s = act_quant(x)
            x = q.float() * s.float().repeat_interleave(32, -1)
        x = x.float()
        out = torch.empty(*x.shape[:-1], end - start, dtype=torch.float32)
        for lo in range(start, end, self.rows):
            hi = min(lo + self.rows, end)
            w = self.decode(name, lo, hi)
            if bf16_weight and self.mode == "ReferenceFp8":
                w = w.bfloat16().float()
            out[..., lo - start : hi - start] = x @ w.T
        if self.mode == "ReferenceFp8":
            out = out.to(output_dtype or torch.bfloat16)
        return out
