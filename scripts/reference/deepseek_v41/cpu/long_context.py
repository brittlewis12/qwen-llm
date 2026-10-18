# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "safetensors>=0.6", "numpy", "sympy", "tokenizers"]
# ///
"""Teacher-forced, attention-only NativeF32 checkpoint replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oracle import HERE, model_args, source_module
from weights import Weights


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        if (
            self.meta["format"] != "dsv41-long-context-v1"
            or self.meta["mode"] != "NativeF32"
        ):
            raise ValueError("unsupported capture contract")
        if self.meta.get("status") != "success":
            raise ValueError("capture status must be success")
        requested = self.meta.get("requested_checkpoints")
        if (
            not isinstance(requested, list)
            or not requested
            or any(type(p) is not int or p < 1 for p in requested)
            or any(a >= b for a, b in zip(requested, requested[1:]))
        ):
            raise ValueError(
                "requested checkpoints must be nonempty and strictly increasing"
            )
        checkpoints = self.meta.get("checkpoints", [])
        if [c["position"] for c in checkpoints] != requested:
            raise ValueError(
                "recorded positions must exactly match requested checkpoints"
            )
        config = json.loads((HERE / "old_config.json").read_text())["text_config"]
        required = set(config["index_source_layer_ids"])
        layers = set()
        for checkpoint in checkpoints:
            current = set(map(int, checkpoint["layers"]))
            if not required.issubset(current):
                raise ValueError("each checkpoint must include all eight index sources")
            if any(not 2 <= layer < 40 for layer in current):
                raise ValueError("target layers must be backbone compressed attention")
            layers.update(current)
        self.inputs = {}
        for layer, spec in self.meta["inputs"].items():
            rows = np.load(self.root / spec["file"], mmap_mode="r", allow_pickle=False)
            if rows.dtype != np.dtype("<f4") or rows.ndim != 2 or rows.shape[1] != 5120:
                raise ValueError("inputs must be little-endian F32 [positions,5120]")
            self.inputs[int(layer)] = (spec["start"], rows)
        for layer in layers:
            if layer not in self.inputs:
                raise ValueError(f"missing input layer={layer}")
            start, rows = self.inputs[layer]
            if start != 0 or len(rows) <= requested[-1]:
                raise ValueError(f"incomplete input coverage layer={layer}")

    def rows(self, layer, start, end):
        base, rows = self.inputs[layer]
        if not base <= start <= end <= base + len(rows):
            raise ValueError(f"missing input layer={layer} positions=[{start},{end})")
        value = torch.from_numpy(rows[start - base : end - base].copy()).unsqueeze(0)
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite normalized input")
        return value


def bind_attention(m, args, store, layer):
    # Only parameter allocation is meta: caches and rotary tables remain CPU.
    with m.set_dtype(torch.float32):
        attn = m.Attention(layer, args)
    for name, obj in attn.named_modules():
        obj.cpu_name = f"layers.{layer}.attn" + ("." + name if name else "")
        if isinstance(obj, m.Linear):
            continue
        for key, param in list(obj.named_parameters(recurse=False)):
            value = store.tensor(obj.cpu_name + "." + key)[:].float()
            if value.shape != param.shape:
                raise ValueError("parameter shape mismatch")
            setattr(obj, key, torch.nn.Parameter(value, requires_grad=False))
    return attn


def configure(m, store):
    original = m.Linear.__init__

    def init(obj, *args, **kwargs):
        with torch.device("meta"):
            original(obj, *args, **kwargs)

    m.Linear.__init__ = init
    for cls in (m.Linear, m.ColumnParallelLinear, m.RowParallelLinear):
        cls.forward = lambda obj, x: store.project(obj.cpu_name + ".weight", x)
    m.act_quant = m.fp4_act_quant = lambda x, *a, **kw: x
    m.cpu_grouped = lambda attn, x: torch.stack(
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


def seed_history(m, attn, rows, position, batch):
    """Rebuild only positions strictly before the query; never read native caches."""
    if attn.is_kv_source:
        ratio = attn.compress_ratio
        batch = max(ratio, batch // ratio * ratio)
        for start in range(0, position, batch):
            end = min(position, start + batch)
            # Aligned chunks use the source prefill pooling path, not token projections.
            latent = attn.compressor(rows(start, end), 0)
            if latent is None or not latent.size(1):
                continue
            freqs = attn.freqs_cis[start : end - (end - start) % ratio : ratio]
            key = attn.indexer.k_norm(attn.indexer.wk(latent))
            m.apply_rotary_emb(key[..., -attn.rope_head_dim :], freqs)
            first, count = start // ratio, latent.size(1)
            attn.indexer.k_cache[:, first : first + count] = key
            m.apply_rotary_emb(latent[..., -attn.rope_head_dim :], freqs)
            attn.compress_kv_cache[:, first : first + count] = latent
    start = max(0, position - attn.window_size + 1)
    if start < position:
        x = rows(start, position)
        kv = attn.kv_norm(attn.wkv(x))
        m.apply_rotary_emb(
            kv[..., -attn.rope_head_dim :], attn.freqs_cis[start:position]
        )
        slots = torch.arange(start, position) % attn.window_size
        attn.window_kv_cache[:, slots] = kv


def metrics(actual, expected, atol, rtol):
    if (
        actual.shape != expected.shape
        or not torch.isfinite(actual).all()
        or not torch.isfinite(expected).all()
    ):
        return {"pass": False, "reason": "shape/nonfinite"}
    delta = (actual - expected).double()
    return {
        "pass": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (
            delta.norm() / expected.double().norm().clamp_min(1e-30)
        ).item(),
    }


def selection_diagnostics(m, attn, x, position):
    index = attn.indexer
    qr = attn.q_norm(attn.wq_a(x))
    q = index.wq_b(qr).unflatten(-1, (index.n_local_heads, index.index_head_dim))
    m.apply_rotary_emb(
        q[..., -index.rope_head_dim :], attn.freqs_cis[position : position + 1]
    )
    count = (position + 1) // index.compress_ratio
    weights = index.weights_proj(x) * (index.softmax_scale * index.n_heads**-0.5)
    scores = (
        torch.einsum("bshd,btd->bsht", q, m.shared_attn.index_k[:, :count]).relu()
        * weights.unsqueeze(-1)
    ).sum(2)
    block_tie = False
    if index.is_candidate_source:
        blocks = torch.nn.functional.pad(
            scores, (0, -count % index.candidate_block_size), value=-torch.inf
        )
        blocks = blocks.unflatten(-1, (-1, index.candidate_block_size)).amax(-1)
        blocks[..., -1] = torch.inf
        ranked = blocks.sort(descending=True).values.flatten()
        k = index.candidate_topk_blocks
        block_tie = len(ranked) > k and bool(ranked[k - 1] == ranked[k])
    if index.uses_candidates:
        scores = scores.masked_fill(~m.shared_attn.candidates, -torch.inf)
    ranked = scores.sort(descending=True).values.flatten()
    k = min(index.index_topk, count)
    gap = float(ranked[k - 1] - ranked[k]) if 0 < k < count else None
    return {
        "selection_boundary_gap": gap,
        "selection_boundary_tie": gap == 0,
        "candidate_boundary_tie": block_tie,
    }


@torch.inference_mode()
def replay(model, capture, batch=256, atol=0.02, rtol=0.02):
    cap = Capture(capture)
    m = source_module(stable_router=True)
    config = json.loads((HERE / "old_config.json").read_text())
    positions = [c["position"] for c in cap.meta["checkpoints"]]
    args = model_args(m, config, max(positions) + 1)
    store = Weights([model], 128, "NativeF32")
    configure(m, store)
    reports = []
    for checkpoint in cap.meta["checkpoints"]:
        p = checkpoint["position"]
        if p < 1:
            raise ValueError("long-context checkpoints require position >= 1")
        m.shared_attn = m.SharedAttentionRuntime()
        layers = sorted(map(int, checkpoint["layers"]))
        required = set(args.index_source_layers)
        if not required.issubset(layers):
            raise ValueError("each checkpoint must include all eight index sources")
        owner = index_owner = None
        for layer in layers:
            if not 2 <= layer < 40:
                raise ValueError("target layers must be backbone compressed attention")
            attn = bind_attention(m, args, store, layer)
            seed_history(m, attn, lambda a, b: cap.rows(layer, a, b), p, batch)
            output = attn(cap.rows(layer, p, p + 1), p).flatten()
            if attn.is_kv_source:
                owner = layer
            if attn.is_index_source:
                index_owner = layer
            native = checkpoint["layers"][str(layer)]
            expected = torch.from_numpy(
                np.load(cap.root / native["output"], allow_pickle=False).copy()
            ).flatten()
            ids = (m.shared_attn.topk_idxs.flatten() - args.window_size).tolist()
            row = {
                "position": p,
                "layer": layer,
                "owner": owner,
                "index_source": index_owner,
                "selection_exact": ids == native["compressed_ids"],
                "owner_exact": owner == native["kv_owner"]
                and index_owner == native["index_source"],
                "output": metrics(output, expected, atol, rtol),
            }
            if attn.is_index_source:
                row.update(selection_diagnostics(m, attn, cap.rows(layer, p, p + 1), p))
            if layer >= args.candidate_source_layer:
                mask = m.shared_attn.candidates.flatten()
                blocks = mask[:: args.candidate_block_size].nonzero().flatten().tolist()
                row["candidates_exact"] = blocks == native["candidate_blocks"]
                row["candidate_discard"] = int((~mask).sum())
            row["selection_discard"] = (p + 1) // attn.compress_ratio - len(ids)
            reports.append(row)
            del attn
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    rows = replay(args.model, args.capture, args.batch, args.atol, args.rtol)
    print(json.dumps(rows, indent=2))
    raise SystemExit(
        0
        if all(
            r["selection_exact"]
            and r["owner_exact"]
            and r.get("candidates_exact", True)
            and r["output"]["pass"]
            for r in rows
        )
        else 1
    )


if __name__ == "__main__":
    main()
