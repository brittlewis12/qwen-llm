# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "safetensors>=0.6", "numpy", "sympy", "tokenizers"]
# ///
"""Validate the full old REAP text graph using pinned headers, never tensor data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import torch

from bindings import validate_bindings
from oracle import load_tokenizer, model_args, source_module
from weights import canonical

REAP = (
    "LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E",
    "f792019bcb369768220f01088f0f7af230786982",
)
BASE = ("deepseek-ai/DeepSeek-V4.1-Flash", "314113f8a7dadb388215396ba5fbee2c610a04d6")


def fetch_range(url, start, end):
    # Unique queries also prevent a CDN's cached eight-byte probe serving a header request.
    url += f"?cpu_header={start}-{end}"
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise RuntimeError(
                f"Server did not honor bounded Range: {response.status} {content_range}"
            )
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("Truncated or oversized header range")
        return data, int(content_range.split("/")[-1])


def cached_json(cache, pin, name, offline=False, header=False):
    root = cache / pin[1]
    path = root / (name + ".header.json" if header else name)
    if path.exists():
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError(path)
    url = f"https://huggingface.co/{pin[0]}/resolve/{pin[1]}/{name}"
    if header:
        prefix, size = fetch_range(url, 0, 7)
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 16 * 1024 * 1024 or length + 8 > size:
            raise ValueError(f"Unsafe safetensors header length: {length}")
        data, second_size = fetch_range(url, 8, length + 7)
        if size != second_size:
            raise ValueError("Shard length changed between requests")
        obj = {
            "header": json.loads(data),
            "file_bytes": size,
            "header_bytes": length + 8,
            "header_sha256": hashlib.sha256(prefix + data).hexdigest(),
        }
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Metadata document exceeds bounded fetch size")
        obj = json.loads(data)
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return obj


def descriptor_model(config, mode="NativeF32", max_seq_len=32):
    module = source_module()
    args = model_args(module, config, max_seq_len)
    # Hash state has no checkpoint parameters. Avoid needing token strings just to
    # check all real parameter descriptors; EngramLayout and its embeddings remain.
    module.NgramHashState = lambda *a, **kw: None
    dtype = torch.float32 if mode == "NativeF32" else torch.bfloat16
    with module.set_dtype(dtype), torch.device("meta"):
        model = module.Transformer(args)
    return model, module, args


def check(cache, offline=False):
    config = cached_json(cache, REAP, "config.json", offline)
    index = cached_json(cache, REAP, "model.safetensors.index.json", offline)[
        "weight_map"
    ]
    required = {
        canonical(k): (k, v)
        for k, v in index.items()
        if canonical(k).startswith(("layers.", "embed.", "head.", "norm."))
    }
    shards = sorted({v for _, v in required.values()})

    def header(file):
        pin = (
            BASE
            if file
            in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
            else REAP
        )
        return file, cached_json(cache, pin, file, offline, header=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        headers = dict(pool.map(header, shards))
    descriptors = {}
    widths = {
        "F32": 4,
        "BF16": 2,
        "F16": 2,
        "F8_E4M3": 1,
        "F8_E8M0": 1,
        "I8": 1,
        "U8": 1,
    }
    for name, (original, shard) in required.items():
        item = headers[shard]
        desc = item["header"][original]
        start, end = desc["data_offsets"]
        count = 1
        for dim in desc["shape"]:
            count *= dim
        if (
            end - start != count * widths[desc["dtype"]]
            or not 0 <= start <= end <= item["file_bytes"] - item["header_bytes"]
        ):
            raise ValueError(f"Invalid physical descriptor: {name}")
        descriptors[name] = desc
    counts = {}
    for mode in ("NativeF32", "ReferenceFp8"):
        model, module, args = descriptor_model(config, mode)
        used = validate_bindings(model, module, descriptors)
        counts[mode] = len(used)
        if mode == "NativeF32":
            lazy = (
                module.Linear,
                module.ParallelEmbedding,
                module.ParallelEngramEmbedding,
                module.ParallelHead,
            )
            resident_bytes = sum(
                p.numel() * 4
                for obj in model.modules()
                if not isinstance(obj, lazy)
                for p in obj.parameters(recurse=False)
            )
        del model, module
    cached_json(cache, REAP, "tokenizer.json", offline)
    tokenizer = load_tokenizer(cache / REAP[1])
    module = source_module()
    layout = module.EngramLayout.from_args(args)
    state = module.NgramHashState(args, layout, tokenizer)
    hashes = torch.cat(
        (state(torch.tensor([[0, 1234]]), 0), state(torch.tensor([[1]]), 2)), dim=1
    )
    for i, rows in enumerate(args.engram_num_embeddings):
        if not ((hashes[:, :, i] >= 0) & (hashes[:, :, i] < rows)).all():
            raise ValueError("Real tokenizer Engram hash exceeds table descriptor")
    report = {
        "layers": args.n_layers,
        "routed_experts_per_layer": args.n_routed_experts,
        "hidden_dim": args.dim,
        "engram_rows": args.engram_num_embeddings,
        "bound_tensors": counts,
        "required_shards": len(shards),
        "header_bytes": sum(h["header_bytes"] for h in headers.values()),
        "tensor_payload_bytes_downloaded": 0,
        "native_resident_parameter_bytes": resident_bytes,
        "engram_revision": BASE[1],
        "reap_revision": REAP[1],
        "ignored_vl_gate_biases": len(descriptors) - len(used),
        "tokenizer_vocab_size": len(tokenizer),
        "compressed_tokenizer_vocab_size": args.engram_compressed_vocab_size,
        "tokenizer_and_engram_hash_ranges_verified": True,
        "tensor_payloads_verified": False,
    }
    (cache / "readiness.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require previously cached pinned metadata",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(check(args.cache_dir, args.offline), indent=2))


if __name__ == "__main__":
    main()
