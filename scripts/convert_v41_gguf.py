#!/usr/bin/env -S uv run --no-project python
"""Output-only V4.1 conversion using the reviewed llama.cpp source patch.

Defaults to a guarded, metadata-only preflight. No dependency or model downloads.
See convert_v41_gguf.md for source preparation and the existing worker environment.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PIN = "2076412598e99127ea2554467b2b7530856895d3"


def check_source(root: Path) -> str:
    """Optional source audit, not a model/runtime identity gate."""
    manifest = json.loads((HERE / "convert_v41_gguf.source.json").read_text())
    states = []
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest == expected["canonical"]:
            states.append("canonical")
        elif digest == expected["patched"]:
            states.append("patched")
        else:
            raise ValueError(f"Source audit mismatch: {name}")
    diff = HERE / "convert_v41_gguf.patch"
    if hashlib.sha256(diff.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        raise ValueError("Sidecar patch does not match source manifest")
    state = "patched" if "patched" in states else "canonical"
    command = ["git", "apply", "--check"]
    if state == "patched":
        command.append("--reverse")
    subprocess.run([*command, str(diff)], cwd=root, check=True)
    return state


def import_converter(root: Path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "gguf-py")]
    import gguf
    import torch
    from conversion.deepseek import DeepseekV41Model

    return gguf, torch, DeepseekV41Model


@contextlib.contextmanager
def headers_only(gguf):
    import numpy as np
    import safetensors

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Weight payload access or output opening during metadata-only preflight"
        )

    with contextlib.ExitStack() as stack:
        for owner, name in (
            (gguf.utility.LocalTensor, "mmap_bytes"),
            (safetensors, "safe_open"),
            (np, "memmap"),
            (gguf.GGUFWriter, "open_output_file"),
        ):
            stack.enter_context(patch.object(owner, name, forbidden))
        yield


def validate_headers(model):
    """Check scale layouts before the converter removes the scale entries."""
    import math
    import re

    h = model.hparams
    q = h["quantization_config"]
    br, bc = q["weight_block_size"]
    projection_shapes = {}
    for name, gen in model.model_tensors.items():
        if not name.endswith(".scale"):
            continue
        weight_name = name.removesuffix(".scale") + ".weight"
        weight = model.model_tensors[weight_name]()
        scale = gen()
        if len(weight.shape) != 2 or len(scale.shape) != 2:
            raise ValueError(f"Expected two-dimensional weight/scale: {name}")
        rows, cols = weight.shape
        if ".engram.embed." in name:
            expected = (rows, cols // 32)
            if cols % 32:
                raise ValueError(f"Engram columns must be a multiple of 32: {name}")
        elif match := re.fullmatch(
            r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.scale", name
        ):
            expected = (rows, cols // 16)
            if cols % 16:
                raise ValueError(
                    f"MXFP4 packed columns must be a multiple of 16: {name}"
                )
            key = (match[1], match[3])
            if projection_shapes.setdefault(key, tuple(weight.shape)) != tuple(
                weight.shape
            ):
                raise ValueError(f"Inconsistent expert shape: {name}")
        else:
            expected = (math.ceil(rows / br), math.ceil(cols / bc))
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"Scale layout mismatch: {name}: {scale.shape} != {expected}"
            )
    for bid, rows in zip(
        h["engram_layer_ids"], h["engram_num_embeddings"], strict=True
    ):
        name = f"layers.{bid}.engram.embed.weight"
        shape = tuple(model.model_tensors[name]().shape)
        if shape != (rows, h["engram_head_dim"]):
            raise ValueError(f"Engram table/config mismatch: {name}: {shape}")
        if name.removesuffix(".weight") + ".scale" not in model.model_tensors:
            raise ValueError(f"Missing FP8 Engram scale: {name}")


def plan(writer, gguf) -> dict:
    totals = {}
    tensors = []
    payload = 0
    for shard in writer.tensors:
        for name, info in shard.items():
            payload += writer.ggml_pad(info.nbytes, writer.data_alignment)
            totals[info.dtype.name] = totals.get(info.dtype.name, 0) + info.nbytes
            tensors.append(
                {
                    "name": name,
                    "shape": list(info.shape),
                    "dtype": info.dtype.name,
                    "bytes": info.nbytes,
                }
            )
    # Serialize the canonical headers in memory, never opening an output file.
    preview = copy.copy(writer)
    from gguf.gguf_writer import WriterState

    with io.BytesIO() as buffer:
        preview.fout = [buffer]
        preview.state = WriterState.EMPTY
        preview.open_output_file = lambda path=None: None
        preview.write_header_to_file()
        preview.write_kv_data_to_file()
        preview.write_ti_data_to_file()
        header = writer.ggml_pad(buffer.tell(), writer.data_alignment)
    return {
        "tensor_count": len(tensors),
        "dtype_bytes": totals,
        "tensor_bytes_with_padding": payload,
        "header_bytes_with_padding": header,
        "output_bytes": header + payload,
        "temporary_payload_bytes": 0,
        "tensors": tensors,
    }


def validate_engram_metadata(model, gguf) -> dict:
    import struct

    h = model.hparams
    kv = model.gguf_writer.kv_data[0]
    arch = model.gguf_writer.arch
    token_map = kv[gguf.Keys.Engram.TOKEN_MAP.format(arch=arch)].value
    primes = kv[gguf.Keys.Engram.PRIMES.format(arch=arch)].value
    offsets = kv[gguf.Keys.Engram.OFFSETS.format(arch=arch)].value
    buckets = (h["engram_max_ngram_size"] - 1) * h["engram_n_heads"]
    if (
        len(token_map) != h["vocab_size"]
        or len(set(token_map)) != h["engram_compressed_vocab_size"]
    ):
        raise ValueError("Engram token map does not match configured vocabulary")
    if len(primes) != buckets * len(h["engram_layer_ids"]) or len(offsets) != len(
        primes
    ):
        raise ValueError("Engram bucket count does not match configuration")
    for i, rows in enumerate(h["engram_num_embeddings"]):
        total = 0
        for j in range(i * buckets, (i + 1) * buckets):
            if offsets[j] != total:
                raise ValueError("Engram bucket offsets are not contiguous")
            total += primes[j]
        if total != rows:
            raise ValueError(
                f"Engram bucket sizes {total} do not match table rows {rows}"
            )
    return {
        "architecture": arch,
        "expert_count": h["n_routed_experts"],
        "layer_count": h["num_hidden_layers"],
        "token_map_entries": len(token_map),
        "compressed_vocab_size": h["engram_compressed_vocab_size"],
        "token_map_i32le_sha256": hashlib.sha256(
            struct.pack(f"<{len(token_map)}i", *token_map)
        ).hexdigest(),
        "engram_table_rows": h["engram_num_embeddings"],
        "engram_bucket_layout_validated": True,
    }


def prepare(root: Path, model_dir: Path, outfile: Path, chunk_rows: int):
    gguf, torch, model_class = import_converter(root)
    with headers_only(gguf), torch.inference_mode():
        config = json.loads((model_dir / "config.json").read_text())
        if config.get("architectures") != ["DeepseekV41ForCausalLM"]:
            raise ValueError(
                "This driver converts DeepseekV41ForCausalLM main models only"
            )
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for filename in set(index.values()):
            path = Path(filename)
            if path.name != filename or path.suffix != ".safetensors":
                raise ValueError(
                    f"Expected local safetensors shard basename: {filename}"
                )
        model = model_class(
            model_dir,
            gguf.LlamaFileType.MOSTLY_BF16,
            outfile,
            eager=False,
            use_temp_file=False,
        )
        model._V41_ENGRAM_CHUNK_ROWS = chunk_rows
        validate_headers(model)
        model.prepare_tensors()
        model.prepare_metadata(vocab_only=False)
        report = plan(model.gguf_writer, gguf)
        report["metadata_validation"] = validate_engram_metadata(model, gguf)
        report["engram_chunk_rows"] = chunk_rows
        report["engram_chunk_f32_bytes"] = (
            chunk_rows * model.hparams["engram_head_dim"] * 4
        )
    return model, report


def write_output(model, outfile: Path, expected_bytes: int):
    """Keep one partial inode; publish with a no-overwrite hard link on success."""
    if outfile.exists() or outfile.is_symlink():
        raise FileExistsError(outfile)
    partial_dir = outfile.with_name(outfile.name + ".incomplete")
    if shutil.disk_usage(outfile.parent).free < expected_bytes + 1024**3:
        raise OSError("Insufficient free space for final GGUF plus 1 GiB margin")
    partial_dir.mkdir(mode=0o700)
    partial = partial_dir / outfile.name
    writer = model.gguf_writer
    try:
        import torch

        with torch.inference_mode():
            writer.write_header_to_file(path=partial)
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=True)
        writer.flush()
        for fp in writer.fout:
            os.fsync(fp.fileno())
        writer.close()
        if partial.stat().st_size != expected_bytes:
            raise ValueError("Written GGUF size differs from metadata-only plan")
        os.link(partial, outfile)
        partial.unlink()
        partial_dir.rmdir()
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        required=True,
        help="Local patched source; never downloaded or modified",
    )
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="Audit sidecar patch against pinned sources, then exit",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Metadata only (default); no payload reads or output writes",
    )
    mode.add_argument(
        "--convert",
        action="store_true",
        help="Explicitly authorize payload reads and final output writing",
    )
    parser.add_argument("--show-tensors", action="store_true")
    args = parser.parse_args()
    root = args.llama_cpp.resolve()
    if args.check_source:
        print(json.dumps({"pin": PIN, "source": check_source(root)}))
        return
    if args.model is None or args.outfile is None:
        parser.error("--model and --outfile are required")
    if not 1 <= args.chunk_rows <= 65536:
        parser.error("--chunk-rows must be in 1..65536")
    outfile = args.outfile.parent.resolve() / args.outfile.name
    model_dir = args.model.resolve()
    if outfile.is_relative_to(model_dir):
        parser.error("Output must be outside the source model/cache directory")
    if args.convert and (
        outfile.exists()
        or outfile.is_symlink()
        or outfile.with_name(outfile.name + ".incomplete").exists()
    ):
        parser.error("Output or incomplete run already exists; no overwrite or resume")
    logging.basicConfig(level=logging.INFO)
    model, report = prepare(root, model_dir, outfile, args.chunk_rows)
    report["mode"] = "convert" if args.convert else "metadata-only"
    if not args.show_tensors:
        report.pop("tensors")
    print(json.dumps(report, indent=2), flush=True)
    if args.convert:
        write_output(model, outfile, report["output_bytes"])


if __name__ == "__main__":
    main()
