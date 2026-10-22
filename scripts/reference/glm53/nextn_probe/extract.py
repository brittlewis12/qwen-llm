#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf>=0.19.0"]
# ///
"""CPU-only, byte-preserving GLM block-45 extraction. Never opens a GPU backend."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import gguf
import numpy as np


ARCH = "glm5-next"
PROVENANCE = "qwen.nextn_probe.provenance"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(path):
    s = path.stat()
    return dict(
        path=str(path),
        bytes=s.st_size,
        device=s.st_dev,
        inode=s.st_ino,
        mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns,
    )


def digest(data):
    view = memoryview(data).cast("B")
    h = hashlib.sha256()
    for start in range(0, len(view), 8 << 20):
        h.update(view[start : start + (8 << 20)])
    return h.hexdigest()


def value(reader, key, default=None):
    field = reader.get_field(key)
    return default if field is None else field.contents()


def sources(first):
    # Do not resolve: publisher rewrite symlinks may point at non-split filenames.
    first = Path(os.path.abspath(first))
    before = stamp(first)
    reader = gguf.GGUFReader(first, "r")
    count = value(reader, "split.count", 1)
    require(type(count) is int and count > 0, "invalid split.count")
    require(value(reader, "split.no", 0) == 0, "--source must be shard one")
    paths = [first]
    if count > 1:
        match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
        require(
            match is not None and int(match[2]) == count,
            "split source must retain its -00001-of-NNNNN.gguf entrypoint name",
        )
        paths = [
            first.with_name(f"{match[1]}-{i + 1:05}-of-{count:05}.gguf")
            for i in range(count)
        ]
    stamps = [before] + [stamp(p) for p in paths[1:]]
    readers = [reader] + [gguf.GGUFReader(p, "r") for p in paths[1:]]
    total = sum(len(r.tensors) for r in readers)
    for i, r in enumerate(readers):
        require(
            r.endianess == gguf.GGUFEndian.LITTLE,
            "only little-endian GGUF is supported",
        )
        require(value(r, "split.count", 1) == count, "inconsistent split.count")
        require(value(r, "split.no", 0) == i, "inconsistent split.no")
        require(
            value(r, "split.tensors.count", total) == total,
            "missing shard or inconsistent split.tensors.count",
        )
    return paths, stamps, readers


def metadata(readers):
    result = {}
    source_arch = value(readers[0], "general.architecture")
    require(source_arch in (ARCH, "glm5next"), "expected GLM5-Next architecture")
    for reader in readers:
        for key, field in reader.fields.items():
            if key.startswith(("GGUF.", "split.")) or key == PROVENANCE:
                continue
            entry = (
                field.contents(),
                field.types[0],
                field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None,
            )
            if key == "general.architecture":
                require(
                    entry[0] == source_arch, "conflicting architecture between shards"
                )
                entry = (ARCH, entry[1], entry[2])
            if source_arch == "glm5next" and key.startswith("glm5next."):
                key = ARCH + key[len("glm5next") :]
            require(
                key not in result or result[key] == entry,
                f"conflicting metadata: {key}",
            )
            result[key] = entry
    return result


def tensor_shapes(meta):
    def number(key):
        x = meta[f"{ARCH}.{key}"][0]
        require(type(x) is int and x > 0, f"invalid {key}")
        return x

    require(
        number("block_count") == 46 and number("nextn_predict_layers") == 1,
        "expected 45 trunk blocks plus one NextN block; do not renumber the head",
    )
    h, vocab = number("embedding_length"), number("vocab_size")
    q, kv = number("attention.q_lora_rank"), number("attention.kv_lora_rank")
    heads = number("attention.head_count")
    kd, vd = number("attention.key_length_mla"), number("attention.value_length_mla")
    ih, idim = (
        number("attention.indexer.head_count"),
        number("attention.indexer.key_length"),
    )
    pool = number("attention.indexer.kpool")
    e, f = number("expert_count"), number("expert_feed_forward_length")
    sf = f * number("expert_shared_count")
    require(
        len(meta["tokenizer.ggml.tokens"][0]) == vocab, "tokenizer/vocabulary mismatch"
    )
    kv_heads = meta[f"{ARCH}.attention.head_count_kv"][0]
    require(len(kv_heads) == 46 and kv_heads[45] == 1, "NextN must be MLA")
    shapes = {
        "attn_norm.weight": [h],
        "ffn_norm.weight": [h],
        "attn_q_a.weight": [h, q],
        "attn_q_a_norm.weight": [q],
        "attn_q_b.weight": [q, heads * kd],
        "attn_kv_a_mqa.weight": [h, kv],
        "attn_kv_a_norm.weight": [kv],
        "attn_k_b.weight": [kd, kv, heads],
        "attn_v_b.weight": [kv, vd, heads],
        "attn_output.weight": [heads * vd, h],
        "indexer.attn_q_b.weight": [q, ih * idim],
        "indexer.attn_k.weight": [h, idim],
        "indexer.k_norm.weight": [idim],
        "indexer.k_norm.bias": [idim],
        "indexer.proj.weight": [h, ih],
        "indexer_compressor_gate.weight": [h, idim],
        "indexer_compressor_ape.weight": [idim, pool],
        "ffn_gate_inp.weight": [h, e],
        "exp_probs_b.bias": [e],
        "ffn_gate_exps.weight": [h, f, e],
        "ffn_up_exps.weight": [h, f, e],
        "ffn_down_exps.weight": [f, h, e],
        "ffn_gate_shexp.weight": [h, sf],
        "ffn_up_shexp.weight": [h, sf],
        "ffn_down_shexp.weight": [sf, h],
        "nextn.eh_proj.weight": [2 * h, h],
        "nextn.enorm.weight": [h],
        "nextn.hnorm.weight": [h],
        "nextn.shared_head_norm.weight": [h],
    }
    return {
        **{f"blk.45.{k}": v for k, v in shapes.items()},
        "token_embd.weight": [h, vocab],
        "output_norm.weight": [h],
        "output.weight": [h, vocab],
    }


def inventory(readers, shapes):
    seen, selected = set(), []
    for shard, reader in enumerate(readers):
        for t in reader.tensors:
            require(t.name not in seen, f"duplicate tensor: {t.name}")
            seen.add(t.name)
            if t.name.startswith("blk.45."):
                require(
                    t.name in shapes,
                    f"unsupported NextN tensor: {t.name}; do not silently drop it",
                )
            if t.name in shapes:
                require(list(t.shape) == shapes[t.name], f"wrong shape: {t.name}")
                require(
                    t.data_offset + t.n_bytes <= reader.data.size,
                    f"truncated tensor: {t.name}",
                )
                selected.append((shard, t))
    require(
        {t.name for _, t in selected} == shapes.keys(),
        f"missing required head/fallback tensors: {sorted(shapes.keys() - seen)}",
    )
    return selected


def extract(source, output):
    output = Path(os.path.abspath(output))
    require(not os.path.lexists(output), f"output already exists: {output}")
    require(output.parent.is_dir(), "output parent directory must already exist")
    paths, stamps, readers = sources(source)
    meta = metadata(readers)
    shapes = tensor_shapes(meta)
    selected = inventory(readers, shapes)
    records = [
        dict(
            name=t.name,
            shape=list(map(int, t.shape)),
            dtype=t.tensor_type.name,
            bytes=int(t.n_bytes),
            source_shard=shard,
            sha256=digest(t.data),
        )
        for shard, t in selected
    ]
    provenance = dict(
        schema="qwen.glm-nextn-extraction.v1",
        sources=stamps,
        source_header_sha256=[digest(r.data[: r.data_offset]) for r in readers],
        gguf_python=str(Path(gguf.__file__).resolve()),
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metadata_changes=[
            "remove split.*",
            "canonicalize glm5next namespace if present",
        ],
        tensors=records,
        tensor_count=len(records),
        payload_bytes=sum(r["bytes"] for r in records),
    )
    # Stage privately on the destination filesystem; hard-link publication never replaces a file/symlink.
    with tempfile.TemporaryDirectory(
        prefix=".nextn-extract-", dir=output.parent
    ) as tmp:
        stage = Path(tmp) / "head.gguf"
        writer = gguf.GGUFWriter(stage, ARCH)
        writer.data_alignment = meta.get("general.alignment", (32,))[0]
        for key, (val, typ, subtype) in meta.items():
            if key == "general.architecture":
                continue
            writer.add_key_value(key, val, typ, subtype)
        writer.add_string(PROVENANCE, json.dumps(provenance, sort_keys=True))
        for _, t in selected:
            # Supply logical reversed dimensions, not the quantized uint8 byte shape.
            writer.add_tensor_info(
                t.name,
                tuple(map(int, reversed(t.shape))),
                np.dtype("float32"),
                t.n_bytes,
                raw_dtype=t.tensor_type,
            )
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()
            for _, t in selected:
                writer.write_tensor_data(
                    t.data, tensor_endianess=gguf.GGUFEndian.LITTLE
                )
        finally:
            writer.close()
        os.chmod(stage, 0o600)
        check = gguf.GGUFReader(stage, "r")
        require(
            not any(k.startswith("split.") for k in check.fields),
            "output still declares shards",
        )
        copied = inventory([check], shapes)
        require(len(check.tensors) == 32, "output contains non-head tensors")
        expected = {r["name"]: r for r in records}
        for _, t in copied:
            r = expected[t.name]
            require(
                t.tensor_type.name == r["dtype"]
                and t.n_bytes == r["bytes"]
                and digest(t.data) == r["sha256"],
                f"copy verification failed: {t.name}",
            )
        require([stamp(p) for p in paths] == stamps, "source changed during extraction")
        with stage.open("rb") as f:
            os.fsync(f.fileno())
        os.link(stage, output)
    return dict(output=str(output), bytes=output.stat().st_size, **provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="GGUF first shard; preserve supplied symlink name",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new private single-file GGUF; must not exist",
    )
    args = p.parse_args()
    try:
        print(json.dumps(extract(args.source, args.output), indent=2))
    except (ValueError, KeyError, OSError) as error:
        p.exit(1, f"extraction refused: {error}\n")


if __name__ == "__main__":
    main()
