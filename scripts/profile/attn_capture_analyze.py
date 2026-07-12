#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path, prefix_bytes: int | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    remaining = prefix_bytes
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if remaining is not None:
                chunk = chunk[:remaining]
                remaining -= len(chunk)
            digest.update(chunk)
            size += len(chunk)
            if remaining == 0:
                break
    return size, digest.hexdigest()


def check_file(root: Path, entry: dict) -> None:
    path = root / entry["name"]
    hashes = entry["hashes"]
    size, digest = sha256(path)
    if size != hashes["byte_length"] or digest != hashes["sha256"]:
        raise ValueError(f"full-file identity mismatch: {path}")
    prefix_digest = hashes.get("prefix_32768_sha256")
    if prefix_digest is not None:
        prefix_size, actual = sha256(path, 32768 * 2 * 256 * 2)
        if prefix_size != 32768 * 2 * 256 * 2 or actual != prefix_digest:
            raise ValueError(f"32K-prefix identity mismatch: {path}")


def mmap_tensor(root: Path, entry: dict) -> np.memmap:
    dtype = {"f16le": "<f2", "f32le": "<f4", "i32le": "<i4"}[entry["dtype"]]
    tensor = np.memmap(root / entry["name"], dtype=dtype, mode="r")
    shape = tuple(entry["shape"])
    if tensor.size != math.prod(shape):
        raise ValueError(f"shape/length mismatch: {entry['name']}")
    tensor = tensor.reshape(shape)
    if not np.isfinite(tensor).all() or not np.any(tensor != 0):
        raise ValueError(f"non-finite or all-zero tensor: {entry['name']}")
    return tensor


def validate_manifest(manifest: dict) -> None:
    architecture = manifest["architecture"]
    if architecture != {
        "name": "Qwen3.6-A3B",
        "q_heads": 16,
        "kv_heads": 2,
        "group": 8,
        "head_dim": 256,
        "kv_dtype": "f16",
    }:
        raise ValueError("unexpected architecture contract")
    blocks = manifest["blocks"]
    tier = manifest["capture_tier"]
    context = manifest["context"]
    contracts = {
        "smoke": (context < 32768, [{"block": 3, "kv_slot": 0}]),
        "guard_32k": (context == 32768, [{"block": 3, "kv_slot": 0}]),
        "canonical": (
            context == 131072,
            [
                {"block": 3, "kv_slot": 0},
                {"block": 19, "kv_slot": 4},
                {"block": 39, "kv_slot": 9},
            ],
        ),
    }
    if tier not in contracts:
        raise ValueError("unknown capture tier")
    valid_context, expected_blocks = contracts[tier]
    if not valid_context or blocks != expected_blocks:
        raise ValueError("capture tier and block-to-KV mapping disagree")
    if tier != "smoke" and manifest["selector_environment"]:
        raise ValueError("non-smoke capture contains environment overrides")
    tokens = manifest["tokens"]
    if tokens["name"] != "tokens.i32le" or tokens["dtype"] != "i32le":
        raise ValueError("unexpected token metadata")
    if tokens["shape"] != [context]:
        raise ValueError("token shape disagrees with context")
    positions = manifest["unique_positions"]
    for view in manifest["views"]:
        if len(view["positions"]) != 64 or len(view["deduplicated_rows"]) != 64:
            raise ValueError("capture view does not contain 64 queries")
        for position, row in zip(
            view["positions"], view["deduplicated_rows"], strict=True
        ):
            if positions[row] != position or position >= view["context"]:
                raise ValueError("capture view row mapping is invalid")
    expected_names = {"tokens.i32le"}
    for block in blocks:
        index = block["block"]
        expected_names.update(
            f"block-{index}-{kind}.{dtype}"
            for kind, dtype in [
                ("k", "f16le"),
                ("v", "f16le"),
                ("q", "f32le"),
                ("o", "f32le"),
            ]
        )
    entries = [manifest["tokens"], *manifest["tensors"]]
    names = [entry["name"] for entry in entries]
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise ValueError("capture tensor file set is incomplete or duplicated")
    for entry in manifest["tensors"]:
        kind = entry["name"].split("-")[-1].split(".")[0]
        if kind in {"k", "v"}:
            expected = ("f16le", [context, 2, 256], [1024, 512, 2])
        else:
            expected = ("f32le", [len(positions), 16, 256], [16384, 1024, 4])
        if (entry["dtype"], entry["shape"], entry["strides_bytes"]) != expected:
            raise ValueError(f"unexpected tensor metadata: {entry['name']}")
    block_slots = {block["block"]: block["kv_slot"] for block in blocks}
    seen = set()
    observed_paths = set()
    for record in manifest["producer_provenance"]:
        identity = (record["block"], record["position"])
        if (
            record["semantics"] != "prefill_causal_query"
            or block_slots.get(record["block"]) != record["kv_slot"]
            or record["position"] not in positions
            or record["causal_length"] != record["position"] + 1
            or identity in seen
        ):
            raise ValueError("invalid or duplicate producer provenance")
        seen.add(identity)
        observed_paths.add(record["path"])
        matrix_fields = (
            record["online_matrix"]
            and record["query_rows"] is not None
            and record["packed_rows"] is None
            and record["packed_qt"] is None
            and record["nwg"] is None
            and record["tile_c"] is None
            and record["group_tile"] is None
        )
        packed_fields = (
            not record["online_matrix"]
            and record["query_rows"] is None
            and record["packed_rows"] is not None
            and record["packed_qt"] is not None
            and record["nwg"] is not None
            and record["tile_c"] is None
            and record["group_tile"] is None
        )
        fallback_fields = (
            not record["online_matrix"]
            and record["query_rows"] is None
            and record["packed_rows"] is None
            and record["packed_qt"] is None
            and record["nwg"] is not None
            and record["tile_c"] is not None
            and record["group_tile"] is not None
        )
        valid = {
            "matrix": matrix_fields,
            "packed": packed_fields,
            "decode_fallback": fallback_fields,
        }.get(record["path"], False)
        if not valid:
            raise ValueError("producer path and topology fields disagree")
        if tier != "smoke" and not (
            record["path"] == "matrix"
            and record["matrix_causal_skip"]
            and record["query_tiled"]
        ):
            raise ValueError(
                "non-smoke producer topology is not frozen production matrix"
            )
    expected_records = len(blocks) * len(positions)
    if len(seen) != expected_records:
        raise ValueError("producer provenance is incomplete")
    if observed_paths != set(manifest["resolved_attention_paths"]):
        raise ValueError("resolved paths disagree with producer provenance")


def selected_rows(view: dict, max_queries: int) -> list[int]:
    rows = view["deduplicated_rows"]
    if len(rows) <= max_queries:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_queries, dtype=int)
    return [rows[index] for index in sorted(set(indices.tolist()))]


def replay_head(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scores = k.astype(np.float64) @ q.astype(np.float64)
    scores /= math.sqrt(q.size)
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= np.sum(weights)
    return weights @ v.astype(np.float64)


def validate_replay(
    root: Path,
    manifest: dict,
    tensors: dict[str, dict],
    max_queries: int,
    poison_suffix: bool,
) -> dict:
    positions = manifest["unique_positions"]
    provenance = {
        (row["block"], row["position"]): row for row in manifest["producer_provenance"]
    }
    expected_records = len(manifest["blocks"]) * len(positions)
    if len(provenance) != expected_records:
        raise ValueError("duplicate or missing producer provenance")
    worst_cosine = 1.0
    worst_max_abs = 0.0
    sum_squared = 0.0
    compared = 0
    replayed_queries = 0
    for block in manifest["blocks"]:
        index = block["block"]
        k = mmap_tensor(root, tensors[f"block-{index}-k.f16le"])
        v = mmap_tensor(root, tensors[f"block-{index}-v.f16le"])
        q = mmap_tensor(root, tensors[f"block-{index}-q.f32le"])
        output = mmap_tensor(root, tensors[f"block-{index}-o.f32le"])
        for view in manifest["views"]:
            for row in selected_rows(view, max_queries):
                position = positions[row]
                if position not in view["positions"]:
                    continue
                causal_length = position + 1
                record = provenance[(index, position)]
                if record["causal_length"] != causal_length:
                    raise ValueError("causal length disagrees with captured position")
                if poison_suffix and causal_length < len(k):
                    poisoned_k = np.array(k)
                    poisoned_v = np.array(v)
                    poisoned_k[causal_length:] = np.nan
                    poisoned_v[causal_length:] = np.nan
                    if np.isfinite(poisoned_k[causal_length:]).any():
                        raise ValueError("causal poison did not take effect")
                    k_prefix = poisoned_k[:causal_length]
                    v_prefix = poisoned_v[:causal_length]
                else:
                    k_prefix = k[:causal_length]
                    v_prefix = v[:causal_length]
                cosine_limit, abs_limit = (
                    (0.9999, 2e-2) if record["path"] == "matrix" else (0.99999, 2e-3)
                )
                for head in range(16):
                    expected = replay_head(
                        q[row, head], k_prefix[:, head // 8], v_prefix[:, head // 8]
                    )
                    actual = output[row, head].astype(np.float64)
                    denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
                    cosine = float(np.dot(expected, actual) / denominator)
                    max_abs = float(np.max(np.abs(expected - actual)))
                    if (
                        not np.isfinite(cosine)
                        or cosine < cosine_limit
                        or max_abs > abs_limit
                    ):
                        raise ValueError(
                            f"replay mismatch block={index} position={position} head={head} "
                            f"path={record['path']} cosine={cosine:.9f} max_abs={max_abs:.6g}"
                        )
                    worst_cosine = min(worst_cosine, cosine)
                    worst_max_abs = max(worst_max_abs, max_abs)
                    sum_squared += float(np.sum(np.square(expected - actual)))
                    compared += expected.size
                replayed_queries += 1
    return {
        "replayed_queries": replayed_queries,
        "compared_values": compared,
        "min_head_cosine": worst_cosine,
        "max_abs": worst_max_abs,
        "rmse": math.sqrt(sum_squared / compared),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and replay an attention capture"
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--max-queries-per-view", type=int, default=6)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--poison-suffix", action="store_true")
    args = parser.parse_args()
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["claim_scope"] != "prefill_causal_query"
    ):
        raise ValueError("unsupported capture manifest contract")
    validate_manifest(manifest)
    entries = [manifest["tokens"], *manifest["tensors"]]
    for entry in entries:
        check_file(args.capture, entry)
    tensors = {entry["name"]: entry for entry in manifest["tensors"]}
    max_queries = 1_000_000 if args.all_queries else args.max_queries_per_view
    if max_queries <= 0:
        raise ValueError("max queries must be positive")
    result = validate_replay(
        args.capture, manifest, tensors, max_queries, args.poison_suffix
    )
    result.update(
        {
            "capture": str(args.capture),
            "context": manifest["context"],
            "paths": manifest["resolved_attention_paths"],
            "file_count": len(entries),
            "status": "pass",
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
