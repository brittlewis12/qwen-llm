# /// script
# requires-python = ">=3.12"
# ///

import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path


SEED = 20260901
HIDDEN_SIZE = 5120
LAYER = 46
LABEL = "seeded-isotropic-gaussian"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_to_bf16_rne(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()


def write_frozen(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to rewrite frozen file: {path}")
        return
    path.write_bytes(content)


def main() -> None:
    root = Path(__file__).resolve().parent
    rng = random.Random(SEED)
    values = [rng.gauss(0.0, 1.0) for _ in range(HIDDEN_SIZE)]
    norm = math.sqrt(math.fsum(value * value for value in values))
    values = [value / norm for value in values]
    bf16 = [f32_to_bf16_rne(value) for value in values]
    bf16_bytes = b"".join(struct.pack("<H", value) for value in bf16)
    decoded = [bf16_to_f32(value) for value in bf16]
    decoded_norm = math.sqrt(math.fsum(value * value for value in decoded))
    decoded_mean = math.fsum(decoded) / HIDDEN_SIZE

    metadata = {
        "model_id": "Qwen/Qwen3.6-27B",
        "coordinate": "post_block_residual",
        "construction": "python_mt19937_gauss_f64_unit_l2_bf16_rne",
        "seed": str(SEED),
        "python_version": sys.version.split()[0],
        "hidden_size": str(HIDDEN_SIZE),
        "layer": str(LAYER),
        "decoded_bf16_sha256": hashlib.sha256(bf16_bytes).hexdigest(),
        "decoded_bf16_l2": format(decoded_norm, ".17g"),
        "decoded_bf16_mean": format(decoded_mean, ".17g"),
    }
    header = {
        "__metadata__": metadata,
        "layers": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]},
        "word_ids": {"dtype": "I64", "shape": [1], "data_offsets": [8, 16]},
        "templates": {
            "dtype": "BF16",
            "shape": [1, 1, HIDDEN_SIZE],
            "data_offsets": [16, 16 + len(bf16_bytes)],
        },
    }
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode()
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    safetensors = b"".join(
        [
            struct.pack("<Q", len(header_bytes)),
            header_bytes,
            struct.pack("<q", LAYER),
            struct.pack("<q", 0),
            bf16_bytes,
        ]
    )
    write_frozen(root / "direction.safetensors", safetensors)
    write_frozen(root / "labels.tsv", f"0\t{LABEL}\n".encode())
    write_frozen(
        root / "provenance.json",
        canonical_json(
            {
                "schema": "qwen.lens.seeded_isotropic_direction",
                "schema_version": 1,
                **metadata,
                "weights_file": "direction.safetensors",
                "weights_sha256": hashlib.sha256(safetensors).hexdigest(),
                "labels_file": "labels.tsv",
            }
        ),
    )
    plan = {
        "version": 1,
        "lenses": [
            {
                "kind": "published_full_transport",
                "id": "neuronpedia-j1000",
                "artifact": "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/neuronpedia-j-lens-n1000-native-v1",
                "token_ids": [815],
                "allow_unvalidated_transfer": True,
            },
            {
                "kind": "workspace_template",
                "id": "isotropic-sham",
                "weights": "direction.safetensors",
                "labels": "labels.tsv",
            },
        ],
        "directions": [
            {
                "id": "isotropic-layer46",
                "lens": "isotropic-sham",
                "row": {"kind": "label", "label": LABEL},
                "normalization": "unit_l2",
            }
        ],
        "operations": [
            {
                "id": "isotropic-layer46-add",
                "scope": {
                    "layers": {"kind": "values", "values": [LAYER]},
                    "prefill": {"kind": "all"},
                    "decode": {"kind": "all"},
                },
                "action": {
                    "kind": "residual_l2_fraction",
                    "direction": "isotropic-layer46",
                    "coefficient": 0.45,
                },
            }
        ],
        "readouts": [
            {
                "id": "error-coordinate",
                "lens": "neuronpedia-j1000",
                "scope": {
                    "layers": {"kind": "values", "values": [LAYER]},
                    "prefill": {"kind": "all"},
                },
                "top_k": 1,
            }
        ],
    }
    write_frozen(root / "plan.json", canonical_json(plan))


if __name__ == "__main__":
    main()
