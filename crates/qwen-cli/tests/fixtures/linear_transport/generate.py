"""Independent standard-library producer fixtures; no Rust serializer or fit registry."""

import hashlib
import json
from pathlib import Path
import random
import struct


def publish(name, method, seed, hidden, layers, sources, target, identities):
    rng = random.Random(seed)
    matrices = []
    for layer in sources:
        values = [
            float(row == col) if layer in identities else rng.uniform(-0.5, 0.5)
            for row in range(hidden)
            for col in range(hidden)
        ]
        matrices.append(struct.pack("<" + "e" * len(values), *values))
    payload = b"".join(matrices)
    manifest = {
        "schema": "llm.lens.linear_transport",
        "schema_version": 1,
        "status": "complete",
        "transport": {
            "operator": "post_block_linear",
            "method": method,
            "source_layers": sources,
            "target_layer": target,
            "orientation": "target_source",
            "bias": "none",
            "output": "deployed_native",
            "identity_layers": identities,
        },
        "model": {
            "architecture": "qwen35",
            "n_layers": layers,
            "hidden_size": hidden,
            "vocab_size": 17 + seed,
            "source_checkpoint": {"id": "independent/fixture", "revision": str(seed)},
        },
        "payload": {
            "path": "transport.f16le",
            "dtype": "f16_le",
            "shape": [len(sources), hidden, hidden],
            "byte_length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "matrix_sha256": [
                hashlib.sha256(matrix).hexdigest() for matrix in matrices
            ],
        },
        "provenance": {"seed": seed, "fixture_generator": "python-stdlib"},
        "qualification": {
            "validated": True,
            "note": "deliberate producer claim, not authority",
        },
    }
    directory = Path(__file__).parent / name
    directory.mkdir()
    with (directory / "transport.f16le").open("xb") as output:
        output.write(payload)
    with (directory / "lens.json").open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    publish("seed41_h2", "experimental/seed41", 41, 2, 3, [2, 0], 1, [])
    publish("seed7_h3", "orthogonal-regression+beta", 7, 3, 4, [3, 1, 2], 2, [2])
