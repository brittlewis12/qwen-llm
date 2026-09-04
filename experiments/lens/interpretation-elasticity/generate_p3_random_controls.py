import hashlib
import json
import math
import random
import struct
from pathlib import Path


ROOT = Path(__file__).parent
OUTPUT = ROOT / "controls" / "p3-random-v1"
MASTER_SEED = 20260903
MODELS = {
    "qwen3.6-27b": (5120, [22, 32, 42, 52, 56]),
    "muse-glimmer-30b": (6656, [27, 35, 43, 49]),
}


def normal_values(rng: random.Random, count: int) -> list[float]:
    values = []
    while len(values) < count:
        u1 = (rng.getrandbits(53) + 1) / ((1 << 53) + 1)
        u2 = rng.getrandbits(53) / (1 << 53)
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        values.append(radius * math.cos(angle))
        if len(values) < count:
            values.append(radius * math.sin(angle))
    return values


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for model, (width, layers) in MODELS.items():
        for layer in layers:
            label = (
                f"interpretation-elasticity|p3-random-v1|{MASTER_SEED}|{model}|{layer}"
            )
            seed_sha256 = hashlib.sha256(label.encode("ascii")).hexdigest()
            rng = random.Random(int(seed_sha256, 16))
            values = normal_values(rng, width)
            norm = math.sqrt(math.fsum(value * value for value in values))
            payload = b"".join(struct.pack("<f", value / norm) for value in values)
            stored = struct.unpack(f"<{width}f", payload)
            stored_norm = math.sqrt(math.fsum(value * value for value in stored))
            path = OUTPUT / f"{model}.layer-{layer}.f32le"
            path.write_bytes(payload)
            records.append(
                {
                    "model": model,
                    "layer": layer,
                    "hidden_size": width,
                    "path": path.name,
                    "byte_length": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "seed_label": label,
                    "seed_sha256": seed_sha256,
                    "stored_l2_norm": stored_norm,
                }
            )
    manifest = {
        "schema": "interpretation_elasticity.random_controls",
        "schema_version": 1,
        "master_seed": MASTER_SEED,
        "generator": "python_random_mt19937_getrandbits53_box_muller_then_f32le_unit_l2",
        "reroll_policy": "never",
        "records": records,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT / "manifest.json")


if __name__ == "__main__":
    main()
