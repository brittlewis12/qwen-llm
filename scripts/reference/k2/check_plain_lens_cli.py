# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Opt-in K2 final-tokenizer checkpoint CLI correctness, not a benchmark.

Every serial child uses the production Metal lease/memory gate itself. Never
hold an outer lease (that would deadlock the children), bypass a busy lease,
or operate any other owner's processes. API validation is forced on.
"""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory; existing results are never overwritten",
    )
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(mode=0o700)
    env = {**os.environ, "MTL_DEBUG_LAYER": "1"}
    base = [
        str(binary),
        "read-full",
        "--model",
        str(model),
        "--identity-cache",
        str(output / "identity-cache"),
    ]

    def run(name, options, failure=None):
        command = base + options
        result = subprocess.run(command, env=env, capture_output=True, timeout=120)
        (output / f"{name}.stderr").write_bytes(result.stderr)
        (output / f"{name}.stdout").write_bytes(result.stdout)
        (output / f"{name}.command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        if failure is not None:
            assert result.returncode != 0, name
            assert failure.encode() in result.stderr, (name, result.stderr)
            assert b"Metal API Validation Enabled" not in result.stderr, name
            return None
        assert result.returncode == 0, (name, result.stderr)
        assert b"Metal API Validation Enabled" in result.stderr, name
        document = json.loads(result.stdout)
        assert document["deployed_model"]["architecture"] == "k2-horizon"
        assert document["observer"]["fitted_artifact"] is None
        return document

    plain = ["--logit-lens", "--top-k", "1"]
    text = "The capital of France is"
    baseline = run("all-sites", plain + ["--prompt", text])
    ids = baseline["input"]["token_ids"]
    assert ids == [0, 864, 7169, 331, 8465, 395]
    assert [r["source_layer"] for r in baseline["results"]] == list(range(36))
    assert baseline["results"][-1]["top_k"][0]["token_id"] == 11511

    bos = run(
        "serialized-bos",
        plain + ["--prompt", "<|ifm|begin_of_text|>" + text, "--no-special-tokens"],
    )
    assert bos["input"]["token_ids"] == ids
    assert bos["input"]["add_special_tokens"] is False
    assert baseline["input"]["add_special_tokens"] is True
    assert bos["results"] == baseline["results"]

    selected = [35, 0, 17]
    bundle = output / "literal-bundle"
    literal = run(
        "literal-suffix",
        plain
        + [
            "--token-ids",
            ",".join(map(str, ids + [42, 17])),
            "--position",
            "5",
            "--layers",
            ",".join(map(str, selected)),
            "--include-vector",
            "--full-output",
            str(bundle),
        ],
    )
    assert literal["input"]["source"] == "token_ids"
    assert literal["input"]["add_special_tokens"] is None
    assert literal["input"]["token_ids"] == ids + [42, 17]
    assert literal["input"]["input_token_count"] == 8
    assert (
        literal["input"]["executed_token_count"]
        == literal["deployed_model"]["executed_capacity"]
        == 6
    )
    assert literal["deployed_model"]["requested_layer_order"] == selected
    assert literal["deployed_model"]["runtime_capture_layer_order"] == sorted(selected)
    metadata = json.loads((bundle / "metadata.json").read_bytes())
    assert metadata["readout"] == literal
    assert metadata["payload"]["source_layers"] == selected
    assert metadata["payload"]["shape"] == [3, 250624]
    payload = (bundle / "logits.f32le").read_bytes()
    assert len(payload) == 3 * 250624 * 4
    for slot, layer in enumerate(selected):
        result = literal["results"][slot]
        assert result["source_layer"] == layer
        assert result["top_k"] == baseline["results"][layer]["top_k"]
        vector = result["transported_vector"]["values"]
        assert len(vector) == 4096 and all(map(math.isfinite, vector))
        logits = struct.unpack_from("<250624f", payload, slot * 250624 * 4)
        assert all(map(math.isfinite, logits))
        top = max(range(len(logits)), key=lambda i: (logits[i], -i))
        assert top == result["top_k"][0]["token_id"]
        assert abs(logits[top] - result["top_k"][0]["logit"]) < 1e-5

    run(
        "reject-transfer",
        plain + ["--token-ids", "0", "--allow-unvalidated-transfer"],
        "do not accept --allow-unvalidated-transfer",
    )
    run(
        "reject-position",
        plain + ["--token-ids", ",".join(["0"] * 33)],
        "selected position must be below 32",
    )
    run(
        "reject-missing-asset",
        ["--token-ids", "0", "--full-lens", str(output / "missing-asset")],
        "open K2 data-only linear transport",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_directory": str(output),
                "content_blake3": baseline["deployed_model"]["content_blake3"],
                "checks": [
                    "all36sites",
                    "serialized_bos",
                    "literal_ids_with_suffix",
                    "layer_order",
                    "native_vectors",
                    "full_logits_bundle",
                    "pre_gpu_rejections",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
