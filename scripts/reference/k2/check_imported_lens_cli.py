# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Opt-in imported K2 lens CLI checks with synthetic assets, never checkpoint conversion.

Children run serially and acquire their own production Metal leases and memory
gates. This script neither holds a conflicting outer lease nor bypasses a busy
lease. Each run requires a fresh output directory; evidence is never overwritten.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess


H = 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(mode=0o700)
    env = {
        **os.environ,
        "MTL_DEBUG_LAYER": "1",
        "QWEN_CHECKPOINT_MODEL_IDENTITY": "hashed",
    }
    base = [
        str(binary),
        "read-full",
        "--model",
        str(model),
        "--token-ids",
        "0,42,17",
        "--top-k",
        "3",
        "--include-vector",
        "--identity-cache",
        str(output / "identity-cache"),
    ]

    def run(name, options, failure=None):
        command = base + options
        result = subprocess.run(command, env=env, capture_output=True, timeout=120)
        (output / f"{name}.stdout").write_bytes(result.stdout)
        (output / f"{name}.stderr").write_bytes(result.stderr)
        (output / f"{name}.command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        if failure:
            assert result.returncode != 0 and failure.encode() in result.stderr, (
                name,
                result.stderr,
            )
            assert b"Metal API Validation Enabled" not in result.stderr, name
            return None
        assert result.returncode == 0, (name, result.stderr)
        assert b"Metal API Validation Enabled" in result.stderr, name
        return json.loads(result.stdout)

    plain = run(
        "plain",
        [
            "--logit-lens",
            "--layers",
            "35,0",
            "--full-output",
            str(output / "plain-bundle"),
        ],
    )
    binding = {
        "gguf_content_blake3": plain["deployed_model"]["content_blake3"],
        "tokenizer_metadata_id": plain["deployed_model"]["tokenizer_metadata_id"],
    }
    identity = bytearray(H * H * 2)
    for i in range(H):
        struct.pack_into("<e", identity, (i * H + i) * 2, 1.0)

    def asset(name, matrices, sources, exact, identity_layers):
        directory = output / name
        directory.mkdir(mode=0o700)
        payload = b"".join(matrices)
        manifest = {
            "schema": "llm.lens.linear_transport",
            "schema_version": 1,
            "status": "complete",
            "transport": {
                "operator": "post_block_linear",
                "method": "synthetic_cli_correctness_not_fitted",
                "source_layers": sources,
                "target_layer": 35,
                "orientation": "target_source",
                "bias": "none",
                "output": "deployed_native",
                "identity_layers": identity_layers,
            },
            "model": {
                "architecture": "k2-horizon",
                "n_layers": 36,
                "hidden_size": H,
                "vocab_size": 250624,
            },
            "payload": {
                "path": "transport.f16le",
                "dtype": "f16_le",
                "shape": [len(sources), H, H],
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "matrix_sha256": [hashlib.sha256(m).hexdigest() for m in matrices],
            },
            "provenance": {
                "kind": "synthetic_not_fitted",
                "generator": "check_imported_lens_cli.py",
            },
            "qualification": {"scientific_transport_quality": "not_evaluated"},
        }
        if exact:
            manifest["model"]["exact_binding"] = binding
        (directory / "transport.f16le").write_bytes(payload)
        (directory / "lens.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return directory, manifest

    exact_dir, manifest = asset(
        "exact-identity", [identity, identity], [35, 0], True, [35, 0]
    )
    exact = run(
        "exact",
        ["--full-lens", str(exact_dir), "--full-output", str(output / "exact-bundle")],
    )
    assert [r["source_layer"] for r in exact["results"]] == [35, 0]
    assert exact["transfer"]["status"] == "exact_deployment_binding_matched"
    assert exact["transfer"]["override_applied"] is False
    assert exact["artifact"]["payload_sha256"] == manifest["payload"]["sha256"]
    assert exact["artifact"]["matrix_sha256"] == manifest["payload"]["matrix_sha256"]
    assert exact["artifact"]["payload_blake3"] != exact["artifact"]["payload_sha256"]
    assert (
        exact["deployed_model"]["content_identity"]["bytes_hashed"]
        == model.stat().st_size
    )
    assert (output / "exact-bundle/logits.f32le").read_bytes() == (
        output / "plain-bundle/logits.f32le"
    ).read_bytes()
    for actual, control in zip(exact["results"], plain["results"], strict=True):
        assert actual["top_k"] == control["top_k"]
        assert (
            actual["transported_vector"]["values"]
            == control["transported_vector"]["values"]
        )
        assert actual["transported_vector"]["operation"] == "post_block_linear"

    unbound_dir, _ = asset(
        "unbound-identity", [identity, identity], [35, 0], False, [35, 0]
    )
    run(
        "reject-unacknowledged",
        ["--full-lens", str(unbound_dir)],
        "require --allow-unvalidated-transfer",
    )
    transferred = run(
        "acknowledged",
        [
            "--full-lens",
            str(unbound_dir),
            "--allow-unvalidated-transfer",
            "--layers",
            "0,35",
        ],
    )
    assert (
        transferred["transfer"]["status"] == "source_deployment_equivalence_unverified"
    )
    assert transferred["transfer"]["override_applied"] is True
    assert [r["source_layer"] for r in transferred["results"]] == [0, 35]
    for actual, control in zip(
        transferred["results"], reversed(exact["results"]), strict=True
    ):
        assert actual == control

    nonsymmetric = bytearray(identity)
    struct.pack_into("<e", nonsymmetric, 2, 2.0)
    struct.pack_into("<e", nonsymmetric, H * 2, -3.0)
    asym_dir, _ = asset("exact-nonsymmetric", [nonsymmetric], [35], True, [])
    asymmetric = run("nonsymmetric", ["--full-lens", str(asym_dir)])
    original = plain["results"][0]["transported_vector"]["values"]
    expected = original.copy()
    expected[0] += 2.0 * original[1]
    expected[1] -= 3.0 * original[0]
    actual = asymmetric["results"][0]["transported_vector"]["values"]
    assert all(
        math.isfinite(a) and abs(a - b) <= 2e-5 * (1 + abs(b))
        for a, b in zip(actual, expected, strict=True)
    )
    assert abs(actual[0] - (original[0] - 3.0 * original[1])) > 1e-3

    bad = copy.deepcopy(manifest)
    bad["model"]["exact_binding"]["gguf_content_blake3"] = "0" * 64
    (exact_dir / "lens.json").write_text(json.dumps(bad))
    run(
        "reject-mismatched-even-with-override",
        ["--full-lens", str(exact_dir), "--allow-unvalidated-transfer"],
        "deployment exact binding mismatch",
    )
    bad = copy.deepcopy(manifest)
    bad["transport"]["target_layer"] = 34
    (exact_dir / "lens.json").write_text(json.dumps(bad))
    run(
        "reject-target",
        ["--full-lens", str(exact_dir)],
        "expected runtime profile/target layer",
    )
    bad = copy.deepcopy(manifest)
    bad["payload"]["sha256"] = "0" * 64
    (exact_dir / "lens.json").write_text(json.dumps(bad))
    run(
        "reject-digest",
        ["--full-lens", str(exact_dir)],
        "whole payload SHA256 mismatch",
    )
    (exact_dir / "lens.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_directory": str(output),
                "binding": binding,
                "checks": [
                    "identity_full_logits",
                    "exact_binding",
                    "explicit_transfer",
                    "default_and_requested_order",
                    "nonsymmetric_orientation",
                    "wrong_binding_not_overridable",
                    "target_gate",
                    "digest_gate",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
