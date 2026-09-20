# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Opt-in declared-context CLI checks; serial children own production leases. No speed claim."""

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qwen", "bench", "lens", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=1024)
    args = parser.parse_args()
    capacity = args.capacity
    assert 258 <= capacity <= 524288
    model = args.model.resolve(strict=True)
    qwen, bench, lens = [
        getattr(args, n).resolve(strict=True) for n in ("qwen", "bench", "lens")
    ]
    output = args.output.absolute()
    output.mkdir(mode=0o700)
    env = {
        **os.environ,
        "MTL_DEBUG_LAYER": "1",
        "QWEN_CHECKPOINT_MODEL_IDENTITY": "hashed",
    }

    def run(name, command, failure=None, env_overrides=None):
        command = list(map(str, command))
        child_env = {**env, **(env_overrides or {})}
        result = subprocess.run(
            command, env=child_env, capture_output=True, timeout=300
        )
        (output / f"{name}.command.json").write_text(json.dumps(command, indent=2))
        (output / f"{name}.environment.json").write_text(
            json.dumps(
                {
                    "MTL_DEBUG_LAYER": child_env["MTL_DEBUG_LAYER"],
                    "QWEN_MATVEC_Q8_0_LCPP": child_env.get("QWEN_MATVEC_Q8_0_LCPP"),
                },
                indent=2,
            )
        )
        (output / f"{name}.stdout").write_bytes(result.stdout)
        (output / f"{name}.stderr").write_bytes(result.stderr)
        if failure is not None:
            assert result.returncode != 0 and failure.encode() in result.stderr, (
                name,
                result.stderr,
            )
            assert b"Metal API Validation Enabled" not in result.stderr, name
        else:
            assert result.returncode == 0, (name, result.stderr)
            assert b"Metal API Validation Enabled" in result.stderr, name
        return result

    bbase = [
        bench,
        "--allow-dirty",
        "k2-request",
        "--model",
        model,
        "--runs",
        "1",
        "--no-warmup",
        "--output",
        "json",
    ]
    rbase = [qwen, "run", "--model", model]
    lbase = [
        lens,
        "read-full",
        "--model",
        model,
        "--identity-cache",
        output / "identity-cache",
        "--layers",
        "35",
        "--top-k",
        "1",
    ]
    prompt = " ".join(["a"] * (capacity - 1))
    reference = None
    for name, text, sampled, count in [
        ("boundary", prompt, 1, capacity),
        ("transition", " ".join(["a"] * (capacity - 2)), 2, capacity - 1),
        ("response", " ".join(["a"] * (capacity - 257)), 257, capacity - 256),
    ]:
        doc = json.loads(
            run(
                f"bench-{name}",
                bbase
                + ["--raw-prompt", text, "--tokens", sampled, "--capacity", capacity],
            ).stdout
        )
        assert len(doc["request"]["prompt_token_ids"]) == count
        sample = doc["samples"][0]
        assert sample["committed_positions"] == capacity
        assert sample["prompt_forwards"] == count
        assert sample["outcome"]["transition_forwards"] == sampled - 1
        assert doc["qualification"]["performance_claim"] is False
        assert doc["method"]["prefill_execution"] == {
            "mode": "q8_lcpp_token_batch",
            "chunk_tokens": 32,
            "commands": (count + 31) // 32,
            "temporary_activation_bytes": 7602304,
        }
        ids = sample["outcome"]["sampled_token_ids"]
        assert len(ids) == sampled
        emitted = bytes.fromhex(sample["outcome"]["emitted_bytes_hex"])
        stats_path = output / f"run-{name}.stats.jsonl"
        result = run(
            f"run-{name}",
            rbase
            + [
                "--raw-prompt",
                text,
                "-n",
                sampled,
                "--max-context-tokens",
                capacity,
                "--request-stats-jsonl",
                stats_path,
            ],
        )
        assert result.stdout == emitted + b"\n"
        stats = json.loads(stats_path.read_text())
        assert (
            stats["diagnostics"]["k2_horizon"]["prefill"]
            == doc["method"]["prefill_execution"]
        )
        assert stats["usage"] == {"input_tokens": count, "output_tokens": sampled}
        fingerprint = hashlib.sha256(
            b"qwen-generated-token-ids-v1\0"
            + struct.pack("<Q", len(ids))
            + struct.pack(f"<{len(ids)}i", *ids)
        ).hexdigest()
        assert stats["output_fingerprint"]["value"] == fingerprint
        if name == "boundary":
            reference = doc

    run(
        "reject-run-capacity-plus-one",
        rbase + ["--raw-prompt", prompt, "-n", "2", "--max-context-tokens", capacity],
        f"requires {capacity + 1} forwards",
    )
    run(
        "reject-run-capacity",
        rbase + ["--raw-prompt", "a", "-n", "1", "--max-context-tokens", "524289"],
        "capacity 524289",
    )
    for name, options in [
        ("capacity", ["--tokens", "1", "--capacity", "524289"]),
        ("sampled", ["--tokens", "524289"]),
    ]:
        run(
            f"reject-bench-{name}",
            bbase + ["--token-ids", "0"] + options,
            "checkpoint context 524288",
        )
    run(
        "reject-bench-prompt",
        bbase + ["--raw-prompt", prompt, "--tokens", "2", "--capacity", capacity],
        f"requires {capacity + 1} forwards",
    )

    fallback = json.loads(
        run(
            "serial-fallback-bench",
            bbase + ["--token-ids", "0,42", "--tokens", "1"],
            env_overrides={"QWEN_MATVEC_Q8_0_LCPP": "0"},
        ).stdout
    )
    assert fallback["method"]["prefill_execution"] == {
        "mode": "serial_single_token",
        "chunk_tokens": 1,
        "commands": 2,
        "temporary_activation_bytes": 0,
    }
    fallback_lens = json.loads(
        run(
            "serial-fallback-lens",
            lbase + ["--token-ids", "0,42", "--position", "1", "--logit-lens"],
            env_overrides={"QWEN_MATVEC_Q8_0_LCPP": "0"},
        ).stdout
    )
    assert (
        fallback_lens["deployed_model"]["prefill"]
        == fallback["method"]["prefill_execution"]
    )
    ids = reference["request"]["prompt_token_ids"]
    assert len(ids) == capacity and ids[0] == 0
    tokens = ",".join(map(str, ids))
    long_input = [
        "--token-ids",
        tokens,
        "--position",
        capacity - 1,
        "--max-tokens",
        capacity,
    ]
    plain = json.loads(
        run(
            "plain-boundary",
            lbase
            + long_input
            + ["--logit-lens", "--full-output", output / "plain-bundle"],
        ).stdout
    )
    assert plain["input"]["executed_token_count"] == capacity
    assert plain["deployed_model"]["executed_capacity"] == capacity
    assert (
        plain["deployed_model"]["prefill"] == reference["method"]["prefill_execution"]
    )
    assert plain["deployed_model"]["execution_topology"] == "q8_lcpp_token_batch"
    assert (
        plain["results"][0]["top_k"][0]["token_id"]
        == reference["samples"][0]["outcome"]["sampled_token_ids"][0]
    )

    asset = output / "identity-asset"
    asset.mkdir(mode=0o700)
    matrix = bytearray(4096 * 4096 * 2)
    for i in range(4096):
        struct.pack_into("<e", matrix, (i * 4096 + i) * 2, 1.0)
    digest = hashlib.sha256(matrix).hexdigest()
    manifest = {
        "schema": "llm.lens.linear_transport",
        "schema_version": 1,
        "status": "complete",
        "transport": {
            "operator": "post_block_linear",
            "method": "synthetic_boundary_check_not_fitted",
            "source_layers": [35],
            "target_layer": 35,
            "orientation": "target_source",
            "bias": "none",
            "output": "deployed_native",
            "identity_layers": [35],
        },
        "model": {
            "architecture": "k2-horizon",
            "n_layers": 36,
            "hidden_size": 4096,
            "vocab_size": 250624,
            "exact_binding": {
                "gguf_content_blake3": plain["deployed_model"]["content_blake3"],
                "tokenizer_metadata_id": plain["deployed_model"][
                    "tokenizer_metadata_id"
                ],
            },
        },
        "payload": {
            "path": "transport.f16le",
            "dtype": "f16_le",
            "shape": [1, 4096, 4096],
            "byte_length": len(matrix),
            "sha256": digest,
            "matrix_sha256": [digest],
        },
        "provenance": {
            "kind": "synthetic_not_fitted",
            "generator": "check_guarded_capacity.py",
        },
        "qualification": {"scientific_transport_quality": "not_evaluated"},
    }
    (asset / "transport.f16le").write_bytes(matrix)
    (asset / "lens.json").write_text(json.dumps(manifest))
    imported = json.loads(
        run(
            "imported-boundary",
            lbase
            + long_input
            + ["--full-lens", asset, "--full-output", output / "imported-bundle"],
        ).stdout
    )
    assert imported["transfer"]["status"] == "exact_deployment_binding_matched"
    assert imported["deployed_model"]["executed_capacity"] == capacity
    assert imported["deployed_model"]["prefill"] == plain["deployed_model"]["prefill"]
    assert (output / "plain-bundle/logits.f32le").read_bytes() == (
        output / "imported-bundle/logits.f32le"
    ).read_bytes()
    invalid_input = [
        "--token-ids",
        tokens + ",42",
        "--position",
        capacity + 1,
        "--max-tokens",
        capacity + 1,
    ]
    for name, mode in [
        ("plain", ["--logit-lens"]),
        ("imported", ["--full-lens", asset]),
    ]:
        run(
            f"reject-{name}-invalid-position",
            lbase + invalid_input + mode,
            "selected position exceeds input or context",
        )
    manifest["model"]["exact_binding"]["gguf_content_blake3"] = "0" * 64
    (asset / "lens.json").write_text(json.dumps(manifest))
    run(
        "reject-binding-override",
        lbase + long_input + ["--full-lens", asset, "--allow-unvalidated-transfer"],
        "deployment exact binding mismatch",
    )
    del manifest["model"]["exact_binding"]
    (asset / "lens.json").write_text(json.dumps(manifest))
    run(
        "reject-unbound",
        lbase + long_input + ["--full-lens", asset],
        "require --allow-unvalidated-transfer",
    )
    transfer = json.loads(
        run(
            "explicit-transfer-boundary",
            lbase + long_input + ["--full-lens", asset, "--allow-unvalidated-transfer"],
        ).stdout
    )
    assert transfer["transfer"]["override_applied"] is True
    assert transfer["results"][0]["top_k"] == plain["results"][0]["top_k"]
    report = {
        "status": "passed",
        "capacity": capacity,
        "performance_claim": False,
        "artifact_directory": str(output),
        "serve_followup": "K2_BOUNDARY_EVIDENCE points to this directory for the leased ephemeral HTTP test",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
