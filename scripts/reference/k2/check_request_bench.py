# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Instrumented K2 benchmark-accounting smoke, NOT a performance measurement.

Serial CLI children own their production leases/memory gates. Requires a fresh
evidence directory and records the explicit dirty-build override. No outer lease.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


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
    env = {**os.environ, "MTL_DEBUG_LAYER": "1"}

    def run(name, options, failure=None, model_path=model):
        command = [
            str(binary),
            "--allow-dirty",
            "k2-request",
            "--model",
            str(model_path),
            "--output",
            "json",
        ] + options
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
        doc = json.loads(result.stdout)
        assert (
            doc["schema"] == "qwen.k2_horizon.request_benchmark"
            and doc["schema_version"] == 1
        )
        assert doc["qualification"]["all_repetitions_identical"] is True
        for field in [
            "steady_state_qualified",
            "llama_bench_comparable",
            "kernel_only",
            "performance_claim",
        ]:
            assert doc["qualification"][field] is False
        assert doc["instrumentation"]["MTL_DEBUG_LAYER"] == "1"
        assert doc["model"]["execution_capacity"] == doc["request"]["capacity"]
        assert doc["model"]["output_head_shape"] == [4096, 250624]
        assert doc["model"]["kv_storage"] == "f16"
        ids = doc["request"]["prompt_token_ids"]
        assert (
            doc["request"]["prompt_token_ids_sha256_i32le"]
            == hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
        )
        all_runs = doc["samples"] + ([doc["warmup"]] if doc["warmup"] else [])
        assert all(r["outcome"] == all_runs[0]["outcome"] for r in all_runs)
        for row in all_runs:
            result = row["outcome"]
            generated = result["sampled_token_ids"]
            assert row["prompt_forwards"] == len(ids)
            assert result["transition_forwards"] == len(generated) - 1
            assert row["committed_positions"] == len(ids) + len(generated) - 1
            assert (
                row["request_wall_ns"]
                >= row["first_sample_ready_request_wall_ns"]
                >= row["session_allocation_wall_ns"] + row["prefill_wall_ns"]
            )
            assert (
                row["transition_forward_wall_ns"]
                <= row["generation_wall_ns"]
                <= row["request_wall_ns"]
            )
            assert (
                row["session_allocation_wall_ns"]
                + row["prefill_wall_ns"]
                + row["sampler_setup_wall_ns"]
                + row["generation_wall_ns"]
                <= row["request_wall_ns"]
            )
            assert (
                result["sampled_token_ids_sha256_i32le"]
                == hashlib.sha256(
                    struct.pack(f"<{len(generated)}i", *generated)
                ).hexdigest()
            )
            assert (
                result["emitted_bytes_sha256"]
                == hashlib.sha256(
                    bytes.fromhex(result["emitted_bytes_hex"])
                ).hexdigest()
            )
        return doc

    text = "The capital of France is"
    raw = run("raw", ["--raw-prompt", text, "--tokens", "8", "--runs", "2"])
    ids = raw["request"]["prompt_token_ids"]
    assert ids == [0, 864, 7169, 331, 8465, 395]
    assert len(raw["samples"]) == 2 and raw["warmup"] is not None
    assert raw["request"]["capacity"] == 13
    generated = raw["samples"][0]["outcome"]["sampled_token_ids"]
    # Request-stats uses a domain and count; the benchmark's raw i32le digest
    # intentionally has a different name/layout. Compare the actual run contract.
    run_fingerprint = hashlib.sha256(
        b"qwen-generated-token-ids-v1\0"
        + struct.pack("<Q", len(generated))
        + struct.pack(f"<{len(generated)}i", *generated)
    ).hexdigest()
    assert (
        run_fingerprint
        == "1bd2f3ebab9ec6045cd9d465ab2c5b61f2ce082f39c517a16981929c1da22723"
    )
    assert (
        raw["samples"][0]["outcome"]["emitted_text_lossy"]
        == " Paris. The capital of Germany is Berlin"
    )
    assert raw["samples"][0]["outcome"]["termination"] == "token_limit"
    serialized = run(
        "serialized-bos",
        [
            "--raw-prompt",
            "<|ifm|begin_of_text|>" + text,
            "--no-special-tokens",
            "--tokens",
            "8",
            "--runs",
            "1",
            "--no-warmup",
        ],
    )
    assert (
        serialized["warmup"] is None
        and serialized["request"]["add_special_tokens"] is False
    )
    assert serialized["samples"][0]["outcome"] == raw["samples"][0]["outcome"]
    literal = run(
        "literal-one-sample",
        [
            "--token-ids",
            ",".join(map(str, ids)),
            "--tokens",
            "1",
            "--runs",
            "1",
            "--no-warmup",
        ],
    )
    assert literal["request"]["add_special_tokens"] is None
    assert literal["aggregate"]["transition_forwards_per_second"] is None
    assert literal["samples"][0]["outcome"]["sampled_token_ids"] == [11511]
    run(
        "reject-capacity",
        ["--token-ids", "0", "--tokens", "1", "--capacity", "257"],
        "must fit both 256",
    )
    run("reject-id", ["--token-ids", "0,250624", "--tokens", "1"], "outside vocabulary")

    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    fixture = (
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 1)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string("qwen35")
    )
    fixture += bytes((-len(fixture)) % 32)
    other = output / "wrong-family-header-only.gguf"
    other.write_bytes(fixture)
    run(
        "reject-family",
        ["--token-ids", "0", "--tokens", "1"],
        "requires a dense K2 Horizon model",
        other,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_directory": str(output),
                "measurement_claim": "instrumented_accounting_only",
                "checks": [
                    "warmup_and_repeat_identity",
                    "raw_run_fingerprint",
                    "serialized_bos",
                    "literal_ids",
                    "zero_transition_rate",
                    "prefix_and_phase_accounting",
                    "host_only_invalid_request_rejection",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
