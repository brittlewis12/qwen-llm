# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Opt-in K2 no-tools CLI/template parity; children own production GPU leases."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir(mode=0o700)
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "crates/qwen-llm/tests/fixtures/k2_chat_hf.json"
        ).read_bytes()
    )
    env = {**os.environ, "MTL_DEBUG_LAYER": "1"}

    def run(name, options, *, failure=None, gpu=True):
        command = [str(binary), *map(str, options)]
        result = subprocess.run(command, env=env, capture_output=True, timeout=180)
        (output / f"{name}.command.json").write_text(json.dumps(command))
        (output / f"{name}.stdout").write_bytes(result.stdout)
        (output / f"{name}.stderr").write_bytes(result.stderr)
        if failure:
            assert result.returncode != 0 and failure.encode() in result.stderr, (
                name,
                result.stderr,
            )
            assert b"Metal API Validation Enabled" not in result.stderr
        else:
            assert result.returncode == 0, (name, result.stderr)
            assert (b"Metal API Validation Enabled" in result.stderr) == gpu
        return result

    info = json.loads(run("info", ["info", "--json", "-m", model], gpu=False).stdout)
    assert info["capabilities"]["template"]["status"] == "identified"
    assert info["capabilities"]["reasoning"]["levels"] == ["high", "medium", "low"]
    assert info["capabilities"]["execution"]["serve"]["chat"] is False
    profile = info["capabilities"]["template"]["profile"]
    assert profile["template_sha256"] == fixture["template_sha256"]
    assert profile["generation_config_sha256"] == fixture["generation_config_sha256"]
    base = ["run", "-m", model, "-n", "8", "--temp", "0"]
    for name in [
        "effort-high",
        "effort-medium",
        "effort-low",
        "history-think_fast-text",
        "system-unicode",
    ]:
        case = next(c for c in fixture["cases"] if c["name"] == name)
        path = output / f"{name}.messages.json"
        path.write_text(json.dumps(case["messages"], ensure_ascii=True))
        stats = output / f"{name}.stats.jsonl"
        rendered = run(
            name,
            base
            + [
                "--messages",
                path,
                "--reasoning-effort",
                case["reasoning_effort"],
                "--request-stats-jsonl",
                stats,
            ],
        )
        document = json.loads(stats.read_bytes())
        diagnostic = document["diagnostics"]["k2_horizon"]["chat"]
        assert diagnostic["profile"] == profile
        assert diagnostic["stops"] == [1, 250019]
        ids = case["token_ids"]
        assert (
            diagnostic["prompt_token_ids_sha256_i32le"]
            == hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()
        )
        assert document["usage"]["input_tokens"] == len(ids)
        assert document["input"]["template"] == profile["renderer"]
        raw_stats = output / f"{name}.raw.stats.jsonl"
        raw = run(
            name + "-raw",
            base
            + [
                "--raw-prompt",
                case["rendered"],
                "--no-special-tokens",
                "--request-stats-jsonl",
                raw_stats,
            ],
        )
        raw_doc = json.loads(raw_stats.read_bytes())
        # The short corpus must not reach chat's extra stop; otherwise this is not
        # a like-for-like output control and the script must report that fact.
        assert rendered.stdout == raw.stdout
        assert document["output_fingerprint"] == raw_doc["output_fingerprint"]
        assert "chat" not in raw_doc["diagnostics"]["k2_horizon"]
        if name == "effort-high":
            user = run("user", base + ["--user", case["messages"][0]["content"]])
            assert user.stdout == rendered.stdout
    for name, options, error in [
        ("no-thinking", ["--user", "-", "--no-thinking"], "no released --no-thinking"),
        (
            "effort-none",
            ["--user", "-", "--reasoning-effort", "none"],
            "reasoning effort must be",
        ),
    ]:
        run(name, base + options, failure=error)
    for name, doc in [
        ("tools", {"messages": [{"role": "user", "content": "x"}], "tools": []}),
        (
            "missing-thinking",
            [
                {"role": "assistant", "content": "prior"},
                {"role": "user", "content": "next"},
            ],
        ),
        (
            "developer",
            [{"role": "developer", "content": "x"}, {"role": "user", "content": "y"}],
        ),
    ]:
        path = output / f"invalid-{name}.json"
        path.write_text(json.dumps(doc))
        run("reject-" + name, base + ["--messages", path], failure="K2 chat:")
    report = {
        "status": "passed",
        "scope": "verified_final_no_tools_cli_template_and_raw_controls",
        "profile": profile,
        "http_chat": False,
        "output_partitioning": False,
        "performance_claim": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
