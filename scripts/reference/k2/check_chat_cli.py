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
    p.add_argument("--http-evidence", type=Path)
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
    execution = info["capabilities"]["execution"]
    assert execution["serve"]["chat"] is True
    assert execution["artifact"]["core"]["status"] == "passed"
    assert execution["artifact"]["generation"]["status"] == "passed"
    assert execution["request_device"]["status"] == "not_evaluated"
    for lane in ["run", "serve", "bench", "lens"]:
        assert execution[lane]["status"] == "conditional"
        assert execution[lane]["artifact_admission"]["status"] == "passed"
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
        tag = {
            "high": "ifm|think",
            "medium": "ifm|think_fast",
            "low": "ifm|think_faster",
        }[case["reasoning_effort"]]
        raw_text = raw.stdout.removesuffix(b"\n").decode("utf-8", errors="replace")
        raw_text = raw_text.removeprefix(f"<{tag}>")
        reasoning, closed, visible = raw_text.partition(f"</{tag}>")
        assert diagnostic["reasoning_closed"] == bool(closed)
        assert diagnostic["output"] == "reasoning_stderr_answer_stdout"
        assert rendered.stdout == (visible + "\n" if visible else "").encode()
        assert reasoning.encode() in rendered.stderr
        if not closed:
            assert (
                b"incomplete response: token budget exhausted before reasoning closed"
                in rendered.stderr
            )
        assert document["output_fingerprint"] == raw_doc["output_fingerprint"]
        assert "chat" not in raw_doc["diagnostics"]["k2_horizon"]
        if name == "effort-high":
            user = run("user", base + ["--user", case["messages"][0]["content"]])
            assert user.stdout == rendered.stdout
    completed_stats = output / "completed.stats.jsonl"
    completed = run(
        "completed",
        [
            "run",
            "-m",
            model,
            "-n",
            "128",
            "--temp",
            "0",
            "--user",
            "What is 2+2? Answer briefly.",
            "--reasoning-effort",
            "low",
            "--request-stats-jsonl",
            completed_stats,
        ],
    )
    completed_doc = json.loads(completed_stats.read_bytes())
    assert (
        completed_doc["diagnostics"]["k2_horizon"]["chat"]["reasoning_closed"] is True
    )
    assert completed.stdout.strip() == b"4"
    assert b"incomplete response" not in completed.stderr
    if args.http_evidence:
        evidence = json.loads(args.http_evidence.read_bytes())
        assert evidence["status"] == "passed"
        for case in evidence["cases"]:
            if case["effort"] == "low" and case["budget"] == 128:
                response = case["response"]
                assert response["status"] == "completed"
                assert (
                    completed.stdout
                    == (response["output"][1]["content"][0]["text"] + "\n").encode()
                )
                assert (
                    completed_doc["usage"]["input_tokens"]
                    == response["usage"]["input_tokens"]
                )
                assert (
                    completed_doc["usage"]["output_tokens"]
                    == response["usage"]["output_tokens"]
                )
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
        "http_completed_parity": args.http_evidence is not None,
        "output_partitioning": True,
        "performance_claim": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
