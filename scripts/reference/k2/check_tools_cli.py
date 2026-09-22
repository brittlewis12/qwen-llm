# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Opt-in CLI tool round trip; CLI children own production GPU leases. No tool execution."""

import argparse
import json
import os
from pathlib import Path
import subprocess

PROMPT = "Call lookup_code with key orbital to obtain its code. Do not guess the code. After the tool result arrives, repeat that code as your final answer."
TOOL = {
    "type": "function",
    "name": "lookup_code",
    "description": "Retrieve the code for a key.",
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--call-format", choices=["xml", "json", "xml_typed"], default="xml"
    )
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)

    def run(name, document):
        messages = args.output / f"{name}.messages.json"
        messages.write_text(json.dumps(document))
        command = [
            str(binary),
            "run",
            "-m",
            str(model),
            "--messages",
            str(messages),
            "--reasoning-effort",
            "low",
            "--max-context-tokens",
            "2048",
            "-n",
            "512",
            "--temp",
            "0",
            "--request-stats-jsonl",
            str(args.output / f"{name}.stats.jsonl"),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            env={**os.environ, "MTL_DEBUG_LAYER": "1"},
            timeout=180,
        )
        (args.output / f"{name}.command.json").write_text(json.dumps(command))
        (args.output / f"{name}.stdout").write_bytes(result.stdout)
        (args.output / f"{name}.stderr").write_bytes(result.stderr)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"Metal API Validation Enabled" in result.stderr
        envelope = json.loads(result.stdout)
        assert envelope["status"] == "completed", envelope
        assert envelope["x_k2"] == {
            "tool_presentation_format": "markdown",
            "tool_call_format": args.call_format,
        }
        return envelope

    document = {
        "input": [{"role": "user", "content": PROMPT}],
        "tools": [TOOL],
        "x_k2": {"tool_call_format": args.call_format},
    }
    first = run("call", document)
    calls = [item for item in first["output"] if item["type"] == "function_call"]
    assert len(calls) == 1 and calls[0]["name"] == "lookup_code", first
    assert json.loads(calls[0]["arguments"]) == {"key": "orbital"}, calls
    document["input"].extend(first["output"])
    document["input"].append(
        {
            "type": "function_call_output",
            "call_id": calls[0]["call_id"],
            "output": "copper-731",
        }
    )
    final = run("result", document)
    assert not any(item["type"] == "function_call" for item in final["output"]), final
    answer = "".join(
        part["text"]
        for item in final["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    assert "copper-731" in answer, final
    report = {
        "status": "passed",
        "call_format": args.call_format,
        "caller_supplied_result": True,
        "tool_executed_by_engine": False,
        "answer": answer,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
