#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Request:
    arrival_ms: int
    tokens: int
    request_id: str
    prompt_tokens: int


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "<channel|>" in text:
        return text.split("<channel|>")[-1].strip()
    return text.strip()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(paths))


def load_game(path: Path) -> tuple[dict, list[dict]]:
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return {}, value
    if not isinstance(value, dict):
        raise SystemExit(f"{path} is not a messages array or metadata wrapper")
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise SystemExit(f"{path} does not contain a messages array")
    meta = value.get("meta", {})
    return meta if isinstance(meta, dict) else {}, messages


def parse_created_at(meta: dict) -> int | None:
    raw = meta.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        instant = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=dt.timezone.utc)
    return int(instant.timestamp() * 1000)


class TokenCounter:
    def __init__(
        self,
        model: Path | None,
        qwen_bench: Path,
        estimate_chars_per_token: float,
    ) -> None:
        self.model = model
        self.qwen_bench = qwen_bench
        self.estimate_chars_per_token = estimate_chars_per_token
        self.cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        if self.model is None:
            count = max(1, round(len(text) / self.estimate_chars_per_token))
        else:
            count = self._count_with_qwen_bench(text)
        self.cache[text] = count
        return count

    def _count_with_qwen_bench(self, text: str) -> int:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            temp = Path(f.name)
        try:
            run = subprocess.run(
                [
                    str(self.qwen_bench),
                    "tok",
                    "-m",
                    str(self.model),
                    "--file",
                    str(temp),
                    "--iters",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            temp.unlink(missing_ok=True)
        for line in run.stdout.splitlines():
            match = re.search(r"\btokens=(\d+)", line)
            if match:
                return int(match.group(1))
        raise SystemExit("qwen-bench tok output did not include a token count")


def session_base_ms(
    created_ms: int | None,
    min_created_ms: int | None,
    session_index: int,
    session_gap_ms: int,
    use_created_at: bool,
) -> int:
    if use_created_at and created_ms is not None and min_created_ms is not None:
        return created_ms - min_created_ms
    return session_index * session_gap_ms


def build_requests(
    paths: list[Path],
    counter: TokenCounter,
    arrival_model: str,
    fixed_gap_ms: int,
    session_gap_ms: int,
    max_requests: int | None,
    strip_assistant_thinking: bool,
) -> list[Request]:
    metas_and_messages = [(*load_game(path), path) for path in paths]
    created_values = [parse_created_at(meta) for meta, _, _ in metas_and_messages]
    min_created_ms = min((v for v in created_values if v is not None), default=None)
    use_created_at = arrival_model == "session-created-fixed-gap"

    out: list[Request] = []
    global_i = 0
    for session_i, (meta, messages, path) in enumerate(metas_and_messages):
        base_ms = session_base_ms(
            parse_created_at(meta),
            min_created_ms,
            session_i,
            session_gap_ms,
            use_created_at,
        )
        prompt_tokens = 0
        assistant_i = 0
        for msg_i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role == "assistant":
                completion = (
                    strip_think(content) if strip_assistant_thinking else content
                )
                if arrival_model == "burst":
                    arrival_ms = 0
                elif arrival_model == "fixed-gap":
                    arrival_ms = global_i * fixed_gap_ms
                else:
                    arrival_ms = base_ms + assistant_i * fixed_gap_ms
                out.append(
                    Request(
                        arrival_ms=arrival_ms,
                        tokens=counter.count(completion),
                        request_id=f"{path.stem}:{msg_i}",
                        prompt_tokens=prompt_tokens,
                    )
                )
                assistant_i += 1
                global_i += 1
                if max_requests is not None and global_i >= max_requests:
                    return sorted(out, key=lambda row: (row.arrival_ms, row.request_id))
            prompt_tokens += counter.count(content)
    return sorted(out, key=lambda row: (row.arrival_ms, row.request_id))


def write_trace(
    rows: list[Request],
    output: Path | None,
    provenance: dict[str, str],
) -> None:
    lines = [
        "# qwen_request_trace_version=2",
        *[f"# {key}={value}" for key, value in provenance.items()],
        "arrival_ms\ttokens\tid\tprompt_tokens",
    ]
    lines.extend(
        f"{row.arrival_ms}\t{row.tokens}\t{row.request_id}\t{row.prompt_tokens}"
        for row in rows
    )
    text = "\n".join(lines) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build replay-economics request traces from game transcripts."
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model", type=Path, help="optional GGUF for exact token counts"
    )
    parser.add_argument(
        "--qwen-bench", type=Path, default=Path("target/release/qwen-bench")
    )
    parser.add_argument(
        "--arrival-model",
        choices=["burst", "fixed-gap", "session-created-fixed-gap"],
        default="burst",
    )
    parser.add_argument("--fixed-gap-ms", type=int, default=1000)
    parser.add_argument("--session-gap-ms", type=int, default=0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--estimate-chars-per-token", type=float, default=4.0)
    parser.add_argument(
        "--strip-assistant-thinking",
        action="store_true",
        help="count visible assistant text instead of full generated text",
    )
    args = parser.parse_args()

    if args.fixed_gap_ms < 0 or args.session_gap_ms < 0:
        raise SystemExit("gap values must be non-negative")
    if args.estimate_chars_per_token <= 0.0:
        raise SystemExit("--estimate-chars-per-token must be positive")

    inputs = expand_inputs(args.input)
    if not inputs:
        raise SystemExit("no input files matched")
    counter = TokenCounter(args.model, args.qwen_bench, args.estimate_chars_per_token)
    rows = build_requests(
        inputs,
        counter,
        args.arrival_model,
        args.fixed_gap_ms,
        args.session_gap_ms,
        args.max_requests,
        args.strip_assistant_thinking,
    )
    provenance = {
        "provenance": "scenario"
        if args.arrival_model != "session-created-fixed-gap"
        else "semi-synthetic",
        "arrival_source": args.arrival_model,
        "content_source": "game_transcripts",
        "completion_source": "assistant_transcript_messages",
        "token_count_source": "qwen-bench tok" if args.model else "char_estimate",
        "is_empirical_arrival": "false",
        "inputs": str(len(inputs)),
        "requests": str(len(rows)),
    }
    write_trace(rows, args.output, provenance)


if __name__ == "__main__":
    main()
