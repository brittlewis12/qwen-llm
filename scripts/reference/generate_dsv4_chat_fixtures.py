# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

"""Generate DeepSeek V4 0731 chat-encoding byte fixtures.

Each case executes the pinned vLLM and SGLang release-derived encoders and
requires byte-identical prompts before pinning. The fixture therefore records
two independently maintained executable implementations, not a transcription.
Only the ordinary chat subset plus release thinking modes are exercised:
tools, developer, latest-reminder, tasks, and continuation stay out of scope.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates/qwen-cli/tests/fixtures/deepseek_v4_0731_chat_fixtures_v1.json"

VLLM_REVISION = "b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07"
VLLM_ENCODER = "vllm/tokenizers/deepseek_v4_encoding.py"
VLLM_ENCODER_SHA256 = "20eb61abe97be7607fd12e2b929faef91743cd2699ad9a4e032b54237d137694"
SGLANG_REVISION = "58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8"
SGLANG_ENCODER = "python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
SGLANG_ENCODER_SHA256 = (
    "012e4dc254c4046f600674eaa799d59dffe5e4a46450da7b4629cf16706fc6c3"
)


def checkout(env_name: str, default: str, revision: str) -> Path:
    path = Path(os.environ.get(env_name, os.path.expanduser(default))).resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_name} checkout missing: {path}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != revision:
        raise SystemExit(f"{path} is at {head}, expected pinned revision {revision}")
    return path


def load_module(name: str, path: Path, expected_sha256: str):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"{path} sha256 {digest} does not match pinned {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assistant(content: str, reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        # vLLM reads `reasoning`; SGLang reads `reasoning_content`. Each
        # encoder ignores the other's key, so setting both drives the same
        # semantic input through both implementations.
        message["reasoning"] = reasoning
        message["reasoning_content"] = reasoning
    return message


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def system(content: str) -> dict:
    return {"role": "system", "content": content}


SINGLE = [user("Hello")]
SYSTEM_SINGLE = [system("Be exact."), user("Hello")]
MULTI = [
    system("Be exact."),
    user("Hello"),
    assistant("Hi!"),
    user("上海 🙂"),
]
MULTI_WITH_REASONING = [
    system("Be exact."),
    user("Hello"),
    assistant("Hi!", reasoning="hidden plan"),
    user("上海 🙂"),
]
MULTI_TWO_ROUNDS = [
    user("one"),
    assistant("first", reasoning="alpha"),
    user("two"),
    assistant("second", reasoning="beta"),
    user("three"),
]
MULTI_MISSING_REASONING = [
    user("one"),
    assistant("done"),
    user("two"),
]

# (name, messages, thinking_mode, reasoning_effort, drop_thinking)
CASES = [
    ("chat_single_user", SINGLE, "chat", None, True),
    ("chat_system_user", SYSTEM_SINGLE, "chat", None, True),
    ("chat_multi_turn", MULTI, "chat", None, True),
    (
        "chat_multi_turn_drops_reasoning_fields",
        MULTI_WITH_REASONING,
        "chat",
        None,
        True,
    ),
    ("thinking_single_user", SINGLE, "thinking", None, True),
    ("thinking_high_single_user", SINGLE, "thinking", "high", True),
    ("thinking_max_single_user", SINGLE, "thinking", "max", True),
    ("thinking_max_system_user", SYSTEM_SINGLE, "thinking", "max", True),
    ("thinking_drop_multi_turn", MULTI_WITH_REASONING, "thinking", None, True),
    ("thinking_drop_two_rounds", MULTI_TWO_ROUNDS, "thinking", "high", True),
    ("thinking_preserve_multi_turn", MULTI_WITH_REASONING, "thinking", "high", False),
    ("thinking_preserve_two_rounds", MULTI_TWO_ROUNDS, "thinking", "high", False),
    (
        "thinking_preserve_missing_reasoning",
        MULTI_MISSING_REASONING,
        "thinking",
        "high",
        False,
    ),
    (
        "thinking_max_preserve_multi_turn",
        MULTI_WITH_REASONING,
        "thinking",
        "max",
        False,
    ),
]


def main() -> None:
    check = "--check" in sys.argv[1:]
    vllm_dir = checkout("DSV4_VLLM_DIR", "~/code/vllm", VLLM_REVISION)
    sglang_dir = checkout("DSV4_SGLANG_DIR", "~/code/sglang", SGLANG_REVISION)
    vllm_encoding = load_module(
        "dsv4_vllm_encoding", vllm_dir / VLLM_ENCODER, VLLM_ENCODER_SHA256
    )
    sglang_encoding = load_module(
        "dsv4_sglang_encoding", sglang_dir / SGLANG_ENCODER, SGLANG_ENCODER_SHA256
    )

    cases = []
    prompts: dict[str, str] = {}
    for name, messages, thinking_mode, reasoning_effort, drop_thinking in CASES:
        rendered = {}
        for source, module in (("vllm", vllm_encoding), ("sglang", sglang_encoding)):
            rendered[source] = module.encode_messages(
                json.loads(json.dumps(messages)),
                thinking_mode=thinking_mode,
                drop_thinking=drop_thinking,
                reasoning_effort=reasoning_effort,
            )
        if rendered["vllm"] != rendered["sglang"]:
            raise SystemExit(
                f"case {name}: vLLM and SGLang disagree\n"
                f"vllm:   {rendered['vllm']!r}\n"
                f"sglang: {rendered['sglang']!r}"
            )
        prompts[name] = rendered["vllm"]
        cases.append(
            {
                "name": name,
                "thinking_mode": thinking_mode,
                "reasoning_effort": reasoning_effort,
                "drop_thinking": drop_thinking,
                "messages": messages,
                "prompt": rendered["vllm"],
            }
        )

    # The release contract makes `high` template-invisible: it must be
    # byte-identical to thinking mode without an effort level.
    if prompts["thinking_high_single_user"] != prompts["thinking_single_user"]:
        raise SystemExit("thinking+high diverged from thinking+None bytes")
    # History under drop-thinking must render byte-identically to chat mode
    # up to the final transition token.
    chat = prompts["chat_multi_turn_drops_reasoning_fields"]
    drop = prompts["thinking_drop_multi_turn"]
    if chat.removesuffix("</think>") != drop.removesuffix("<think>"):
        raise SystemExit("thinking-drop history diverged from chat-mode history")

    fixture = {
        "fixture": "deepseek_v4_0731_chat_fixtures_v1",
        "sources": {
            "vllm": {
                "revision": VLLM_REVISION,
                "file": VLLM_ENCODER,
                "sha256": VLLM_ENCODER_SHA256,
            },
            "sglang": {
                "revision": SGLANG_REVISION,
                "file": SGLANG_ENCODER,
                "sha256": SGLANG_ENCODER_SHA256,
            },
        },
        "cases": cases,
    }
    encoded = json.dumps(fixture, ensure_ascii=False, indent=2) + "\n"

    if check:
        current = OUTPUT.read_text(encoding="utf-8")
        if current != encoded:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as handle:
                handle.write(encoded)
                fresh = handle.name
            raise SystemExit(
                f"{OUTPUT} drifted from regenerated fixtures; fresh copy at {fresh}"
            )
        print(f"{OUTPUT} matches regenerated fixtures ({len(cases)} cases)")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(encoded, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
