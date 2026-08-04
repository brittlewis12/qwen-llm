# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

"""Generate DeepSeek V4 0731 chat-encoding byte fixtures.

Every case is rendered by two independently maintained executable release
encoders where the tier is expressible, and pinned only on byte agreement:

- vLLM release-derived encoder at the three-tier reasoning-effort revision
  (`77434861`): the primary contract.
- SGLang release-derived encoder at its pinned pre-three-tier revision. Its
  old two-tier surface maps onto the new contract (old default/`high` ==
  new `low` bytes, old `max` == new `high` bytes), so it cross-checks every
  tier except the new `max` text, which postdates it. Cases the old surface
  cannot express are pinned by structural invariants derived from two-source
  cases plus the sha-pinned effort text; re-pin SGLang and retire those
  waivers when it adopts the three-tier contract.

Encoder sources are read revision-addressed (`git show REV:PATH`) from the
local checkouts, so working-tree state and current HEAD never matter; the
pinned revisions only need to exist in each repository's object store.

Only the ordinary chat subset plus release thinking tiers are exercised:
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

VLLM_REVISION = "77434861904a9f01ea4818fe9f0c7b2a5c05686e"
VLLM_ENCODER = "vllm/tokenizers/deepseek_v4_encoding.py"
VLLM_ENCODER_SHA256 = "25dc8cbd63023db12076a082f320882a36201acd058577644b0e607265c5e2cd"
SGLANG_REVISION = "58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8"
SGLANG_ENCODER = "python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
SGLANG_ENCODER_SHA256 = (
    "012e4dc254c4046f600674eaa799d59dffe5e4a46450da7b4629cf16706fc6c3"
)
BOS = "<｜begin▁of▁sentence｜>"

# The old two-tier SGLang encoder expressed today's tiers under other names.
SGLANG_TIER_REMAP = {None: None, "low": None, "high": "max"}

# Cases whose tier the old SGLang surface cannot express, pinned instead by
# structural invariants asserted after rendering (see the contract-invariant
# block).
STRUCTURALLY_PINNED = {
    "thinking_max_single_user",
    "thinking_max_system_user",
    "thinking_max_preserve_multi_turn",
}


def checkout(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name, os.path.expanduser(default))).resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_name} checkout missing: {path}")
    return path


def load_pinned_module(
    name: str,
    repository: Path,
    revision: str,
    file_path: str,
    expected_sha256: str,
    scratch: Path,
):
    """Imports a module from `git show revision:path`, independent of the
    repository's working tree and HEAD."""
    blob = subprocess.run(
        ["git", "show", f"{revision}:{file_path}"],
        cwd=repository,
        capture_output=True,
        check=True,
    ).stdout
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha256:
        raise SystemExit(
            f"{repository} {revision}:{file_path} sha256 {digest} does not "
            f"match pinned {expected_sha256}"
        )
    staged = scratch / f"{name}.py"
    staged.write_bytes(blob)
    spec = importlib.util.spec_from_file_location(name, staged)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assistant(content: str, reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        # vLLM reads `reasoning`; SGLang and the llama.cpp template read
        # `reasoning_content`. Each implementation ignores the other's key.
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
    ("thinking_default_single_user", SINGLE, "thinking", None, True),
    ("thinking_low_single_user", SINGLE, "thinking", "low", True),
    ("thinking_high_single_user", SINGLE, "thinking", "high", True),
    ("thinking_high_system_user", SYSTEM_SINGLE, "thinking", "high", True),
    ("thinking_max_single_user", SINGLE, "thinking", "max", True),
    ("thinking_max_system_user", SYSTEM_SINGLE, "thinking", "max", True),
    ("thinking_drop_multi_turn", MULTI_WITH_REASONING, "thinking", "low", True),
    ("thinking_drop_two_rounds", MULTI_TWO_ROUNDS, "thinking", "low", True),
    ("thinking_preserve_multi_turn", MULTI_WITH_REASONING, "thinking", "low", False),
    ("thinking_preserve_two_rounds", MULTI_TWO_ROUNDS, "thinking", "low", False),
    (
        "thinking_preserve_missing_reasoning",
        MULTI_MISSING_REASONING,
        "thinking",
        "low",
        False,
    ),
    (
        "thinking_high_preserve_multi_turn",
        MULTI_WITH_REASONING,
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
    vllm_dir = checkout("DSV4_VLLM_DIR", "~/code/vllm")
    sglang_dir = checkout("DSV4_SGLANG_DIR", "~/code/sglang")

    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        vllm_encoding = load_pinned_module(
            "dsv4_vllm_encoding",
            vllm_dir,
            VLLM_REVISION,
            VLLM_ENCODER,
            VLLM_ENCODER_SHA256,
            scratch,
        )
        sglang_encoding = load_pinned_module(
            "dsv4_sglang_encoding",
            sglang_dir,
            SGLANG_REVISION,
            SGLANG_ENCODER,
            SGLANG_ENCODER_SHA256,
            scratch,
        )
        generate(check, vllm_encoding, sglang_encoding)


def generate(check: bool, vllm_encoding, sglang_encoding) -> None:
    cases = []
    prompts: dict[str, str] = {}
    for name, messages, thinking_mode, reasoning_effort, drop_thinking in CASES:
        thinking = thinking_mode == "thinking"
        rendered: dict[str, str] = {}
        rendered["vllm"] = vllm_encoding.encode_messages(
            json.loads(json.dumps(messages)),
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            reasoning_effort=reasoning_effort,
        )
        if not thinking or reasoning_effort in SGLANG_TIER_REMAP:
            rendered["sglang"] = sglang_encoding.encode_messages(
                json.loads(json.dumps(messages)),
                thinking_mode=thinking_mode,
                drop_thinking=drop_thinking,
                reasoning_effort=SGLANG_TIER_REMAP[reasoning_effort]
                if thinking
                else None,
            )
        reference = rendered["vllm"]
        for source, prompt in rendered.items():
            if prompt != reference:
                raise SystemExit(
                    f"case {name}: {source} disagrees with vllm\n"
                    f"vllm:   {reference!r}\n"
                    f"{source}: {prompt!r}"
                )
        if len(rendered) < 2 and name not in STRUCTURALLY_PINNED:
            raise SystemExit(f"case {name} has fewer than two sources")
        prompts[name] = reference
        cases.append(
            {
                "name": name,
                "thinking_mode": thinking_mode,
                "reasoning_effort": reasoning_effort,
                "drop_thinking": drop_thinking,
                "sources": sorted(rendered),
                "messages": messages,
                "prompt": reference,
            }
        )

    # Contract invariants across tiers.
    if prompts["thinking_default_single_user"] != prompts["thinking_low_single_user"]:
        raise SystemExit("thinking default diverged from explicit low tier")
    high_text = vllm_encoding.REASONING_EFFORT_PROMPTS["high"]
    max_text = vllm_encoding.REASONING_EFFORT_PROMPTS["max"]
    if prompts["thinking_high_single_user"] != prompts[
        "thinking_low_single_user"
    ].replace(BOS, BOS + high_text, 1):
        raise SystemExit("high tier is not low tier plus the high effort prompt")
    if prompts["thinking_max_single_user"] != prompts[
        "thinking_low_single_user"
    ].replace(BOS, BOS + max_text, 1):
        raise SystemExit("max tier is not low tier plus the max effort prompt")
    # Max-tier cases have one executable source until SGLang adopts the
    # three-tier contract; pin each structurally to a two-source rendering
    # with the sha-pinned effort text swapped.
    if prompts["thinking_max_system_user"] != prompts[
        "thinking_high_system_user"
    ].replace(high_text, max_text, 1):
        raise SystemExit("max-system does not match high-system with swapped text")
    if prompts["thinking_max_preserve_multi_turn"] != prompts[
        "thinking_high_preserve_multi_turn"
    ].replace(high_text, max_text, 1):
        raise SystemExit("max-preserve does not match high-preserve with swapped text")
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
                "tier_remap": {"low": "default", "high": "max"},
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
