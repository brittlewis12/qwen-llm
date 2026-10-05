# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2==3.1.6", "tokenizers==0.23.2"]
# ///
"""GLM-5.3-Flash text-chat template oracle; no weights or model execution.

CPU-only and offline. Reads the local, hash-pinned upstream chat template,
generation config and tokenizer, and the chat template embedded in the
local UD-IQ3_XXS GGUF (read from its metadata, never hashed whole). Nothing
is downloaded.

    uv run --offline scripts/reference/generate_glm53_chat_fixtures.py

Each case renders with the upstream template under transformers' Jinja
settings, and again with the GGUF-embedded template; the record notes
where the two disagree. `native` records what qwen-llm's hand-written
renderer does with the case: render it byte-for-byte, or refuse it.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from jinja2 import TemplateError, sandbox
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/glm53_chat_hf.json"
HF_DIR = Path("/Volumes/wdblack/weights-archive/glm-5.3-flash-fp8")
GGUF = Path(
    "/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/glm5-next/"
    "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf"
)
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
HASHES = {
    "chat_template.jinja": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
    "generation_config.json": "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
}
GGUF_TEMPLATE_SHA256 = (
    "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724"
)


def fail(message):
    raise TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # transformers.utils.chat_template_utils: same signature and defaults.
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def environment():
    # transformers' _compile_jinja_template: ImmutableSandboxedEnvironment,
    # trim_blocks, lstrip_blocks, loopcontrols. The {% generation %} tracker
    # is irrelevant: neither template uses it.
    env = sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = fail
    env.filters["tojson"] = tojson
    return env


def read_pinned(name):
    data = (HF_DIR / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != HASHES[name]:
        raise RuntimeError(f"pinned artifact hash drift: {name} {digest}")
    return data, digest


def gguf_chat_template(path):
    """Return `tokenizer.chat_template` from GGUF metadata (v2/v3)."""
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as f:

        def read(n):
            data = f.read(n)
            if len(data) != n:
                raise RuntimeError("truncated GGUF metadata")
            return data

        def u32():
            return struct.unpack("<I", read(4))[0]

        def u64():
            return struct.unpack("<Q", read(8))[0]

        def string():
            return read(u64())

        def skip(kind):
            if kind in scalar:
                f.seek(scalar[kind], 1)
            elif kind == 8:
                f.seek(u64(), 1)
            elif kind == 9:
                inner, count = u32(), u64()
                if inner in scalar:
                    f.seek(scalar[inner] * count, 1)
                else:
                    for _ in range(count):
                        skip(inner)
            else:
                raise RuntimeError(f"unknown GGUF value type {kind}")

        if read(4) != b"GGUF" or u32() not in (2, 3):
            raise RuntimeError("not a GGUF v2/v3 file")
        u64()  # tensors
        for _ in range(u64()):
            key = string().decode()
            kind = u32()
            if key == "tokenizer.chat_template":
                if kind != 8:
                    raise RuntimeError("tokenizer.chat_template is not a string")
                return string()
            skip(kind)
    raise RuntimeError("GGUF has no tokenizer.chat_template")


def user(text):
    return {"role": "user", "content": text}


def system(text):
    return {"role": "system", "content": text}


def assistant(content, **fields):
    return {"role": "assistant", "content": content, **fields}


HISTORY = [
    user("Q1"),
    assistant("A1", reasoning_content="R1"),
    user("Q2"),
    assistant("A2", reasoning_content="R2"),
    user("Q3"),
]


def cases():
    """(name, messages, variables, native) where native is "render" or "refuse:<why>"."""
    out = []

    def add(name, messages, native="render", **variables):
        variables.setdefault("add_generation_prompt", True)
        out.append((name, messages, variables, native))

    q = [user("What is 2 + 2?")]
    add("effort-absent", q)
    for effort in ["low", "high", "max"]:
        add(f"effort-{effort}", q, reasoning_effort=effort)
    # The template maps every other value to Max; qwen-llm refuses them.
    for effort in ["medium", "none", "xhigh", "Low", "HIGH", ""]:
        add(f"effort-{effort or 'empty'}", q, "refuse:effort", reasoning_effort=effort)
    add("effort-null", q, reasoning_effort=None)

    add("system-user", [system("You are a careful assistant."), user("Hello")])
    add(
        "system-user-unstripped",
        [system("  Be precise.\n"), user("cafe\u0301 / \u4f60\u597d\n")],
        reasoning_effort="high",
    )
    add(
        "system-mid-conversation",
        [system("A"), user("Q"), assistant("x"), system("B"), user("Q2")],
    )
    add("empty-user", [user("")])
    add(
        "literal-markers",
        [
            user(
                "[gMASK]<sop><|system|>s<|user|>u<|assistant|><think>t</think>"
                "<|observation|><tool_call>c</tool_call>"
            )
        ],
    )
    add(
        "unicode",
        [
            system("\U0001f9ed \u65e5\u672c\u8a9e"),
            user(
                "\u0645\u0631\u062d\u0628\u0627 \U0001f468\u200d\U0001f469\u200d\U0001f467"
            ),
            assistant("\u00e9\u00e8", reasoning_content="\u03c0 \u2248 3.14159"),
            user("\u2764\ufe0f"),
        ],
        reasoning_effort="low",
    )

    # Assistant history: reasoning sources and the content strip.
    add("history-reasoning", HISTORY)
    add(
        "history-reasoning-empty",
        [user("Earlier"), assistant("Answer", reasoning_content=""), user("Next")],
    )
    add(
        "history-reasoning-null",
        [user("Earlier"), assistant("Answer", reasoning_content=None), user("Next")],
    )
    add("history-missing", [user("Earlier"), assistant("Answer"), user("Next")])
    add(
        "history-missing-after-reasoning",
        [
            user("Q1"),
            assistant("A1", reasoning_content="R1"),
            user("Q2"),
            assistant("A2"),
            user("Q3"),
        ],
    )
    add(
        "history-reasoning-unstripped",
        [
            user("Earlier"),
            assistant("  \n Answer \t\n", reasoning_content="\n  kept  \n"),
            user("Next"),
        ],
    )
    add(
        "history-content-whitespace-only",
        [user("Earlier"), assistant(" \n\t ", reasoning_content="r"), user("Next")],
    )
    add(
        "history-content-empty",
        [user("Earlier"), assistant("", reasoning_content="r"), user("Next")],
    )
    add(
        "history-content-null",
        [user("Earlier"), assistant(None, reasoning_content="r"), user("Next")],
    )
    # Python's str.strip() also strips U+001C..U+001F (bidi B/S), which are
    # not Unicode White_Space; U+200B and U+180E are stripped by neither.
    add(
        "history-strip-python-whitespace",
        [
            user("Earlier"),
            assistant(
                "\x1c\x1d\x85\xa0\u3000\u2028Answer\u2029\u202f\u205f\x1e\x1f\x0b\x0c",
                reasoning_content="r",
            ),
            user("Next"),
        ],
    )
    add(
        "history-strip-non-whitespace",
        [
            user("Earlier"),
            assistant("\u200bAnswer\u180e\ufeff", reasoning_content="r"),
            user("Next"),
        ],
    )
    # Inline <think> in assistant content when reasoning_content is absent.
    for name, content in [
        ("inline-think", "<think>Inline reasoning</think>\n\nAnswer"),
        ("inline-close-only", "Reasoning text</think>Answer"),
        ("inline-multiple-close", "<think>a</think>b</think>c"),
        ("inline-nested-open", "<think>x<think>y</think>z"),
        ("inline-open-only", "<think>unclosed reasoning"),
        ("inline-empty", "<think></think>Answer"),
    ]:
        add(f"history-{name}", [user("Earlier"), assistant(content), user("Next")])
    add(
        "history-inline-with-reasoning-content",
        [
            user("Earlier"),
            assistant("<think>i</think>visible", reasoning_content="r"),
            user("Next"),
        ],
    )
    add(
        "history-inline-with-null-reasoning",
        [
            user("Earlier"),
            assistant("<think>i</think>visible", reasoning_content=None),
            user("Next"),
        ],
    )

    # clear_thinking: drops reasoning at or before the last user turn.
    add("clear-thinking-true", HISTORY, clear_thinking=True)
    add("clear-thinking-false", HISTORY, clear_thinking=False)
    add(
        "clear-thinking-inline",
        [user("Earlier"), assistant("<think>gone</think>Answer"), user("Next")],
        clear_thinking=True,
    )
    add(
        "clear-thinking-trailing-assistant",
        HISTORY[:4],
        clear_thinking=True,
        add_generation_prompt=False,
    )
    # No user turn: last_user_index is -1, so every assistant keeps reasoning.
    add(
        "clear-thinking-no-user",
        [system("S"), assistant("A", reasoning_content="R")],
        clear_thinking=True,
        add_generation_prompt=False,
    )
    add("no-generation-prompt", HISTORY[:2], add_generation_prompt=False)
    add(
        "no-generation-prompt-user-last",
        HISTORY[:3],
        add_generation_prompt=False,
    )
    # The template treats an empty list as no tools; a document that carries
    # a tools key is refused natively, empty or not.
    add("tools-empty-list", q, "refuse:tools", tools=[])
    # A non-string reasoning_content falls through to the inline split.
    add(
        "history-reasoning-nonstring",
        [user("Earlier"), assistant("Answer", reasoning_content=42), user("Next")],
        "refuse:input",
    )

    # Shapes the template renders but qwen-llm refuses.
    add("assistant-last", HISTORY[:2], "refuse:last_turn")
    add("system-last", [user("Q"), system("S")], "refuse:last_turn")
    add("no-messages", [], "refuse:empty")
    add(
        "developer-role",
        [{"role": "developer", "content": "D"}, user("Q")],
        "refuse:role",
    )
    add(
        "text-parts",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ],
        "refuse:content_parts",
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=GGUF)
    args = parser.parse_args()

    template_bytes, template_sha = read_pinned("chat_template.jinja")
    generation_bytes, generation_sha = read_pinned("generation_config.json")
    generation = json.loads(generation_bytes)
    assert generation["eos_token_id"] == [154820, 154827, 154829], generation
    assert generation["temperature"] == 1.0 and generation["top_p"] == 0.95, generation
    tokenizer_bytes, tokenizer_sha = read_pinned("tokenizer.json")
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())

    gguf_bytes = gguf_chat_template(args.gguf)
    gguf_sha = hashlib.sha256(gguf_bytes).hexdigest()
    if gguf_sha != GGUF_TEMPLATE_SHA256:
        raise RuntimeError(f"GGUF chat template drift: {gguf_sha}")

    env = environment()
    upstream = env.from_string(template_bytes.decode())
    embedded = env.from_string(gguf_bytes.decode())

    records = []
    for name, messages, variables, native in cases():
        record = {"name": name, "native": native, "messages": messages, **variables}
        try:
            rendered = upstream.render(messages=messages, **variables)
        except TemplateError as error:
            record["error"] = str(error)
            records.append(record)
            continue
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        assert ids == tokenizer.encode(rendered, add_special_tokens=True).ids, name
        record["rendered"] = rendered
        record["token_ids"] = ids
        try:
            gguf_rendered = embedded.render(messages=messages, **variables)
        except TemplateError as error:
            gguf_rendered = f"<error: {error}>"
        if gguf_rendered != rendered:
            record["gguf_rendered"] = gguf_rendered
        records.append(record)

    document = {
        "schema": "glm53.text_chat_template_oracle.v1",
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": REVISION,
        "template_sha256": template_sha,
        "gguf_template_sha256": gguf_sha,
        "gguf_template_chars": len(gguf_bytes.decode()),
        "engine": "jinja2 3.1.6 ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols); transformers tojson",
        "tokenizer": "tokenizers 0.23.2, add_special_tokens=False (== True)",
        "tokenizer_sha256": tokenizer_sha,
        "generation_config": generation,
        "generation_config_sha256": generation_sha,
        # Every code point Python's str.strip() removes (str.isspace): the
        # template strips historical assistant content with it.
        "python_isspace": [c for c in range(0x110000) if chr(c).isspace()],
        "cases": records,
    }
    OUTPUT.write_text(json.dumps(document, ensure_ascii=True, indent=1) + "\n")
    disagreements = [r["name"] for r in records if "gguf_rendered" in r]
    print(
        json.dumps(
            {
                "template_sha256": template_sha,
                "gguf_template_sha256": gguf_sha,
                "generation_config_sha256": generation_sha,
                "cases": len(records),
                "errors": [r["name"] for r in records if "error" in r],
                "gguf_disagreements": disagreements,
                "output": str(OUTPUT),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
