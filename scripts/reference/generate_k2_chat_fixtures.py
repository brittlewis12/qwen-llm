# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2==3.1.6", "tokenizers==0.22.2"]
# ///
"""Pinned IFM chat/tool Jinja and CPU tokenizer oracle; no weights or model execution."""

import hashlib
import argparse
import json
import math
import random
import struct
from pathlib import Path
import urllib.request

from jinja2 import nodes, sandbox, TemplateError
from jinja2.ext import Extension
from tokenizers import Tokenizer
from generate_k2_tokenizer_fixtures import fetch, REVISIONS, CACHE


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        token = next(parser.stream)
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(
            self.call_method("passthrough"), [], [], body
        ).set_lineno(token.lineno)

    def passthrough(self, caller):
        return caller()


def fail(message):
    raise TemplateError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Generate separate native-tool contract fixtures",
    )
    args = parser.parse_args()
    revision = REVISIONS["posttrained"]
    root = Path(__file__).resolve().parents[2]
    tokenizer = Tokenizer.from_file(str(fetch(revision, "tokenizer.json")))
    config = json.loads(fetch(revision, "tokenizer_config.json").read_bytes())
    path = CACHE / revision / "chat_template.jinja"
    url = f"https://huggingface.co/IFM/K2-Horizon-7B/resolve/{revision}/chat_template.jinja"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("template exceeds download bound")
        path.write_bytes(data)
    source = path.read_bytes()
    generation_path = CACHE / revision / "generation_config.json"
    if not generation_path.exists():
        with urllib.request.urlopen(
            f"https://huggingface.co/IFM/K2-Horizon-7B/resolve/{revision}/generation_config.json",
            timeout=90,
        ) as response:
            generation_bytes = response.read(4097)
        if len(generation_bytes) > 4096:
            raise ValueError("generation metadata exceeds bound")
        generation_path.write_bytes(generation_bytes)
    generation_bytes = generation_path.read_bytes()
    if (
        hashlib.sha256(generation_bytes).hexdigest()
        != "2da7d47641f4509da4ae47711e31d8b5f0f3f801ee08d87e7e9f07f814bdc4a3"
    ):
        raise ValueError("pinned generation metadata digest drift")
    generation = json.loads(generation_bytes)
    assert generation == {"bos_token_id": 0, "eos_token_id": [1, 250019]}
    digest = hashlib.sha256(source).hexdigest()
    if digest != "a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590":
        raise ValueError("pinned template digest drift")
    env = sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[Generation, "jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = fail
    env.filters["tojson"] = lambda value, **kw: json.dumps(
        value, ensure_ascii=False, **kw
    )
    template = env.from_string(source.decode())
    cases = []

    def case(name, messages, effort="high", **options):
        variables = {
            "messages": messages,
            "reasoning_effort": effort,
            "add_generation_prompt": True,
            **options,
        }
        record = {"name": name, **variables}
        try:
            rendered = template.render(**variables, bos_token=config["bos_token"])
            bos = config["bos_token"]
            assert rendered.startswith(bos)
            body = rendered[len(bos) :]
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            assert ids == tokenizer.encode(body, add_special_tokens=True).ids
            record.update(rendered=rendered, body=body, token_ids=ids)
        except TemplateError as error:
            record["error"] = str(error)
        cases.append(record)

    for effort in ["high", "medium", "low", "none", "xhigh"]:
        case(
            "effort-" + effort, [{"role": "user", "content": "What is 2 + 2?"}], effort
        )
    case(
        "system-unicode",
        [
            {"role": "system", "content": "  Be precise.\n"},
            {"role": "user", "content": "cafe\u0301 / \u4f60\u597d\n"},
        ],
    )
    case(
        "literal-markers",
        [
            {
                "role": "user",
                "content": "<|ifm|begin_of_text|><|ifm|endoftext|><|ifm|im_start|>user<|ifm|im_end|><ifm|think>not history</ifm|think><ifm|tool_calls>literal</ifm|tool_calls>",
            }
        ],
    )
    case("empty-user", [{"role": "user", "content": ""}])
    for alias in [
        "think",
        "think_fast",
        "think_faster",
        "reasoning_content",
        "reasoning",
    ]:
        for content in ["", "A prior calculation.\n"]:
            case(
                f"history-{alias}-{'empty' if not content else 'text'}",
                [
                    {"role": "user", "content": "Earlier"},
                    {"role": "assistant", "content": "Answer", alias: content},
                    {"role": "user", "content": "Continue"},
                ],
                "medium",
            )
    case(
        "history-priority",
        [
            {"role": "user", "content": "Earlier"},
            {
                "role": "assistant",
                "content": "Answer",
                "think": "",
                "think_fast": "ignored",
                "reasoning": "ignored too",
            },
            {"role": "user", "content": "Continue"},
        ],
    )
    case(
        "history-missing",
        [
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Continue"},
        ],
    )
    case(
        "history-nonstring",
        [
            {"role": "assistant", "content": "Answer", "think": 42},
            {"role": "user", "content": "Continue"},
        ],
    )
    if args.tools:
        from k2_tool_fixture_cases import add_cases

        cases.clear()
        add_cases(case)
        from k2_tool_schema_cases import add_schema_cases

        add_schema_cases(case)
        for record in cases:
            if "error" in record:
                continue
            variables = {
                k: v
                for k, v in record.items()
                if k not in {"name", "rendered", "body", "token_ids"}
            }
            module = template.make_module(
                {**variables, "bos_token": config["bos_token"]}
            )
            tools = variables.get("tools") or variables["messages"][0].get("tools", [])
            call_format = variables.get("tool_call_format", "xml")
            first = variables["messages"][0]
            system_content = (
                first.get("content", "") if first["role"] == "system" else ""
            )
            record["tool_system"] = str(
                module.render_system_with_tools(
                    tools,
                    system_content,
                    variables.get("tool_presentation_format", "markdown"),
                    call_format,
                )
            )
            record["tool_definitions"] = str(
                module.render_tool_presentation(
                    tools, variables.get("tool_presentation_format", "markdown")
                )
            )
            record["call_blocks"] = [
                str(
                    module.render_tool_calls_block(
                        message["tool_calls"], call_format, tools
                    )
                )
                for message in variables["messages"]
                if message.get("tool_calls")
            ]
            record["tool_results"] = [
                str(module.render_tool_response_messages(message["content"]))
                for message in variables["messages"]
                if message["role"] == "tool"
            ]
    document = {
        "schema": "k2.no_tools_template_oracle.v1",
        "revision": revision,
        "template_url": url,
        "template_sha256": digest,
        "engine": "jinja2 3.1.6 ImmutableSandboxedEnvironment; generation blocks are output-preserving",
        "tokenizer": "tokenizers 0.22.2; pinned tokenizer metadata from generate_k2_tokenizer_fixtures.py",
        "bos": config["bos_token"],
        "generation_config": generation,
        "generation_config_sha256": hashlib.sha256(generation_bytes).hexdigest(),
        "cases": cases,
    }
    output = root / "crates/qwen-llm/tests/fixtures/k2_chat_hf.json"
    if args.tools:
        document["schema"] = "k2.tools_template_oracle.v1"
        numbers = [
            1.2345678901234567e-7,
            1.2345678901234567e20,
            2.2250738585072014e-308,
            1.7976931348623157e308,
            5e-324,
            -5e-324,
            0.0,
            -0.0,
        ]
        for boundary in [1e-4, 1e16, 1e-5, 1e15]:
            numbers.extend(
                [
                    math.nextafter(boundary, 0),
                    boundary,
                    math.nextafter(boundary, math.inf),
                ]
            )
        rng = random.Random(0x4B32)
        while len(numbers) < 276:
            number = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
            if math.isfinite(number):
                numbers.append(number)
        document["json_number_oracle"] = [
            {"value": number, "expected": json.dumps(number, ensure_ascii=False)}
            for number in numbers
        ]
        document["empty_call_block"] = str(
            module.render_tool_calls_block([], "xml", [])
        )
        output = output.with_name("k2_tools_hf.json")
    output.write_text(json.dumps(document, ensure_ascii=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "template_sha256": digest,
                "trimmed_template_sha256": hashlib.sha256(source.strip()).hexdigest(),
                "generation_config_sha256": hashlib.sha256(
                    generation_bytes
                ).hexdigest(),
                "cases": len(cases),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
