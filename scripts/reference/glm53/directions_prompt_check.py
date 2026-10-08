# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2==3.1.6"]
# ///
"""GLM direction study: do the native lens runs feed the model exactly the
prompt the archived llama.cpp sweep fed it?

The archived sweep (`glm-5.3-flash-huihui-rank1/scripts/sweep.py`) asked
llama-server (`--jinja`, the GGUF-embedded template) for
`/apply-template` with one user message and `reasoning_effort: low`, then
appended `</think>`, and tokenized with special-token parsing. This script
renders the same with jinja2 from the template embedded in the local GGUF
(read from shard 1's metadata, never hashing weights), appends `</think>`,
tokenizes with llama.cpp's own `llama-tokenize` (vocabulary only), and
compares every token ID with each native run artifact's `prompt_token_ids`.
CPU-only and offline.

    uv run --offline scripts/reference/glm53/directions_prompt_check.py RUNS_DIR
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import jinja2
import jinja2.sandbox

SHARD1 = "/Volumes/wdblack/weights-archive/glm-5.3-flash-ud-iq3_xxs/glm5-next/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf"
TOKENIZE = Path.home() / "code/llama.cpp/build-glm5/bin/llama-tokenize"


def gguf_string(path, wanted):
    """One string metadata value from a GGUF header (v3)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, _tensors, kv = struct.unpack("<IQQ", f.read(20))
        assert version == 3

        def string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode()

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(kind):
            if kind == 8:
                string()
            elif kind == 9:
                inner, (count,) = (
                    struct.unpack("<I", f.read(4))[0],
                    struct.unpack("<Q", f.read(8)),
                )
                for _ in range(count):
                    skip(inner)
            else:
                f.read(sizes[kind])

        for _ in range(kv):
            key = string()
            (kind,) = struct.unpack("<I", f.read(4))
            if key == wanted:
                assert kind == 8
                return string()
            skip(kind)
    raise SystemExit(f"{wanted} not found")


def main():
    runs = Path(sys.argv[1])
    template = gguf_string(SHARD1, "tokenizer.chat_template")
    # As transformers (and llama.cpp's --jinja) compile chat templates, and as
    # scripts/reference/generate_glm53_chat_fixtures.py does.
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda v, ensure_ascii=True: json.dumps(
        v, ensure_ascii=ensure_ascii
    )
    compiled = env.from_string(template)
    prompts = [
        line.split("\t", 1) for line in (runs / "prompts.tsv").read_text().splitlines()
    ]
    rows, ok = [], True
    for pid, question in prompts:
        text = (
            compiled.render(
                messages=[{"role": "user", "content": question}],
                reasoning_effort="low",
                add_generation_prompt=True,
            )
            + "</think>"
        )
        out = subprocess.run(
            [str(TOKENIZE), "-m", SHARD1, "--ids", "--log-disable", "--stdin"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        llama_ids = json.loads(out.strip().splitlines()[-1])
        native = json.loads((runs / f"{pid}-a0.json").read_text())
        native_ids = native["prompt_token_ids"]
        equal = llama_ids == native_ids
        ok &= equal
        first = next(
            (i for i, (a, b) in enumerate(zip(llama_ids, native_ids)) if a != b), None
        )
        rows.append(
            {
                "id": pid,
                "tokens": len(native_ids),
                "equal": equal,
                "first_difference": first,
                "rendered_tail": text[-60:],
            }
        )
        print(
            f"{pid}: {len(native_ids)} tokens, equal {equal}"
            + ("" if equal else f", first difference at {first}")
        )
    (runs / "prompt-check.json").write_text(
        json.dumps({"rows": rows, "all_equal": ok}, indent=1) + "\n"
    )
    print("all equal:", ok)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
