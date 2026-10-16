# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Freeze public source tokens and identify Maya using GGUF headers only."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

REVISION = "28e45c3a"
SOURCES = [
    "crates/qwen-llm/src/glm5_next_metal/snapshot.rs",
    "crates/qwen-llm/src/glm5_next_metal/packed.rs",
    "crates/qwen-llm/src/glm5_next_metal.rs",
    "crates/qwen-llm/src/glm5_next/memory.rs",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def header(path):
    with path.open("rb") as f:

        def scalar(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            n = scalar("Q")
            if n > 16777216:
                raise ValueError("Unexpected GGUF string length")
            return f.read(n).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = scalar("I"), scalar("Q")
                if count > 1000000:
                    raise ValueError("Unexpected GGUF metadata array length")
                return [value(element) for _ in range(count)]
            return scalar(
                {
                    0: "B",
                    1: "b",
                    2: "H",
                    3: "h",
                    4: "I",
                    5: "i",
                    6: "f",
                    7: "?",
                    10: "Q",
                    11: "q",
                    12: "d",
                }[kind]
            )

        if f.read(4) != b"GGUF" or scalar("I") != 3:
            raise ValueError("Expected GGUF v3")
        tensors, fields = scalar("Q"), scalar("Q")
        meta = {}
        for _ in range(fields):
            key = string()
            meta[key] = value(scalar("I"))
        names = []
        for _ in range(tensors):
            names.append(string())
            dimensions = scalar("I")
            for _ in range(dimensions):
                scalar("Q")
            scalar("I")
            scalar("Q")
        n = f.tell()
        f.seek(0)
        checksum = digest(f.read(n))
    st = path.stat()
    return (
        meta,
        names,
        {
            "path": str(path),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "header_bytes": n,
            "header_sha256": checksum,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shards = sorted(
        args.model.parent.glob("GLM-5.3-Flash-Maya-M-IQ2_S-*-of-00003.gguf")
    )
    if len(shards) != 3 or args.model != shards[0]:
        raise ValueError("Expected all three Maya shards, with shard one selected")
    identities, names, vocab = [], [], None
    for path in shards:
        meta, tensor_names, identity = header(path)
        if meta.get("general.architecture", "glm5-next") != "glm5-next":
            raise ValueError("Unexpected architecture")
        if path == shards[0] and "general.architecture" not in meta:
            raise ValueError("Missing architecture in first shard")
        if "tokenizer.ggml.tokens" in meta:
            vocab = len(meta["tokenizer.ggml.tokens"])
        identities.append(identity)
        names.extend(tensor_names)
    if not vocab or len([n for n in names if n.startswith("blk.45.")]) != 29:
        raise ValueError("Missing expected vocabulary or NextN inventory")
    artifact = {
        "vocab_size": vocab,
        "header_bytes": 24,
        "modelshards": identities,
        "identity_scope": "Full GGUF headers, shard sizes and mtimes; payloads not rehashed",
    }
    (args.out / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    source_meta, pieces = [], []
    for path in SOURCES:
        data = subprocess.check_output(
            ["git", "show", f"{REVISION}:{path}"], cwd=args.repo
        )
        source_meta.append({"path": path, "sha256": digest(data)})
        pieces.append(f"\n--- {path} ---\n" + data.decode())
    parts = {
        "prefix": "[gMASK]<sop><|system|>Reasoning Effort: Low<|system|>"
        "You are reviewing Rust inference code. Ground your answer in the provided source."
        "<|user|>Review these source excerpts:\n",
        "body": "".join(pieces),
        "suffix": "\n[End of excerpt; final file may be truncated.]\n"
        "Explain why truncating the token position alone cannot restore GLM's state after "
        "a rejected speculative token. Identify which state must be preserved or rebuilt, "
        "and distinguish the current full-prefix snapshot from an efficient hypothetical "
        "speculative checkpoint. Give a concrete review of 200-300 words."
        "<|assistant|><think>",
    }
    token_parts = {}
    for name, text in parts.items():
        (args.out / f"{name}.txt").write_text(text)
        run = subprocess.run(
            [
                str(args.tokenizer),
                "-m",
                str(args.model),
                "--stdin",
                "--ids",
                "--no-bos",
                "--no-escape",
            ],
            input=text.encode(),
            capture_output=True,
            check=False,
            timeout=120,
        )
        (args.out / f"tokenize-{name}.log").write_bytes(run.stderr)
        run.check_returncode()
        token_parts[name] = json.loads(run.stdout)
    budget = 32768 - len(token_parts["prefix"]) - len(token_parts["suffix"])
    if budget <= 0 or len(token_parts["body"]) < budget:
        raise ValueError("Source excerpts do not fill the fixed prompt")
    tokens = (
        token_parts["prefix"] + token_parts["body"][:budget] + token_parts["suffix"]
    )
    if len(tokens) != 32768 or not all(0 <= t < vocab for t in tokens):
        raise ValueError("Invalid token fixture")
    payload = (json.dumps(tokens) + "\n").encode()
    (args.out / "prompt.json").write_bytes(payload)
    manifest = {
        "source_revision": REVISION,
        "source_files": source_meta,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": digest(args.tokenizer.read_bytes()),
        "part_lengths": {k: len(v) for k, v in token_parts.items()},
        "body_tokens_used": budget,
        "prompt_sha256": digest(payload),
        "construction": "Parts tokenized separately; body token-prefix truncated to fill 32768; no BOS insertion",
        "purpose": "Fixed greedy latency and stream agreement, not a task-quality benchmark",
    }
    (args.out / "prompt-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
