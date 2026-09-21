# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.22.2"]
# ///
"""Generate CPU-only evidence, downloading pinned tokenizer metadata if absent.

No model weights are downloaded, converted, or executed. Vocabulary-only GGUFs
are temporary tokenizer test containers, not runnable checkpoint conversions.
"""

import hashlib
import json
from pathlib import Path
import struct
import urllib.request

from tokenizers import Regex, Tokenizer, pre_tokenizers

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "target/profiles/k2-tokenizer"
OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/k2_tokenizer_hf.json"
REVISIONS = {
    "pretrain": "04fb6c8d11cd7d52d8ea81cbc0e286459079b6a5",
    "posttrained": "2c9659a84c4eea6f9f60462221fe762c8c84d75c",
}
HASHES = {
    (
        REVISIONS["pretrain"],
        "tokenizer.json",
    ): "030b626c1d36427eb9620b9e263c391e442957338261c86cd9d7b94b325f1d8c",
    (
        REVISIONS["pretrain"],
        "tokenizer_config.json",
    ): "2481b319071cc584e32c57a7e49deef00753a62b4cba52861d663a00f9673c3e",
    (
        REVISIONS["posttrained"],
        "tokenizer.json",
    ): "838d767b7c9925ff257feb20eaa4299a8e3cc35bb3d805589c373f51d2cc3cb6",
    (
        REVISIONS["posttrained"],
        "tokenizer_config.json",
    ): "068cfdcf2bcef44fd77f935a9fb41b4d45af547fd41b95079817bd40b24fe518",
}
TEXTS = [
    "",
    "Hello, world!",
    "The capital of France is",
    "1234567890 1234 12",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "\t  leading \r\n\n trailing  ",
    "a\u00a0b\u2028c\u0085d",
    "\0a\0",
    "caf\u00e9",
    "cafe\u0301",
    "A\u030a\u0323",
    "\u212b",
    "\ufb01",
    "\u1100\u1161\u11a8",
    "\u0301\u0308!a",
    "!\u0301! abc",
    "a\u200cb\u200dc",
    " \u200d!",
    "x\U0001f469\u200d\U0001f4bby",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0915\u093f\u0924\u093e\u092c",
    "\u4f60\u597d\u4e16\u754c\u65e5\u672c\u8a9e",
    "\u0661\u0662\u0663\u0664\u00b2\u2163",
    "I'm I'M we'd WE'VE it'll",
    "'\u017f 'S 'RE 'VE 'LL",
    "'\u212a 'I\u0307 'D",
    '\\n \\" {} [] -- !!!\n\n',
    "<|begin_of_text|>a<|endoftext|>",
    "<|ifm|begin_of_text|>a<|ifm|endoftext|>",
    "<|ifm|im_start|>user\nHi<|ifm|im_end|>",
    "<ifm|tool_calls><ifm|tool_call>f</ifm|tool_call></ifm|tool_calls>",
    "<|fim_prefix|>before<|fim_suffix|>after<|fim_middle|>",
    "reserved_special_token_think_start e\u0301 reserved_special_token_think_end",
    "e\u0301<|ifm|endoftext|>a\u030a",
    "<|endoftext|><|endoftext|>",
]


def fetch(revision, name):
    path = CACHE / revision / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/IFM/K2-Horizon-7B/resolve/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=90) as source:
            data = source.read(64 * 2**20 + 1)
        if len(data) > 64 * 2**20:
            raise RuntimeError("tokenizer artifact exceeds bounded download")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[(revision, name)]:
        raise RuntimeError(f"pinned artifact hash mismatch: {path}")
    return path


def vocabulary_fixture(document, destination):
    """Tokenizer-only GGUF test container, no model tensors or weights."""
    tokens = [f"[PAD{i}]" for i in range(250624)]
    types = [5] * len(tokens)
    for token, index in document["model"]["vocab"].items():
        tokens[index], types[index] = token, 1
    for token in document["added_tokens"]:
        assert not any(
            token[k] for k in ("normalized", "lstrip", "rstrip", "single_word")
        )
        tokens[token["id"]] = token["content"]
        types[token["id"]] = 3 if token["special"] else 4
    merges = [
        " ".join(m) if isinstance(m, list) else m for m in document["model"]["merges"]
    ]
    metadata = {
        "general.architecture": (8, "k2-horizon"),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "k2-horizon"),
        "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.merges": (9, (8, merges)),
        "tokenizer.ggml.bos_token_id": (4, 0),
        "tokenizer.ggml.eos_token_id": (4, 1),
        "tokenizer.ggml.add_bos_token": (7, True),
        "tokenizer.ggml.add_eos_token": (7, False),
    }

    def encode(kind, item):
        if kind == 8:
            data = item.encode()
            return struct.pack("<Q", len(data)) + data
        if kind == 9:
            subtype, items = item
            return struct.pack("<IQ", subtype, len(items)) + b"".join(
                encode(subtype, i) for i in items
            )
        return struct.pack({4: "<I", 5: "<i", 7: "<?"}[kind], item)

    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, (kind, item) in metadata.items():
        data += encode(8, key) + struct.pack("<I", kind) + encode(kind, item)
    destination.write_bytes(data + b"\0" * (-len(data) % 32))


def main():
    results = []
    for label, revision in REVISIONS.items():
        path = fetch(revision, "tokenizer.json")
        config_path = fetch(revision, "tokenizer_config.json")
        document = json.loads(path.read_bytes())
        config = json.loads(config_path.read_bytes())
        tokenizer = Tokenizer.from_file(str(path))
        splitter = pre_tokenizers.Split(
            Regex(document["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]),
            behavior="isolated",
        )
        vocabulary_fixture(document, path.with_name("vocab-only.gguf"))
        cases = []
        for text in TEXTS:
            cases.append(
                {
                    "text": text,
                    "normalized": tokenizer.normalizer.normalize_str(text),
                    "pieces": [
                        p
                        for p, _ in splitter.pre_tokenize_str(
                            tokenizer.normalizer.normalize_str(text)
                        )
                    ],
                    "ids": tokenizer.encode(text, add_special_tokens=False).ids,
                    "ids_with_special": tokenizer.encode(
                        text, add_special_tokens=True
                    ).ids,
                    "decoded": tokenizer.decode(
                        tokenizer.encode(text, add_special_tokens=False).ids,
                        skip_special_tokens=False,
                    ),
                }
            )
        record = {
            "profile": label,
            "repository": "IFM/K2-Horizon-7B",
            "revision": revision,
            "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "normalizer": document["normalizer"],
            "pre_tokenizer": document["pre_tokenizer"],
            "post_processor": document["post_processor"],
            "decoder": document["decoder"],
            "model_options": {
                k: v
                for k, v in document["model"].items()
                if k not in ("vocab", "merges")
            },
            "special_config": {
                k: config.get(k)
                for k in [
                    "add_bos_token",
                    "add_eos_token",
                    "bos_token",
                    "eos_token",
                    "sep_token",
                    "add_sep_token",
                ]
            },
            "added_token_flags": sorted(
                {
                    json.dumps(
                        {k: v for k, v in t.items() if k not in ("id", "content")},
                        sort_keys=True,
                    )
                    for t in document["added_tokens"]
                }
            ),
            "non_special_added_tokens": [
                t["content"] for t in document["added_tokens"] if not t["special"]
            ],
            "cases": cases,
        }
        results.append(record)
        print(json.dumps({**record, "cases": cases[:2]}, indent=2), flush=True)
    OUTPUT.write_text(
        json.dumps(
            {"generator": "tokenizers==0.22.2", "profiles": results},
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
