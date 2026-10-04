# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Generate GLM-5.3-Flash (`glm4` pre-tokenizer) HF tokenizer fixtures.

CPU-only and offline: reads the local, hash-pinned tokenizer.json and
tokenizer_config.json; nothing is downloaded and no model is executed.

    uv run --offline scripts/reference/generate_glm4_tokenizer_fixtures.py [HF_DIR]
"""

import hashlib
import json
import random
import sys
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "crates/qwen-llm/tests/fixtures/glm4_tokenizer_hf.json"
DEFAULT_DIR = Path("/Volumes/wdblack/weights-archive/glm-5.3-flash-fp8")
HASHES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
}
TEXTS = [
    "",
    "Hello, world!",
    "The capital of France is Paris.",
    # Contractions: HF folds case (and U+017F long s) inside (?i:...).
    "I'm I'M we'd WE'VE it'll They're can't DON'T",
    "'\u017f 'S 'RE 'VE 'LL",
    "'\u017ftop 'Stop it'\u017f",
    "'\u212a 'I\u0307 'D",
    "rock'n'roll o'clock ''' '",
    # Digits group in threes; \p{N} includes No/Nl.
    "1234567890 1234 12 1",
    "x\u00b2+y\u00b2 \u00bd \u00be \u2153 \u216b \u2460\u2461\u2462",
    "\u0661\u0662\u0663\u0664\u0665 \u0966\u0967\u0968\u0969",
    "\uff13\uff0e\uff11\uff14\uff11\uff15\uff19 \uff11\uff12\uff13\uff14",
    "v1.2.3-rc4 0x1F 3.14e-10",
    # CJK and full-width punctuation.
    "\u4f60\u597d\u4e16\u754c\uff0c\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u300c\u3053\u3093\u306b\u3061\u306f\u300d\uff01",
    "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8\uc785\ub2c8\ub2e4.",
    "\uff08\u5168\u89d2\u62ec\u53f7\uff09\u3010\u6807\u9898\u3011\uff1a\u5185\u5bb9\uff1b\u201c\u5f15\u53f7\u201d",
    # Combining marks are \p{M}, not \p{L}: they split letter runs.
    "\u0915\u093f\u0924\u093e\u092c \u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u092e\u0947\u0902 \u0932\u093f\u0916\u093e \u0917\u092f\u093e \u0935\u093e\u0915\u094d\u092f\u0964",
    "cafe\u0301 caf\u00e9 e\u0301\u0301 A\u030a\u0323",
    "\u0301\u0308!a",
    "!\u0301! abc",
    "a\u0301b\u0327c \u0301x",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a",
    "\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd",
    # Emoji, ZWJ/ZWNJ (Cf: punctuation class here, unlike K2).
    "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 family \U0001f469\U0001f3fd\u200d\U0001f52c \u2764\ufe0f \U0001f1fa\U0001f1f8",
    "x\U0001f469\u200d\U0001f4bby",
    "a\u200cb\u200dc",
    " \u200d!",
    # Whitespace runs, tabs, CR LF, trailing spaces, NBSP and friends.
    "\t  leading \r\n\n trailing  ",
    "a\u00a0b\u2028c\u0085d\u3000e",
    "line1\nline2\r\nline3\n\n\n",
    "   ",
    "\n",
    "x  \n  y\t\t\tz   ",
    "\u2009thin\u2003em\u200bzwsp\u180emvs\u00a0\u00a0nbsp",
    "\0a\0\x01\x7f",
    # Code.
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    '```rust\nfn main() {\n    println!("{}", 1 + 2);\n}\n```',
    'json: {"tools":[{"name":"search","parameters":{"query":"hi"}}]}',
    '<div class="x">&nbsp;</div> // comment /* c */',
    "if (a != b && c >= d) { x->y = z::w; }\n\treturn;",
    # ignore_merges: whole pieces that are vocab tokens skip BPE.
    "kohol",
    " kohol",
    "wirkungen",
    "tiquetas",
    "ramientas",
    "kohol wirkungen tiquetas ramientas",
    "Herramientas etiquetas Auswirkungen alkohol",
    # Special (control) and user-defined added tokens.
    "[gMASK]<sop><|system|>You are helpful.<|user|>Hi<|assistant|><think></think>Hello",
    "<|assistant|><think>reason</think><tool_call>search<arg_key>q</arg_key><arg_value>hi</arg_value></tool_call>",
    "<|observation|><tool_response>ok</tool_response><|assistant|>",
    "<|endoftext|><|endoftext|>",
    "text<|user|>more /nothink",
    "[MASK][sMASK]<eop><|begin_of_image|><|image|><|end_of_image|>",
    "<|code_prefix|>a<|code_suffix|>b<|code_middle|>",
    "<think <think>> </think > <|user| [PAD154856]",
    " <|user|> \n<think>\n",
    "Mixed \u4e2d\u6587 and English 123 with \u00e9mojis \U0001f642!",
]
CHUNKS = [
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "a",
    "Z",
    "foo",
    "Bar",
    "'s",
    "'LL",
    "'\u017f",
    "1",
    "12345",
    "\uff11\uff12",
    "\u00b2",
    "\u4e2d\u6587",
    "\u3002",
    "\u0915\u093f",
    "e\u0301",
    "\u0301",
    "\U0001f642",
    "\u200d",
    "\u00a0",
    "\u3000",
    "{",
    "}",
    "::",
    "->",
    '"',
    "kohol",
    "tiquetas",
    "<|user|>",
    "<think>",
    "[gMASK]",
    "<sop>",
]


def generated_texts():
    rng = random.Random(0x6C4D_5300)
    return [
        "".join(rng.choice(CHUNKS) for _ in range(rng.randint(1, 24)))
        for _ in range(32)
    ]


def load(directory, name):
    path = directory / name
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != HASHES[name]:
        raise RuntimeError(f"pinned artifact hash mismatch: {path} {digest}")
    return data


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    raw = load(directory, "tokenizer.json")
    config = json.loads(load(directory, "tokenizer_config.json"))
    document = json.loads(raw)
    assert document["normalizer"] is None
    assert document["model"]["ignore_merges"] is True
    tokenizer = Tokenizer.from_str(raw.decode())
    # Same vocabulary with ignore_merges disabled, to prove which cases
    # actually exercise the whole-piece lookup.
    plain_document = json.loads(raw)
    plain_document["model"]["ignore_merges"] = False
    plain = Tokenizer.from_str(json.dumps(plain_document))
    pattern = document["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    splitter = pre_tokenizers.Split(Regex(pattern), behavior="isolated")

    cases = []
    for text in TEXTS + generated_texts():
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        case = {
            "text": text,
            "pieces": [p for p, _ in splitter.pre_tokenize_str(text)],
            "ids": ids,
            "ids_with_special": tokenizer.encode(text, add_special_tokens=True).ids,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
        }
        plain_ids = plain.encode(text, add_special_tokens=False).ids
        if plain_ids != ids:
            case["ids_without_ignore_merges"] = plain_ids
        cases.append(case)

    record = {
        "generator": "tokenizers==0.23.2",
        "source": str(directory),
        "tokenizer_sha256": HASHES["tokenizer.json"],
        "tokenizer_config_sha256": HASHES["tokenizer_config.json"],
        "normalizer": document["normalizer"],
        "pre_tokenizer": document["pre_tokenizer"],
        "post_processor": document["post_processor"],
        "decoder": document["decoder"],
        "model_options": {
            k: v for k, v in document["model"].items() if k not in ("vocab", "merges")
        },
        "vocab_size": len(document["model"]["vocab"]),
        "special_config": {
            k: config.get(k)
            for k in [
                "add_bos_token",
                "add_eos_token",
                "bos_token",
                "eos_token",
                "pad_token",
            ]
        },
        "added_tokens": [
            {"id": t["id"], "content": t["content"], "special": t["special"]}
            for t in document["added_tokens"]
        ],
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(record, indent=1, ensure_ascii=True) + "\n")
    exercised = sum("ids_without_ignore_merges" in c for c in cases)
    print(f"wrote {OUTPUT} ({len(cases)} cases, {exercised} exercise ignore_merges)")


if __name__ == "__main__":
    main()
