# /// script
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Regenerate the small Rust fixture with the pinned upstream function, not a port."""

import ast
import hashlib
import json
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models

root = Path(__file__).resolve().parents[4]
source = root / "scripts/reference/deepseek_v41/fixtures/inference/engram.py"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
)
function = next(
    n
    for n in ast.parse(source.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == "build_compressed_token_map"
)
namespace = {}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
    namespace,
)

# Byte-level BPE tokens exercise accents, case, whitespace, compatibility forms,
# special tokens, and incomplete UTF-8 bytes without fetching a full tokenizer.
byte_values = (
    list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
)
codes = byte_values[:]
extra = 0
for byte in range(256):
    if byte not in byte_values:
        byte_values.append(byte)
        codes.append(256 + extra)
        extra += 1
alphabet = dict(zip(byte_values, map(chr, codes)))
texts = [
    " The",
    "the",
    "THE",
    "caf\u00e9",
    "CAFE",
    "\uff23\uff21\uff26\uff25",
    "e\u0301",
    "\u00c9",
    " ",
    "\t",
    "\r\n",
    "  ",
    "",
    "\u00a0",
    "\ue000",
    "\u0130",
    "\u00df",
    "<special>",
]
raw = ["".join(alphabet[b] for b in text.encode()) for text in texts]
raw += [alphabet[0xC3], alphabet[0xA9], alphabet[0xFF]]
vocab = dict(zip(raw, range(len(raw))))
backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
backend.decoder = decoders.ByteLevel()
backend.add_special_tokens(["<special>"])


class Wrapper:
    backend_tokenizer = backend

    def __len__(self):
        return backend.get_vocab_size(with_added_tokens=True)


lookup, size = namespace["build_compressed_token_map"](Wrapper())
fixture = {
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "tokenizer": json.loads(backend.to_str()),
    "lookup": lookup,
    "compressed_vocab_size": size,
}
Path(__file__).with_suffix(".json").write_text(
    json.dumps(fixture, ensure_ascii=True, indent=2) + "\n"
)

# Optional local tokenizer.json regenerates the digest, never the large lookup.
if len(sys.argv) == 2:
    data = Path(sys.argv[1]).read_bytes()
    backend = Tokenizer.from_str(data.decode())
    Wrapper.backend_tokenizer = backend
    lookup, size = namespace["build_compressed_token_map"](Wrapper())
    summary = {
        "repository": "LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E",
        "revision": "f792019bcb369768220f01088f0f7af230786982",
        "tokenizer_sha256": hashlib.sha256(data).hexdigest(),
        "raw_vocab": len(lookup),
        "compressed_vocab": size,
        "compressed_pad_id": lookup[2],
        "lookup_u32le_sha256": hashlib.sha256(
            b"".join(struct.pack("<I", n) for n in lookup)
        ).hexdigest(),
    }
    Path(__file__).with_name("v41_engram_full_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
