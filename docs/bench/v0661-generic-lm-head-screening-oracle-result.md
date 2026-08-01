# v0.661 Generic Lm-Head Screening Oracle Result

Status: **INVALID** with `authority=[]`. The sole frozen acquisition stopped
during authenticated GGUF parsing, before runtime construction, Metal model
loading, tokenization, capture, screening, or ledger evaluation. It contains no
lm-head mechanism observation and neither advances nor kills the candidate.

## Sealed Evidence

The packet contains only `self-tests.json`, `artifact-manifest.json`, and
`decision.json`; its preregistered artifact directories are empty. The decision
records:

```text
disposition: INVALID
failure: decode: failed to fill whole buffer
authority: none
```

`decision.json` SHA-256 is
`d4628a76ed85d84018d6e0b4e53d5805bfb5b41398448d6bef13825f82397cbc`.
`artifact-manifest.json` SHA-256 is
`70412eda74485ea7a55008c8605e62424dfdbb5102e5c252f87d9f32e06a436b`.
The manifest authenticates the sole nonterminal file and the decision embeds
the same manifest digest.

The command was launched from clean commit
`b4b4210c3ce2a8cdd3d90908bcdd53fe64c58b5c`. Because the failure preceded the
capture manifest, that association is operator provenance rather than a build
identity persisted inside the packet.

## Cause

`sha256_file_handle` seeks a `File::try_clone` to byte zero and reads the full
model. Darwin clones share one open-file-description cursor, so the retained
descriptor is left at EOF. `GgufFile::from_opened_file` then maps successfully
from the descriptor, but its streaming metadata parser clones the same EOF
cursor without seeking. `GGUFContainer` fails its first four-byte `read_exact`,
which yields the sealed error.

Thus complete file hashing and mmap header prevalidation ran, but the first
opened-GGUF parse did not complete. No second parse, `Runtime::metal`, model
binding, GPU command, or model output was reached.

## Decision

Do not rerun v0.661 or use its false mechanism flags as observations. A
separately preregistered v0.662 may seek the streaming parser descriptor to byte
zero, prove the behavior from an EOF-positioned fixture, preserve all inherited
gates, and use a new schema and packet root. It should also persist build and
executable identity before model parsing so another early stop is
self-contained.

Artifacts:
`target/profiles/v0661-generic-lm-head-screening-oracle-a3b-p1/`.
Post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
