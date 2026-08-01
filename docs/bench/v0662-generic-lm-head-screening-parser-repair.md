# v0.662 Generic Lm-Head Screening Parser Repair

Status: preregistration. v0.661 is sealed `INVALID`; no v0.662 implementation,
packet root, model observation, screening result, or authority exists.

## Inheritance And Question

Inherit `docs/bench/v0661-generic-lm-head-screening-oracle.md` in full,
including its model, prompt, six captures, production path, prophecy firewall,
exact arithmetic, byte ledger, gates, decision precedence, and narrow authority.
The mechanism question and all numeric thresholds are unchanged.

v0.661 failed before model construction because complete hashing left a shared
open-file-description cursor at EOF and the opened-GGUF streaming parser did not
rewind it. v0.662 changes only this parser contract and early provenance:

1. `open_one_shard_file` must seek its metadata-parser descriptor explicitly to
   byte zero before constructing `BufReader`/`GGUFContainer`. Mmap validation
   remains descriptor-bound and offset-independent.
2. A pre-model runtime self-test must open a valid minimal single-shard fixture,
   move a shared clone's cursor to EOF, rename the backing path so the diagnostic
   path is absent, and prove `GgufFile::from_opened_file` still parses the
   retained descriptor from byte zero. This one test jointly covers cursor
   rewind and no path reopen.
3. Add typed `readiness.json`. After exact argument, clean build, empty
   `QWEN_*`, and executable-identity validation, publish it before self-tests or
   any model-file operation. It contains schema, complete build identity,
   executable identity, operator attestation, exact command arguments, and the
   predecessor decision SHA-256. It contains no environment values.

`readiness.json` is mandatory in every complete, coverage, or post-readiness
`INVALID` inventory, and later capture-manifest build/executable identities must
equal it exactly. A failure before readiness publication leaves the root
consumed and unsealed; it must not publish an unauthenticated `INVALID`.

No fallback reopen, path-based parser, removed hash, relaxed identity check,
changed stop vector, extra model, alternate prompt, adaptive screen, or second
acquisition is permitted.

## Repair Gates

Before the sole acquisition:

- the EOF-positioned opened-file regression passes in debug and release;
- the existing path-replacement identity and split-file rejection tests pass;
- all v0.661 oracle release tests pass unchanged;
- `cargo check --locked -p qwen-cli --bin qwen-bench` and `git diff --check`
  pass; and
- independent review confirms that the successor changes no mechanism input,
  arithmetic, ledger, disposition, or authority.

The packet schema becomes `qwen-lm-head-screening-oracle/v0662`. The fixed
artifact inventory adds `readiness.json`; every other v0.661 path and schema is
inherited subject only to the identity equality above. The initially absent root
is
`target/profiles/v0662-generic-lm-head-screening-oracle-a3b-p1`.

## Sole Acquisition

```sh
target/release/qwen-bench lm-head-screening-oracle \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt \
  --tokens 128 --capture-calls 0,1,7,31,63,127 \
  --packet-dir target/profiles/v0662-generic-lm-head-screening-oracle-a3b-p1 \
  --attest-no-other-user-gpu-workload
```

Require the same clean release identity, empty inherited `QWEN_*` controls,
host-validity gates, exact artifact contracts, and terminal authority as v0.661.
The old root remains untouched. An unsealed or `INVALID` v0.662 root is consumed
and requires a new preregistration.

Predecessor decision SHA-256:
`d4628a76ed85d84018d6e0b4e53d5805bfb5b41398448d6bef13825f82397cbc`.
Predecessor post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
