# v0.664 Generic Lm-Head Screening Prompt-Digest Repair

Status: preregistration. v0.663 is sealed `INVALID`; no v0.664 implementation,
packet root, model observation, screening result, or authority exists.

## Inheritance And Question

Inherit v0.661 through v0.663 in full. Preserve the model, prompt, tokenizer
call, production capture path, exact arithmetic, byte ledger, mechanism gates,
decision precedence, parser/readiness contracts, dual-name identity, and narrow
authority.

v0.663 authenticated and loaded the complete model, then failed because its
prompt helper prepended a domain string while the frozen production digest is
raw concatenated signed-i32 little-endian. v0.664 changes only this digest
contract.

Add one shared `qwen-llm` helper with exactly this definition:

```text
SHA256(token[0].to_le_bytes() || ... || token[n].to_le_bytes())
```

It has no domain prefix and no encoded count. Production `qwen`, the existing
Metal sampled-structural fixture, and the oracle prompt gate/evidence must all
call this shared helper. Remove the duplicate production-main implementation.

Rename the oracle's current domain-separated helper so its generated-stream
scope is explicit. Keep its bytes and use unchanged only for per-capture
generated-prefix identity and terminal generated-output identity. Do not alter
`main.rs` generated-token hashing, integrated-grammar formats, or any other
domain-specific digest.

Split the oracle's combined prompt count/digest predicate into distinct errors.
The tokenizer invocation remains `encode(prompt, false)`: the exact target
fixture yields the same frozen 419 IDs under both special-token settings, but
v0.664 claims no broader policy equivalence.

## Repair Gates

Before the sole acquisition:

- shared-helper tests cover empty input and signed `[1,-2,248319]`, whose raw
  digest is
  `3f37364bc87f9ff835c64d4bdb3d993fe35e530097697da8cbacb5e6f92119d5`;
- a test calling the renamed private oracle helper freezes the inherited
  domain-separated digest for that vector as
  `4b93ca7810c97ba477771dba407b0e8b9dd27e743f21d3a1fa9242d01fe06a9d`
  and proves it differs from the raw helper;
- production CLI and oracle prompt call sites use the shared helper, while only
  the two frozen oracle generated-stream call sites retain the private helper;
- all inherited oracle release tests plus the new focused digest test,
  debug/release shared-helper tests, opened-file tests, and the relevant
  production CLI digest test pass;
- `cargo check --locked -p qwen-cli --bin qwen-bench`, formatting, and
  `git diff --check` pass; and
- independent review confirms no model input, model command, arithmetic, ledger,
  disposition, or authority changed.

Every pre-prompt model and identity constant was reached and passed in v0.663.
The post-acquisition vocab-only diagnostic independently reproduced 419 IDs and
both predicted hashes. No other fixture mismatch is presently identified.

The packet schema becomes `qwen-lm-head-screening-oracle/v0664`. The fixed
artifact inventory and readiness/capture binding remain v0.663's. The initially
absent root is
`target/profiles/v0664-generic-lm-head-screening-oracle-a3b-p1`.

## Sole Acquisition

```sh
target/release/qwen-bench lm-head-screening-oracle \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt \
  --tokens 128 --capture-calls 0,1,7,31,63,127 \
  --packet-dir target/profiles/v0664-generic-lm-head-screening-oracle-a3b-p1 \
  --attest-no-other-user-gpu-workload
```

Require the same clean release identity, empty inherited `QWEN_*` controls,
host-validity gates, artifact contracts, and terminal authority. Earlier roots
remain untouched. An unsealed or `INVALID` v0.664 root is consumed and requires
a new preregistration.

Predecessor decision SHA-256:
`759750cfb5a8af2621e3c8c0dc9a8d120ad7a1df46ddeda223451f61e731dbcc`.
v0.663 post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
