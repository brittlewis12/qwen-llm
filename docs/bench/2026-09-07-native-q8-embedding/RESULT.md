# Dense Q8 native embedding: resource and bitwise oracle PASS

## Result

The existing `QWEN_NATIVE_QUANT_EMBED=1` path removes exactly **3,734,732,800
Metal-allocated bytes** on the installed Qwen3.8-27B-Q8_0 artifact. No new kernel,
copy policy, residency mechanism or default allowlist change is implemented.
Test-only `6e36f048` records actual storage and bitwise correctness before any
endpoint qualification. The separately measured CLI latency packet is held:
`docs/bench/2026-09-07-native-q8-cli/RESULT.md`.

## CPU ownership witness

GGUF metadata and model binding, without Metal initialization, establish:

- Dense64, hidden5120, vocabulary248320, FFN17408, 24 query / 4 KV heads, head256.
- Input embedding is Q8_0 `[5120,248320]`, aligned to its 32-element blocks.
- Input and output embeddings are untied. The distinct LM head is also Q8_0.
- Native Q8 gather is structurally supported; this dense Q8 profile remains
  auto-unpromoted. Current dense Q4_K/Q6_K and MoE Q8 defaults are not changed.
- `mtp_bound=true`, with one declared MTP layer. This is metadata, not evidence
  that MTP inference executes or is qualified by this test.

## Real storage, two separate model processes

A forces native embedding `0`, B forces `1`. Each release test loads only one model.
No simultaneous baseline/candidate residency or whole-model wiring is used.

| Quantity | A: expanded F32 | B: native Q8 |
| --- | ---: | ---: |
| GGUF input embedding bytes | 1350860800 | 1350860800 |
| Resident embedding tensor bytes | 5085593600 | 1350860800 |
| Actual embedding backing bytes | 5085593600 | 1350860800 |
| Actual model allocation delta from context baseline | 32321110016 | 28586377216 |

The full model allocation reduction exactly equals the one-tensor logical
reduction. It is not double-counted with the source mapping or conversion ledger.
Input/output Metal buffers remain distinct; LM-head dtype and bytes are unchanged.
This is an allocation/backing result, not peak RSS, physical residency, process
footprint, startup latency or an explanation of earlier host compression.

## Bitwise witness

Thirteen rows cover first/last vocabulary entries, boundary-adjacent IDs, ordinary
and special IDs, and a repeated row. Scalar and packed gathers agree bitwise in
each arm. Cross-arm SHA256 hashes agree for all gathered F32 values.

A complete code ChatML prompt has 25 tokens. Both arms use the existing ordinary
packed prefill and then 64 greedy outputs, with the final emitted token left
unconsumed. Cross-arm comparisons agree for:

- full prompt-final F32 logits;
- every greedy-step full-logit hash and all 64 generated token IDs;
- complete K/V and GDN state/conv arenas after prefill and after decode;
- snapshot identities, consumed tokens, per-layer positions and arena lengths.

These are full-payload hash comparisons, not cosine thresholds. They witness
bitwise equality for this fixture, not all inputs or execution modes. MTP,
DFlash, sampled decode, tied embeddings and other artifacts are not qualified.

## Authority and attempts

Constructor-only observations are 7582.436 / 2848.223 ms. These fixed-order timings
exclude the runtime prefetch/setup contract and have no endpoint authority. The
later CLI packet demonstrates why they must not be projected onto process wall.

CPU metadata, both release oracles, four native-embedding policy/support tests,
formatting and diff checks pass. The first release compilation exceeds a 120-second
build timeout before any GPU test launches; compilation succeeds with an adequate
timeout. A module-order formatting check is repaired before commit. Neither is
candidate performance evidence.

Raw: `target/profiles/native-q8-embedding/` contains `PROTOCOL.md`, `metadata.log`,
`build.log`, `A.json`, `B.json`, both GPU logs and `compare.py`. Independent review
accepts resource/correctness qualification while requiring actual process-first
output, complete wall and loaded-generation guards before any latency/default claim.
