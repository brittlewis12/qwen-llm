# DeepSeek V4 Packed All-IQ3 Route Census

Status: exact current-asset schedule fixture `GO`. This packet records schedule
provenance only; it contains no performance claim.

## Question

The 16 `IQ3_XXS/IQ3_XXS/IQ3_XXS` routed layers still execute 1,036
per-expert bucket chains for an N=128 packed prompt. A model-free direct gate
for a mapped bank-axis replacement needs the exact assignment populations,
source rows, destination slots, and width-32 panels from the accepted product
condition rather than a reconstruction from bucket totals.

Capture those routes once, prove they rebuild the ordinary schedule, then
remove the asset harness and use only the canonical fixture in candidate work.

## Frozen Contract

- Base revision: `8dbd084089236d7e4c9e0bb16c9151d2935faaed`.
- Asset: pinned 97.05 GiB current IQ3_XXS GGUF, content ID
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`.
- Prompt: `[35,201,200,34]` repeated to 128 tokens, little-endian token digest
  `b57816bcb0d5fdf5a8e2ddc7a0afe9e57fb0ca6ffc2b849285e1635d04772843`.
- Execution: one unsampled packed forward with ordinary Rust routing, the
  qualified grouped-IQ2 production policy, and zero grouped-IQ3 invocations.
- Product identity: exact logits, normalized hidden, causal, prefix,
  compatibility, committed-token, model, and prompt digests must match the
  accepted prior packet before fixture emission.
- Cohort: exact IQ3_XXS gate/up/down dtypes and ordered layer IDs
  `1,4,5,8,9,20,24,27,28,29,30,31,32,33,34,40`.
- Per-layer active bucket counts must equal
  `23,38,46,68,73,72,71,79,75,79,84,77,65,62,59,65`.
- Every layer must expose 768 in-range route IDs; each token's six IDs must be
  distinct. Independently reconstructed counts and active buckets must match
  production metadata exactly.
- The emitted JSON carries all counts and slot-major route IDs plus derived
  width-32 panel and padding totals. Count and route hashes bind domain, model,
  prompt, geometry, ordered layers, counts, and routes.

Any mismatch panics before JSON emission. The capture authorizes only a
canonical model-free schedule fixture, not a kernel, timing result, or product
switch.

## Source Freeze

- `deepseek_v4_metal.rs`:
  `666d39bfd7143101b8015877569491a1fa55c9a71497a27bc538b112145ac7f3`
- Release diagnostics test executable:
  `e3885041032a181bc73ccfb9643496277ca133f1421441268df395947b280a94`
- Complete `experiment-source.diff`:
  `58eb87e1fb078b3f65de5f1fdbe1853c7a5cd7f6fbefffed87daa1395271a9ae`

The existing exact route reconstruction test, release diagnostics compilation,
strict release diagnostics library Clippy, formatting, and patch whitespace
pass. CX session `019fd685-acb3-7890-83c9-192cbea48c6e` returns static GO.

## Sole Command

```bash
set -o pipefail && cargo test --release -p qwen-llm \
  --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_all_iq3_route_census \
  -- --ignored --exact --nocapture --test-threads=1 \
  2>&1 | tee \
  "docs/bench/2026-08-06-dsv4-packed-all-iq3-route-census/capture.log"
```

## Result

The sole capture passes in 6.77 seconds in-test. It records 12,288 exact
slot-major routes across 16 layers and reproduces the accepted 1,036 active
layer-expert buckets.

The exact width-32 schedule contains 1,187 panels and 25,696 padded columns:
12,288 useful assignments occupy 32.3505% of 37,984 launched columns. There are
151 continuation panels beyond the first panel for each active expert, or
12.7228% of panel count.

The count payload hashes to
`72b5d2dba179d5d65334ec413bd82df784b9001d545b8c6e7b74aa29f309ff7c`.
The route payload hashes to
`ee106712f42aed80cc559414140327537dd9f256110b6acffee6a1b893319582`.
The extracted evidence JSON and committed fixture are byte-identical.

## Decision

Retain the canonical fixture and its independent validator; remove the
one-shot asset harness. The fixture authorizes construction of the separately
reviewed model-free 16-layer bank-axis all-IQ3 falsifier. It does not authorize
that kernel, infer a saving, or change the production packed-prefill policy.

## Final Provenance

- Capture log:
  `1598af9575169832ea4cc544a67e5718169498e19f8c76168af8d565d78c4cf4`
- Evidence JSON and committed fixture:
  `df9ac5e48e72c9a09b2685462ff52db9540e29b81b3b9c5588046db37cc76091`
- Retained `retained-source.diff`:
  `13833c4fececf41ac27e4b0d11fb268294aae79354a3545913431e8b87ff942e`
- Retained `deepseek_v4_metal/prefill.rs`:
  `cccaf7cc51fa1c028d43a1c8ff3445084fa37f2606cd201f6ba5d137e1aa91b3`
- Retained release diagnostics test executable:
  `cd8c4f96a73e4fd87ec417eae614158f53fc6fc943b437edaebcec507db1ed68`

The retained canonical fixture test, strict release diagnostics library Clippy,
formatting, patch whitespace, and all archived hashes pass after harness
removal.
