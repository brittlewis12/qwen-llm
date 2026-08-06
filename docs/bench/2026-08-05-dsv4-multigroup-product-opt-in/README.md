# DeepSeek V4 Multi-Group Selector Product Opt-In

Date: 2026-08-05

Status: `GO` for an explicit, off-by-default M4 Max CLI policy. Radix4 remains
the default. This checkpoint adds no new speed, quality, real-prefix, packed,
device, asset, or capacity claim.

## Decision

The 32-group selector already cleared its exact operation, crossover, session
ownership, admission, synthetic real-weight integration, and whole-token gates.
The remaining question was whether that hidden session experiment had a narrow
enough operator contract to leave diagnostics without becoming a default-on
claim.

Expose exactly two CLI values:

```text
--deepseek-v4-multigroup-selector=off
--deepseek-v4-multigroup-selector=qualified-experimental
```

`off` is the default. `qualified-experimental` is process-wide for single-turn
and resident JSONL execution. It is accepted only for a native DeepSeek V4 run
on an exact `Apple M4 Max` device and a request-sized session that can actually
reach the frozen selector band.

The hidden Rust mutator remains doc-hidden. This is a bounded CLI product seam,
not a generally qualified library policy.

## Geometry Contract

The engine owns one geometry check shared by the CLI and selector:

```text
query_count = 1
top_k = 512
196,608 <= physical_capacity_rows <= 262,144
max_visible_rows = forward_limit / 4
max_visible_rows >= 196,608
max_visible_rows >= physical_capacity_rows - physical_capacity_rows / 4
```

The CLI applies device and geometry qualification immediately after session
planning and before snapshot loading, admission, or residency realization. A
786,431-forward plan owns a rounded 196,608-row slab but can expose only 196,607
rows, so it fails. The first accepted budget is 786,432 forwards. One million
forwards maps to 250,112 physical rows and 250,000 reachable rows; the promoted
1,048,576 ceiling maps to 262,144 of each.

The three-quarter term is currently redundant with the minimum-visible and
maximum-capacity bounds, but remains part of the contract as a widening guard.

## Session Contract

- Single-turn seals the policy immediately after session construction and
  before snapshot restore, snapshot publication, or prompt execution.
- Resident JSONL seals every newly constructed request session. Reusing immutable
  weight residency cannot leak selector policy or invocation state.
- Packed prefill and every dynamically ineligible singleton position retain
  radix4. The qualified selector can run only after its complete predicate
  becomes true.
- Snapshot-v1 remains policy-neutral. Selector mode, private scratch, generation,
  and invocation telemetry are session state, not causal state.
- Existing session scratch remains always priced for supported capacities; this
  slice adds no Metal buffer or admission byte and changes no memory plan or
  snapshot ABI.
- The engine counts successful encodes separately for multi-group selection and
  ineligible singleton radix4 fallback. Disabled sessions must report zero for
  both.

Each session emits versioned JSON telemetry at sealing and completion. The
policy record distinguishes requested, sealed, device-qualified, physically
allocated, and maximally reachable state. The completion record reports actual
multi-group and ineligible-singleton radix4 invocation counts so `sealed` cannot
be mistaken for `used`.

## Reused Evidence

This decision reuses, without replaying, the existing packets:

- `docs/bench/2026-08-05-dsv4-multigroup-threshold-ceiling/README.md`
- `docs/bench/2026-08-05-dsv4-multigroup-full-selector/README.md`
- `docs/bench/2026-08-05-dsv4-multigroup-crossover/README.md`
- `docs/bench/2026-08-05-dsv4-multigroup-integration/README.md`

The final integration packet establishes at least 0.624/0.627 ms GPU/wall
saving per eligible layer in the frozen model-free band and 13.867/17.573 ms
whole-token GPU/wall saving in one current-asset synthetic zero-causal-state
fixture at position 786,431. That fixture preserves bit-identical logits and
final normalized hidden values, matching causal, prefix, and compatibility
digests, committed tokens, exact generation ownership, and a separate untimed
exact decision transcript.

That packet explicitly is not real-prompt continuation evidence. Its repeated
5.43 GB restores produced a 120.7 GB, 884-second offline process, so this
product-surface decision does not repeat it absent implementation, asset, or
device drift.

## Claim Limits

This checkpoint authorizes an explicit M4 Max operator opt-in only. It does not
authorize:

- default-on selection;
- another Metal device or a non-Metal backend;
- physical capacity above 262,144 compressed rows;
- packed, multi-query, ranked-output, or shallow multi-group selection;
- real-prompt long-context quality or retrieval claims;
- general current-asset throughput from the synthetic deep fixture; or
- a new speedup claim from this source-only product slice.

Unsupported device, model family, malformed policy, unreachable visibility, or
late/repeated sealing fails closed. There is no force mode.

## Focused Validation

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::multigroup_selector_product_geometry_requires_reachable_visibility \
  -- --exact

cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::multigroup_selector_invocation_telemetry_distinguishes_fallback \
  -- --exact

cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::sparse_csa_multigroup_scratch_is_explicit_and_capacity_bounded \
  -- --exact

cargo test --release -p qwen-cli --bin qwen \
  tests::deepseek_v4_multigroup_selector_cli_contract_is_explicit_and_bounded \
  -- --exact
```

All four focused release tests pass. The CLI test covers default and explicit
values, malformed values, missing generation scope, non-DeepSeek rejection,
M4 Max qualification, a nonqualified Apple device, unreachable rounded
capacity, first reachable capacity, JSONL policy acceptance, versioned telemetry
goldens, and disabled/sealed completion invariants.

The complete active selector family reports seven passed and three explicitly
ignored profilers in 0.22 seconds. The complete CLI binary suite reports 73
passed and two fixture-dependent ignores. Release workspace all-target,
all-feature check and strict `qwen-llm`/`qwen-cli` Clippy both pass, as do
formatting and patch whitespace.

CX design review: `019fcf7d-e9d4-7150-b496-e70a31958e80`.
