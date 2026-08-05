# DeepSeek V4 Packed Grouped All-IQ3 Widening

Status: exact widening is established, but production promotion is `HOLD`.
The mapped IQ3 kernel, arena proof, exact differentials, and test-only policy
remain as diagnostics evidence. The ordinary Apple M4 Max policy continues to
group only the 25 IQ2_XS/IQ2_XS/IQ3_XXS layers.

## Question

The promoted packed expert path consumes the exact Rust expert-major schedule
for 25 current-asset layers in two grouped dispatches. Sixteen more layers use
IQ3_XXS for gate, up, and down. This packet asks whether those layers can use a
four-dispatch grouped path while preserving every observed output and causal
state bit and delivering a stable incremental win over the promoted IQ2-only
baseline.

The widening deliberately excludes the two MXFP4-down layers and does not
reopen GPU route arithmetic. Its frozen N=128 wall gate requires:

- eight samples per arm under a balanced schedule;
- no more than 5% first-half/second-half drift in either arm;
- at least 5% saving in each matched half; and
- candidate p95 no greater than control p95.

The same promotion decision also requires credible GPU attribution. Exactness
alone authorizes retaining the implementation, not enabling it.

## Candidate

The candidate adds a separate mapped IQ3_XXS matrix entry point rather than
changing the promoted grouped-down kernel ABI or arithmetic body. Each compact
tile carries explicit source-row and destination-slot maps. One thread validates
the complete map into threadgroup memory, all threads cross one barrier, and an
invalid map returns uniformly before input or output pointer arithmetic.

An all-IQ3 layer executes four serial dispatches:

1. mapped IQ3 gate projection into the gate half of the existing output arena;
2. mapped IQ3 up projection into its disjoint up half;
3. the existing exact clamped SwiGLU into grouped-inner scratch; and
4. mapped IQ3 down projection from grouped-inner rows to route slots.

The production geometry has `H=4096`, `F=2048`, and therefore `H=2F`. A host
proof verifies that the gate and up views share one Metal buffer, are exactly
adjacent and disjoint, cover the complete output arena, and contain the exact
required element counts. The down dispatch starts only after gate and up have
been consumed. No allocation or admission total changes: the candidate reuses
the promoted 6,291,456-byte grouped-inner buffer and existing expert output
arena.

The new Rust policy variant exists only in tests and `dsv4-diagnostics` builds.
An ordinary build embeds the separate Metal function in its metallib, but has no
Rust dispatcher that can select it. The current production policy, environment
control, fallback behavior, and snapshot ABI are unchanged.

## Fixture

- Base revision: `da9e5d02c439542d4feab34d7fae81cf051d0213`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Asset size: 104,207,848,032 bytes across four pinned shards
- Prefix pattern: exact token IDs `[35, 201, 200, 34]`
- Prefix sizes: N=12, 32, and 128 from a fresh session
- Continuation: canonical snapshot restore, then exact token ID 35
- Session limit: 129 forwards for the N=128 case

The controls force the already-promoted IQ2 grouped policy: 25 IQ2 grouped
invocations and zero IQ3 grouped invocations. The candidate retains those 25
invocations and adds 16 all-IQ3 invocations, for a calculated 114 grouped
dispatches versus the control's 50.

Representative command shapes:

```text
cargo test --release -p qwen-llm --lib packed_grouped_

cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  packed_grouped_

QWEN_DSV4_PACKED_GROUPED_IQ3_EXPERIMENT=1 \
QWEN_DSV4_PACKED_GROUPED_ONLY_N=128 \
QWEN_DSV4_PACKED_GROUPED_SAMPLES=8 \
QWEN_MATMAT_IQ3_XXS_MM=1 \
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_grouped_expert_integration_packet \
  -- --ignored --exact --nocapture
```

## Correctness

The model-free mapped-projection differential deliberately reverses source rows
so that a source cannot be inferred from `destination_slot / 6`. Its output
matches the current gather plus per-bucket IQ3 matrix path exactly, including
both guards.

The complete four-dispatch differential covers N=1/12/31/32/33/64/128,
deterministic nonzero IQ3 banks, hot and sparse schedules, and expert IDs 0 and
255 at `H=512`, `F=256`, `E=256`, and `K=6`. It checks each stage separately:

- gate is exact while up remains poison;
- up is exact while gate remains unchanged;
- SwiGLU is exact while both projection halves remain unchanged;
- down is exact while grouped-inner remains unchanged; and
- one-command execution is exact with every outer guard intact.

Current-asset N=12/32/128 controls and candidates preserve bit-exact logits,
normalized hidden output, and restored continuation logits. They also preserve
causal, prefix, compatibility, and continuation-causal digests and committed
tokens.

| N | Logit SHA-256 | Causal digest | Continuation causal digest |
|---:|---|---|---|
| 12 | `6c5dbb88a1767fb98fd3e5eb976b310c6f97a9732f1ffabedfb2a9e82a673ed2` | `37dcd04a641cb0cd3763adc7b2267e1423b0ec6ec39cc965c30fddbf16888a9b` | `42c56a22e6887a4818cbc9bfa0a74e5593211922a6a096c84c96037ea5a49b5c` |
| 32 | `7c072707db22b5202757a8fffd3d8cf056564b62620d6b793db916258074a7bb` | `ef14d3f12c306e2a154f48fcb9d10e8b26d0b3e0be904f5a6bb09f43895da532` | `a857a160e70ffe0a9b54505dd021d7a2658da1c80e2ebfd662042c76fdf072b4` |
| 128 | `45c414408cdc688eeeb4f40166013b2ac7261e5a2c2655cdf9f51e818941f2d5` | `3de61f2054ff7e540be5beb113c5c3884875b7018de800696aa58a0393535035` | `7bc950b04b735a172ccb3c5b467488f548407b8b38f497b304d5d4bef7ae0dd8` |

## Timing Campaigns

The first one-sample exact bracket showed directional incremental wall savings
of 12.973%, 12.378%, and 8.679% at N=12, 32, and 128. It established a reason to
measure, not a promotion result.

### Rejected blocked R5

The first five-sample campaign ran complete policy blocks:

```text
control before: [2239.465, 2204.263, 2213.495, 2227.360, 2224.490]
candidate:      [2025.717, 2019.240, 2010.441, 2014.115, 2033.646]
control after:  [3042.224, 2613.958, 2242.376, 2523.730, 2525.943]
```

The candidate saved 9.227% against the faster control median, but control drift
was 12.692%. Policy order confounded the result, so it was rejected.

### Rejected repeated-ABA R5

A second five-sample campaign interleaved repeated A/B/A triplets:

```text
control before: [2455.229, 2236.914, 2237.299, 2231.651, 2249.466]
candidate:      [1996.974, 2004.792, 2026.521, 2435.041, 2042.274]
control after:  [2218.840, 2494.147, 2473.624, 2615.744, 3090.428]
```

The candidate saved 9.421%, but control drift remained 10.857%. This campaign
was also rejected rather than used to relax the gate.

### Sealed balanced R8

The final protocol was preregistered before execution: one untimed
`ABBA BAAB`, then timed `(ABBA BAAB) x 2`, conventional even medians, no more
than 5% drift in either arm, at least 5% saving in both halves, and candidate
p95 no greater than control p95.

```text
control first:  [2226.791, 2659.569, 2593.801, 3026.125]
candidate:      [2421.158, 2019.354, 2295.123, 2428.935,
                 2019.871, 2416.562, 2446.036, 2022.764]
control second: [2650.060, 3065.651, 2641.342, 2240.930]
```

Control medians are 2,626.685 and 2,645.701 ms, with 0.721% drift. Candidate
half medians are 2,358.140 and 2,219.663 ms, with 6.050% drift. Observed
half-specific savings are 10.224% and 16.103%; candidate/control overall p95 is
2,446.036/3,065.651 ms.

Every exactness and invocation assertion passed before timing adjudication. The
candidate-stationarity assertion then failed the frozen 5% gate. The later
saving and p95 values are retained arithmetic, not passed assertions. Per the
preregistered rule, this campaign receives no retry under the same measurement
condition.

### GPU Attribution

A separate single traced A/B/A bracket records direction, not a balanced GPU
promotion gate:

| Metric | IQ2 baseline | IQ2 + all-IQ3 | IQ2 baseline | Saving |
|---|---:|---:|---:|---:|
| Post-route GPU | 1,057 ms | 848 ms | 1,057 ms | 19.773% |
| Affected 16 layers | 369.3 ms | 160.0 ms | 368.1 ms | 56.534% |
| Unchanged 27 layers | 687.6 ms | 688.0 ms | 688.4 ms | -0.058% |
| Total packed GPU | 2,122 ms | 1,905 ms | 2,115 ms | 9.929% |

The total-minus-post-route region remains about 1.06 seconds and does not move
materially. Exactness also passes in this traced bracket. These numbers establish
a credible GPU mechanism, but their single-bracket shape does not substitute for
the failed balanced wall gate.

## Decision

Hold production/default enablement. Preserve the mapped kernel, exact arena
proof, model-free differential, current-asset integration, negative timing
campaigns, and test-only policy as a reusable checkpoint. Ordinary execution
continues to group only the 25 promoted IQ2_XS/IQ2_XS/IQ3_XXS layers.

Do not retry the same timing condition or build a fused IQ3 gate/up kernel now.
Reopen production widening only after a materially changed measurement
condition can support one preregistered balanced GPU-and-wall gate with both
arms at or below 5% drift.

The higher-value next measurement is attribution of the roughly 1.057-second
pre-expert GPU region into attention, compressor/mHC, chronological row work,
and router projection. Then attribute the unchanged roughly 688 ms post-route
region before changing it. Keep the two MXFP4-down layers and GPU route
arithmetic unchanged.

## Evidence

Final reviewed source SHA-256:

- `deepseek_v4_metal.rs`:
  `789ec182abf2d71a8ce98b16fd27a4cad296267b96c473c942723130fa9a6e38`
- `prefill.rs`:
  `1f294bf299324f4b4a2445cb15bf216147b69bf4721e1d7ff051c2d529380bf8`
- `mat_mat_iq3.metal`:
  `5e81b574ee9cf1636513d4945fc3e87b2540b23d1bbc2a96dfc2924aeb0db8b2`

Final release diagnostic test executable SHA-256:
`4fff50420702dbf41ffcbda0a75837f49f265a45a0b5effc3d97e28766257421`.

Local raw logs and SHA-256:

- First exact bracket: `integration-first.log`,
  `41f94fd9480355df8d69c8a8c4c50fee1e4c15142adb10ca1c6c7175eed260ea`
- Rejected blocked R5: `integration-n128-r5.log`,
  `c42b85c936ed18d8e26a78de85381d538c9a62806e8af8e30627fb8b0b6beeef`
- Rejected repeated-ABA R5: `integration-n128-r5-interleaved.log`,
  `44fba9d72f30c1eb328ade1f3a47a80ffefad306bc3848e951096816554f38a8`
- Sealed balanced R8: `integration-n128-r8-balanced.log`,
  `db479ee1fdcbc8c6f7466da76d225e1ea2fd828d96d5de230b6a0a5fffb62fb0`
- Traced bracket: `integration-n128-trace.log`,
  `d79bc4744161756f55caa46a2cf718924ead9c8b0bdfac6b0d7f378f95b472bd`

The logs live under `target/profiles/dsv4-packed-grouped-iq3/`. Focused default
and diagnostic release tests, ordinary and diagnostic session-memory inventory,
formatting, patch whitespace, and strict release workspace all-target/all-feature
Clippy pass. CX design and final review session:
`019fcf7d-e9d4-7150-b496-e70a31958e80`.
