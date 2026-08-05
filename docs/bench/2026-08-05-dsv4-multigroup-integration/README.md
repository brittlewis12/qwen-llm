# DeepSeek V4 Multi-Group Selector Integration

Date: 2026-08-05

Status: bounded off-by-default singleton route `GO`; production default `HOLD`.

## Question

The exact 32-group selector and its capacity/visibility crossover had cleared in
test-owned buffers. This checkpoint asks whether the same topology remains exact
and profitable after it owns real session scratch, admission bytes, generation
state, and the production singleton dispatch seam.

## Integration

The ordinary selector remains radix4. A hidden explicit opt-in, sealed at
position zero, authorizes the candidate only when all frozen conditions hold:

```text
query_count = 1
top_k = 512
visible_rows <= capacity_rows <= 262,144
visible_rows >= 196,608
visible_rows >= capacity_rows - capacity_rows / 4
```

Packed prefill, shallow singleton selection, ranked diagnostics, and every
ineligible singleton remain on the existing path. Eligible capacities own five
optional buffers even while routing is disabled: records, partition plan, state,
private byte mask, and private IDs. At 262,144 rows they add 267,432 logical
bytes, all present in admission inventory. The invocation owner cannot produce
zero and fails before any `u32` generation can be reused.

## Model-Free Gate

Three capacity/visibility cells run mixed and all-tied scores. Each case performs
an exact current/candidate check, 40 warm pairs, then 24
current/candidate/current samples through `DeepSeekV4SparseCsaScratch` itself.
Every mask, cache-order ID, count, status, and fresh generation is checked.

```bash
cargo test --release -p qwen-llm --lib \
  deepseek_v4_metal::tests::profile_sparse_csa_multigroup_integrated_gate \
  -- --ignored --exact --nocapture
```

Savings use the faster control median.

| Capacity | Visible | Case | Candidate GPU | GPU saving | Candidate wall | Wall saving |
|---:|---:|---|---:|---:|---:|---:|
| 196,608 | 196,608 | mixed | 0.573 ms | 0.858 ms | 0.715 ms | 0.869 ms |
| 196,608 | 196,608 | tied | 0.829 ms | 0.720 ms | 0.986 ms | 0.729 ms |
| 250,112 | 196,608 | mixed | 0.653 ms | 0.779 ms | 0.848 ms | 0.760 ms |
| 250,112 | 196,608 | tied | 0.913 ms | 0.635 ms | 1.098 ms | 0.634 ms |
| 262,144 | 196,608 | mixed | 0.663 ms | 0.772 ms | 0.830 ms | 0.788 ms |
| 262,144 | 196,608 | tied | 0.927 ms | 0.624 ms | 1.104 ms | 0.627 ms |

All candidate median/p95 values remain below the inherited 1.35/1.40 ms
ceilings. Maximum control drift is 1.874%, and every GPU/wall saving exceeds
0.50 ms.

Two five-warm-pair attempts immediately after the 120.7 GB live campaign are
retained as negative measurement evidence. Their first radix4 bracket walked
down for 24 consecutive samples and failed the 5% control-drift guard while the
candidate stayed stable. Increasing only the untimed stabilization to 40 pairs
made the bracket stationary; no production code or promotion threshold changed.

## Live Gate

The current August 4 IQ3_XXS real weights run at the first eligible singleton
boundary, position 786,431. The fixture is synthetic: every causal arena starts
at zero, the phase is advanced directly, and token ID 35 is repeated in the
fabricated prefix transcript. That state is captured and restored through the
canonical snapshot-v1 path with 5,431,592,700 payload bytes into a full
262,144-row session. This is not real-prompt continuation evidence.

Current A, experimental, and current B each execute two warm and eight timed
one-command tokens from that exact synthetic deep-position fixture. A separate
untimed instrumented token captures all 21 CSA selected-ID/score decisions and
43 MoE routes; the complete decision hash does not describe a timed collapsed
execution.

The experiment is enabled before restore. Every candidate execution advances
the generation owner by exactly 21; both controls advance it by zero. Across all
arms, logits, final normalized hidden bits, causal and prefix digests, committed
tokens, and the complete decision transcript are exact.

| Metric | Current A | Candidate | Current B |
|---|---:|---:|---:|
| GPU median | 181.056 ms | 167.189 ms | 183.631 ms |
| Wall median | 1,352.152 ms | 1,321.883 ms | 1,339.456 ms |

Candidate saving against the faster control is 13.867 ms GPU and 17.573 ms wall,
above the 21 x 0.50 = 10.5 ms gate. GPU/wall control drift is 1.412%/0.943%.
The absolute wall values are not a product decode baseline: repeated 5.43 GB
snapshot restores deliberately drive a 120.7 GB offline promotion process. The
884-second test is therefore a one-time code/device/asset gate, not a routine
development loop. It reports zero swaps.

Memory admission also reconciles: 112,775,036,928 planned bytes including
reserve versus 112,233,611,264 observed at first forward.

## Identity

- Model content: `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`
- Current census: `cbfddbea4260cbaffb02429f0d9593d6e00ec08eb9a5b8d56ea83e8d22860889`
- Metallib: `89f8862d54aa5f2a3aad594f2b791221f2288765f4aa84f6781961296e2d50f0`
- Input causal: `d07671952860a611666e520c9e51b9d419c2b98c56b9cd1f0a830a25feb15093`
- Completed causal: `a87d1f6623bab8e137b43ab2103923347b8de9577ce43e1e95e18f0873a57a7c`
- Logits: `959a5e7bee641cce3eed319a5ebf7d4b73dd7ed6563991daae15597ddde003e5`
- Final hidden: `26d90c291675cf3b1593a7ab563266e372187fd09fb7e6220656ff729da81dda`
- Decisions: `c81842c7a895254defac7473ca0ff70657a134ee647786d5b8a0b61618dbe0bb`

`model-free.log` and `live.log` retain raw timing arrays. The two
`model-free-control-drift*.log` files retain the rejected brackets. Hashes and
byte counts are frozen in `summary.json`.

## Decision

Promote the integrated seam as a qualified, off-by-default current-device
experiment. Do not widen capacity, device scope, packed routing, real-prefix
quality claims, or enable it by default from this packet alone. The model-free
and real-weight synthetic-state gates establish that the real production
consumer executes the candidate, remains bit-exact for this fixture, preserves
snapshot-v1, and clears its whole-token savings threshold.

## Provenance

The live executable is 17,130,712 bytes with SHA-256
`54902cc9f0c2224254dd55c432b02a43e9a8f4098366fe909f9884fc78a39ee8`.
The final model-free executable is 16,672,600 bytes with SHA-256
`27d1a3cd8de7af5ef40bb8af117bfd570bce20acacdf5925a63fb7749681e887`.
Local copies are retained under
`target/evidence/dsv4-multigroup-integration/`; hashes, not ignored target
artifacts, are the durable binding.
The reviewed final Rust source is 1,075,998 bytes with SHA-256
`2afd55c9de26bb2482d6ff3d50ed7d3bad3b3cb835faedf94997ddb9677bf6d7`.

The live run preceded three post-live source-only changes: hiding the public
experiment from generated docs, diagnostics-gating two live-test helpers, and
raising only the model-free profiler's untimed warm pairs from five to forty
with its expected-generation assertion. The production path and live harness
did not change. `summary.json` binds both executables, final source, feature
sets, Rust/LLVM, Metal target, SDK, OS, architecture, and device.

CX review: `019fcf7d-e9d4-7150-b496-e70a31958e80`.
