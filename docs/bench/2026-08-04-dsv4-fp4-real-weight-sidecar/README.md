# DeepSeek V4 Real-Weight FP4 Sidecar

Date: 2026-08-04

Status: diagnostics sidecar and selection counterfactual `GO`; production FP4
selection, cache migration, and snapshot v2 remain `HOLD`.

## Scope

This checkpoint replaces the packed shadow's synthetic K input with exact
real-weight lineage from the refreshed DeepSeek V4 Flash-0731 asset. Every CSA
indexer row is packed at its post-Hadamard, pre-F16 publication seam into
separate E2M1 value and UE8M0 scale planes. Per-row status makes publication a
transaction: unavailable, writing, then ready or a frozen failure code.

The query follows the same official BF16-before-amax contract. One preflight
checks exact visibility, all 64 Q statuses, and every visible K status before
the packed scorer and selector can expose IDs. Invalid observer operands produce
a report with no shadow decision. Invalid counterfactual operands poison rather
than commit the causal token.

The counterfactual borrows the FP4 selector's IDs, count, and visibility without
overwriting the authoritative F16 selector records. Selected attention still
reads the established F16 compressed history. This isolates the decision change
from a cache-representation migration.

All Rust allocations, dispatches, and behavior are behind
`dsv4-diagnostics`. The repository's single unconditional Metal build still
embeds the inert kernels in the default metallib; the default Rust graph has no
call edge or sidecar allocation.

## Identity

- Base revision:
  `f7df1aeec29b14049ca2fc34e05aca274ea09b86`.
- Six-file campaign source SHA-256:
  `02c9e295fea1bcd1a64293da57c8e492cd9881250c90f250b0a290936d6797ef`.
- DeepSeek Metal source SHA-256:
  `bd6e7ce08c84dbfad93999ef6008310153fccf5dca6c848e1ccae9d332747b8a`.
- Embedded metallib SHA-256:
  `45c446a9151bae8a217d223554b28b5c6e3a1042d5faa6fe7b01ba72db28ded0`.
- Test executable SHA-256:
  `587285f83affde9fe2233d59de0f7e6eb4b72a1647f2e838d2a0b333d51d3cb2`.
- Refreshed model-content BLAKE3:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`.
- Token transcript SHA-256 over little-endian u32 IDs:
  `5ef390b5ff3dcb4e14c0fb1dc18d703bea7db08f1c2c20ac12a93f56cbb40125`.
- Hardware: Apple M4 Max, registry ID `4294968482`.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`, LLVM 22.1.6.
- Metal: Apple metal 32023.864, target `air64-apple-darwin24.6.0`.
- SDK: macOS 26.2.

The schema-v2 source packet is SHA-256
`bd26a4bfaadd44088cc60074eb07343f0660d5f1586477c23832b279a273a383`.
It is 4,256,143 bytes and deliberately compacted for Git rather than committed
as 225,035 lines. `summary.json` retains identities, gates, hashes, timings,
state domains, aggregate report metrics, and every exchanged cutoff ID. Its
SHA-256 is
`2361ba7c3f91aa3e1ae0101386cf68200a88ce77dd6e1ee0d4d7226d2f1ac05e`.
The short unedited test log is retained as `run.log`.

Runtime compiler strings are PATH observations rather than proof of the build
tools. The exact executable and embedded-metallib hashes bind the artifacts that
actually ran.

## Protocol

One residency executes three fresh sessions in order:

1. control A: authoritative F16 selection plus the FP4 observer;
2. candidate: authoritative scoring plus FP4-ID/F16-cache selection;
3. control B: the same authoritative observer as control A.

Each session advances `[35, 201, 200, 34] * 512` to position 2,048. One packed
four-token chunk captures the first sparse query at position 2,051, then token
35 captures singleton position 2,052. Eight more singleton tokens provide the
repeated GPU bracket. The complete transcript contains 2,061 forwards.

Exact command:

```bash
env -u DSV4_FP4_OBSERVE_ONLY \
  DSV4_MODEL=/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
  DSV4_FP4_PACKET=/Users/tito/code/qwen-llm/target/dsv4-current-fp4-selection-counterfactual-final.json \
  cargo test --release -p qwen-llm --features dsv4-diagnostics \
  --test deepseek_v4_position_zero_live \
  current_deepseek_v4_fp4_selection_counterfactual_packet \
  -- --ignored --exact --nocapture
```

The test finished in 203.60 seconds. It writes the pass packet only after all
quality, determinism, lifecycle, and timing gates pass.

## Operand And Selector Evidence

Every one of the 21 CSA layers in all four captured reports has:

- 64 ready Q heads and every visible K row ready;
- selected count 512 and selector status zero;
- either an exact selected mask or one reciprocal rank-512/rank-513 exchange;
- maximum selected-set symmetric difference two.

| Report | Exact layers | One exchange | Median score rel-RMS | Max score rel-RMS |
|---|---:|---:|---:|---:|
| candidate packed | 8 | 13 | 0.07114 | 0.20362 |
| control packed | 8 | 13 | 0.07121 | 0.20455 |
| candidate singleton | 11 | 10 | 0.09809 | 0.24306 |
| control singleton | 11 | 10 | 0.09825 | 0.24338 |

The original exact-decision falsifier therefore remains failed: official FP4
QAT is not numerically identical to the retained b10222 F16 scorer. No identity
gate was weakened. The separately frozen counterfactual asks whether those
strictly cutoff-local exchanges preserve downstream behavior.

## Whole-Token Evidence

| Endpoint | Argmax control/candidate | Cosine | Relative RMS | Max abs |
|---|---:|---:|---:|---:|
| packed position 2,051 | 35 / 35 | 0.9999999671 | 0.0002605744 | 0.0035267 |
| singleton position 2,052 | 201 / 201 | 0.9999999958 | 0.0000915797 | 0.0019851 |

Frozen gates are argmax preservation, cosine at least 0.999999, relative RMS at
most 0.001, and maximum absolute error at most 0.01. Both endpoints clear every
gate with substantial room.

Control A and B are bit-identical in packed/singleton logits, FP4 reports,
decision transcript, prefix digest, and causal digest. Their logit hashes are:

- packed:
  `cd13112e0e1c20a9d62bb35a20459d487fd792c70c026b0cac541e7f8642f165`;
- singleton:
  `958f3b880ad4dcd424437b0dd092438e7a8ec72e2d8f2f23135102e122539540`.

The candidate consumes exactly 21 packed and 42 cumulative singleton CSA
selections. Domain-separated BLAKE3 traces bind execution kind, position,
layer, exact visibility, and all 512 cache-order IDs for every consumption:

- packed trace:
  `d12af05682f275ce43a5442a40c62c499af280b5446fda839a25e788a75360ea`;
- singleton cumulative trace:
  `cbbecb529beb5f66f63c184c62398b6f8c50e06445733dff341ff7a357814138`.

Candidate causal state differs as expected and is bound under a separate
counterfactual domain. It cannot be exported to or restored from snapshot v1.
An ordinary observer restore invalidates all sidecar status and capture lineage;
re-arming fails immediately, while an ordinary F16 continuation remains valid.

## Timing And Memory

| Arm | Prefix, ms | Packed, ms | Repeated singleton GPU median, ms |
|---|---:|---:|---:|
| control A | 65,340.9 | 895.7 | 46.513 |
| candidate | 66,326.1 | 911.8 | 47.721 |
| control B | 65,219.4 | 904.3 | 46.673 |

Control drift is 0.344%, below the frozen 5% validity limit. Candidate is 1.129
ms, or 2.42%, above the control midpoint. This is not a production performance
falsifier: every arm computes both F16 and FP4 scores and captures full reports,
while the repeated timing tokens span only 513-515 visible compressed rows. The
packet establishes real-weight decision and quality behavior, not the terminal
no-double-score saving projected by the packed matrix shadow.

The refreshed 2,061-forward plan prices 104,202,649,600 residency bytes and
185,581,568 session bytes, or 104,925,102,080 bytes including the 512 MiB
reserve. Diagnostics add 1,190,424 logical session bytes at the 3,073-forward
test geometry. At full context they add 398,481,944 logical bytes; all buffers
are predeclared in admission accounting.

## Correctness Gates

- Transactional publication writes `WRITING`, then payload, then a device
  barrier, then final status.
- Preflight rejects unavailable/writing Q or K, underreported visibility, and
  over-capacity visibility before score or selection can read packed scales.
- Fail-closed score/selector composition yields zero mask/count and status one.
- Packed chunks with more than one sparse query are rejected before token
  staging or session mutation.
- Ineligible observer reports complete and a subsequent ready capture succeeds.
- Restore disables every sidecar and invalidates capture lineage.
- Counterfactual snapshots are rejected without mutating position.
- Feature-on and feature-off memory inventories remain exact and non-aliasing.

Focused release tests pass 19/19 with one explicit profiler ignored. The live
target compiles independently, and default and diagnostics release checks pass.
The pre-campaign source reviews returned `GO`:

- CX session `019fcf7d-e9d4-7150-b496-e70a31958e80`;
- source review session `ses_0306334c8ffeB030bWXC9iRnYV`.

The ignored-test annotation still says three 2,053-token sessions, while the
source constant and packet correctly bind 2,061 forwards. It is retained as a
known cosmetic typo because changing campaign source after execution would
invalidate the source hash.

## Decision

Promote the real-weight sidecar, transactional status contract, observer, and
FP4-ID/F16-cache counterfactual as diagnostics infrastructure. The packet shows
that official FP4 QAT creates only cutoff-local selector exchanges at the first
sparse boundary and preserves downstream logits far inside the frozen gates.

Do not switch production selection or define snapshot v2 from this shallow
double-score packet. The next FP4 promotion must remove authoritative scorer
work in a timed experimental arm, retain one paired dual-score audit endpoint,
and measure a useful deep/current-product state. Retain the F16 indexer cache as
the differential until that packet clears.

Multi-group exact selection may proceed in parallel: the admitted terminal FP4
scorer is already below the retained radix4 selector, and its threshold/tie
contract is independent of the eventual cache ABI. GPU-resident deterministic
packed routing remains the leading TTFT lane.
