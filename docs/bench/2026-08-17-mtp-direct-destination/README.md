# MTP Direct F32 Destination

Date: 2026-08-17

Status: **GO** after control-only Amendment A1. The default-on direct destination
passes every amended correctness, timing, order, and memory gate.

## Question

The default A3B MTP MoE policy dequantizes each expert bank into a host
`Vec<f32>`, then copies that complete vector into final Shared Metal storage.
Can the existing checked dequantizer instead populate the final Metal buffer
directly, preserving every output bit while deleting the temporary and copy?

This is a cold load-path representation deletion, not a kernel, quantization,
or inference-semantic change. F32 source tensors retain their existing path.

## Geometry And Prior

Fixture:

`/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`

The default-F32 MTP banks are:

| Bank | Source | Shape | Final F32 bytes |
|---|---|---|---:|
| gate | Q4_K | `[2048, 512, 256]` | 1,073,741,824 |
| up | Q4_K | `[2048, 512, 256]` | 1,073,741,824 |
| down | Q5_K | `[512, 2048, 256]` | 1,073,741,824 |
| total | - | - | 3,221,225,472 |

Only one bank's host temporary is live at a time, so the predicted peak-memory
deletion is about 1 GiB. The earlier 256 MiB primitive floor measured a 12.261
ms paired saving and a 268,451,840-byte RSS deletion. Linear scaling therefore
screens this candidate near 137 ms across three banks, but that extrapolation
has no acceptance authority.

## Candidate

- Control: `QWEN_MTP_DIRECT_F32_DEST=0`, preserving host dequantization plus
  `MetalTensor::from_bytes`.
- Candidate: default-on `QWEN_MTP_DIRECT_F32_DEST=1`, allocating final writable
  F32 Metal storage and calling `dequant_to_f32_into` against it.
- The direct path must check element and byte arithmetic, destination range,
  offset, and F32 alignment before exposing a `MaybeUninit<f32>` slice.
- The environment variable is a rollback lever and is recorded in benchmark
  output; tests call the two implementations explicitly rather than mutating a
  process-cached flag.

## Correctness Protocol

1. Load each of the three banks sequentially through the staged and direct
   implementations, hashing every final byte. Require equal shape, dtype,
   length, and BLAKE3 digest for every bank.
2. Run the production normal lazy-D1 MTP benchmark with greedy token IDs.
   Require `identical=true`, equal target token digest, and identical MTP bank
   policy/dtypes/bytes. This path consumes the loaded MTP banks but does not
   invoke the independently failing packed-N2 terminal-state assay described in
   Amendment A1.
3. Keep all GPU/model work serialized through the normal process lease.

Any byte, token, or ledger mismatch kills and removes the candidate.

## Amendment A1: Packed-N2 Baseline Is Invalid For This Gate

After the protocol-only commit and exact bank oracle, the first staged control
arm ran; no candidate arm had run. It emitted the exact target token stream and
wrote a complete result, but the existing A3B packed-N2 terminal resume audit
failed after 16 continuation steps:

- KV cosine `0.9994732413`, below the assay's `0.99999` threshold;
- continuation max-abs `1.6023061`, above `0.05`;
- GDN state max-abs `0.35898465`, above `0.01`;
- continuation argmax still equal.

The staged control therefore cannot satisfy the original absolute gate. This is
an assay/baseline failure independent of destination construction; the direct
candidate had not been executed, and the full-byte bank oracle already proves
the two constructors equal. The invalid control sample is excluded from timing
and preserved under the external run root.

A1 restarts all six pairs from zero and removes `--mtp-physical-n 2`, selecting
the normal lazy-D1 product path. The exact byte oracle remains the primary
correctness gate. The production run adds target-stream and bank-ledger checks;
it makes no terminal-resume claim. The load timing and memory decision rules are
unchanged.

## Timing Protocol

Build one release binary, then run six fresh-process pairs in alternating order:
`AB, BA, AB, BA, AB, BA`, where A is staged and B is direct. Each arm runs:

```sh
target/release/qwen-bench mtp \
  --model "$MODEL" \
  --spec-tokens 1 \
  --tokens 16 --no-warmup \
  --include-token-ids --output "$OUT"
```

`/usr/bin/time -l` wraps every arm. The implementation records `mtp_load_ms`
around `MetalMtpHead::load`, excluding base-model load and later inference.

Primary endpoints:

- paired `control_mtp_load_ms - candidate_mtp_load_ms`;
- maximum RSS and peak footprint deletion;
- exact semantic gates above.

Whole-process wall time is supporting evidence because base-model load and
inference are outside the changed boundary.

## Decision Rule

Retain only if all correctness gates pass and:

- candidate wins at least five of six pairs;
- median paired MTP-load saving is positive in both AB and BA strata;
- overall median paired MTP-load saving is positive; and
- median maximum-RSS deletion is at least 512 MiB.

Otherwise remove the implementation and bank the negative result. A noisy
whole-process wall result does not veto a passing isolated load boundary, but
it cannot be promoted as an end-to-end speed claim.

## Result

All six fresh-process pairs completed under one exact build/runtime source-state
digest. The direct path won every pair:

| Endpoint | Staged | Direct | Candidate result |
|---|---:|---:|---:|
| Median MTP load | 579.631 ms | 314.819 ms | 265.030 ms paired saving |
| MTP load ratio | 1.000x | 0.543x | **1.841x speedup** |
| Whole-process wall | 3.670 s | 3.455 s | 0.270 s paired saving |

The marginal medians imply a **45.69%** load reduction; the primary median
paired saving is 45.72% of the control median. Candidate wins are `6/6`; AB
median saving is `265.515 ms` and BA median saving is `264.545 ms`.
Whole-process wall tracks the isolated deletion, but remains supporting evidence
rather than a general inference-speed claim.

Memory is consistent with the deleted representation:

| Endpoint | Staged median | Direct median | Median paired deletion |
|---|---:|---:|---:|
| Maximum RSS | 49,251,901,440 B | 48,252,731,392 B | **999,383,040 B** |
| Peak footprint | 26,608,315,408 B | 25,970,625,060 B | **637,075,972 B** |

The RSS deletion is 953.09 MiB, consistent with removing one sequential 1 GiB
host bank temporary. The implementation also deletes 3 GiB of aggregate bank
population copies across gate, up, and down.

## Exactness

The ignored fixture oracle hashes every byte after loading each 1 GiB bank by
both constructors. Shape, F32 dtype, byte count, and digest all match:

| Bank | Source | BLAKE3 |
|---|---|---|
| gate | Q4_K | `860565f930abb0e362f96e1662f3d7905915b268fb972da161ec8363b89580bd` |
| up | Q4_K | `2f35551e2ea074a75bf7bed23ecb5bdc8739b2063c7d28df3380fb03b6a4e2dd` |
| down | Q5_K | `8983881b26d9eb1e6cd0ab34e8126ead7971b57c251ab8e217a4de67d1a43559` |

All 12 production arms report `identical=true`, the same F32/F32/F32
3,221,225,472-byte bank ledger, and target token digest
`3af19c4e6c6bc7ca867fd41488b65ef22933de2c31e6c5541c43654591a856a6`.
Every arm's build and runtime source digest is
`git-source-sha256-v2:500f12d186993a989dd51a6a20dbd4a25e0ab8f4f00d8052e8d88081bbe48dc7`.
The stable benchmarked implementation commit is
`c918ff0dc3987c88317f225858d31f2b62430eec`; the evidence follow-up changes no
runtime or harness code.
The oracle embeds that same build identity, a timestamp inside its manifest
event, and test-binary SHA-256
`e447b4f9919e8e362b0ca59a2edc1352215139d9a29bdc694df756e0904bf283`.

Authority is narrow: timing covers this A3B fixture's default-F32 MTP load;
production semantics cover one 16-token lazy-D1 prompt; full-byte evidence
covers the three converted expert banks. This does not claim a native-bank
speedup, an inference-throughput change, or coverage of every possible MTP
source dtype/model geometry.

The final chronology brackets build identity before and after the oracle and
campaign. Its staged packed-N2 control reproduces Amendment A1's exact failure
as event 2, before the oracle and every direct arm; it does not enter timing
vectors. `results.json` retains the full chronology, oracle, excluded control,
and admitted packets. `analyze.py` rejects source drift, fixture absence,
schedule drift, non-treatment environment drift, protocol drift, or an
unauthenticated token fixture rather than emitting a disposition.

## Retained Change

- Non-F32 MTP tensors dequantize into final writable Shared Metal storage.
- `QWEN_MTP_DIRECT_F32_DEST=0` retains the staged rollback path.
- `qwen-bench mtp` records isolated MTP load time, direct-path state, and target
  token digest.
- F32 source tensors and native MTP bank policies keep their prior storage path.

## Safety Boundary

No residency set, `requestResidency`, pre-wire, `mlock`, cache-bypass read,
uncached read, or residency-coupled A10B pread path is authorized. The ordinary
pageable GGUF loader remains unchanged. PID 8770 is user-owned and must not be
touched. No process may be killed to conduct this experiment.
