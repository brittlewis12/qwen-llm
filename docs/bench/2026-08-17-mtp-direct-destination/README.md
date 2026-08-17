# MTP Direct F32 Destination

Date: 2026-08-17

Status: **PREREGISTERED**. No timing result has been admitted yet.

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
2. Run the production MTP benchmark with greedy token IDs and terminal resume
   audit enabled. Require `identical=true`, equal target token digest,
   `resume_audit_pass=true`, and identical MTP bank policy/dtypes/bytes.
3. Keep all GPU/model work serialized through the normal process lease.

Any byte, token, state, or ledger mismatch kills and removes the candidate.

## Timing Protocol

Build one release binary, then run six fresh-process pairs in alternating order:
`AB, BA, AB, BA, AB, BA`, where A is staged and B is direct. Each arm runs:

```sh
target/release/qwen-bench mtp \
  --model "$MODEL" \
  --spec-tokens 1 --mtp-physical-n 2 \
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

## Safety Boundary

No residency set, `requestResidency`, pre-wire, `mlock`, cache-bypass read,
uncached read, or residency-coupled A10B pread path is authorized. The ordinary
pageable GGUF loader remains unchanged. PID 8770 is user-owned and must not be
touched. No process may be killed to conduct this experiment.
