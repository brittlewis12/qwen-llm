# DeepSeek Packed Scratch Overlay KILL

Date: 2026-08-11

Status: `KILL`. A phase-disjoint query-to-MoE overlay saves exactly 512 MiB per
session in the tested K160 configuration, but regresses packed prefill. The
experimental implementation was deleted.

## Question

Can the fixed 512 MiB raw-query scratch allocation be reused after the
pre-expert command completes for the routed expert output plus routed and final
hidden rows, reducing the large per-session footprint without moving arithmetic?

The experiment kept compact GPU routing incompatible because that path may merge
the producer and consumer phases into one command. The default separate-buffer
layout remained the control.

## Protocol

- Base: `a6d19b7ad5d5b8f46b55448b02a7557cdad9c87c` plus the bounded uncommitted
  overlay; the code was removed after closure.
- Model: DeepSeek V4 Flash 0731 REAP K160 Q3_K/Q4_K.
- Host: Apple M4 Max, all model/GPU work serialized under the process Metal
  lease, `QWEN_DSV4_RESIDENCY_SET=0`.
- Candidate rollback during the experiment:
  `QWEN_DSV4_PACKED_SCRATCH_OVERLAY=0`.
- Correctness: hash complete stdout for each control/candidate process.
- Endpoints: model prefill, complete pair wall, session allocation delta, and
  peak Metal allocation.

The final concurrency fixture contains two identical 4,201-token prompts,
requests two output tokens each, uses `--concurrency 2`, and disables prefix
fanout so both sessions execute complete private prefills.

## Results

The one-session control/candidate/control screen produced the same stdout
SHA-256, `04faa0b65b45aca62525c1a5ad1e6d7467be3b1e5a9ee2301d7d98dee6d9aab3`.
The candidate averaged about `16,369.9 ms` prefill against `16,219.6 ms`
control, a roughly `0.9%` regression, while reducing the observed session
allocation by exactly `536,870,912` bytes.

The decisive B=2 control/candidate/control bracket produced the same stdout
SHA-256 in all three processes:
`e62ef3aa2baeb77a56421e15fe82fd6bccc8b08a03bb3a6e23b829c3b2594f20`.

| Arm | Private prefill | Pair wall | Session delta | Peak Metal |
|---|---:|---:|---:|---:|
| Control 1 | `32,999.537 ms` | `33,354.710 ms` | `8,727,298,048` | `10,382,446,480` |
| Overlay | `34,150.112 ms` | `34,444.049 ms` | `7,653,556,224` | `9,250,689,200` |
| Control 2 | `33,273.731 ms` | `33,733.855 ms` | `8,727,298,048` | `10,325,922,040` |

Against the interpolated controls, the overlay regresses private prefill by
`3.06%` and pair wall by about `2.7%`. It saves exactly `1,073,741,824` bytes of
two-session allocation and approximately 1 GiB of observed peak Metal footprint.

## Decision

Do not retain shared-resource scratch aliasing as an opt-in. The fixed memory
saving does not pay for slower inference or the added lifetime and compact-route
qualification surface.

Reopen only for chunk-sized allocation, or for another representation that
reduces physical capacity without imposing shared-resource alias cost on the hot
prefill path.
