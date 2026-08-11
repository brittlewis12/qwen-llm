# Exact Q6 B=16 Two-Token Reuse KILL

Date: 2026-08-11

Status: `KILL`; experimental kernel, host wrapper, test arm, and selector deleted.

## Question

The exact Qwen MoE B=16 packet misses its queue-crossover gate by only `0.0272`
token/s. Its exact token-axis Q6_K head reads the 248,320-row output matrix once
per cohort lane. Can one threadgroup process two token rows, reuse each Q6 row,
retain singleton arithmetic, and create practical product-integration margin?

The mechanism gate required at least 2% whole-model candidate throughput.
Anything smaller cannot pay for scheduler/JSONL integration around a candidate
that already sits at the crossover.

## Attribution

The existing `decode-proj-batch` probe measures the A3B head at B=16:

| Mode | GPU ms/B16 step | Aggregate repeated-byte GB/s |
|---|---:|---:|
| 16 singleton Q6_K heads | 13.2133 | 505.2 |
| Q6_K matrix head | 1.7833 | 3742.9 |

The matrix arm changes accumulation order and is not admissible, but its 11.4 ms
gap proves the head is material. The exact arm's repeated-byte rate already
exceeds the 474 GB/s stream anchor, evidence that independently dispatched token
rows receive cache/fabric reuse even before explicit threadgroup grouping.

## Candidate

The temporary T=2 kernel halves grid Y. Each threadgroup visits one Q6 output-row
pair for two activation rows, retaining per token:

- the singleton lane and K-stripe map;
- Q6 bit extraction and scale application;
- four-component accumulation grouping;
- per-lane K iteration order and final `simd_sum` reduction.

It supports odd token and output tails. A release GPU test over B=16, three Q6
blocks, and seven output rows compares both the original token-axis kernel and T=2
against singleton output bits exactly.

## Whole-Model Screen

The source-identical control/candidate/control screen uses the exact qkv/z/out
GDN organization, exact Q6 head arithmetic, a 1,024-token frontier, two warmups,
and 16 measured generated transitions.

| Arm | Candidate wall, ms | Candidate token/s | Exact causal state |
|---|---:|---:|---:|
| control A, T=1 | 115.8303 | 138.133 | yes |
| candidate, T=2 | 116.2271 | 137.662 | yes |
| control B, T=1 | 117.1521 | 136.575 | yes |

T=2 is `1.0022x` the linear midpoint of the controls, a 0.22% movement and far
below the 2% mechanism gate. All 18 same-history transitions have finite,
bit-exact logits and residuals, matching generated-ID hashes, and byte-identical
final active KV/GDN/conv/token state.

## Decision

Delete the candidate. The larger work unit reduces nominal row visits but raises
live activation and accumulator state; the existing grid already obtains enough
cross-token cache/fabric reuse that T=2 does not reduce wall. T=4 has a worse
register-pressure prior and is not authorized by this result.

Reopen the exact head only with a different mechanism that preserves singleton
reduction order while predicting at least 2% charged whole-model movement. Raw
screen outputs remain disposable under `target/profiles/qwen-moe-q6-t2-*`.
