# Dense serial-tail scratch omission: KEEP for resource reduction

Baseline `5703ef75`; candidate `6c2a3ba1`. Separate release binaries on Apple
M4 Max, ordinary loading, serial A-B-B-A per lane. All acquisition attempts,
frozen requests/protocol, SSE events, host observations, and the committed-source
allocation inventory remain under the dedicated worktree's
`target/profiles/serial-tail-scratch/`.

## Mechanism and review

Before allocation, the immutable retained prefix lookup exposes its restored
state position. Dense requests requiring 1-48 actual forwards now omit matrix
prefill scratch from both admission and allocation. Exact-final-logit hits stay
scratch-free; zero rows without logits, MoE, and 49+ rows retain the old plan.
The existing serial/packed execution boundary and arithmetic do not change.

The source review specifically challenged pending tokens, cache eviction,
capture completeness, and fallback admission. Matched length is not consumed
length: a pending token counts among the forwards. The lookup pins its snapshot
through allocation. Fresh DFlash capture failure uses the same allocation helper
on retry, avoiding a previously unpriced packed-scratch allocation in the new
serial-only lane. No shared scratch pool, width retile, or KV capacity change.

## Actual allocation inventory

Committed-source test on Qwen3.8-27B Q8_0, with capacity prompt+128. Values are
Metal `current_allocated_size` deltas for the real scratch-plus-sequence
allocator. Each arm returns to its prior counter after drop.

| Prompt shape | Old bytes | Serial-only bytes | Removed bytes |
| --- | ---: | ---: | ---: |
| 32 | 215252992 | 194887680 | 20365312 |
| 8208 | 2041364480 | 730710016 | 1310654464 |
| 32784 | 5740740608 | 2341322752 | 3399417856 |

Both long shapes exceed the 256 MiB resource gate. These are allocated Metal
resource counts, not RSS, physical residency, or peak-process-memory results.
The reduced admission requirement is real; no artificially pressured-host
admission experiment or filesystem-cache-cold claim was made.

## Request flows

Two processes per arm on Q8 + DFlash2 Q8 and Q4 no-drafter. The natural reference
seed is 8,810 tokens. Exact hit reuses 8,810; blue/green/code continuations report
20/20/33 unmatched tokens plus the unconsumed pending token. The packed guard has
149 unmatched tokens plus pending. All 56 responses preserve output hashes and
usage/cache counts; the code request emits 128 tokens and the green turn is
sampled at temperature 0.7, seed 1729.

The Q8 code continuation's complete wall moves `10668.744 -> 10210.264 ms`
(4.297%, positive in both orders, baseline spread 1.066%). Its TTFT controls
spread 8.478%, so the apparent 5.4% TTFT saving is not promoted. The drafter is
loaded but these completed-turn restorations report `decode_path=serial` in
both arms; fresh and exact-hit requests exercise DFlash. Do not label the code
row a restored-speculative speedup.

Other useful constraints: Q8 blue wall is +0.225%; Q4 warm-code TTFT is +0.159%.
Q4 fresh-short wall improves 3.052% with 2.699% baseline spread. Q4's unchanged
fresh long lane regresses 2.934% wall / 2.998% TTFT, narrowly inside the 3% guard;
this is not strict Pareto evidence. Exact-hit, packed-guard, and several warm
rows have >5% control spread and supply no speedup authority. No rows were
discarded or selectively rerun. The primary promotion reason remains the
1.31/3.40 GB allocation removal, not a blanket latency claim.

100 serve tests pass; two normally ignored tests were executed separately.
Coverage includes no-scratch allocation at widths 1/2/16/48, bitwise final logits
and KV/GDN state, real pending-token cache restore, and the 48/49 boundary.
The pending snapshot restores eight consumed tokens while matching nine;
requests of lengths 9/56/57 require 1/48/49 forwards and allocate accordingly.

Disposition: keep the narrow omission; do not widen serial execution or retile
49-256 from this result. Future admission can use the released resource headroom,
but active speculative traffic still needs its existing correctness contract.
