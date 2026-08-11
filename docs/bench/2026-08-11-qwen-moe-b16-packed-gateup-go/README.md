# Exact Qwen MoE B=16 Packed Gate/Up

Date: 2026-08-11

Status: `GO` for a product-shaped fixed-B=16 Qwen MoE JSONL executor. The final
campaign clears every frozen performance and exactness gate. This packet changes
benchmark infrastructure only; ordinary inference remains unchanged.

## Question

The exact B=16 organization reaches roughly 12% over independent queues but
misses its frozen product-spike margin. Prior B=8 work established that packed Q4
routed gate/up is bit-exact while alternate Q5 down arithmetic is not. Does
composing only that exact routed stage with the repaired exact GDN and Q6 head
create enough margin to authorize product scheduler work?

## Candidate

For each block, the candidate:

1. evaluates production route/top-k/shared-gate arithmetic for every lane;
2. packs each lane's post-mixer hidden row and ordered top-k IDs;
3. runs the existing exact packed Q4 gate/up kernel concurrently with each
   lane's production shared-expert gate/up;
4. copies each ordered routed inner slice back to its owning session;
5. resumes production routed/shared down and final residual waves unchanged.

KV, GDN, route weights, shared-gate state, pending tokens, and all down/final
scratch remain sequence-owned. Command-encoder boundaries enforce route, pack,
gate/up, unpack, down, and final dependencies. The concurrent pass has disjoint
writes and read-only weights.

## Prior Evidence

The loaded-once route replay over all 40 Q4 layers moves routed gate/up from
`0.9649` to `0.6254 ms/token` at B=16, a 35.2% stage reduction. This predicts
about `5.4 ms` from a roughly 113 ms exact cohort step.

A short source-identical control/candidate/control bracket reports candidate
throughput `138.997 / 148.082 / 137.656` token/s. The candidate is `1.0705x` the
interpolated controls. A 66-transition bracket reports
`141.768 / 145.126 / 136.381`, or `1.0435x` interpolated. Both candidate runs are
strictly exact in every compared logit, residual, selected ID, and final causal
snapshot.

These observations authorize the final campaign but have no decision authority.

## Frozen Final Gate

Run three serialized fresh processes from one binary and source-state identity,
each with a 1,024-token frontier, two warmups, and 64 measured generated
transitions. Every process must:

- run `exact=all`, the exact Q6 head, and packed Q4 gate/up;
- pass all 66 same-history finite and bitwise checks plus final snapshot equality;
- reach at least `140.9072` candidate token/s (`1.12x` the frozen queue control);
- retain at least `1.25x` paired throughput over its serialized arm.

The median candidate must reach `143.4234` token/s (`1.14x` the frozen queue
control), leaving two percentage points beyond the prior integration gate. Any
failed exactness check or median miss keeps productization HOLD. No retries may
replace a completed valid process.

## Final Campaign

All three processes use source state
`git-source-sha256-v2:c4d25af85f2131d7145a1d4b16cd70aafcadce733415584ff0bf53077c85c36a`.

| Run | Serial tok/s | Candidate tok/s | Paired speedup | Versus queue |
|---:|---:|---:|---:|---:|
| 1 | 106.811 | 147.562 | 1.3815x | 1.172896x |
| 2 | 112.108 | 150.820 | 1.3453x | 1.198792x |
| 3 | 111.712 | 150.300 | 1.3454x | 1.194659x |

Median candidate throughput is `150.300` token/s, `6.8766` token/s above the
`143.4234` gate. Every process exceeds the `140.9072` per-run floor and the
`1.25x` paired-speedup floor.

Each run checks all 66 same-history transitions: logits and final residuals are
finite and bit-exact, all 1,056 selected IDs and their SHA-256 agree, and each
lane's final snapshot matches consumed/pending tokens, positions, active K/V,
complete GDN convolution/recurrent state, and final logits byte-for-byte. Across
the campaign that is 198 exact generated transitions and 48 exact final causal
snapshots.

Run 3 experienced unrelated host pressure during prefill/restore
(`3,596.987/1,948.532 ms` versus roughly `660/156 ms`), but its charged decode
candidate remained `150.300` token/s and exact. No final process was replaced.

The three raw records are tracked beside this README.

## Decision

Advance a sibling Qwen MoE fixed-B=16 executor, not a width-parameterized dense
executor. Preserve exact Q8 GDN projections, packed Q4 routed gate/up, exact Q6
head, production route/shared/down/final arithmetic, and sequence-private causal
state. Product integration must add shared-prefix fanout, model-derived admission,
input-order JSONL output, cancellation and frontier poisoning, and serial fallback
for incomplete cohorts. Scheduler overhead must be remeasured before promotion.

## Commands

```text
cargo fmt --all
cargo check -p qwen-cli --bin qwen-bench
cargo clippy -p qwen-cli --bin qwen-bench -- -D warnings
cargo clippy -p qwen-llm --lib -- -D warnings
cargo build --release -p qwen-cli --bin qwen-bench

QWEN_BENCH_GDN_REPLAY_EXACT=all \
QWEN_BENCH_MOE_BATCHED_HEAD=0 \
QWEN_BENCH_MOE_PACKED_GATEUP=1 \
./target/release/qwen-bench --allow-dirty \
  decode-moe-gdn-repair \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --frontier-tokens 1024 --prefill-chunk 1024 \
  --warmup-steps 2 --steps 64
```
