# Native Split Decode And Leverage Rechart

**Authority correction:** the native test did not hold the production GPU
lease; `cfg(test)` contexts used isolated per-process locks. Timing and coarse
attribution below are provisional, not exclusive speedup/delivery authority.
Numerical/state checks remain observed passes. See [current status](PRODUCT.md).

Decision: **native bounded qualification PASS; GO to experimental opt-in delivery.**
Implementation `0557d812`, local M4 Max, existing UD-Q3_K_XL. Production math
and defaults remain unchanged: this checkpoint uses a test-only body override.
No new weights, scalar full-prefill reference, or competitor run.

## Native Protocol

One current packed prefill of the existing SSH fixture's 2179-token prompt,
then its first four teacher-forced tokens. The SHA256
`874537119c68f6c566c4288ba17c1099694416edb001c4003249570894438e97`
identifies the complete 2211-token fixture, not just the prefix.

The in-process checkpoint is used only while its owning runner remains alive;
it is not a public/durable snapshot format. It restores all persistent GPU
bytes, final hyper residual, logits, full PLE history, and session/QSA committed
lengths. Baseline -> restore -> baseline reproduces all four full-vocabulary
rows and hyper residuals, plus final persistent state, bitwise before testing B.

One 1,585,152-byte scratch stays allocated in both arms and is reused serially
through all 12 QSA layers. Persistent GPU state is 173,414,400 bytes across 121
tensors; the four-copy CPU snapshot budget is below 1 GiB. A dispatch census
witnesses 48 split and 48 merge dispatches, with no incumbent singleton QK, in
the four candidate forwards. IDs cover all 2048/2049/2050/2051 residues and cross
pooled-key publication. Selection, KV publication, projection, and state
transaction code remain intact.

Frozen gates, all passing:

- Each full-vocabulary row: same argmax, cosine >0.99999999, relative RMS <1e-4,
  maximum absolute delta <1e-3. Observed worst RMS `5.719671e-6`, max absolute
  `1.411438e-4`; argmax IDs are 198, 951, 1133, 1141 in both arms.
- Hyper residuals: relative RMS <=3e-4 / absolute <=0.01. Observed maxima
  `6.063125576e-6` / `1.621246338e-5`.
- Persistent F32 state: relative RMS <=3e-4 / absolute <=0.01. Observed maxima
  `3.72524936e-6` / `6.866455078e-5`.
- New F16 KV/compressed rows, tested separately from the old prefix: relative
  RMS <=1e-3 / absolute <=0.03125. Observed maxima `4.937311356e-5` / `0.001953125`.
- Old QSA prefixes remain byte-identical to the checkpoint; unused suffixes
  agree between arms. All compared values are finite. Warm/timed replay remains
  bitwise identical to each arm's own untimed rows/hyper residuals.

Four teacher-forced tokens establish bounded composition agreement, not broad
semantic/perplexity authority, exact sampling, or default-promotion authority.

## Parent Timing

Release Rust build, one fixed warm four-forward ABBA followed by one measured
four-forward ABBA. Restore and intervening readbacks/assertions are outside
the clocks. The wall metric sums the four existing executor timings; it is not
externally timed request/sequence throughput. Test-hook validation is charged.

| Metric, four forwards | A1 ms | B1 ms | B2 ms | A2 ms | Time saved | A spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 231.341792 | 193.915458 | 192.155250 | 231.385375 | 16.566233% | 0.0188375% |
| Executor wall | 235.684084 | 198.530498 | 196.816083 | 235.768251 | 16.142831% | 0.0357054% |

Both axes pass the frozen mean and pairwise floors (5% GPU, 3% executor wall)
and <=5% control spread. The mean saving is approximately 9.58 ms GPU and
9.51 ms executor wall per forward in this packet. Do not translate this into
a general generated-token rate.

The entire test completes in 33.75 seconds. Its one first-use prefix costs
30.531 seconds executor wall /4.181 seconds command GPU across three commands;
that mixed first-use observation is not a balanced prefill benchmark.

Raw authority: `target/profiles/2026-09-14-qwen4exp-native-split-01.log` in the
optimization worktree. Reproduce without downloading any model:

```sh
cargo test --release -p qwen-llm --lib native_split_decode_shared_prefix -- \
  --ignored --exact qwen4exp_runtime::tests::split_decode::native_split_decode_shared_prefix \
  --nocapture --test-threads=1
```

## Rechart From Coarse Native Profiles

Two additional first-token profiles restore the same checkpoint and reproduce
their own arm's full logits/hyper residual bitwise. These are diagnostic stage
observations, not a second balanced performance experiment.

| Complete stage bucket | Count | A GPU ms | B GPU ms |
|---|---:|---:|---:|
| Layers zero-one | 1 | 2.080209 | 1.960042 |
| Post-PLE GDN-containing blocks | 34 | 30.914917 | 31.054499 |
| QSA-containing blocks | 12 | 24.728959 | 15.303501 |
| Tail | 1 | 1.085375 | 1.087541 |

The QSA-containing block reduction of 9.425 ms is consistent with the native
parent saving. The remaining 31.054 ms in GDN-containing blocks includes HC,
projections and FFN: it is **not recurrence time**. No current per-substage
decode budget is inferred from old packed-prefill percentages or the invalid
August named-stage screen. `summarize_native.py` summarizes these existing log
rows and state maxima without GPU work; it is not an independent gate validator.

## One Bounded RMS Attempt, Then Stop

Candidate source is preserved at `30f65ca5` and subsequently removed from the
active tree. Lane zero computes each head's original scalar RMS sum, broadcasts
it, and one SIMDgroup produces RoPE/query/raw-gate outputs. The test covers 12
independent layer inputs x3 calls x4 positions (0/2179/32767/131071), including
zero/tiny/large inputs, offset canaries, exact raw-gate extraction, immutable
weights/inputs, and bitwise repeated outputs. **144 component comparisons pass.**

Frozen useful floor: at least 0.5 ms saved across the complete 36-call chain,
in the mean and both pairs, A spread <=5%. Fixed warm 12-repeat ABBA followed
by measured 12-repeat ABBA at position2179 gives GPU milliseconds:

`0.581524 / 0.605031 / 0.588042 / 0.434545`.

Timing is **INCONCLUSIVE** with 28.93% A spread; neither candidate observation
shows a gain over either control. This is not a quantified regression or proof
that all cooperative normalization is unhelpful. It is no reason to spend more
time here now: the candidate is removed, source/results preserved, no width
sweep, control-fishing retries, or model replay. Test wall0.29s. Raw log:
`target/profiles/2026-09-14-qwen4exp-rms-broadcast-01.log`.

## Updated Leverage Map

1. **Deliver split attention behind an explicit default-off session option.**
   Initially restrict to singleton24/2/256, F16 caches, and2048-2051 active IDs.
   Account for one session scratch shared serially across layers, retain it
   through completion, and preserve the incumbent outside the qualified range.
   Exercise the actual product bindings with the same existing prefix and its
   32 continuation tokens under unchanged gates, plus the four-forward timing
   bracket. No second reference prefill or repeated coarse profiles needed.
2. **Price the remaining HC/FFN projection bodies.** HC up has K=320: ten Q8
   blocks feed the generic GEMV's 32 initial block slots, leaving two SIMDgroups
   with no weight-loop work. A rank-specialized up-plus-gated-mean body is a
   concrete mechanism, not another launch-only fusion. Native routed IQ3/IQ4
   FFN remains a likely large budget; get a bounded complete-MoE observation
   before choosing another expert kernel topology.
3. **Park more attention-body and exact RMS-broadcast tuning.** At longer
   contexts, index scoring still grows with context/4 while selected attention
   saturates near2051 IDs. Observe it at a naturally available long checkpoint;
   the 2179-token packet does not settle the 32K/128K indexer budget.

Independent read-only reviews: `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef` (native
preflight/outcomes/delivery boundary), `01a0a136-57fe-7310-9aac-975bc5a195b3`
(leverage rechart and RMS preflight).
