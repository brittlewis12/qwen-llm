# Flash-Next Donor Investigation And Split Decode Screen

Follow-up: [native qualification and leverage rechart](NATIVE.md) passes a
bounded real-model comparison; production opt-in delivery remains next.

Decision: **KEEP research prototype; GO to bounded native qualification.**
No new model download, competitor build, full-model run, default change, or
end-to-end speedup claim. Implementation: `cea41b05`; separate incumbent Metal
scratch-alignment repair: `09553fb9`. Local base: `2ac9f268`, Apple M4 Max.

## Competitive Evidence

The [M4 Max 64GB community report](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2/discussions/2)
contains actual ordinary and MTP measurements. Its older fork `a30ed07`, with
IQ2_XXS/Q2_K experts and Q4_1 PLE, reports 366.2-378.7 prefill tokens/s at 32K,
40.0 ordinary decode and 51.9 MTP decode. At 128K it reports 369.5 prefill,
38.5 ordinary decode and 50.4 MTP decode. Later comments disclose memory-pressure
scatter and serving/recovery issues, some subsequently fixed upstream.

These are reports, not controlled local engine comparisons. They show a reported
decode-throughput gap motivating investigation, but do **not** establish a threefold prefill deficit on
our chip. Our recorded 425-476 prefill and 16-21 decode use a different quant
and different contexts/prompts. The tweet's >1400 prefill cannot be transferred
to M4 Max without qualification. No matched benchmark is needed to investigate
source mechanisms, and smaller donor weights are not a prerequisite.

## Source Findings

Pinned donor: [antirez/ds4 9139e2ae58a41503968a500f36f75895c1ba63fc](https://github.com/antirez/ds4/tree/9139e2ae58a41503968a500f36f75895c1ba63fc).
Read-only source checkout: `/tmp/qwen-flashnext-ds4-9139e2a`.

1. **Split decode QSA first.** `ds4.c:qwen4_graph_attention_core` (line 58141)
   reaches `ds4_metal.m:ds4_gpu_qwen4_attn_decode_tensor` (48994), selecting the
   ordinary portable split kernel, not M5 NAX. In `metal/qwen4.metal:1733`, four
   SIMDgroups each own three query heads, sharing K/V loads. Up to 64 key splits
   replace a long serial value chain; line 1820 merges max/denominator/numerator
   partials. Local singleton QSA still materializes logits and launches only
   24 value threadgroups, each walking up to 2051 IDs. Local packed GQA4 reuse
   does not fix that singleton path. This is distinct from rejected gathered
   cache or dense-masked attention.
2. **Once-per-head QSA RMS next.** Donor `kernel_qwen4_attn_prep` at 1072 uses
   head-cooperative normalization. Local `kernels/qwen4exp.metal:559` and `:901`
   repeat a scalar D-element RMS sum in every output thread. Delete redundant
   arithmetic; compare a cooperative sum with an exact-order broadcast option.
   Do not copy donor pooled-key rounding: local rounds the mean to F16 before
   RMS, while donor retains an F32 mean.
3. **HC rank-320 gate consumption is a distinct, lower-priority reopening.**
   Donor `kernel_qwen4_hc_gate_mix` (223) uses eight lanes per branch and consumes
   the projection immediately. Local `qwen4exp_metal::encode_read` materializes
   the wide gate projection through generic GEMV. A projection-body saving can
   reopen this; dispatch-only fusion cannot. Local HC normalization already
   avoids some duplicated work present in donor decode.
4. **GDN R4 state ownership is portable but demoted.** Donor
   `kernel_qwen4_gdn_scan_r4` (666), selected in `ds4_metal.m:48672`, shares Q/K
   across four value rows with vector state accesses. Local has one value row
   per SIMDgroup. State bytes do not disappear, and old local complete-middle
   and blocked-recurrence failures constrain expected upside.
5. **Do not mechanically port generic Q8 or grouped MoE.** Donor Q8 GEMV and
   local Q8 already use the same essential two-output-row, four-SIMDgroup
   topology. Both engines have grouped half-staged expert MMA. Donor token-tile
   scheduling can be adapted around local IQ3/IQ4 readers, but local retile
   evidence and prior small-bucket kills still matter. NAX and quant-specific
   readers are not evidence of missing portable execution by themselves.

Independent local review additionally identified existing wide-Q8 prefill and
strict F32 router-projection donors that could serve GDN projections. Chunkwise
WY recurrence is a separate algebraic direction, not a retry of serial blocked
recurrence. Those remain behind the two concrete QSA mechanisms.

## Prototype Contract

`kernels/research/qwen4exp_qsa_split.metal` and test-only
`crates/qwen-llm/src/qwen4exp_qsa/tests/split_decode.rs` implement fixed released
24-query-head / 2-KV-head / 256-dimension attention. Retain local strided QK
accumulation, post-dot scaling, sigmoid expression, direct selected IDs and F16
caches. Online PV and split merging change arithmetic; no bitwise equivalence
claim. Donor attribution and MIT notice are included.

At most 64 splits use `24*64*258*4 = 1,585,152` scratch bytes, owned only by the
component fixture. This is not a per-layer or production session allocation.
At 2051 IDs, 64 splits of capacity 33 include a five-ID split and an empty final
split. Empty partials overwrite their storage and merge without NaNs.

The 12 cases cover 0/1/31/32/33/128/2048/2051 IDs, zero and peaky Q, gates at
+/-100, valid ascending selected four-token blocks with gaps, mixed invalid and
permuted IDs, and all-invalid IDs. The independent F64 oracle uses the actual
half-rounded caches, scalar QK, global softmax, and F64 PV. Both incumbent and
candidate must satisfy absolute error <= `3e-5 + 3e-5*abs(reference)` and
relative RMS <= `3e-5`; direct candidate/incumbent delta is diagnostic.

Output/logit/partial poison, active-partial overwrite, inactive-scratch poison,
offset canaries, bytewise K/V immutability, and query/ID/gate immutability pass.
Maximum candidate/F64 absolute error is `1.147929483e-6` (peaky Q); ordinary
2051-ID error is `1.766001653e-8`.

## Attempts And Timing

Raw logs remain under `target/profiles/2026-09-14-qwen4exp-*` in the worktree.
The final harness is reproducible without model files:

```sh
cargo test -p qwen-llm --lib split_decode_attention_screen -- \
  --ignored --exact qwen4exp_qsa::tests::split_decode::split_decode_attention_screen \
  --nocapture --test-threads=1
```

- Initial compile failed for missing Pod derives; repaired before execution.
- `split-screen-01.log`: capacity 4099 rejected before attention dispatch;
  repaired to ratio-aligned 4100. No performance or numerical result.
- `split-screen-02.log`: numerical PASS; long-ID timing **INCONCLUSIVE**, with
  A-control spreads 82.48% and 53.98%. Not retroactively promoted.
- `split-screen-03.log`: separate confirmation protocol, same inputs/kernels
  and gates, adding a fixed untimed 12-chain ABBA bracket immediately before
  the measured 12-chain ABBA. Numerical PASS, test wall 1.74 seconds.

Each arm is one serial command with 12 complete two-kernel chains. These are
not 12 independent ABBA observations. The frozen useful-screen gate requires
20% GPU savings in the mean and both pairs, with A spread <=5%, for both long
ID counts. Count 128 is diagnostic. All timed replays match their own arm's
untimed output exactly.

| IDs | A1 GPU ms | B1 GPU ms | B2 GPU ms | A2 GPU ms | Mean saving | A spread |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.148271 | 0.105531 | 0.107813 | 0.150490 | 28.590% | 1.4853% |
| 2048 | 0.498135 | 0.070958 | 0.070865 | 0.497969 | 85.762% | 0.0335% |
| 2051 | 0.493424 | 0.070438 | 0.072406 | 0.493125 | 85.521% | 0.0605% |

Amortized encoding/submission/wait wall at 2048 IDs is 0.550410/0.118490/
0.116594/0.545556 ms; at 2051 IDs 0.541979/0.109955/0.119858/0.541427 ms.
It excludes command-buffer/encoder creation and uses Rust's unoptimized test
profile. GPU kernels use the normal optimized Metal build. Neither wall nor
GPU results establish native forward throughput or 12-layer savings.

## Metal Validation Repair

The separate incumbent API-validation run aborted: its logical nine-float
threadgroup scratch was bound as 36 bytes, but Metal requires a multiple of 16.
`09553fb9` rounds the singleton and packed-fallback allocation/preflight to 48
bytes without changing the nine-float indexing or arithmetic. GQA4's 144-byte
allocation was already aligned and is unchanged.

After repair, the released incumbent component and all 12 split cases pass
with `MTL_DEBUG_LAYER=1`. Those runs check API validation, not shader-validation
instrumentation. The debug-layer timing output is supplementary, not the
primary speed authority. The original abort log is retained.

`packed-fallback-validation.log` initially failed its hardcoded GQA4 census
expectation after the fallback GPU command and bytewise comparisons succeeded.
The test now asserts the selected value kernel and its launch geometry for both
configurations, rather than always expecting GQA4. Separate fallback and default
API-validation runs pass (0.36/0.38 seconds). The earlier census failure remains
recorded, not relabeled as a pass or arithmetic failure.

## Next Native Packet

Use existing `2026-08-29-qwen4exp-selected-semantic/natural-ssh.u32le`: current
packed prefill of its 2179-token prompt once, then the first four teacher-forced
targets, exercising all selected-ID residues and pooled-key publication.
Reuse a common in-memory checkpoint, not scalar full-prefill or new weights.
Snapshot/restore must include persistent GPU bytes **and** session/QSA committed
lengths and PLE history. Prove baseline restore/replay bitwise before comparing
candidate forwards. Bind one shared test scratch at the attention-body seam,
after KV publication and before output projection; keep it allocated in both
arms. Check full logits, downstream state, unchanged old cache prefixes, actual
12-layer dispatch reachability and warmed native timing. Freeze model-level
numerical gates before running; primitive agreement alone cannot promote it.

Read-only reviews: `01a0a136-57fe-7310-9aac-975bc5a195b3` (local opportunities),
`01a0a13a-f2f8-7673-ace3-4fbfd25a3aef` (reachable donor, preflight and outcome).
