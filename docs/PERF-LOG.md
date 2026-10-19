# Performance Log

Current and historical evidence index for performance work. Entries in the
current month are newest first. Earlier periods are preserved in dated archive
files and their headings remain directly linkable.

See [PERF-ROADMAP](PERF-ROADMAP.md) for the current state and evidence index.

## Period archive

- [2026-09](archive/perf-log/2026-09.md)
- [2026-08](archive/perf-log/2026-08.md)
- [2026-07](archive/perf-log/2026-07.md)
- [2026-06](archive/perf-log/2026-06.md)
- [2026-05](archive/perf-log/2026-05.md)

## 2026-10

## 2026-10-19 — Descriptor-Derived GGUF Population Plan; Owned Arena Archived

- Removed the three artifact-keyed parallel population profiles and the dense
  27B retained-layout sentinel. Load-time plans sort direct tensor requests by
  shard and offset, choose contiguous byte-balanced worker cuts, size rounded
  destinations from the host page size, check every allocation against Metal's
  maximum buffer length, and admit the plan using its computed destination
  bytes plus the shared 256 MiB session reserve.
- Header-only fixture comparisons use the published A3B, dense 27B, and A10B
  filenames resolved through `QWEN_MODEL_SEARCH_PATH`. Dense 27B and A10B keep
  their historical cuts, tasks, and worker byte counts. A3B moves one
  2,097,152-byte tensor from worker 4 to worker 3: the third cut changes from
  539 to 540 and the maximum worker-byte deviation remains 68,568,960 bytes.
  The A3B page-rounded plan still allocates 2,758,144 padding bytes over 232
  tensors. The computed headroom requirements are 22,391,974,400 bytes for
  A3B, 17,074,685,952 for dense 27B, and 77,287,432,192 for A10B.
- Bit-exact load witnesses pass for A3B and dense 27B with both mmap-copy and
  pread. They compare logits, continuation state, and every planned resident
  source byte against the ordinary copied path, and require the generic plan
  marker and copied load ledger.
- The force-only owned arena is archived. v0.598 passed its bit-exact state and
  fresh first-byte gates, but loaded decode was 1204.60/1390.20/1390.25 ms for
  copied/owned/retained, and complete request wall was 1.107–1.125× slower for
  owned in every block. The experiment concluded that its one-window resource
  topology caused the steady-state tax and recommended topology-preserving
  parallel copy. Live arena allocation, A3B arena constants, and the frozen
  `qwen bench gguf-arena-floor` implementation are deleted; the v0.597/v0.598
  measurements and review remain archived in their existing bench records.
- Five interleaved A3B `MetalModel::load()` A/B pairs, main/branch in ms:
  `601.326/603.619`, `610.678/605.922`, `604.982/603.296`,
  `606.383/601.733`, `608.289/604.588`. Medians are `606.383/603.296 ms`
  (branch `0.51%` faster). At the timing check, `uptime` reported
  `3.42/4.59/4.23`; the process scan found no cargo, test, Qwen, or model
  inference processes. The measured production Rust source delta is `-5,981`
  lines from `f2f6b6d2`, excluding test source files.

## 2026-10-18 - Archived v0.664 LM-Head Screening Code

The concluded v0.664 screening oracle and its `qwen-bench` subcommand were
removed from the live workspace. The last commit containing the implementation
is `ac70437fd3fc56f462f326d087cef705b2241cbd`; recover it from git if needed.
The protocol and result records remain under `docs/bench/`.

## 2026-10-18 - Archived v0.656 Integrated Grammar-Row Code

The concluded v0.656 integrated grammar-row runner, its private runtime helper,
and the `qwen-bench integrated-grammar-row` subcommand were removed from the
live workspace. The last commit containing the live implementation is
`66c4be5785307c8eb616e12278c6c55e933ebbd4`; recover it from git if needed.
The KILL result and protocol remain under `docs/bench/` as historical evidence.
## 2026-10-17 - Three-Way GLM Pilot Finds No Q4 Quality Advantage

- Preregistered eight public engineering cases, including two approximately
  9.7K-token archives, at `b57ce0b8`. Same M5 Ultra, native `d7b2a05b`, low
  reasoning, greedy sampling, Fast/Half, budgets, and no prefix reuse across
  Maya-M, stock UD-IQ3_XXS, and stock UD-Q4_K_XL. Full artifact hashes pass.
- All 24 responses complete naturally; all actual prefill widths are 2048.
  Zero swap at before/after observations; existing compressor occupancy remains
  nonzero without new compression events. Q4 fits without tuning.
- Two fresh blinded reviewers score Maya 23/23, IQ3 23/24, and Q4 22/21, each
  out of 32. Separate exact-fact all-correct cases are 6/8, 5/8, and 5/8.
  Reviewers agree on 22/24 scores; no averaging or significance claim.
- Weighted real-response generation is 47.25/49.14/42.42 tok/s for Maya/IQ3/Q4;
  weighted prefill is 499.91/506.06/485.25 tok/s. Eight-request totals are
  177.28/169.16/179.22 seconds. Q4 emits fewer tokens, so its slower generation
  does not translate directly into the same whole-request latency penalty.
- Q4 misses exhaustion caching and chooses a wrong long-context base; Maya has
  a bytes/str UTF-8 implementation bug; IQ3 has a generation-key cache-hit bug.
  These are static-review findings, not execution-test passes or broad rankings.
- Keep Maya as incumbent: this small low-reasoning pilot does not show enough
  Q4 benefit to justify approximately 74-78 GiB extra storage and 10-14% lower
  generation throughput. IQ3 remains competitive. No backend changes/promotions.
  Evidence: `docs/bench/2026-10-16-glm-quality-comparison/`.

## 2026-10-16 - M5 Ultra GLM Maya First Light Complete

- The same three-shard Maya-M IQ2_S artifact passed full SHA-256 verification
  on the 256 GiB M5 Ultra. Clean `d7b2a05b` native binaries and runtime source
  match; no kernel changes, device overrides, or wired-limit adjustments.
- Three warmed repetitions: tg128/tg256 at depth zero are 56.10/55.23 tok/s;
  tg128 after an untimed 8192-token prefix is 51.80 tok/s. Synthetic pp128,
  pp512, and pp2048 are 256.98, 466.90, and 448.83 tok/s respectively.
- The real low-reasoning Rust smoke is correct and stops at EOS: 54 prompt
  tokens, 86 generated, 1014.7 ms prefill, 50.83 decode tok/s, 2791.7 ms total.
  Filesystem cache was warmed by checksums; mapped loader setup is not cold-load
  latency. Simple semantic review is not independent numerical qualification.
- Runner and queue exit zero; swap/compressor usage remain zero before/after.
  No fresh matched M4 GLM suite or upstream MTP result is claimed. The worker
  is left idle with binaries and weights ready, not running a persistent server.
- Evidence: `docs/bench/2026-10-16-m5-ultra-first-light/m5-glm/` and its parent
  README. Completes the GLM transfer/run marked pending in the preceding entry.

## 2026-10-16 - M5 Ultra First Light: Native Qwen Runs Without Kernel Changes

- M5 Ultra, 80 GPU cores, 256 GiB, macOS 27.0.1. Native binaries
  built at clean `d7b2a05b` on M4 Max, with embedded Metal libraries, run against
  a matching source checkout. No device overrides or wired-limit changes.
- Qwen3.5-0.8B Q8_0, same binary/artifact, three warmed repetitions: M5 decode
  435-439 versus M4 261-263 tok/s (~1.67x); pp4096 16174 versus 8700 (~1.86x).
- Qwen3.6-35B-A3B UD-Q4_K_M: M5 tg128/256 169.94/168.62 versus M4
  112.64/111.96 tok/s (~1.51x); pp2048 3695.20 versus 1829.43 (~2.02x).
  At depth 8192, tg128 is 138.96 versus 99.86 tok/s (~1.39x).
- These are synthetic compute suites, not sampled streaming rates. The real
  Qwen35B Rust smoke is correct and ends at EOS: 79 generated tokens,
  133.47 decode tok/s, 339.1 ms TTFT excluding 2708.0 ms model load.
- M5 runs between transfers with zero swap/compressor before and after. M4
  remains an active desktop; its Qwen35B repeat overlaps the outgoing network
  copy, not checksum work. Ratios are provisional, not broad M5 qualification.
- GLM Maya is still transferring; no GLM or MTP result is implied. Evidence:
  `docs/bench/2026-10-16-m5-ultra-first-light/`. No kernel changes or promotions.

## 2026-10-13 - ReplaySSM Precision Separation Works; Keep Production FP32

- Implemented actual Metal KDA replay: FP32 checkpoint plus cached per-key decay,
  key and correction, followed by physical F16-read/FP32-anchor and shared-F16
  variants. L8 flushes recompute delta corrections from raw inputs, excluding
  cached LP corrections. Production state, snapshots and dispatch stay unchanged.
- FP32 L1/L8 outputs, conv and reconstructed states match incumbent bits across
  residues/flushes. Eleven KDA tests pass under leased Metal validation, including
  the independent CPU contract and poisoned-LP-correction flush isolation.
- Three synthetic 64K CPU streams against persistent F64: separation substantially
  reduces slow-decay F16 feedback, modestly helps ordinary forgetting, and ties
  under strong rotating decay. All 24 configurations are finite. Eight portable
  Python tests pass. This is F16 RTN, not a paper FP8/NVFP4 replication or model KL.
- Physical Metal 1024-step slow fixture: output RMS shared F16 3.11855e-4,
  separate 5.41567e-5 (5.758x lower), ordinary FP32 1.81584e-6. Separate anchor
  matches FP32 bits after completed windows. All errors finite; synthetic
  lower_bound=-0.001 is explicitly distinct from the release -5.
- Release-bound synthetic batch-one leaf, 34 distinct 64-head states, complete
  windows: GPU B/A FP32 replay 1.1878, separate 1.0366, shared 1.1776. Separate
  pairs disagree; no convincing latency win. Matrices+records use 136/161.5/
  238.0664/102.0664 MiB for incumbent/replay/separate/shared. These are logical
  bytes and leaf timings, not model throughput, RSS or measured DRAM traffic.
- Keep ordinary FP32: separate is less accurate at readout, larger, and has no
  demonstrated speed advantage over it. Shared saves roughly 25% recurrent
  storage but pays speed/precision costs. Archive GPU code at `a84d1c57`, remove
  it from active builds, retain CPU study and evidence. Speculative rollback is
  a separate hypothesis requiring a concrete copy cost, not a repeat of this one.
  Evidence: `docs/bench/2026-10-12-glm-replayssm/`.

## 2026-10-12 - Document Checkpoints Promoted After Default Validation

- Experiment archived at `42b3cdad`; production now enables the existing-chunk
  checkpoint for all Fast precision settings
  under existing prefix-reuse, Fast-snapshot and cache-budget controls. No new
  knob; Exact/serial unchanged. Cache origin is distinct from shared-split
  execution. Same shared cut/rows/precision/token prefix and proven full chunks
  preserve same-cold-schedule equality; partial alignment cannot restore origin.
- Initial private real-source A/B: branch reuse 51 -> 41,011 tokens; prefill
  217,658.66 -> 2,522.78 ms, first decoded piece 217,727.52 -> 2,555.85 ms,
  backend wall 240,292.96 -> 22,665.94 ms. Seed and branch logits/greedy bytes
  match exactly. Seed truncates equally at 1024 tokens; branch ends normally
  after 576. One ordered backend A/B, not HTTP latency or universal throughput.
- Seed capture 76.221 ms, extra 643,637,452 cache-accounted bytes, not RSS.
  Session allocation/precision unchanged; optional denial skips capture without
  reshaping workspace or chunks. Live wins ties; shared/restored entries protected.
- Default override `None`: `gpu_chunk_checkpoint_cancel_resume` passes under
  Metal validation in 53.60 s. Shared 13, committed 2061, target 4109, prompt
  4112; live resume publishes, deep restore reads suffix 3, full logits/greedy
  bytes/end match, pins return to zero. Separate injected denial at cap 8/suffix
  65 preserves width/output; not an actual pressure experiment.
- CLI 807 passed / 45 ignored in 29.32 s; primitive state oracles passed in
  139 s; all-target check passed. [Evidence and telemetry](bench/2026-10-12-glm-document-checkpoint/README.md)
  distinguish the initial private timing packet from default validation.
  The branch lever is complete; no further arbitrary experiment or quality
  rerun is queued. Earlier answer errors and parked kernels remain recorded.
- Integrated current main's portable-fixture changes at `405c16ce`; expanded
  full contracts pass after a build-timeout retry. Merged CLI: 805 passed,
  47 ignored (two fixture tests reclassified); client scripts: 17 passed.
  Production release CLI rebuilt; no pushes or environment interventions.

## 2026-10-12 - Maya 41K Repository Context And Live Follow-Up Complete

- CLI `ec96cc45`, driver `9cd2915e`; one real-source packet, two requests,
  65536 capacity, default Fast Half, automatic cache/workspace, low effort,
  fresh sampling at temperature 1.0 / top_p 0.95 without overrides. Both
  responses completed at EOS, not truncated; no request/teardown errors.
- Initial input 41,420 tokens: first delta 187.2918 s, first answer 188.5597 s,
  wall 245.4655 s, output 1,613 tokens. Follow-up input 43,138, live cached
  43,032, prefilled 106: first delta 1.2314 s, first answer 6.3004 s,
  wall 43.0456 s, output 1,164. Decode 27.73/27.84 tokens/s. Cold load
  68.265 s and warm-up 36.573 s precede requests; cause not inferred.
- Admitted rows 2048, session ledger 3,548,266,496 bytes. Cache budget
  7,149,666,304 bytes; actual shared cut 51, one 153,264,332-byte entry.
  This demonstrates long live reuse, not restoration of a 41K document.
- Independent review found material answer errors: decode cancellation drops
  live state but leaves snapshots; Fast canonicality checks segment starts,
  not cut endpoints; tick-5 cancellation is at the cut; Exact can restore
  the transcript suffix (`a_len - 2`). The 2048-row cancellation citation is
  `gpu_adaptive_prefill_cold_hit_and_continuation`. No task-quality pass.
- [Evidence and corrected guidance](bench/2026-10-12-glm-maya-repository-context/README.md)
  retain raw answers, hashes and timings unchanged. One example is not broader
  quality evidence; no rerun to paint it green. No new task is underway.
  Research is considering the Fast shared-only cut as a long-document branch
  lever, but a candidate needs concrete branch cost and a compatible mechanism;
  there is no measured branch cliff yet.

## 2026-10-12 - Maya Practical Reuse And Harness Refresh

- CLI `ec96cc45`, harness `f36f52b1`; 10 mock tests passed for the refreshed
  screen's outcomes, replay, failure evidence and owned teardown. Default Half,
  automatic cache, 32768 context capacity and 2048 workspace rows.
- Cold load 50.711 s plus warm-up 1.129 s precede requests. First deltas:
  A1 50.016 s, A2 live 1.473 s, B snapshot 0.622 s, C snapshot 0.538 s,
  A return 2.051 s. One 11,104-token entry uses 272.36 MiB; the 6.66 GiB
  cache budget is not allocated usage. Return restores the shared prefix.
- A/B/A return are useful and correct for the canned fixture. C is unfinished
  after three continuations: unsupported task/rg/main.rs read, then pending
  cat main.rs. All 11 argument objects parse; nine completed HTTP responses
  do not make C a task-quality pass. No request/teardown errors; SIGINT exit 130.
- [Recipe and original evidence](bench/2026-10-12-glm-maya-agent-reuse/README.md)
  preserve those fixture failures. No paired speedup claim against historical
  UD on a different, pre-snapshot binary. HC remains parked at `ec96cc45`,
  source archived at `08ec6267`.
- The subsequent 64K-capacity real repository task and follow-up are now
  complete; see the checkpoint above for live reuse and factual answer errors.
  No new kernel search or duplicate serving work.

## 2026-10-12 - F32 HC MMA Leaf Win Does Not Convert To Whole Prefill

- Reused the existing F32-operand MMA template with guarded M24 fragments,
  normalized inputs and no new GPU buffers. Dedicated test-only Fast Half
  permission leaves production Q6 MMA and Exact unchanged.
- Actual final-block K16384/M24/N2048 leaf favors MMA in both pairs, with large
  scalar timing drift. Independent F64 checks cover all 24 channels at ten tokens;
  scalar replay confirms the captured binding. Primitive guards/oracles pass.
- Ordinary whole N2048 ABBA: A 8097.769/8430.192 ms, B 8409.403/8436.907 ms;
  A/B 0.981103x, both pairs favor incumbent (second effectively tied). Non-HC
  dispatch signatures match. No demonstrated useful end-to-end conversion.
- Each arm reproduces its logits through three common continuation tokens;
  between-arm KL up to 0.007581 is descriptive, not the rejection criterion.
  Park without a tile sweep or repeated timing until favorable. Source/harness
  archived at `08ec6267`, removed from active builds; raw evidence retained in
  `docs/bench/2026-10-12-glm-hc-f32-mma/`. 96 grouped and 49 GLM tests pass under
  leased Metal validation; all-target release checks pass.
- Next: exercise current Maya cold/live/snapshot conversation reuse and publish
  usable configuration guidance, rather than another low-ceiling kernel search.

## 2026-10-12 - KDA Component Budget Does Not Justify Recurrence Redesign

- Split packed KDA into five existing-recorder spans; ordinary execution keeps
  one encoder and the same three kernels/arguments/order. Shared validation
  preflights all PSOs before dispatch. No new activation memory or GPU arithmetic.
- Maya Half N2048 across 34 blocks: input projections 1244.76 ms, prepare 105.13,
  recurrence 340.05, output norm 12.71, output projection 392.54. Ordinary wall
  7684.56/8031.44 ms; profiled 8631.17 ms (9.8% overhead), 984 timestamps used.
  This is diagnostic attribution with perturbation, not ordinary-wall fractions.
- Full model profiled/ordinary logits match bits. Four leased KDA tests pass,
  including independent CPU chunk/decode and seven-buffer output/state equality
  across sampled encoder boundaries at rows 1/37 with nonzero incoming state.
- Projections dominate; park a large chunkwise recurrence algorithm rather than
  claim its budget is the whole KDA parent. Next is one small F32 normalized-HC
  MMA primitive-reuse bet, with explicit lineage permission and M24 tail guards.
  Evidence: `docs/bench/2026-10-12-glm-prefill-cost/maya-2048-kda-components.jsonl`.

## 2026-10-12 - Avoid Serial Transcript Segments In Short Qwen Prefill

- Default-enable the existing bounded single-chunk route for fresh greedy
  19-48-token requests on the covered dense-27 geometry without a drafter.
  Remove template/LM-head dtype filters: these are not operation requirements.
  Retain the 128 MiB priced scratch cap, admitted fallback and explicit
  `QWEN_SERVE_FRESH_PACKED=0` rollback (invalid values also disable).
- Actual current serving, not the old forced-serial pilot: ordinary dense
  prefill switches above 12 tokens but reevaluates each transcript segment.
  Fresh19 splits into 12+7 serial steps; fresh32/48 pack 25/41 then serialize
  the seven-token header. Single-chunk scratch batches both segments. Request
  `prefill_path=chunked` alone did not describe the executed segment schedule.
- M4 Max, Metal validation, isolated loopback, durable tier off, RAM cache4GiB,
  ABBA fresh server processes, eight identical cases per process. Q8 A=0/B=1;
  Q4 A=0/B=unset verifies the new default. First SSE delta milliseconds:

  | Model / prompt | A1, A2 | B1, B2 |
  | --- | --- | --- |
  | Qwen3.8 Q8 / 19 | 1493.4, 1361.6 | 488.8, 476.9 |
  | Qwen3.8 Q8 / 32 | 684.9, 653.5 | 312.2, 312.9 |
  | Qwen3.8 Q8 / 48 | 761.0, 833.0 | 392.8, 394.0 |
  | Qwen3.6 Q4 / 19 | 896.4, 897.4 | 561.7, 546.2 |
  | Qwen3.6 Q4 / 32 | 578.8, 534.7 | 507.9, 472.2 |
  | Qwen3.6 Q4 / 48 | 668.8, 628.1 | 590.8, 583.0 |

- All 64 responses match within model on output hash, full usage, status and
  incomplete reason/error. Code/prose128 and guard49 are token-limit truncated,
  not completed-task quality evidence. The fixture named `exact-19` restores
  12 transcript tokens and runs seven header tokens, not an exact-logits hit.
  Q8/Q4 full-state/logit numerical and 3/64/64 greedy continuation checks pass.
- Controls retain their routes but drift: guard512 paired TTFT changes are
  -7.5%/-14.0% Q8 and +11.5%/-5.9% Q4. Host VM activity also changes, without
  a localized cause. Do not attribute every saved millisecond to the route or
  claim universal throughput. In particular, Q4 output-heavy wall is near flat.
  This does not rescore the September guard failure or erase its VM outlier.
- Initial tight-loop probe stops on a transient server-busy503 after DONE;
  retained separately. Completed comparisons use an equal, recorded, untimed
  200 ms inter-request pause, not a back-to-back throughput/reliability claim.
  The probe now exposes `--settle-ms`; every owned server exits cooperatively130.
- Reproduction: existing `fresh_packed_endpoint_fixture`, `request_probe.py`,
  and `fresh_short_packed_matches_serial`. Raw local runs:
  `target/profiles/audit-fresh-settled-{A1,B1,B2,A2}` and
  `target/profiles/audit-fresh-q4-{A1,B1,B2,A2}`. cx reviewed the current baseline,
  mechanism, resource repair and bounded rollout; no new qualification framework.
- Final checks: 800 CLI tests pass (43 ignored), formatting and environment
  inventory checks pass; probe rejects a negative pause before file access.

## 2026-10-12 - Strict Router Follows Geometry, Not Sampled Widths

- Current Maya Half N2048 profile (`5d934f4a`) finds42 generic F32 routers,
  roughly519 ms including selection, because strict routing admitted exactly
  N128/N512. Those widths and H4096/E288 were historical coverage, not limits
  of the existing strict-order primitive.
- Forced strict N2048 gains3.36% mean throughput; unforced production B gains
  2.12%, with both wall/GPU pairs positive. N129 boundary control gains0.77%.
  Full projection bits, selected routes/weights and join/three-step logits agree;
  independent sampled F64 dots satisfy the mathematical accumulation bound.
- Replace sampled lists with Fast/F32, rows>=128 and primitive geometry:
  positive u32 dimensions, E divisible by8, plus existing pipeline capability
  fallback. Keep generic below128 based on prior small-batch results. No new
  runtime allocation, environment knob, precision weakening or model whitelist.
- Add actual E288/N2048 and odd-K/E primitive cases;90 GLM tests plus strict
  primitive pass under leased Metal validation. Initial unleased primitive run
  timed out during concurrent validation; serialized retry passes in0.23s.
  Timing packets retain the approximately82.5s warm-only wall gap; warmed gains
  are not cold-TTFT gains. All-target release checks pass.
- Incorporate main session reuse/re-admission work through `65b09776` before
  integration. Next is a bounded KDA internal split tied to one recurrence
  decision, not another broad profiling campaign. Evidence:
  `docs/bench/2026-10-12-glm-router-large/`.

## 2026-10-12 - Re-Admit Qwen's Actual Packed Allocation Fallback

- Short-request investigation found a stale resource assumption: optional
  single-chunk scratch can be smaller than the general packed fallback. The
  fallback is no longer necessarily serial after the dense serial-tail limit
  moved to 12. Following candidate allocation failure, re-admit the full
  baseline sequence/scratch footprint and unchanged durable/transport CPU
  reservation before allocating it. Preserve typed pressure/telemetry errors.
- Successful candidates and already-admitted ordinary paths do not repeat
  admission. Failed candidate resources drop before fresh memory signals are
  sampled. No policy widening or execution change; fallback may still refuse
  conservatively rather than initiate another snapshot-eviction round.
- Three CPU fault tests and six transport-memory tests pass; release server
  build passes; cx review finds no blocker. Mocked allocation failures are not
  real Metal OOM tests.
- Re-ranking: Qwen's recorded allocation costs are about 1-2 ms, not GLM's
  hundreds. Defer sequence retention. The old short-packed pilot still passes
  numerical/state/greedy checks, but forces a serial reference: its historical
  speedup is not today's serving opportunity, since current ordinary serving
  already batches prompts above 12 tokens. Compare actual routes before rollout.

## 2026-10-12 - Retain Healthy GLM Session Storage On Cold Misses

- Actual served cold misses paid roughly 170-280 ms for session destruction and
  allocation before the existing prefill timer. Avoid this work by resetting
  compatible healthy state, not by changing arithmetic or adding a session pool.
  Clear KDA convolution/recurrent state and complete MLA pending rings; retain
  invisible cache tails and scratch. Poison still refuses reset, errors still
  discard sessions, and lineage changes still recreate. Prefix/snapshot hits
  retain precedence; a reset is zero cached tokens, with transport admission
  before mutation and no double charge for resident storage.
- M4 Max, UD-IQ3_XXS, context32768/workspace2048, current Fast/Half defaults,
  Metal validation, loaded weights, no snapshots. Alternating unrelated
  24/2062-token requests; recreate/reset order reverses each round. Final
  first-backend-piece milliseconds (round0 then three warm paired rounds):

  | Prompt | Recreate | Reset |
  | --- | --- | --- |
  | 24 tokens | 701.9, 922.0, 727.7, 771.9 | 548.6, 528.2, 526.4, 528.7 |
  | 2062 tokens | 9597.9, 9380.1, 9910.4, 9895.3 | 9533.8, 9810.8, 9436.5, 9532.3 |

- Warm short pairs save 27.7-42.7% to first backend piece; not HTTP TTFT or
  loading-inclusive latency. Direct reset costs 0.7-11.6 ms versus 179-296 ms
  destruction/allocation in these warm rounds. Whole-request savings exceed
  that isolated interval; do not attribute all the difference to memset alone.
  Long-request timing is mixed: no general prefill-throughput claim.
- Full prefill logit bits match across every recreate/reset arm. Independent
  fresh-session test checks snapshots, full logits and continuation at zero,
  partial pool and sparse-frontier positions after a longer old prompt;
  storage/ledger remain unchanged, repeated reset works and poison refuses.
  Serial fallback also passes. Library reset test and all four GLM serving GPU
  tests pass under Metal validation, including live/snapshot/cancel/retry paths.
  The combined run timed out during snapshots; the isolated rerun passes.
- 798 default CLI tests pass (43 ignored); cx finds no production blocker.
  Strict all-target Clippy remains blocked by existing warnings outside this
  change. Reproduce the paired backend check with ignored
  `gpu_cold_session_recreation_cost` and `GLM53_GGUF`; no production knob added.

## 2026-10-12 - Flash Integer Selector: Ordinary Execution Does Not Convert

- Reused retained `3b21b899` without old parent observers or encoder splitting.
  One ordinary context32768 cell: prefix8192/suffix2048, native UD, same
  checkpoint restored outside every warm/ABBA arm. Actual selector launches
  have 32 queries, compressor capacity8192 and visible blocks2048-2560.
- 768 substitutions (12 layers x64 bands), unchanged surrounding dispatch
  order/geometry. Full logits and persistent state hashes remain exact through
  three common continuation tokens in both packets. Independent GPU oracle
  passes ties, nonfinite values, offsets and immutability checks.
- First wall/GPU mean A/B ratios0.9647/0.9484 with opposite-signed pairs;
  repeat0.9241/0.9034, both pairs favor incumbent. Retain all samples and drift;
  no precise regression estimate or performance promotion. Large-query leaf
  gains did not establish improvement for actual serving launches.
- Source/harness checkpoint `7628319e`; remove live candidate/one-off harness
  rather than widen bands or revive the attribution campaign. Production and
  decode unchanged. Evidence: `docs/bench/2026-10-12-flash-selector/`.

## 2026-10-12 - Enable Half Q6 Latent MMA On Task Evidence

- Refreshed lane from `8d50ec11` through `73d13835`, then `f9abbf19` before
  integration. GLM arithmetic was unchanged; rebuilt for the new MSL3.1 target.
- Fixed quality-v1 subset before outputs: three 64-token real continuations and
  four tools, greedy plus seed11, same Maya weights/workspace2048. All eight
  tool runs per arm strictly correct with identical text/tokens. Mean NLL
  improves 0.009980 nats/token; top-1 hits fall 111 to 109/192. Bounded evidence,
  not a general non-inferiority or quality-upgrade claim.
- Production B without override: throughput ratios 1.068x full512, 1.112x
  full2048, 1.037x ragged129 suffix and 1.118x512 suffix after4096. Independent
  primitive tolerances unchanged; no new GPU buffers or model gates.
- Enable only Fast Half alongside existing Fast F32. Keep DenseF32, Exact,
  Q8, decode and capability/geometry fallbacks unchanged. Test-only scalar8 is
  isolated so promotion cannot silently change its diagnostic arm.
- Historical forced-MMA KL assertion still fails at 0.012874 and is retained.
  Old Fast agreement is not task ground truth. New production AB records KL
  without that gate. UD control timed out after two bit-identical cells amid
  unstable timings/host pressure; retain it and make no UD speed claim.
- Ten grouped-latent and 87 GLM tests, six analyzer tests, all-target checks
  and Maya adaptive serve cold/hit/cancel/retry/continuation validation pass.
  Snapshots are RAM-only and bound to a weights instance; there is no
  demonstrated cross-binary collision requiring a new lineage.
- Evidence: `docs/bench/2026-10-12-glm-latent-quality/`. Task comparison answered
  the open question more directly than another parent-stage profile; actual
  N2048 attribution remains a way to choose the next mechanism, not a
  prerequisite to exploiting the demonstrated one.

## 2026-10-12 - Quant-Compatibility Worktree Disposition

- Reviewed the completed October 10 Flash and GLM experiments against current
  main. Do not merge the old worktree wholesale: its large observer/benchmark
  scaffolding is not a production feature. No new archive format or preservation
  framework; this log carries the conclusions and ordinary Git history carries
  useful source checkpoints.
- Flash encoder splitting: counter-free restored suffixes at 8K/32K improved
  wall time about 4.75%, with exact checked outputs/state. Whole 32K prefill
  averaged 77.254/75.871 seconds, but all four paired changes were -3.115%,
  +0.682%, -4.398%, -0.173%, with unstable ordinary repeats. The isolated wait
  saving did not repeat across sustained chunks; stop expanding this execution
  plumbing. These are not loading-inclusive TTFT measurements.
- Sustained parent observation: numerical/state and local timestamp checks pass;
  whole/decode controls exceed the frozen perturbation bounds. Do not use its
  parent fractions to rank kernels. Retire the attribution campaign, not the
  valid observations it produced.
- Parallel integer scans: exact independent selector checks pass. At N2048 and
  capacity 8192, all six GPU pairs save 6.65-22.25%, with unstable ordinary
  repeats. At capacity 16384 one pair is effectively neutral; a small-query
  control retains a regression. This remains a promising batched candidate,
  not a universal replacement or a demonstrated whole-model win. Its next
  useful test is an ordinary selected-prefill A/B with only the selector changed,
  not another parent-observer prerequisite or a threshold-relaxation exercise.
- GLM quant screen: pinned llama.cpp, eight paired tasks, greedy/low reasoning,
  16K context. At 2048 generated tokens UD/Maya both pass 6/8; at 8192 UD passes
  7/8 and Maya 6/8. UD's deep-get completion becomes correct with more budget;
  Maya finishes but mishandles the dot after a list index. Tools are constrained
  and token-identical. Maya's logged buffers are about 4.80 GiB smaller. This
  supports no general quality winner and says nothing about native Half/MMA.
- Reuse task meanings and existing graders for native quality work, not the
  machine-bound operator, duplicated metadata or bytecode-dependent freezes.
  Its local indentation repair exposed a real C5/C6 bug in the existing mainline
  grader: special tests were inserted outside the generated try block. Repair
  that utility directly; preserve task/prompt hashes and historical scores.
- Read-only verification of the old quality packet passed its 52 adapter and
  25 operator CPU tests. This is historical evidence review, not renewed GPU
  qualification or a claim that the entire dirty tree is ready to land.
- Retired the `qwen-llm-quant-compat` worktree. Selector candidate and standalone
  correctness tests remain at `3b21b899` on `perf/glm-flash-next-prefill`, not
  merged to main; three selector CPU tests pass. Discarded the superseded
  experiment machinery rather than creating another archive.

## 2026-10-11 - Gate/Up and KDA Falsifiers; Retire Live Experiment Debt

- M32/N32 lower-state gate/up: Maya whole ratios 0.9877x at N2048 and
  0.9961x at N512. Recorded leaf and whole join/three-step outputs match bits;
  no performance justification for promotion or a UD sweep.
- Register-history KDA: prepared arrays and final history match bits, including
  short chunks and CPU chunk/decode oracle. Sampled preparation is only 3.2 ms
  per N2048 block; small leaf savings do not explain a noisy 0.9% whole mean.
  Production remains unchanged. No coefficient/barrier/recurrence tuning.
- First KDA packet is incomplete after a slow baseline A1; process later
  disappeared without manual termination/environment changes, cause unknown.
  Whole-only retry completes all 12 sessions with matching output hashes and
  no accumulating per-session GPU allocations. Preserve both raw packets.
- Validated experimental source checkpoint: `d7a46ba7` (154 targeted tests,
  all-target release check). Wide-down source is also retained at `950a2db3`.
  Retire their live shaders, switches, tests and capture machinery; history is
  the reproducibility archive, not an ever-growing active research build.
- After archive cleanup and main refresh through `e7b48483`: 48 grouped-MoE,
  3 KDA and 86 GLM tests pass (137 total), plus all-target release checks.
  Existing unrelated `prompt_prefix_len` unused warning remains.
- Re-ranked next: one actual N2048 cost/dispatch reconciliation, then the
  already-measured Half latent MMA task-quality question. Do not treat old
  Fast output equality as quality. Coordinate Flash work instead of inventing
  further low-budget GLM knobs. Raw evidence: gate-up-state/kda-history packets
  under `docs/bench/2026-10-11-*`.

## 2026-10-11 - Wide Down Falsifier and Maya Deep Decode

- Tested compressed native M64/N64 expert down on real N2048/N512 routes,
  preserving the actual production SmallCounts baseline. Whole UD/Maya ratios
  0.991-1.011x do not establish a useful gain. The 8 KiB result-overlay follow-up
  is also neutral. Candidate remains test-only in the research metallib;
  production dispatch/admission/memory are unchanged.
- All recorded leaf/weighted and join/three-step continuation outputs are
  bit-identical. A cancellation-sensitive scalar-relative oracle was replaced
  by a derived accumulation bound plus the retained absolute and aggregate RMS
  checks; candidate/incumbent bit equality is checked independently first.
- Maya 8K/32K attribution: ordinary decode 34.44/32.85 ms, selected attention
  1.076/1.042 ms, score+selection growth about 0.35 ms. Different subsequent
  teacher-forced positions, not a paired speedup claim. KDA and routed experts
  each remain about 30% of profiled spans.
- Re-ranked next work to lower-state gate/up prefill, then register-resident KDA
  convolution preparation. No threshold sweep or low-budget attention rewrite.
- Final 55 grouped-MoE and 86 GLM tests pass under Metal validation; all-target
  workspace check passes. Three ignored model ABBA packets and the deep packet
  ran separately. Evidence and original-16-KiB reproducer:
  `docs/bench/2026-10-11-glm-down-wide/`.

## 2026-10-11 - GLM Leverage Refresh and Precision-Scoped Latent MMA

- Re-ranked against main `27b49476`: serving workspace selection, wide gate/up,
  native quant baselines, F32 expert down and native HC decode are completed.
  High-count expert down is next, followed by depth-dependent attention.
- Grouped Q6_K latent F32 MMA gives paired explicit-F32 Maya prefill gains of
  5.8% at 512 tokens and 8.5% at 2048. No new persistent scratch or weight
  expansion. Matrix prefixes compose with scalar tails. Half, DenseF32, Exact,
  decode and the Q8 path keep their prior production selection/arithmetic.
- The forced Half/MMA experiment gained roughly 9% but exceeded its 1e-3
  diagnostic KL bound, reaching 0.013. Against native scalar Exact, neither
  Half arm is uniformly closer; all-F32 greatly reduces this amplification.
  Preserve the failed experiment, do not call old Fast ground truth or claim
  task-quality improvement. F32 probes had no top-token changes; max A/B KL
  through three continuation steps was 3e-10/4.8e-5 at 512/2048 respectively.
- Scalar8 retained per-output arithmetic and whole-model bits but had neutral
  timings (0.996-1.006x), so is test-only rather than a new default.
- Final targeted suites: 10 grouped-latent, 86 GLM and 15 HC tests passed under
  Metal validation. The model probes ran separately; the original forced Half
  diagnostic remains a recorded failure, not a passing test claim.
- Packet: `docs/bench/2026-10-11-glm-latent-mma/` (raw attempts, precision probe,
  unchanged UD control and scalar8 falsifier). Updated `PERF-ROADMAP.md` records
  the next implementation bets and distinguishes baseline support from optional
  optimization selection.

## 2026-10-10 - GLM: 32-Token Gate/Up Tile On (Bitwise); Depth Cost Is Flat

- **Wide gate/up tile** (`docs/bench/2026-10-09-glm53-prefill-attribution/`):
  buckets of at least 17 slots take a 32-token clamped grouped SwiGLU tile
  (`kernels/moe_grouped_n32.metal`), bitwise equal per slot to the 16-token
  tile (unit tests; model logits and end state at 512 and 2,048 rows), so
  outputs are unchanged and it is on by default. Fresh 2,048-token prompt:
  -1.6% at 512-row chunks (noise-level), -4.2% at 2,048 rows (expert stage
  -6.5%). With 2,048-row chunks: 10.1 -> 8.3 s.
- **Depth:** prefill at 16K reads at the depth-0 rate (196 tok/s, MLA ~20%
  share, experts 51%); decode is 29.6 / 32.3 / 32.5 / 32.8 ms per token at
  64 / 4K / 16K / 32K. Selected attention bounds GLM's depth cost; routed
  experts remain the lever.
- **Chunk-size default stays 512** pending served-schedule quality, the
  snapshot/admission tradeoff (+1.7 GiB scratch) and router/SmallCounts
  dispatch witnesses (both decline at 1,024/2,048).

## 2026-10-09 - GLM Prefill Is 61% Routed Experts; 2,048-Row Chunks Read Prompts 14-16% Faster

- **Profiler:** `Glm5NextSession::prefill_chunk_stage_profiled` times one
  packed chunk per stage per block (sampled encoders; logits bitwise equal
  to an unprofiled chunk). Packet: `docs/bench/2026-10-09-glm53-prefill-attribution/`.
- **Where prefill goes** (512-row chunk, Fast half): routed experts 61% at
  depth 0 (1.37 s per chunk, ~2.7 ms/token), KDA 22%, MLA 7% rising to 18%
  at depth 4,096, shared expert 4%, mHC 3%. Each expert sees ~14 tokens per
  512-row chunk.
- **Chunk size:** fresh 2,048/4,096-token prompts at 512 / 1,024 / 2,048
  rows: 194 / 206 / 221 and 193 / 209 / 224 tok/s (+14-16% at 2,048 rows)
  for 0.60 / 1.17 / 2.30 GiB of packed scratch.
- **Decode experts** (10 of 30.5 ms/token): codebooks in threadgroup memory
  (10% slower) and a register epilogue (no change) were both falsified,
  bitwise equal, and not kept.
- **Next:** wider grouped gate/up token tiles for large buckets (the
  16-token tile re-dequantizes per 16 tokens), with larger chunks.

## 2026-10-09 - #12 Narrow F32 Kernels Recover About Half the One-Token F32 Penalty, Bitwise Equal to the Wide Tiles

- **What** (`docs/bench/2026-10-09-glm53-fast-f32-operands/`, "Short
  spans"; cx 01a10cc jam and review): `kernel_mat_mat_q6_K_f32_r8c8` and
  `kernel_mat_mat_q8_0_f32_r2c1k64` keep the wide F32 tiles' per-output
  operation sequence over 8-token spans; FastF32 spans of at most 8 rows
  take them (test-selected, as all F32 operands).
- **Equivalence, qualified:** bitwise equal to the wide tiles in unit tests
  (Q6_K K 256-12,288, Q8_0 K 64-4,096, spans 1-9 and 17 at several starts,
  partial output groups, poisoned neighbours; validation and release) and
  in the model (suffixes of 1-9 tokens, spans across and past the sparse
  frontier, a 5-row last chunk: logits, prompt-end and end state; census
  confirms where they ran).
- **Cost** (release): one-token suffix 91 ms half-staged, 129 ms all-F32
  with narrow kernels vs 167 ms wide-only (approximately half of the 76 ms
  penalty recovered in this screen, the target set beforehand); 2-token 166 vs 202; 4-token 207 vs
  242 (Fast 204); 8-token 274 vs 310; 9-token and fresh-prompt controls
  unchanged. Dense F32 alone: +5.1% fresh, +0.2-0.4% on 17-64-token
  suffixes, 128 ms on one token.
- **Next:** one bounded packet on the F32 expert tiles' cost (bucket
  census, fixed-input timings, one variant), then the preregistered
  holdout for the frozen candidates.

## 2026-10-09 - #12 GLM Fast with F32 Matrix Operands: Half the Drift from Exact at +13-16% Prefill; Dense-Only -30% at +4.5%

- **What** (`docs/bench/2026-10-09-glm53-fast-f32-operands/`; cx 01a10cc
  jam, review and confirmation): F32-operand tiles for every quantized matrix
  operand of Fast packed prefill, selected per stage family in tests only
  (product Fast unchanged): the shared Q8_0 tile, a new Q6_K tile
  (`kernel_mat_mat_q6_K_f32_mm64x32`) and F32 copies of the grouped expert
  gate/up and down tiles (`kernels/moe_grouped_f32.metal`; product half
  kernels untouched). A census proves every selected call took its tile
  and covers the same calls as the half run.
- **Accuracy** (six frozen natural cases, 512 rows, against Exact;
  diagnostic): equal-case mean of per-position KL 8.28e-3 (Fast), 8.40e-3
  (Q8_0 only), 5.83e-3 (dense projections: five of six cases lower, H1
  higher), 4.16e-3 (all F32). Under all F32, H3 and H6 match Exact to
  reduction order (worst KL 1.8e-7, 5.5e-8); H1, H2 and H5 keep drift at a
  few positions (largest worst KL 0.116 vs Fast's 0.113), coinciding with
  routing changes that begin at a single row (H1: block 7, spreading to 654
  row-blocks; Fast changes routes from the first MoE block on every case).
  Closer agreement with Exact, not a demonstrated quality improvement.
- **Bitwise properties hold under all F32:** 512 == 128 and 64 == 97 rows;
  capture/restore at every pool residue and around the sparse frontier, a
  cancelled boundary, and 512 vs 128 rows past the frontier, in logits and
  end state. Kernel tests: Q6_K relative RMS 4.3e-7 vs the half tile's
  2.3e-4; experts 1.8e-6 vs 4.7-6.1e-4; per-token outputs independent of
  the dispatch.
- **Cost** (release, ABC-CBA twice): fresh 2,048-token prompt +4.5% (dense),
  +12.7% (all; +16.1% in an earlier run); 64-token suffixes at 2,048 and at
  sparse depth -2.0%/-1.7% (dense) and +15.6%/+15.5% (all); 17-token
  +1.9%/+4.3%; one-token suffix +78-84% (92 -> 164-170 ms: a 32-token tile
  for one token where half Fast uses mat-vec).
- **Next:** a one-row path keeping each candidate's per-token arithmetic;
  one bounded packet on the F32 expert tiles' cost; then freeze the
  surviving configurations (dense, all F32) and run a preregistered fresh
  holdout against Exact under the deployed split schedule, selection rule
  fixed before viewing.

## 2026-10-08 - GLM Fast Snapshots On by Default; Branches and New Sessions 58 s -> 0.5 s; Fast Accuracy Lane Opened

- **Decision:** Fast snapshots are on by default (`b8ab3693`;
  `QWEN_GLM_FAST_SNAPSHOTS=0` turns them off). An explicit exception accepts
  unresolved quality uncertainty in exchange for the measured latency
  benefit (~58 s per branch, new session or return). The preregistered
  comparison against Exact stays recorded as INCONCLUSIVE (one arm 0.0054
  vs 0.005; unsplit Fast's own bound was 0.0046) and is not retroactively
  qualified. Supporting, not qualifying: against unsplit Fast, the split
  arms' mean-NLL bounds lie within the cohort's limits (grid +0.0019,
  tail +0.0040, frontier long +0.0095 upper bounds), with the frontier arm's
  long top-1 still INCONCLUSIVE (six documents).
- **Rule for future schedule changes** (amending the adopted Fast policy),
  preregistered: a schedule-only change (kernels, precision and dispatch
  arithmetic unchanged, established by inspection, since segmentation can
  change dispatch selection) is judged for non-inferiority both against
  the incumbent Fast schedule and against a fixed, versioned reference
  (the quality-v1 Exact reference), so successive changes cannot degrade
  cumulatively; a change to Fast's arithmetic keeps the comparison against
  Exact.
- **Release screen** (`docs/bench/2026-10-08-glm53-fast-snapshot-screen/`,
  opencode-shaped conversation, 11,104-token shared prefix): branch,
  new-session and return requests restore in 5-11 ms and reach their first
  delta in 0.50-1.73 s, against 57.7-59.6 s with snapshots off; one 28 ms
  capture (0.27 GiB) on the cold request.
- **Next: make Fast more accurate** (cx 01a10cc jam): the dense projection
  families (KDA, MLA, shared expert, dense FFN; Q6_K/Q8_0) round both
  activations and dequantized weights to half in their mat-mat tiles, and
  the routed experts do the same; all-F32 Q8_0 tiles already exist (DS4,
  GLM absorption), Q6_K and the expert tiles have none at prefill widths.

## 2026-10-08 - Saluki Now Loads Without Weight Inflation

- Native IQ1_S/IQ1_M matrix execution plus qualified IQ1_M embedding gather
  completes the IQ2 checkpoint:851 direct source-dtype bindings, zero weight
  conversions,7.345690 GiB logical storage. Eliminated58.726997 GiB compared
  with the original plan, including19.193573 GiB from this completion.
- Loaded point samples:Metal7.349045 GiB,physicalfootprint7.513699 GiB,
  RSS14.900940 GiB (copied weights plus mapped source pages). Not peaks,
  not7.35GiB RSS, and no expanded baseline model was loaded.
- All36S/31M tensors receive real-row primitive checks. Native embedding
  matches66,560 independently decoded coefficients over anchors and actual
  prompt IDs. Whole native packed/GEMV comparison:20finite outputs,10same-top1
  comparisons, maxbidirectionalKL5.063457e-7. Ordinary concurrency remains on.
- Fifteen primitive,11role,2adapter and2embedding tests passMetalvalidation;
  independent source/data review approves. Autoembedding is scopeduntied27B;
  expert/head native policies remain unchanged. No whole-speedup/quality claim.
- Raw v2 evidence and limits: `docs/bench/2026-10-08-native-iq-capacity/README.md`.


## 2026-10-08 - Native Dense IQ2 Cuts Saluki Logical Weights by 39.533 GiB

- Role-aware IQ2_XS/IQ2_XXS projection residency replaces persistent F32
  expansion. Output heads, embeddings, MoE projections/banks and the legacy
  native whitelist retain prior policies. Prepared choices are frozen before
  allocation; planner/loader realization matches all851 bindings.
- One normally admitted Saluki load realizes26.539 GiB logical weights:
  784 direct copies,67 remaining IQ1 conversions. Loaded Metal allocation,
  RSS and physical footprint are recorded separately; no inflated baseline
  model was loaded and no measured75-to27GiB RSS claim is made.
- Canonical codec/F64 primitive checks cover all39XS/119XXStensor row samples.
  Whole native packed128 versus native GEMV on N129/131 plus4 common tokens
  yields10 comparisons, maxKL5.728e-7, same top1, finiteoutputs. Native decode
  concurrency and matrix dispatch witnesses pass. This is not broad quality
  or whole-model speedup evidence.
- Final13IQXXS/dispatch and11role tests, XS checks, workspace/release checks
  and independent adversarial review pass. GPU work waits for other owners.
  Evidence: `docs/bench/2026-10-08-native-iq-capacity/README.md`.
- Next: IQ1_M matrices/embedding and IQ1_S matrices; remaining expansion
  ~19.194 GiB. Shared session-pricer undercount595,168 bytes is tracked
  separately; the existing2GiB reserve covered it in qualification.


## 2026-10-08 - #12/#15 Fast Shared-Prefix Split: INCONCLUSIVE; Fast Snapshots Stay Opt-In

- Preregistered `quality-split-v1` (`docs/bench/2026-10-08-glm53-fast-split-quality/`,
  run at `d65a09a2`, rebased as `9148817d`, Metal validation): grid (cut at the deployed agent
  prefix's in-chunk offset), tail (17-20 tokens after the cut) and frontier
  (cuts 2050-2053) arms against a full Exact rerun. Every gated limit passed
  except the tail arm's overall NLL: +0.0011 nats/token, central 90%
  [-0.0034, +0.0054] against 0.005. Tools 12/12 and 1.00 sampled in every
  arm; agent-shaped cases (11,104-token opencode prefix) 8/8 first-call
  checks in every arm.
- Reported: split minus unsplit Fast intervals include zero; no split arm
  is bitwise equal to unsplit Fast on any text item; this run's Exact and
  Fast 512 reproduce the committed quality-v1 metrics exactly (38/38).
- Decision by the rule: Fast snapshots stay opt-in; Exact snapshots are on.
  Next: a fresh confirmatory cohort (`quality-split-v2`, v1 for planning
  only: frozen hash-ordered selection, the same three gates and limits,
  `fast_512` as a control, N fixed in advance from a simulated joint-gate
  power; about 70-100 fresh documents for one gate), then the release
  time-to-first-token screen.

## 2026-10-08 - #14 Output Memory, Packet 3: Qwen/DS4 Tool-Block Byte Ceilings

- Ordinary Qwen, Flash-Next and DeepSeek V4 profiles now carry their
  backend's default token limit and longest decoded piece (computed once at
  load); the output protocol's tool-block ceiling is the resolved limit x
  longest piece x 3 (UTF-8 replacement), as GLM's (`a96df045`, `83dc518f`;
  cx jam and reviews). Normalization refuses an uncomputable ceiling (400)
  wherever the protocol parses tools (always for ordinary Qwen and
  Flash-Next); the protocol fails closed otherwise; the backends resolve
  their token limit through the same helper.
- A block past its ceiling fails the turn as a typed 500 internal
  accounting error (not pressure, no raw-text fallback); reservations never
  pass the ceiling. With packet 2's admission, the Qwen/DS4 tool block is
  bounded both by accepted length and by fresh headroom.
- Scope: the retained tool-block text; UTF-8 assembly, held-back visible
  text and the first marker's copy happen before the block's admission.
  Remaining #14 items: K2 and Muse tool buffers, response accumulation and
  streaming retention, trace bytes, K2's schema-work budget.

## 2026-10-08 - #14 Output Memory, Packet 2: Qwen-XML and DeepSeek-DSML Tool Buffers Admitted

- Measured first (`643fddfc`): Qwen XML and DS4 DSML tool blocks of GLM's
  four shapes through the real handler in one-byte pieces peak at most
  147.5 bytes per block byte, the same as GLM's (the shared argument parse
  and publication dominate), so GLM's retained 256x model applies; four
  shapes are evidence, not a bound.
- Now (`c59aa130`, `9684df1d`; cx reviews): the Qwen/DS4 tool buffer admits
  each 64 KiB step at the checked 256x outstanding peak against fresh
  headroom and again before the parse (GLM's admission now uses the same
  checked pricing); refusals are stored, inert and returned by `finish`.
  A stored Qwen or GLM failure stops a streaming generation at the refusing
  piece and fails it with the typed code, publishing no held tool text;
  non-streaming answers the JSON error after collection.
- Not yet: a request-derived byte ceiling for Qwen/DS4 (their profiles do
  not carry the effective token limit and piece bytes), so admission bounds
  memory against headroom, not accepted output length; K2 and Muse tool
  buffers, response accumulation, streaming retention and trace bytes.

## 2026-10-08 - #14 Output Memory, Packet 1: Admitted Contiguous Non-Streaming Collection, Typed Sink Refusals

- Non-streaming responses held one allocation per output piece and a
  retained event per piece until the body was written, unpriced in every
  family. Measured end to end (test allocator, plain output in one-byte
  pieces, 200,000 bytes): **246.5 bytes per output byte** above a minimal
  response; streaming 5.2.
- Now (`1c1c560c`, `ce4b1e57`; cx 01a10cc jam and reviews): one contiguous
  buffer plus piece end offsets, each growth step admitted before
  allocating (the step's whole peak less what is held, against fresh
  process headroom, Batch 3a's typed refusals; checked arithmetic, typed
  500 `memory_size_overflow`; a failed admitted reservation is a typed 500);
  pieces replay into the partition one at a time with their events fed
  straight into the response object. **16.3 bytes per output byte**; GLM's
  non-streaming tool block worst ratio fell from ~205 to 146 (model 256).
- A refusal raised inside a sink is carried as a typed error, not a
  disconnect: JSON before headers, `response.failed` after, the server
  failure reported once (direct and across the HTTP owner bridge).
- Scope: admitted collection only. Response accumulation and terminal
  copies, streaming retention, non-GLM tool parsers (Qwen/DS4, K2, Muse)
  and trace bytes remain unpriced (next packets, in that order after the
  tool parsers); K2's schema-work budget runs in parallel.

## 2026-10-08 - #12 Fast Policy Adopted; Drift Tests Migrated; Fast Split Qualification Preregistered

- **Adopted** (the position below, with the review qualifications): Fast
  stays serve's default; bitwise properties are hard (reuse, cancel/resume
  and restore within one lineage and schedule; Exact equals serial decode
  and is selectable per request); any Fast change or new schedule that is
  not bit-identical passes the preregistered quality cohort (a fresh
  holdout for substantial or tuned changes); Fast-vs-Exact and warm-vs-cold
  drift of the measured class are reported, not bounded; DFlash keeps its
  serial-equivalent verification requirement.
- **Investigation trigger** (explicitly chosen, provisional, not a quality
  limit): worst-position KL(Exact || Fast) > 0.33 on the frozen natural
  cases, Fast 512 rows, cold and warm (`reuse_natural_evaluate`, report
  schema v2: per-arm change from the hash-pinned 2026-10-07 baseline; 97
  rows diagnostic). Non-finite logits or metrics always fail; flips are
  counted by argmax inequality with both regrets.
- **Tests migrated** (`4490f96a` and review fixes): the warm/cold reproducer
  is now `fast_reuse_schedule_sensitivity_across_reuse_boundaries`
  (cancel/resume bitwise hard, drift reported); the chunking test is
  `packed_fast_chunkings_keep_bit_identities_with_drift_alarms` (oracle and
  bit identities hard, drift reported, state bounds kept as named
  regression alarms). The trigger applies only where preregistered.
- **Fast split qualification preregistered** (`quality-split-v1`, cx review
  before any GPU run): a full Exact rerun; Fast 512 control; split arms at
  the deployed agent prefix's in-chunk offset (11,104-token opencode
  prefix, offset 352), with a short final segment, and across the sparse
  frontier (long items); the tool tasks at their real 255-token boundary;
  four agent-shaped cases at the deployed geometry (descriptive). Gates:
  quality-v1's limits for every split arm, and the tool screen; otherwise
  Fast snapshots stay opt-in.

## 2026-10-08 - #15 GLM Serve Snapshots: Exact On by Default, Fast Opt-In Pending Its Quality Arm

- GLM serve keeps a RAM snapshot cache (`--snapshot-cache-mib`, shared
  expiry and eviction flags; no durable tier). The renderer records the end
  of the shared instructions-and-tools prefix and the start of the
  generation header while appending; serve verifies each against the
  prompt's tokens. A request of a snapshot schedule splits its prefill at
  those cuts on hit, miss, full cache or denied capture, so a restore
  continues exactly the trajectory a miss runs (cx 01a10cc jam). Entries
  are namespaced by (lineage, schedule) under one budget; reuse takes the
  longer of the live-session extension and the best compatible snapshot.
- Exact (`x_qwen.prefill_lineage: "exact"`) cuts at both boundaries and is
  on whenever the cache has a budget: packed Exact equals serial decode, so
  restores equal cold runs. Fast (the default lineage) stays off; its
  shared-prefix split is a new schedule that enters the #12 quality cohort
  as a candidate arm before it can be a default. `QWEN_GLM_FAST_SNAPSHOTS=1`
  opts in for that qualification and for #13's cost screen. Fast restores
  only at the request's own verified cut and publishes only states on the
  canonical schedule (review: an older shorter cut or an off-grid live
  session would otherwise publish a different trajectory). Pressure
  refusals of the transport or a new session release the cache and retry
  once.
- GPU gate (Metal validation, 1,173-token shared prefix): Exact miss, shared
  hit and transcript hit equal cold runs bitwise (logits and bytes); Fast
  hit equals a miss on the same split, as do a full cache and denied
  captures; Fast entries never serve Exact; split prefills cancelled after
  one chunk resume onto the same schedule in both lineages; a shorter cached
  cut is ignored, an off-grid live state is not published, boundary-less
  Fast requests neither restore nor capture, and a continued request cannot
  evict the shared entry it passed. Observed, not relied on: for this
  prompt the Fast split's logits equaled the unsplit Fast prefill's.
- Next: the Fast shared-split quality arm (preregistered: segmentation
  algorithm, split points below and across the frontier and at pool
  residues, all strata, real rendered tool boundaries, an ~11K agent-shaped
  case), then #13's screen with the opt-in for cost.

## 2026-10-08 - #12 GLM Fast Prefill Passes a Preregistered Quality Cohort; Half-Rounded Activations Alone Reproduce Its Drift

- **Quality** (`docs/bench/2026-10-08-glm53-fast-quality/`; cohort and
  limits committed before the GPU run, analysis amended after cx review and
  before viewing): Fast at 512 rows vs Exact on 38 continuation items from
  22 repository documents (prose, code, long past the sparse frontier).
  Overall NLL +0.0004 nats/token (central 90% [-0.0039, +0.0046]; the upper
  bound 0.0046 sits close to the 0.005 limit); every stratum within 0.010;
  top-1 accuracy unchanged; tool screen 12/12 greedy and 1.00 sampled for
  both lineages. QUALIFIES on this cohort and screen, not as general
  answer-quality equivalence. Fast at 97 rows passes overall; its long
  stratum is inconclusive.
- **Mechanism** (rounding probe, two frozen cases): Exact with only its
  activations rounded through half precision drifts from Exact by worst KL
  0.164 (H2) and 0.154 (H5), against Fast's 0.113 and 0.095; Fast is as far
  from rounded Exact as each is from Exact. Activation rounding alone is a
  sufficient source of drift of this size; its share within Fast and what
  F32 staging would leave are unmeasured. The F32-staging experiment is
  deprioritized given the cohort result, not refuted.
- **Policy position** (proposed here; adopted later the same day, entry
  above; reviewed by cx 01a10cc):
  keep Fast as serve's default and replace the one-prompt agreement bound
  (KL 2e-2, regret 0.2) with separate properties. Bitwise: reuse,
  cancellation-resume and snapshot restore equal the uninterrupted run
  within one lineage and one schedule (segmentation and serial-versus-packed
  spans); Exact equals token-by-token decoding and stays selectable per
  request (`x_qwen.prefill_lineage`, `f1b00b35`); bit-identity claims for
  Fast kernel changes are shown. Gate: any Fast change or new schedule that
  is not bit-identical passes the preregistered cohort's limits, with a
  fresh holdout for substantial changes (this cohort is now visible).
  Accepted and reported: Fast-vs-Exact and warm-vs-cold drift of the
  measured class (worst-position KL to ~0.2, flips with regret up to 1.50).
  Investigation trigger (explicitly chosen and provisional, not a derived
  envelope): on the frozen natural cases, worst KL above ~0.33 (twice the
  half-rounding drift), with per-case baselines and both choice regrets
  kept beside it; non-finite values always fail. DFlash keeps its own
  serial-equivalent verification requirement.
- Evaluated before the sparse IQ3_S down retile landed; that packet observed
  bit-identical outputs, so the results are expected to carry over.

## 2026-10-08 - #15 GLM RAM Snapshots: Engine Capture and Restore Bitwise

- `Glm5NextSession::capture_snapshot` / `restore_snapshot`
  (`glm5_next_metal/snapshot.rs`, `6d65d90e`): KDA conv tails and state,
  MLA latent rows `[0, n)`, completed pooled keys and the pending ring with
  its phase; bound to the weights instance, the prefill lineage and a policy
  version; restore validates before writing and refuses with a typed
  mismatch.
- Gates (GPU, Metal validation): restore then suffix equals the
  uninterrupted run bitwise for both lineages, every pool residue, around
  the sparse frontier (Exact n = 2051-2052, Fast 2050-2054), a cancelled
  prefill's committed boundary and a recycled destination of another
  capacity; refusals for lineage, weights instance, policy version and
  capacity. A session refuses a lineage change once it has committed
  tokens (review fix), so a capture cannot carry one lineage's state under
  the other's label. Rerun on the integrated tree (`652742b1`): snapshot
  and intervention gates pass, as do the lib (1,316) and CLI suites.
- Size: about 0.14 GiB of fixed recurrent state plus ~11.7 KiB per token
  (0.14 GiB at 100 tokens, 0.17 GiB at 2,052; estimator 0.27 GiB at the
  11,129-token instructions-and-tools prefix #13 measured replaying in ~61 s).
- Serve does not use it yet; serve integration (capture at the
  renderer-reported system/tools boundary and before the final assistant
  header, compatibility-aware lookup in the shared snapshot cache) is next.

## 2026-10-08 - GLM Native Directions Reproduce the Archived llama.cpp Dose-Response Study

- `qwen-lens run` gains a GLM-5.3-Flash lane: raw directions, operations at
  module sites (mixer, routed-expert, shared-expert and FFN outputs,
  embedding), before/after module readouts, first-token top-k and tracked
  logprobs, typed `</think>` prefill with provenance (`f8a499b1` ..
  `54a66fd9`, the never-merged directions lens work plus GLM sites).
- Sweep (`docs/bench/2026-10-08-glm53-directions/`): projection at the
  mixer and shared-expert outputs of blocks 15-36, 19 doses from -3 to 6,
  8 prompts, prompt token IDs equal to the reconstructed archived
  rendering tokenized by llama.cpp. Against the archived adapter sweep on
  the same IQ3_XXS weights: labels 152/152, every flip interval identical,
  first token 148/152 (near-ties), 8-token text 137/152; "I" logprob
  difference median 0.011, maximum 0.64.
  Behavioral concordance; no cross-engine numerical budget was calibrated.

## 2026-10-08 - MLA Tail Experiment Retained Off; Native IQ Capacity Queued

- Guarded F32 grouped-Q8 tail batching removes repeated GEMV dispatches.
  Whole target GPU saves3.07-6.02% at sampled N32/64/127/255 across two
  streams; deep127 suffix saves6.14-6.76%, excluding its4096-token setup.
  N8 is inconclusive; N128/129/512 are unchanged controls.
- Production remains unchanged. Code-N127 shows endpoint KL up to0.08747
  and a repeatable first-continuation top1 change; incumbent/candidate-side
  regrets differ (0.01762/0.35316). No new bit gate, no claim that A is truth.
  Route-driven numerical/timing amplification is plausible but unmeasured.
- Four all-fragment primitive tests, three scope/split tests and the observer
  test pass. Independent review finds no harness defect but requests targeted
  numerical attribution before promotion. Evidence and next bounded GEMV
  alternative: `docs/bench/2026-10-08-glm-mla-tail/README.md`.
- User-requested Saluki capacity work is next. Metadata inspection confirms
  four unsupported native-storage cohorts, not IQ2_XXS alone:225 tensors,
  3.841 GiB compressed ->62.568 GiB F32; avoidable expansion58.727 GiB.
  Reported75 GiB RSS/7tok/s not reproduced. IQ2_XXS is32.068 GiB of the
  potential saving; existing IQ2_XS kernels and IQ1_M embedding gather also
  matter. Role-aware scope: `docs/archive/IQ-QUANT-NATIVE-CAPACITY-2026-10-12.md`.

## 2026-10-08 - GLM Sparse IQ3_S Expert Down Converts to Whole-Prefill Gains

- GLM Fast now selects M128/N16 down for expert counts 1-16, retaining
  generic down for counts 17+; IQ3_S K2048/M4096/E288, actual rows 32-512.
  Exact, other families/dtypes/widths stay unchanged. Checked capability
  fallback precedes grouped dispatch; no new production allocation.
- Two-corpus domain screen saves 7.34-10.93% GPU at sampled N32-256,
  3.44-4.15% at N512 and 3.84-4.05% on ordinary eight-chunk 4K calls.
  Final production-versus-forced-incumbent packet confirms 8.73-10.89%
  at N32/128, 4.25-4.33% N512, 2.71-3.61% 4K (~0.59-0.78 s).
  The lower 4K result has drift; all paired GPU and wall comparisons win.
- Endpoint and four-token teacher-forced continuation comparisons match:
  120 domain and 96 final comparisons, including incumbent-repeat controls.
  This does not requalify Fast versus Exact or impose a bit-equality gate.
- Eight leased Metal-validation primitive/fallback/isolation tests, family
  policy and diagnostic CPU tests, release build/non-test check and independent
  cx review pass. Shared GPU work waited for the quality evaluation lease.
- Evidence and retained negative concentrated-route controls:
  `docs/bench/2026-10-08-glm-iq3-down-retile/README.md`.
  Next: Flash scheduling on the promoted-router baseline, GLM ragged
  absorption/expansion, then Flash selected-QSA leaf attribution.

## 2026-10-08 - Existing Router Kernels Reach Flash Tails and GLM Packed Chunks

- Flash-Next now admits strict E8P32 at N2045, without changing the
  2048+3+2045 planner. Whole4K prefills save about7% GPU latency on both UD
  and GSQ (~0.63 s); both natural streams and both ABBA pairs improve. A
  repeated UD packet confirms7.22%/7.44%; inflated drifty means are retained
  but not used as headlines. Five witnesses per artifact isolate48 changes.
- GLM Fast uses the existing guarded kernel at N128/N512: final packet
  saves1.43%/1.66% GPU, with paired wall wins. Exact and N32 stay unchanged.
- No new shader or state approximation. Finite outputs, causal metadata and
  continuation checks pass. A test-only observer was reading stale singleton
  hyper scratch after packed execution; corrected to use the actual last
  packed row, preserving the original evidence and its limitations.
- Ten CPU checks, the leased all-width Metal differential, workspace check
  and adversarial review pass. Raw evidence and decision:
  `docs/bench/2026-10-08-prefill-router-transfer/README.md`.
- Next: expert density, then remaining planner benefit measured against
  efficient routing; old mixed-schedule savings are not additive.

## 2026-10-07 - #12 on Natural Trajectories: Fast Fails Like llama.cpp's Batched Prefill; #13 Screen Sizes Prefix Replay

- **#12** (`docs/bench/2026-10-07-glm53-natural-reuse/`): six frozen natural
  cases (the model's own turns, Exact-greedy continuations, fixture
  `reuse-natural-v1.json` committed before any Fast run; serve's CPU round
  trip renders every turn exactly). Exact warm equals cold bitwise in logits
  and state everywhere. Fast vs Exact: worst KL 8e-3 to 0.18, flips with
  regret up to 1.50 (H2 code, H5 long chat); warm vs cold Fast up to 0.165.
  The natural subtask fails; no bound changed.
- **Attribution (H2, 512 rows):** several Fast stage families diverge
  independently (alone: MLA projections 0.22, KDA expansions 0.19, shared
  expert 0.16, KDA projections 0.14, dense FFN 0.10, routed experts 0.07;
  the F32 router and MLA absorption ~1e-2), and replacing any single family
  does not restore the policy. Half staging is a candidate shared
  mechanism, not yet isolated (weight tiles are half too).
- **Reference context:** llama.cpp batched vs its serial run on the same
  tokens: H2 worst 0.17 with 4 flips, H1 4.2e-2; the same order as native
  Fast vs Exact (the oracle batches the continuation too; not an envelope).
- **Exact cost:** fresh prefill 39.5-40.3 tok/s vs Fast 186-197 (4.7-4.9x);
  a 64-token suffix 1.6 s vs 0.8 s.
- **Next:** a bounded F32-staging feasibility experiment before any bound
  change; Exact by default not recommended.
- **#13** (`docs/bench/2026-10-07-glm53-agent-reuse-screen/`, scripted, no
  opencode process): one scripted continued step in the patched client's
  shape reuses 11,185 of 11,284 tokens (1.25 s to first token); a branch, a
  new session with the same instructions and tools, and a return each
  replay ~11K tokens at ~185 tok/s (~61 s). #15 is promoted on this
  measurement (potentially avoidable replay on matching-prefix hits).

## 2026-10-07 - Leverage Map Re-Ranked After Concurrent Flash-Next Work (cx `01a10cc`)

- **Concurrent landings, logged here:**
  - GSQ for Flash-Next (`7c75b21f`, `9f0aca03`, merge `95a83129`; packet
    `docs/bench/2026-10-07-quant-compat/`): current GGML wire types admitted
    with bounded CPU decoding; per-role native dtype admission, Q2_0 kernels,
    generic grouped IQ2_S/IQ3_S experts. Smoke requests only; quality and
    controlled performance are unclaimed.
  - Release sampling for Flash-Next (`09b0c9c4`): omitted fields come from
    its generation_config (temperature 1, top-p 0.95, top-k 20, min-p 0) in
    run and serve, as for Muse and GLM. Later the same evening `60f6071e`
    gave ordinary Qwen, DS4 and K2 their releases' published presets too
    (previously temperature 0, greedy); a holdout qualification remains.
- **New rank:** #12, #13, #16 with #14 (holdout qualification of the new
  sampling defaults inside #16), #18 (new), #15 (conditional on #13), #9, #3(a), #2, #8 (after
  a Flash-Next re-baseline), #19 (new), then #6, #10, #3(b). Tracked: #1's
  open items, #5 v2, #7, #17 (demoted: no 2-8-row hot path shown for
  Flash-Next or K2). GLM NextN stays a paper candidate.
- **Moves:** #12 ahead of #13 (a default failing its declared policy);
  #9 ahead of #2 (the ~3.3 ms expert figure is bandwidth headroom, not a
  demonstrated saving, while an 18-token GLM tool step prefilled in 400 ms).
- **Correction (GSQ packet):** selected packed attention is not gated off
  for GSQ. `packed_selected_capable` compares the prompt extent (or session
  capacity) with the selected output width, 2051
  (`qwen4exp_text_session.rs:328-339`, `qwen4exp_qsa.rs:798-800`); the smoke
  capacities were 256 and 1024. Above 2051 the path is enabled for GSQ and
  unqualified (first child of #18). The packet sentence is replaced.
- **Decline census (#18):** fast-path declines classed S/Q/I/P across
  families. Examples: Flash-Next's IQ3_XXS expert specializations cannot read
  GSQ's IQ2_S/IQ3_S banks (generic kernels run, cost unmeasured; the
  specialized down dtype is IQ4_NL); its HC up-mix contract (Q8_0,
  4x2560, K320) declines GSQ's BF16 HC projections while the load line
  prints `hc_up_mix=true` (permission, not use); GLM coverage gaps refuse at
  binding; GLM Fast (#12) is enabled despite a failed qualification; DS4's
  device, artifact, expert-count and chunk qualifications sit inside #9.
- **#13 sharpened:** GLM's live session reuses only exact extensions, so with
  the unpatched client every step whose history carried dropped reasoning
  was a full re-prefill. Families with LCP truncation keep the common
  prefix; snapshot families restore from the latest compatible checkpoint.
  With the patched client, continued steps should extend; prefill retries
  resume from the committed chunk boundary. The screen counts continued
  steps, branches, shared-prefix new sessions and cold starts separately.
- **#19 DFlash2 beyond Qwen:** incoai publishes drafters for GLM-5.3-Flash,
  GLM-5.3, Muse-Glimmer-30B, Qwen3.6-35B-A3B and Qwen3.8-27B. A bounded
  feasibility packet (Muse first as lower risk): GLM Exact verify2/verify8
  cost (a uniform-routing illustration gives ~58 distinct experts per layer
  for 8 rows against 8 for one decode row), the KDA/MLA checkpoint contract,
  and the drafter's feature convention (upstream averages the mHC streams).
- **Memory statement corrected:** this host's `iogpu.wired_limit_mb` is
  120832 (118 GiB), not the 112 GiB the GLM plan and ledger comment stated.
  The UD-IQ3_XXS trunk leaves ~8.5 GiB; ~7.2 GiB after the fixed session
  terms is a planning estimate assuming 512-row packed scratch, and ~640K
  positions is a cache-only quotient, not an admitted context.
- **Not ranked:** the Maya-S GLM requant (IQ2_XXS gate/up, mostly IQ2_S
  down). The complete artifact is ~22 GiB smaller (89.8 vs 112.1 GiB);
  routed bytes per token would fall only from 2.84 to 2.23 GiB, and the
  decode benefit is unmeasured. Its card reports KL 0.428 against FP8 and
  the worst agreement on tool calls, and IQ2_XXS has no Metal path here.
  Byte reduction alone does not justify the implementation.

## 2026-10-06 - Lane Convergence Batch 2: `qwen run` Output and Stats

Packet `docs/bench/2026-10-06-glm53-lane-audit/` (Batch 2 section).
- **Output:** K2 and GLM `qwen run` print the decoded generated bytes like
  every family, replacing the inherited stderr/stdout split.
  `--format responses` is the explicit structured format (verified K2 and
  GLM chat).
- **Stats:** both lanes use the established stats line;
  `weights_first_use=true` labels no-copy families' single-turn prefill.
- **Live:** K2 chat (text equals raw, all efforts), K2 tools (`--format
  responses`) and GLM run checks all pass on the final binary.
- **Stale check fixed:** K2's chat check had failed since 2026-09-24.

## 2026-10-06 - GLM-5.3 Lane Audit; Convergence Batch 1

Packet `docs/bench/2026-10-06-glm53-lane-audit/` (reviewed adversarially by
cx 01a10cc, with recall checks of recorded intent).
- **Findings:**
  - GLM was modelled on K2, the least representative bring-up, and
    inherited its `qwen run` conventions (split stdout/stderr, JSON tools,
    private refusals).
  - GLM's single live session rested on a wrong claim ("KDA cannot
    rewind"): the Qwen and Flash-Next hybrids snapshot recurrent state at
    boundaries. Corrected in SERVE.md, the plan and the code; GLM snapshots
    are roadmap #15.
  - Established families carry drift too:
    - Muse `qwen run` ignored `--drafter`;
    - Muse, K2 and GLM ignored the snapshot-policy flags;
    - text `qwen info` was wrong for five families;
    - DS4 ignored `--durable-idle-publish-secs`.
- **`c640a113`:** the few-row Q8_0 arm reached Flash-Next and K2
  unqualified, and three route tests failed on main.
- **Batch 1 (`11b488b2`..`89d30867`):**
  - drafter admission settled before family dispatch in run and serve;
  - unsupported snapshot-policy settings refused;
  - GLM `qwen run --messages` renders missing reasoning as serve does;
  - text `qwen info` rendered from the JSON projection, per family;
  - few-row routing contained to Qwen callers.
- **Tests:**
  - qwen-llm 1,251 passed, 0 failed (first clean suite since `c640a113`);
  - qwen-cli all pass;
  - new binary-level tests for drafter refusal and text info.
- **Next:** `qwen run` presentation (batch 2), then errors and admission.
  Roadmap #15-#17.

## 2026-10-06 - GLM-5.3 Review Fixes Qualified Live; #12 Fast Reuse Fails Its Frozen Gate

Packet `docs/bench/2026-10-06-glm53-review-qualification/`. Binaries frozen
at `dfef7dc1`.
- **Gates:** 14/14 pass under the debug layer.
- **Tool loop:** the live tool loop reuses 218 of 236 tokens.
- **Idle residency:** every verdict passes (polling and refusal never renew;
  a completed request and a client abort after submission each renew once).
- **#12 (known failing qualification):**
  - The frozen warm/cold gate fails: ordinary KL 2.13e-2 with a 0.397-regret
    flip; tool loop across the frontier KL 5.6e-2.
  - An Exact reference shows that cold and warm Fast both exceed the
    existing Fast policy: cold ordinary 2.17e-2; tool loop 7.9e-2 and
    0.111, both flipping position 2 at Exact-side regret 0.54.
  - Cancellation and resume on an unchanged schedule are bitwise.
  - No regression from this branch was identified. A default-Fast
    investigation is open; no bound change.

## 2026-10-06 - opencode Reasoning Replay (Map #13): Lost in the Client's Bundled Provider, Fixed in the Fork

Packet `docs/bench/2026-10-06-opencode-reasoning-replay/`. CPU only: a
scripted mock in serve's shapes, with opencode 0.1.0 isolated from the
calling session.
- **Result:** no reasoning item is replayed, in any of 15 requests. That
  holds within one multi-step tool run (the request after a tool result
  lacks the reasoning before the call) and across runs (`-c`), in-process
  (`--local`) and server-attached (`opencode serve` + `run --attach`), with
  the open-responses and openai provider ids.
- **Cause:**
  - opencode stores reasoning parts and passes them to the provider.
  - Its bundled `@ai-sdk/open-responses` 1.0.35 has no reasoning input
    conversion.
  - It cannot load the cached 2.0.29 (AI SDK 6 provider specification),
    the build #4's provider capture used. That packet is corrected.
  - Its bundled `@ai-sdk/openai` keeps only reasoning with OpenAI item ids.
- **Effect:** where a family keeps past reasoning (GLM-5.3 by default),
  each agent step re-renders the previous step with empty reasoning. Live
  reuse stops there, and conditioning differs from generation.
- **Fix (fork `8084ffaf7b`):** a bun patch backports the 2.x reasoning
  case. With it, local and attached runs replay every past reasoning item
  in order (`report-patched.json`). The new provider test fails on the
  unpatched converter.
- **Landed:** on `fork-dev` and installed (`install:local`). The installed
  binary replays every past item.
- **Open:** a live GLM-serve opencode session, unpatched vs patched.

## 2026-10-05 - GLM-5.3-Flash Native Tools (Map #11)

Packet `docs/bench/2026-10-05-glm53-tools/`.
- **Rendering (`871f93e8`):** the jinja oracle gains 29 tool cases:
  definitions, calls with every `tojson` type, results reordered to call
  order, loops with `clear_thinking`, and refusals. The renderer is
  byte-exact on all 56 render cases, and native token ids equal HF's.
  K2's `tojson` and schema-type masks moved to a family-neutral
  `tool_schema` (`a4550d82`).
- **Parsing (`c39e1728`):** a streaming `<tool_call>` parser. It holds
  back markers, buffers the block within a byte budget, types arguments by
  schema so re-rendering a parsed call reproduces its bytes, and treats a
  malformed block as an error, never as text.
- **Serve (`2c9285ff`) and `qwen run` (`f00e5230`):** tools, `function_call`
  and `function_call_output` items, `function_call` output items; the run
  lane prints a Responses object.
- **Live:**
  - the model emitted `get_weather {"city":"Paris","days":2}` through
    serve (JSON and SSE) and `qwen run`;
  - the replayed loop reused 218 of 236 tokens (401 ms vs 1.5 s cold
    prefill), because `<|observation|>` ends the call turn the way
    `<|user|>` ends a chat turn.

## 2026-10-05 - Empty Reasoning Round Trip (Map #4): Provider Exact, Serve 400 Fixed, Client Omits Reasoning

Packet `docs/bench/2026-10-05-empty-reasoning-round-trip/`.
- **Provider (`@ai-sdk/open-responses` 2.0.29, from opencode's cache):**
  - Serve's empty reasoning item becomes a reasoning part with empty text
    and `reasoningContent` metadata, both streamed and not.
  - Replayed with that metadata, it returns exactly as emitted.
  - Replayed without metadata, it becomes `{"type":"reasoning",
    "summary":[]}`, which serve refused with a 400.
- **Fix:** a reasoning item with neither `content` nor `encrypted_content`
  is empty reasoning, in the shared parser and in both K2 parsers.
  Encrypted reasoning stays refused. Tests use the captured shapes.
- **Real client (opencode 0.1.0, `run --local` then `-c`, isolated
  config):**
  - It stores the empty reasoning part without metadata.
  - It replays no reasoning item for the past turn, whether the reasoning
    was empty or not.
  - This conflicts with the 2026-09-26 agent-loop evidence and is tracked
    as map #13.
- **#4 is closed.**

## 2026-10-05 - Out-Of-Band Review Fixes: Residency Lifecycle, Validation Predicates, Lens Capability

An external read-only review of main `77665c2d` found no architectural
issue and five bounded defects, all fixed:

- **Idle residency renewal (#1 qualification, `ad5934f9`):** every admitted
  connection completed through `request_finished`, so model-list polling,
  malformed requests and disconnects opened or renewed the keep-alive
  window. Now only a request that reached model execution (marked at each
  backend's first GPU work) refreshes it.
- **Few-row Q8_0 test (`b426663d`):** its `f64::max` error fold dropped NaN,
  so an all-NaN output could pass. It now asserts length and finiteness,
  with a negative control.
- **Oracle cache (`3a621166`):** verified evidence was cached by path only.
  The key now includes the expected length, SHA-256 and record count, with
  valid-first, conflicting-second controls.
- **Lens capability (`bb6ffc49`):** lens support now follows forward
  preparation (the same `inspect` the plain lens calls), not generation
  stops.
- **GLM readout domain (`8d84abf7`):** a finite residual whose output-norm
  F32 sum of squares overflows used to read out finite zero logits. It is
  now refused by an accumulation-error bound, not a magnitude cutoff. The
  live lens test still reads out bit for bit.

The review's forward items: native GLM tool support, and a Fast-lineage
teacher-forced warm/cold reuse check (KL/regret policy). Their priority
against the leverage map is pending.

## 2026-10-05 - Leverage Map Re-Ranked After P6 (cx Session `01a10cc`)

- **Status:**
  - #1 serve keep-alive has landed across the eligible families;
    qualification items remain.
  - #2 GLM attribution and its four fixes are complete; routed experts are
    the open branch.
  - #5 radix sampler is complete; sampler-v2 is parked.
- **New rank:**
  1. #4 empty reasoning, bounded to 1-2 h and judged on the full round
     trip.
  2. #9 census of DS4 short prefill. A fully wired ~40-token request takes
     1.2-1.3 s; CPU routing as the cause is unproven. The GLM pp32 gap uses
     the same census format; it is only ~49 ms per 32 tokens.
  3. #2 expert limiter profile with `scripts/profile/gpu_limiter_capture.py`
     before any rewrite.
  4. #3(a) Qwen restart qualification.
  5. The GLM short-prefill fix.
  6. #8 attribution.
  7. #6 and #10.
  8. #3(b) Flash-Next codec.
  #7 DFlash continues on main.
- **Protocol (map gates):** order-balanced, repeated A/B blocks with
  contemporaneous controls; MDE from between-block variation (one parent
  tg cell drifted ~4.4% between runs); unchanged cells as controls; GLM
  plus DS4 for shared-kernel changes.
- **Correction to the bitwise dispatch fusion entry:** the threadgroup-grid
  IQ2_S variant did not help, but that alone does not rule out
  constant-memory divergence as a factor. The limiter profile decides.
  A first bitwise candidate is keeping the gate/up kernel's final
  per-simdgroup totals in registers instead of the threadgroup round trip.
- **The 38-40 tok/s GLM decode figure** is an optimistic bandwidth-based
  estimate, not an established ceiling.

## 2026-10-05 - Few-Row Q8_0 MMA (Ported From llama.cpp): DFlash2 Q8 Code 29.8 -> 38.3 tok/s

Qwen3.8-27B Q8_0 + DFlash2 Q8_0, greedy, 256 tokens, M4 Max.
- **Kernel:** `kernels/mat_mat_q8_0_fewrow.metal` adapts llama.cpp PR
  #29869's few-row MMA for Q8_0. NSG simdgroups split K and reduce once.
  Quants fill half A fragments from registers (sign flip, exact 1024 + q). The
  block scale is applied in FP32 after the MMA, and the next block is
  prefetched into registers. It uses upstream's tiling rule and handles any
  n_out and 1..8 rows.
- **Scope:** the small-N table routes Q8_0 at 2..8 rows to it only for
  `encode_mat_mat_dispatch` callers. Explicit-policy callers (DeepSeek V4,
  GLM5-Next) keep their own matrix lineage. Rollback:
  `QWEN_MATMAT_Q8_FEWROW=0`.
- **Correctness:** a GPU unit test matches an f64 CPU reference within
  relative 2-5e-7 at tile-edge shapes (n_out 9 and 7000; 1, 2, 3, 5 and 8
  rows; NSG 8/16/32; NT 1/2/4).
- **Micro** (`matmat-smalln-micro`, production tensors rotated 8 per family):
  target shapes 1.34-1.50x faster than `r1c1k128` (294-352 -> 420-472 GB/s);
  drafter-class shapes 1.7-2.7x. Weighted 8-row matmul per verify fell from
  85.6 to 60.8 ms. The exact-shape gate earlier predicted 23-27 ms.
- **End to end** (census code prompt, A/B bracketed by two old-kernel runs):
  29.8/31.1 -> **38.3 tok/s** with the calibrated guard. Fast mode (guard off)
  reaches 44.1, against llama.cpp v0.6.0's 49-51 on the same model, drafter
  and prompt. Verify dropped from 118-128 to 95.5 ms and draft from 15.6-17.1
  to 11.4 ms. Short code 44.6, reasoning 54.1, 12K code 33.4, prose 17.2
  (backoff). Every run was byte-identical to serial.
- **Numerics improve:** against the serial shadow reference (1,530 rows), the
  top1-top2 margin error fell from median 0.0012 to 0.0003; the near-tie max
  is 0.0029 (from 0.0043-0.0065) and the error stays below 2.9% of the gap
  (from <7%). Zero flips. The 0.1 guard therefore has more headroom.

## 2026-10-05 - GLM P6 Work Integrated Into Main (Rebase Onto `bffe76f9`)

- **What:** `feat/glm53-p6` (29 commits) was rebased onto main `bffe76f9`
  and fast-forwarded into main.
- **Conflict:** only `docs/PERF-LOG.md`, where both sides added entries at
  the top. Both were kept; the P6 entries sit above the DFlash entries, in
  integration order.
- **Commit ids:** the 2026-10-05 packets and entries below cite pre-rebase
  ids, which also appear as build stamps in their JSON records. The map is
  `docs/bench/2026-10-05-glm53-p6-commit-map.md`.
- **Verification on the rebased tree:**
  - fmt clean;
  - qwen-llm lib 1,228 passed; two Flash-Next tests first failed only on
    another process's GPU lease, then passed with
    `QWEN_METAL_LEASE_WAIT=1`;
  - qwen-cli 772 passed.

## 2026-10-05 - GLM Bitwise Dispatch Fusion (KDA q/k/v, Shared SwiGLU): tg128 ~+2%; Routed-Expert Screen

Leverage map (2026-10-05) #2. Packet
`docs/bench/2026-10-05-glm53-bitwise-dispatch-fusion/`.
- **Change (`d786a9f5`):**
  - KDA q/k/v run as one three-matrix Q6_K dispatch, with the single
    kernel's per-row body verbatim.
  - The shared expert uses DS4's fused Q6_K gate/up/clamped-SwiGLU kernel.
- **Exactness:** Bitwise. Unit tests compare against the separate
  dispatches bit for bit. A greedy 96-token GLM decode gives the same
  fingerprint (`199e752f…`) before and after.
- **A/B vs `a9fafb50`:**
  - tg128 depth 0: 31.33 -> 32.16 (repeat 32.12 -> 32.78), about 0.6-0.8
    ms per token;
  - depth 4096: 28.38 -> 30.10, but the parent cell was 1.3 tok/s low
    against its own earlier run.
- **Routed-expert screen (`a9fafb50`,
  `metal::expert::tests::routed_expert_block_dispatch_costs`):**
  - 229-234 µs per block (attribution 238);
  - IQ2_S gate/up SwiGLU 145-147 µs (291-297 GB/s);
  - IQ3_S down 81-84 µs (345-357 GB/s);
  - the cost is in the dequantizing kernels, not in dispatch structure.
  - A threadgroup-memory IQ2_S grid (llama.cpp's approach), with 2 or 4
    simdgroups per threadgroup, was bitwise but 2-4% slower, so divergent
    constant-memory reads are not the limiter. Rejected.
- **P6 so far:** tg128 27.6 -> ~32.5 tok/s at depth 0 and 20.8 -> ~30 at
  depth 4096.
- **Next:** the routed-expert kernels need a limiter profile (ALU vs memory)
  before any rewrite, and DS4 shares them. Smaller items: the dense FFN
  through the same fused kernel (3 blocks), and MLA projections.

## 2026-10-05 - Serve Idle Residency for Every No-Copy Family; DS4 Confirms the Re-wire Cost

Leverage map (2026-10-05) #1. Code `344da7e7`; packet
`docs/bench/2026-10-05-serve-idle-residency-ds4/`.
- **Change:**
  - K2 Horizon, DeepSeek V4, Muse Glimmer and Flash-Next name their no-copy
    weight buffers (`retained_buffers()`, `metal::retained_gguf_buffers`)
    and get GLM's keep-alive hooks. The default is 60 s for all of them.
  - The window stays closed until the first activity (warm-up or first
    request), so a pulse never faults in a cold model.
  - Qwen serve loads Metal-allocated copies (always wired) and refuses a
    nonzero flag.
  - DS4 stays off when its opt-in residency set is active.
- **DS4 pause screen (`dc2d5ada`, fresh 40-token prompts, 4 tokens out):**
  - Keep-alive off: wired memory falls to 4.9 GiB within 10 s, and paused
    requests prefill in 1,994-2,032 ms against 1,244 ms back to back.
  - Default window: 102 GiB stays wired through 10 s and 30 s pauses, and
    paused requests prefill in 1,212-1,285 ms.
  - So a second, dissimilar family pays the same ~0.8 s re-wire and the
    keep-alive removes it.
- **Still open:** observing the pressure suspension under real pressure,
  and idle energy.

## 2026-10-05 - GLM Short-K Q8_0 Mat-Vec for KDA: tg128 +4.7% (30.0 -> 31.4 tok/s)

Leverage map (2026-10-05) #2. Packet `docs/bench/2026-10-05-glm53-kda-short-k/`.
- **Screen (`metal::kda::tests::kda_block_dispatch_costs`):** it reproduces
  attribution (352.5 vs 352 µs per KDA block). q/k/v run at 441 GB/s and
  the output at 441 GB/s, but the five small Q8_0 projections take 74 µs
  for 3.4 MB. Most of that is `ssm_f_b`/`ssm_g_b` (Q8_0 128 -> 8192, 28.8 µs
  each): the generic kernel keeps 16 of 128 lanes busy on a 4-block row.
- **Change (`b7655f61`):** `encode_mat_vec_q8_0_short_k_f32` (K of 32-256,
  2·nb lanes per row, several rows per simdgroup) for those two tensors, at
  decode and in packed Exact rows. The generic dispatcher and other
  families are unchanged.
- **Exactness:** Numerical (requalified). Live gates pass 14/14; Exact
  packed == serial bitwise; near-4096 prompt-end KL 2.95e-8.
- **Screen result:** 28.8 -> 4.7 µs per expansion; KDA block 352.5 -> 302.6 µs.
- **A/B vs `344da7e7`:**
  - tg128 depth 0: 30.03 -> 31.43 tok/s;
  - tg128 depth 4096: 28.46 -> 29.72 tok/s;
  - prefill unchanged.
- **DS4 regression:** the 256-token greedy fingerprint at `dc2d5ada` is
  `26f39112d3959e96`, equal to main's recorded value. The shared
  selected-attention refactor and new kernels leave DS4 bitwise.
- **P6 so far:** tg128 27.6 -> 31.4 tok/s (depth 0) and 20.8 -> 29.7
  (depth 4096); llama.cpp, placement-matched, is 23.7.
- **Next:**
  - routed experts (10.0 ms at 305 GB/s; DS4 shares the all-slot IQ2_S
    and IQ3_S kernels, so a bitwise-first plan is needed);
  - a one-dispatch q/k/v (proxy saves ~0.6 ms);
  - shared expert (2.6 ms at 333 GB/s).

## 2026-10-05 - GLM Fused mHC Pre: tg128 +7.8% (27.5 -> 29.7 tok/s)

Leverage map (2026-10-05) #2. Packet `docs/bench/2026-10-05-glm53-fused-mhc-pre/`.
- **Change (`b1ff3b8e`):** the 90 per-token mHC pre sub-blocks go from five
  serial dispatches to two:
  - split-K Q8_0 partials of the raw residual, per (256-value chunk, row);
  - one 1024-thread finish per row: RMS scale, register-resident
    controls, collapse and block norm.
  Decode and packed prefill share the kernels. DS4 is unchanged.
- **Not bitwise for DS4:** a register-resident controls kernel was tried
  as a bitwise drop-in for both families. Under fast math the combination
  moved by up to 35 ulps, so DS4 keeps its kernel until it has its own
  qualification.
- **Exactness:** Numerical (requalified). Live GLM gates pass 14/14:
  - Exact packed == serial decode bitwise;
  - near-4096 prompt-end KL 9.3e-9, worst 9.0e-5;
  - packed sparse KL 1.7e-5 against a 1e-2 gate.
- **A/B vs `a23a9202`:**
  - tg128 depth 0: 27.54 -> 29.70 tok/s (36.31 -> 33.67 ms);
  - tg128 depth 4096: 26.66 -> 28.45 tok/s;
  - prefill neutral or better (pp128@4096 +4.4%, pp4096 179.1 -> 182.7).
- **Attribution v3:** mHC pre 4.13 -> 1.33 ms per token; unprofiled step
  38.58 -> 35.97 ms (depth 64) and 40.60 -> 38.03 ms (depth 4096).
- **Against llama.cpp (`e1425c0be`, no residency sets):** native tg is now
  29.7 vs 23.7 tok/s.
- **Next:** weight streaming. kda is 12.0 ms at 324 GB/s, against 505 GB/s
  for the same Q6_K kernel on the head. Run a KDA per-dispatch screen
  before choosing between multi-matrix dispatches and the recurrence.

## 2026-10-05 - GLM Split Selected Attention: tg128 at Depth +27.9% (20.8 -> 26.6 tok/s)

Leverage map (2026-10-05) #2. Packet
`docs/bench/2026-10-05-glm53-split-selected-attention/`; attribution v2 in
`docs/bench/2026-10-05-glm53-decode-attribution/`.
- **Change (`ea49595f`, hardened `02d3ccdf`):** selected attention runs as
  128-row partials, one simdgroup per (query, head, split), and an in-order
  merge into the sink state. Packed sparse rows use 16-query sub-batches.
  DS4 still uses the serial kernel.
- **Exactness:** Numerical (requalified). Exact packed prefill still equals
  serial decode bitwise. Live GLM gates pass 14/14:
  - near-4096 prompt-end KL 2.402e-8, worst 3.343e-5, top-1 33/33;
  - packed sparse KL 1.040e-6;
  - frontier decode top-1 46/46.
  The hardening (finite empty state) reproduced these values exactly.
- **A/B vs `479fa9c1`, same session:**
  - tg128 at depth 4096: 20.81 -> 26.62 tok/s (48.06 -> 37.57 ms);
  - tg128 at depth 8192: 20.92 -> 26.41 tok/s;
  - pp32, pp512 and pp4096 neutral within noise (pp4096 181.4 -> 181.7).
- **Attribution v2 at depth 4096:**
  - selected attention 11.60 -> 1.05 ms per token;
  - unprofiled step 51.21 -> 40.60 ms;
  - the depth penalty is now ~2.0 ms (all sparse stages).
- **Next:**
  - mHC pre. The screen gives 28.8 µs per sub-block, 10.4 µs of it in the
    single-thread controls kernel. First, a bitwise register-resident
    controls kernel; then a fused split-K pre, which is numerical.
  - KDA streaming (11.8 ms at 328 GB/s) needs its own per-dispatch screen.

## 2026-10-05 - GLM Serve Idle Residency On By Default (60 s)

Leverage map (2026-10-05) #1.
- **Decision:** with the 1 GiB kill check recovered in every case, GLM serve
  defaults to a 60 s idle-residency window. `--idle-residency-secs 0` (or
  the environment variable) disables it.
- **Rationale for 60 s:** it covers an interactive turn without holding
  ~115 GiB wired for minutes. llama.cpp's 3-minute keep-alive is precedent,
  not a requirement.
- **Still open:** observing the pressure suspension under real pressure,
  idle energy, and extending the mechanism to other families. Each backend
  must name its pageable no-copy weights; a family whose weights are
  Metal-allocated is always wired and gains nothing.

## 2026-10-05 - GLM Decode Attribution: Depth Cost Is Selected Attention; Weight Streaming at ~321 GB/s

Leverage map (2026-10-05) #2. Packet `docs/bench/2026-10-05-glm53-decode-attribution/`.
- **Instrument:** `Glm5NextSession::forward_stage_profiled` runs one command
  with one timestamp-sampled encoder per stage. It is bitwise equal to a
  plain step (`MTL_DEBUG_LAYER=1`). Encoder boundaries cost ~1.5 ms per
  step, so stage shares transfer and absolute sums do not.
- **Depth 64 (unprofiled 38.98 ms/token):**
  - kda 12.0 ms (323 GB/s over 3.9 GB).
  - routed experts 10.1 ms (304 GB/s over 3.1 GB).
  - shared expert 2.6 ms (334 GB/s).
  - mHC pre (attention + ffn) 4.2 ms for 39 MB of weights: 90 small
    stages at ~46 µs.
  - MLA output 1.7 ms (432 GB/s), MLA projection 1.5 ms (320 GB/s).
  - head 1.0 ms (505 GB/s), router 0.8 ms (242 GB/s, F32), dense FFN
    0.8 ms (459 GB/s).
  - Weight streaming overall is ~30.5 ms for ~9.8 GB, about 321 GB/s.
- **Depth 4096 (51.21 ms/token, +12.2 ms):**
  - Selected attention takes 11.6 ms (1.05 ms per MLA block over 2,048
    selected rows, one query). A few MB of reads per block, so it is
    latency- or occupancy-bound.
  - Scoring 0.08 ms, selection 0.55 ms, sparse query 0.34 ms. Every other
    stage is unchanged.
  - The kernel is the family-neutral online selected attention that DS4
    also uses (rising tide).
- **Decision:** next is split-row selected attention for one query: rows
  split across threadgroups with an in-order reduce, as K2's 2026-09-26
  split did, raw-bit equal to serial. Ceiling ~10 ms/token at depth ≥ 2052,
  about 20% of decode there. mHC-pre fusion (~2-3 ms, about 6-8% at depth
  0) and Q6_K/expert streaming efficiency (321 -> ~400 GB/s would be
  ~6 ms) follow.

## 2026-10-05 - Kill Check: Ordinary Command-Buffer Wiring Is Reclaimed After SIGKILL (1 GiB)

Leverage map (2026-10-05) #1. Packet `docs/bench/2026-10-05-residency-kill-check/`.
- **Setup:**
  - Detached observer, verified with a deliberately timed-out tool call.
  - Quiet host: baseline 5.00 GiB wired, 0.25 GB spread.
  - A 1 GiB file-backed no-copy buffer.
  - No residency set, mlock, sysctl change or gate bypass. The child holds
    the production lease. Tolerance 384 MiB.
- **Results (excess after kill at 60 s):**
  - normal exit, wired +1.17 GiB: -46 MiB.
  - SIGKILL after pulsing stopped and the buffer unwired (-0.09 GiB before
    the kill): 29 MiB.
  - SIGKILL between keep-alive pulses (+1.17 GiB): 25 MiB, recovered
    within 1 s.
  - SIGKILL 0.3 s into a ~1.9 s command confirmed in flight (+1.20 GiB):
    765 MiB at 1 s, 196 at 5 s, 107-189 MiB through 60 s, within the
    baseline spread. A re-check minutes later read 4.94 GiB.
- **Interpretation:**
  - Ordinary command-buffer wiring, both idle-pulsed and in flight, was
    reclaimed after SIGKILL at 1 GiB. That supports the keep-alive's
    premise that it adds only state every request already holds.
  - Not proof at 115 GiB; residency sets stay closed.
- **Remaining for default-on:** pressure and idle-energy qualification on
  the serve configuration, and a dissimilar no-copy family.

## 2026-10-05 - Idle Residency Keep-Alive (Opt-In): GLM Serve Prefill After A Pause 1.65 -> 0.60 s

Leverage map (2026-10-05) #1. Opt-in; default-on is held.
- **Mechanism:**
  - `metal::ResidencyKeepAlive` submits a pulse: one ordinary command
    buffer that `useResource(Read)`s the retained no-copy windows and runs a
    one-element fill. It never waits. `poll` reports a failed pulse, or one
    running longer than 10 s as stuck; `settle` bounds shutdown.
  - `serve::idle_residency` (family-neutral) pulses every 500 ms while the
    window is open. The window opens at backend ready and again when any
    request finishes. Pulsing suspends while
    `kern.memorystatus_vm_pressure_level` >= 2 and stops for the server's
    lifetime on any failure.
  - Flag: `qwen serve --idle-residency-secs` (or
    `QWEN_SERVE_IDLE_RESIDENCY_SECS`), default off. GLM-5.3-Flash only;
    other families refuse it. Not a residency set.
- **Evidence (GLM serve, fresh 27-token prompts, 4 output tokens):**

  | Window | Gap before request | Wired before | Prefill |
  |---|---|---|---|
  | none | ≥ 2 s | 5.0-5.2 GiB | 1,603-1,699 ms |
  | 120 s | 0-60 s | 114.6-115.5 GiB | 577-614 ms |
  | 120 s | 150 s (lapsed) | 5.2 GiB | 1,641 ms |
  | 60 s | 5, 0, 30 s | wired | 578-610 ms |
  | 60 s | 90 s (lapsed) | 5.0 GiB | 1,686 ms |

  SIGINT returned wired memory to 4.9-5.2 GiB within 3-5 s every time.
- **Review (cx):** go for the mechanism, no unconditional default-on yet.
  - The pulse holds the state every request already holds (command-buffer
    wiring) and extends it into a bounded idle window. Historical
    mid-request kills are indirect evidence that this is reclaimed, not an
    observation.
  - Default-on needs a bounded kill check of ordinary pulse wiring
    (detached observer, ≤ 1 GiB, a recovery tolerance well below the
    observed wired increase, kills both between pulses and during confirmed
    execution), pressure and idle-energy qualification, and a dissimilar
    family.
  - An initial 60 s window is preferred over llama.cpp's 3 minutes.
- **Next:** the kill check; then the placement-matched short-prompt gap
  inside the #2 attribution.

## 2026-10-05 - Placement Screen: GPU Wiring Follows Activity; ~1 s Re-Wire After 2 s Idle

Leverage map (2026-10-05) #1, first screen. No change landed.
- **Wiring follows GPU activity, in both engines.** System wired memory
  (vm_stat, 2-3 s samples):
  - `qwen-bench suite` on GLM-5.3-Flash holds 115.4 GiB wired throughout
    its run and returns to 5.1 GiB at exit.
  - llama.cpp `e1425c0be` with `GGML_METAL_NO_RESIDENCY=1` climbs 5 -> 115
    GiB over its first ~66 s (first GPU use at ~1.7 GiB/s), dips to 57.7 GiB
    between test phases, and returns to 4.9 GiB within 5 s of exit.
  - So no-copy weights are wired while commands use them, with or without
    residency sets. The earlier "~5 GiB wired mid-serve" sample was taken
    while serve was idle.
- **Re-wire tax (GLM serve, capacity 4096, fresh 27-token prompts, effort
  low, 4 output tokens):**
  - Back-to-back request: wired 115.4 GiB before it, prefill 591 ms.
  - After any idle gap of 2, 5, 10, 30 or 60 s: wired 5.0-5.2 GiB before,
    115.4 GiB after, prefill 1,603-1,699 ms. The 0-gap request right after
    the warm-up also paid it.
  - Page-ins during these requests were 17-3,106 pages (≤ ~49 MB), so the
    weights stayed in the page cache. The ~1.0-1.1 s is re-wiring, not
    disk.
  - Decode is unaffected once wired (115-129 ms for 3 tokens).
- **Implication:** every interactive serve turn after a pause of ≥ 2 s pays
  about 1 s of TTFT. llama.cpp's default residency-set keep-alive (3 min
  heartbeat) exists to avoid this. Eviction (the 13-27 s first prefills) is
  a separate, rarer state.
- **Placement-matched llama.cpp** (`GGML_METAL_NO_RESIDENCY=1`, 3 runs):
  - llama.cpp pp32 66.2 ± 1.1, pp128 141.1 ± 2.5, pp512 165.2 ± 9.8, tg16
    23.7.
  - qwen-llm in the same session: pp32 59.6, pp512 210.2, tg64 27.9 (pp128
    126.4 earlier).
  - The small-prompt gap (llama.cpp +10-12% at pp32/pp128) is real, not
    placement; it moves into the #2 attribution. qwen-llm leads at pp512
    (+27%) and tg (+18%).
- **Next:** prototype an idle keep-alive (an ordinary command buffer that
  marks the weight buffers used, sub-2 s cadence, bounded window,
  default-off) and re-run this gap screen. Its kill safety needs its own
  check: it extends ordinary command-buffer wiring into idle time; it is not
  a residency set.

## 2026-10-05 - Sampler: Radix Full-Vocabulary Order, Still Bitwise v1; GLM Sampled Decode +3.2%

Leverage map (2026-10-05) #5.
- **Instrument:** `qwen-bench sampler-replay` (CPU-only) times the
  production sampler on captured logits rows. `--sidecar` replays each
  recorded request with its seed and requires the recorded tokens.
  Acquisition: `glm5_next_metal::tests::capture_release_sampling_logits`
  (144 GLM rows, release preset, three chat prompts, efforts high, low and
  max).
- **Attribution (top-k off, 154,880 entries):**
  - 2.88 ms per call. The comparator sort is 2.26 ms, probability weights
    0.36, fill 0.12, top-p 0.08.
  - The nucleus is tiny (p50 1, p90 4, max 39 tokens). With top-k 64 the
    call is 0.35 ms.
- **Change:** the full-vocabulary order is built from u64 keys (inverted f32
  total order, then token id) with a stable 16-bit LSD radix. It is the
  same order, so the same weights, sums, cutoff and draw: sampler-v1
  unchanged, with no version bump.
- **Evidence:**
  - An exhaustive-shape order test passes, as do the 32 sampler tests with
    their v1 golden streams.
  - Replay reproduces all 144 recorded tokens, at 2.88 -> 1.65 ms per call.
    8- and 11-bit digits measured 2.25 and 2.81 ms on a host at load ~12.
- **Whole decode (GLM `qwen run`, release preset, seed 7, 256 tokens,
  ABBA):** comparator 25.09 / 25.12, radix 25.78 / 26.00 tok/s, +3.2%
  (about -1.2 ms/token). Output bytes and token fingerprints are identical
  across all four runs.
- **Remaining:** about 1.65 ms per call. A sampler-v2 (total summed outside
  sorted order, nucleus by partial selection) could save about 0.8 ms more
  but changes the replay contract and needs a version bump. Not taken.

## 2026-10-05 - GLM-5.3-Flash Baseline, Placement, And Attribution Priorities

Leverage map (2026-10-05). No optimization win claimed.
- **Bring-up state:** GLM-5.3-Flash (`glm5-next`, UD-IQ3_XXS, 320B.A18B MoE:
  45 executed blocks, 34 KDA + 11 MLA with sparse DSA from visible length
  2052, 288 experts top-8 plus one shared, four-stream mHC) runs native in
  run, bench, serve and plain lens (`docs/archive/GLM53-FLASH-PLAN-2026-10-12.md`, P1-P5).
- **Baseline (`0294d13e`, `qwen-bench suite`, 3 runs, tok/s):**
  - Depth 0: pp32 59.9, pp128 126.4, pp512 197.5 (samples 191-203), tg128
    27.6.
  - Depth 8192: pp32 54.0, pp128 107.4, pp512 174.7, tg128 20.8 (20.7 at 4096
    in the P4 packet; llama.cpp 19.9-20.1 there, and 22.1-22.5 at depth 0).
  - Decode at depth costs ~12 ms/token over depth 0 and is flat from 4096 to
    8192; crossing the 2052 frontier alone costs +4.7 ms (46.5 -> 51.2 ms).
    Neither locates the overhead.
  - Sampled decode with the release preset (temperature 1.0, top-p 0.95, top-k
    and min-p off) 25.3 vs greedy 27.6; top-k off sorts the full 154,880-entry
    vocabulary per token. Trajectories differ, so the gap is not yet
    attributed.
- **Provisional weight-stream model (GGUF tensor inventory, GiB per token):**
  - Dense non-expert 5.38: mostly Q6_K (attn_output 1.42, KDA q/k/v 0.89
    each, dense blocks 0-2 0.35); Q8_0 MLA projections ~0.54; F32 router 0.18.
  - Shared expert 0.81 (Q6_K); head 0.485; routed experts 8/288 x 102.3 =
    2.84 (gate/up IQ2_S, down IQ3_S and IQ4_XS).
  - Total ~9.5, or ~21.6 ms at 474 GB/s, against 36.2 ms measured. This is a
    weight-stream estimate, not a demonstrated ceiling: executed tensors, KDA
    and latent state, intermediates and rereads are not yet counted.
- **Placement:**
  - GLM weights are 109.5 GiB of pageable no-copy file cache; system wired
    memory stays ~5 GiB even mid-serve.
  - Near capacity the page cache churns. A cold run's prefetch read 117.5 GB
    in 71 s and its 23-token prefill then took 26.7 s, consistent with the
    prefetch evicting its own head. A startup after a build re-read 19 GB,
    and fresh 22-28 token serve prefills varied 0.6-1.4 s (swap 2.8 GB in
    use, ~190 MB free).
  - Default-configuration llama.cpp (`e1425c0be`) wires its buffers through
    `MTLResidencySet` with a 3-minute keep-alive. System wired memory went
    5 -> 114.7 GiB while loading, and pp32 was 67.4 / tg16 24.45. That run
    happened under tool supervision with a 200 ms SIGKILL escalation: the
    2026-08-10 stranding hazard. It exited cleanly, and wired memory returned
    to 5.1 GiB.
  - `vm.user_wire_limit` = 116,823,110,451 B (108.8 GiB) < the 109.5 GiB
    trunk.
- **Decision:**
  - Prioritize bounded re-warming (necessary but not sufficient against
    steady-state over-subscription) and one GLM decode attribution packet;
    cheap continuity checks stay high.
  - Primary llama.cpp comparisons run with `GGML_METAL_NO_RESIDENCY=1`;
    residency-on numbers are labelled default-configuration evidence.
  - Whole-model `MTLResidencySet` stays closed. mlock reclamation is bounded
    research under the roadmap's kill-test protocol, separate from
    performance qualification.
- **Next:** sampler CPU replay on identical logits; placement-matched
  llama.cpp pp32; residency/I/O-correlated TTFT; the decode attribution
  packet.

## 2026-10-05 - DFlash Guard Recalibrated To The Flip-Relevant Error: Margin 0.1

Qwen3.8-27B Q8_0 + DFlash2 Q8_0, greedy, 256 tokens, M4 Max.
- **Trigger:** llama.cpp v0.6.0 (few-row MMA) runs DFlash2 on the same model,
  drafter and 7,406-token code prompt at 49-51 tok/s against our 24.5. Both
  engines accept the same drafts (~5.0 emitted/step) and produce identical
  serial text. With the fallback off we reach 36.6. The rest is ~27 ms/step of
  N=8 matmul (kernel gate, same day).
- **Divergence census:** 12 prompts (code, structured, reasoning, prose; 128 to
  7.4K positions). Serial at prefill chunk 128, DFlash without the guard, and
  DFlash with it all matched default serial byte for byte (0/12 each). The
  guard flagged 105 steps in those runs, none of which changed the output.
- **Measurement:** pair each committed row's packed (top1 - top2) gap with a
  serial shadow reference's gap: 3,133 rows from 128 to 13.7K positions. The
  margin error |packed - serial| has median 0.0012-0.0015 in every band. On
  near ties (serial gap < 0.75) it peaks at 0.0065 (<1K), 0.0061 (5.9-7.4K)
  and 0.0043 (11.5-13.7K). Its larger values (up to 0.127) occur only on
  wide-gap rows; overall it stays below 7% of the gap. The old 0.2/0.75 tiers
  came from the max per-logit error over the vocabulary (up to 0.365) and
  flagged ~8% of rows; a gap below 0.1 covers ~0.9%.
- **Change:** margin 0.1 below the 16K speculation stop (~15x the near-tie
  peak); 0.75 stays only for the explicit long lane past 16K, which this
  calibration does not reach. `QWEN_DFLASH_VERIFY_MARGIN` still overrides.
- **Result:** byte-identical on all validation runs (7.4K code, short code,
  prose, three 11.5-13.7K code prompts). Fallbacks per run fell from 16-26 to
  1-4; the census code prompt went 24.2 -> 29.7-31.4 tok/s (fast mode 36.1).
  The guard now costs ~2-15% over no guard, from 15-52%. One validation run
  was contaminated by concurrent load (every phase 2x, prefill included).
- **Fast mode** (guard off) estimates a 3.7e-6 per-token flip rate (95% CI up
  to 1.2e-5, <=1.2% of 1,024-token sessions); it stays opt-in. The next speed
  lever for every mode is the few-row MMA kernel port.

## 2026-10-05 - DFlash Fallback Replays From The First Flagged Row: Q8 Code 1.18x -> 1.50x

Qwen3.8-27B Q8_0 + DFlash2 Q8_0, ~7.4K prompt, greedy, 256 tokens, M4 Max.
- **Census:** code ran 1.19x serial (16.29 tok/s serial), prose 0.90x. Exact
  fallbacks were 49% of code decode: 22 of 51 steps, 292 ms each. Margin
  sensitivity on code: 0.75 -> 19.3, 0.375 -> 24.2, 0.2 -> 28.6 tok/s.
- **Waste:** the first under-margin row averaged index 2.8, yet the fallback
  replayed from row 0; 61 of 102 replayed rows had already passed the margin.
- **Change:** restore to the first flagged row (`n_keep = r`, the
  partial-accept restore) and replay token-major from there. Rows replayed this
  way run over packed in-block state, so any replayed row with exact gap below
  0.15 or non-finite escalates to the whole-block replay
  (`restore_to_pre_block_after_replay`). The replay no longer scatters into the
  capture ring; the append publishes committed rows. Default on;
  `QWEN_DFLASH_VERIFY_FALLBACK_PARTIAL=0` restores whole-block replay.
- **Drift (shadow probe, margin 3.0):** on 91 paired repaired rows, drift vs a
  serial reference was median 0.0047 / max 0.0308 (partial) vs 0.0034 / 0.0196
  (whole-block); whole-block replay itself reached 0.0731. The worst partial row
  (serial gap 0.080) motivated the 0.15 escalation floor.
- **Result:** code 19.24 -> 24.48 tok/s (5 escalations); prose 15.02 -> 15.35
  (fallback 1442 -> 1071 ms; prose stays below serial on acceptance). Stress at
  margin 3.0: 39 fallbacks, 3 escalations, 0 shadow flips. Every run was
  byte-identical to serial.
- **Kernel gate (same day):** llama.cpp's few-row MMA (PR #29869) at exact
  Qwen3.8 shapes runs the Q8_0 N=8 matmuls 1.31-1.47x faster than the mma8v
  table, 26-30% of N=8 matmul time (~23-27 ms per verify). Q4_K_M shapes are
  1.2-1.5x faster except lm_head. Not yet ported.
