# V4.1 concurrent HTTP preparation and exact token-prefix reuse

Bounded stage began 2026-11-14 11:02:34 UTC; stop/report by 13:02:34 UTC. Branch `perf/serve-ttft-instrumentation`, starting tip `e5b163c8a`. This is one composed diagnostic run against the previous clean reference, not a paired promotion gate. Both candidates remain default-off with independent `=0` rollbacks. Checkpoint capture is unchanged.

## Serialization point and local fix

At baseline `e5b163c8a`, `crates/qwen-cli/src/serve/mod.rs:817` forwards an accepted socket as an `Incoming` event. The shared loop runs synchronous `backend.pump()` at `mod.rs:1173`, then handles that event and calls `Connection::start_batched` at `mod.rs:1206`. `transport.rs:498` delegates to the worker-spawning path. Therefore the HTTP thread does not exist while a peer's pump is inside its blocking prefill. There is no tokenizer/prefix-cache mutex held across that prefill and no async executor pool to enlarge. This is dispatch ordering on the inference thread.

`QWEN_SERVE_ASYNC_PREPARE=1` snapshots only immutable CPU request/profile/tokenizer state and the inference thread's wake handle before accepting. The accept thread starts a bounded HTTP worker immediately and sends the connection through the same arrival channel. GPU state never leaves the inference thread. Rendering, tokenization and token-prefix matching can proceed during a peer's native wait. Starting the native forward still requires the inference thread.

At most `max_active` eager workers exist; when their permits are occupied, sockets retain the existing bounded FIFO path. Existing active workers remain bounded by `max_active`, so the combined maximum is twice that count, also capped by accepted connection depth. Permits follow the connection lifetime and release on failure/shutdown. Each early caller conservatively adds `max_active * 128 MiB` to its existing CPU transport reserve for preparations not yet submitted to the scheduler; these are admission allowances, not allocations. Three 48K sessions retain the same engine representation. The test suite checks overlap while a fake inference pump is blocked, cap fallback preserving the original socket, and permit release on shutdown.

Mutable live-KV lookup/restore still consumes inference state in `backend_v41/reuse.rs:181`; its previous append lookup cost was about 0.006 ms. It is not the cause of handler-start delay and has no lock held across peer GPU work. The concurrent prefix matching added here is exact text/token-prefix matching. An asynchronous KV-state consume/restore would need a separate state handoff design; no GPU state race is introduced for a microsecond lookup.

## Exact tokenization reuse

`QWEN_SERVE_TOKEN_PREFIX_CACHE=1` retains the previous full tokenization for four conversation families (first-user-turn hash, as in checkpoint retention), at most 8 MiB of text/IDs/cut payload plus fixed metadata. Exact bytes, never that hash, authorize reuse. Encoding and text scans happen outside the short cache mutex. Concurrent old-entry snapshots are bounded by the bounded HTTP worker count. Oversized entries are not retained; eviction or a missing proof uses the incumbent full encode.

The released JoyAI tokenizer partitions text at explicit special tokens, then pretokenizes/BPE-encodes each ordinary fragment independently. The new `NativeTokenizer::encode_partitioned` records the byte/token offsets at those fragment boundaries without changing the incumbent `encode` implementation. For an extension or fork it finds the exact unchanged byte prefix and resumes only at a recorded boundary at least one maximum-special-token length before the first changed byte. This guard prevents a completed or longer special token from changing the earlier partition. Ordinary BPE/Unicode boundaries are never guessed. Records carry an immutable tokenizer-instance identity, so another vocabulary cannot reuse them.

A prefix ending at a recorded special-fragment boundary has exactly the independent prefix encoding. The V4.1 admission path can therefore get shared/generation cut IDs from the just-prepared full encoding, preserving the existing nonempty/length/`starts_with` checks and falling back to the independent encode at any other cut. No prompt IDs, CED cuts, sampler or numerical kernel is intentionally changed.

CPU property checks use recorded V4.1 primary plan sizes, actual repository source excerpts and reconstructed tool-call transcripts passed through the current V4.1 renderer. They compare full token IDs and generation cuts, with adversarial partial/overlapping special markers, BPE merges, forks, Unicode combining marks/ZWJ, whitespace and tokenizer identity changes. Synthetic native vocabularies include a real `a b -> ab` merge and overlapping control tokens. The M5 CPU witness also uses the released file-A tokenizer with the existing source corpus, comparing full IDs and prefix IDs before any served measurement.

## Frozen measurement

One composed arm, same production file-A IQ2_XS + streamed IQ4_NL Engram, release profile, greedy, three agents, seed 20261112, steady v2 plan, 49,152-token sequential disjoint warm roots with 32-token replies, 15-second ramp and 120-second window. Context 196,608; MAX_ACTIVE=4, SAMPLE_WORKERS=3, PREFILL_FAIR=1; both existing timing probes enabled. Both new knobs are 1. Nice 0 is checked on the actual server. Engram purge must reach zero resident pages. Thermal settle <=1 continuously 60 s; allow 2, abort 3. Each M5 build/model phase uses a waiting lease <=15 minutes, inherited FD through Python. Process disk bytes, load, compiler and thermal flags are retained.

Reference: the clean `e5b163c8a` record, measured Rust source `0a54d2316`, [previous decomposition](2026-11-14-serve-ttft-instrumentation.md). Compare per-class component distributions and common call labels/output bytes. Small-sample p95 and unpaired wall/throughput ratios are descriptive; a full paired gate is next stage if justified.

## Checkpoint async note

The current checkpoint reads a live GPU state that decode will mutate. Running its copy independently while advancing that state is unsafe without an explicit copy-before-write dependency or immutable/COW snapshot. The command queue could overlap independent copies only after establishing those dependencies. Publishing the first token before a final Ready checkpoint, then completing that checkpoint before the next state mutation, is a separate ordering candidate. Earlier semantic checkpoints cannot move across their state advance. No checkpoint code is changed here.

## Validation and results

**Large component reductions; served output-bitwise is not established. Both knobs remain default-off.** The one composed diagnostic run completed. No paired performance gate was attempted and no winner is promoted.

CPU checks: 29 lifecycle/transport/cache tests passed, zero failed/ignored, including six new maintained CPU tests. Full standard contracts pass at `402dfde3c` (the first inventory-scope omission and two test-only native-vocabulary identity initializers were corrected). ENV has both knobs. The measured release runtime source is `724b5782a1a163a0ad560047f9869dc3014fa13f`, tree `4cef3ce80c2ef32aad2c5a86a6d4d29c290c37a4`; later Rust edits are confined to existing `#[cfg(test)]` fixtures and make the concurrency test signal after encoding completes. The M5 build took 3m49s at nice 10. All three metallibs are **byte-identical to the reference**.

Before loading the engine, the released tokenizer CPU witness passed **57 cases, 54 incremental hits, 699 prefix checks**, with `full_ids=true prefix_ids=true`. The process was verified nice 0. The witness reconstructs plan-shaped source transcripts from the existing corpus (source suffixes capped at 8 KiB to bound the CPU check), with explicit adversarial fragments; it is not a recovered production-payload replay. The native renderer CPU property separately checks the current V4.1 tool-transcript renderer and all generation cuts.

The measurement lease ran 11:36:28–11:45:03 UTC, about 8m35s. Actual server PID 58281, nice 0. Purge 0→0 resident pages, msync return 0; no other resident model at loading. Three roots, ramp and full 120-second window completed, no memory refusal or thermal abort. Total sampled process disk-read counter **190,324,699,136 bytes**, versus reference **189,521,977,344 bytes**, both including loading/roots. Load reached about 40–46 during warmup and about 22–34 during the window, versus about 9–13 previously. Compiler processes overlapped both windows. Whole-arm thermal max 2; after-warm settling took 109.743 s to obtain <=1 continuously 60 s. The candidate measurement window spent about 56.6 s at 0 and 63.4 s at 1, none at 2/3. The reference window included 31.6 s at 2, allowed by the same protocol.

Candidate delivered TPS **89.917**, reference **90.325** (0.9955x). This is one unpaired comparison under different host/native wall, not a non-regression verdict.

[All distributions](2026-11-14-v41-serve-prepare/candidate-summary.json), [component comparison](2026-11-14-v41-serve-prepare/comparison.json), [per-call cache and peer-round ledger](2026-11-14-v41-serve-prepare/candidate-calls-with-cache.jsonl), [output-byte checks](2026-11-14-v41-serve-prepare/output-comparison.json), [controls](2026-11-14-v41-serve-prepare/controls.json), [pins/metallibs](2026-11-14-v41-serve-prepare/pin-manifest.json), [tokenizer witness](2026-11-14-v41-serve-prepare/token-prefix-exact.log), [input digests](2026-11-14-v41-serve-prepare/input-manifest.json).

## Component results

All 31 eligible visible TTFTs join with no missing records, parse errors or dropped spans. Per-call non-overlapping partitions balance within 0.001 ms. Reference has 32 eligible visible calls. Candidate short N=19, ~1K N=8, roots N=3, plus one 2,002-row call retained in the ledger. One other offered call, `a1-t11`, was right-censored after only 84.162 ms at the cutoff; no TTFT is invented or assigned to it. It was ~1K in the reference. Partial generations whose first token was observed remain included. Small-sample nearest-rank p95 equals the maximum here; component quantiles do not add.


### Short appends: p50 / p95 milliseconds

| Component | Reference | Candidate |
|---|---:|---:|
| Client TTFT | 497.202 / 1622.344 | 425.649 / 1459.027 |
| Native prefill wall | 368.163 / 658.084 | 374.284 / 669.878 |
| Outside native | 105.692 / 1154.966 | 51.365 / 975.975 |
| Outside native minus logged waits | 79.153 / 90.709 | 35.649 / 70.438 |
| HTTP handler start after TCP accept | 15.191 / 1082.155 | 0.081 / 0.160 |
| Body receive | 0.136 / 0.279 | 0.124 / 0.261 |
| Parse/validate | 0.265 / 0.338 | 0.264 / 0.335 |
| Chat render | 0.085 / 0.126 | 0.096 / 0.124 |
| Tokenization total | 52.086 / 61.514 | 2.681 / 3.826 |
| Full worker encode / cached suffix path | 30.966 / 37.328 | 2.663 / 3.808 |
| Generation-prefix encode / cached cut | 20.580 / 24.859 | 0.014 / 0.019 |
| Prefix match | 0.006 / 0.007 | 0.006 / 0.682 |
| Prefix take/restore, inclusive | 0.017 / 0.021 | 0.015 / 0.695 |
| Checkpoint capture | 21.649 / 24.458 | 27.662 / 58.592 |
| Prepared mailbox wait | 10.602 / 19.264 | 13.395 / 764.344 |
| Scheduler slot wait | 0.005 / 0.057 | 0.005 / 0.009 |
| Admission work, inclusive | 21.201 / 25.473 | 0.391 / 1.138 |
| Admitted → first slice | 0.002 / 0.002 | 0.002 / 141.092 |
| First sample after finish | 0.438 / 0.440 | 0.437 / 0.493 |
| Sample → raw piece | 0.021 / 0.025 | 0.021 / 0.041 |
| Raw piece → visible SSE flush | 0.859 / 1.427 | 0.824 / 1.119 |
| Flush-return → client read | 0.633 / 0.736 | 0.837 / 3.398 |
| All logged ready waits | 25.679 / 1100.053 | 13.462 / 905.537 |
| Wait covered by peer prefill | 0.000 / 1082.090 | 0.000 / 764.435 |
| Wait covered by peer decode | 22.706 / 33.196 | 13.449 / 140.968 |

### ~1K appends: p50 / p95 milliseconds

| Component | Reference | Candidate |
|---|---:|---:|
| Client TTFT | 1108.693 / 1551.415 | 1072.709 / 1579.689 |
| Native prefill wall | 1028.846 / 1468.336 | 1055.647 / 1518.677 |
| Outside native | 85.773 / 99.808 | 42.190 / 61.012 |
| Outside native minus logged waits | 65.895 / 86.098 | 34.525 / 46.529 |
| HTTP handler start after TCP accept | 6.440 / 12.353 | 0.067 / 0.097 |
| Body receive | 0.106 / 0.235 | 0.121 / 0.193 |
| Parse/validate | 0.250 / 0.340 | 0.272 / 0.306 |
| Chat render | 0.082 / 0.099 | 0.074 / 0.113 |
| Tokenization total | 39.495 / 56.179 | 3.163 / 3.937 |
| Full worker encode / cached suffix path | 24.577 / 34.185 | 3.145 / 3.917 |
| Generation-prefix encode / cached cut | 14.614 / 21.775 | 0.013 / 0.016 |
| Prefix match | 0.005 / 0.007 | 0.005 / 0.026 |
| Prefix take/restore, inclusive | 0.017 / 0.019 | 0.015 / 0.036 |
| Checkpoint capture | 21.585 / 24.138 | 26.193 / 38.665 |
| Prepared mailbox wait | 11.647 / 22.078 | 5.157 / 21.605 |
| Scheduler slot wait | 0.005 / 0.019 | 0.005 / 0.006 |
| Admission work, inclusive | 15.295 / 22.387 | 0.393 / 0.404 |
| Admitted → first slice | 0.002 / 0.002 | 0.002 / 0.002 |
| First sample after finish | 0.439 / 0.455 | 0.437 / 0.449 |
| Sample → raw piece | 0.024 / 0.028 | 0.024 / 0.026 |
| Raw piece → visible SSE flush | 1.029 / 1.303 | 0.789 / 0.963 |
| Flush-return → client read | 0.576 / 0.762 | 0.776 / 0.976 |
| All logged ready waits | 18.099 / 27.930 | 5.222 / 21.683 |
| Wait covered by peer prefill | 0.000 / 0.000 | 0.000 / 0.000 |
| Wait covered by peer decode | 17.959 / 27.813 | 5.208 / 21.653 |

### Sequential warm roots: p50 / p95 milliseconds

| Component | Reference | Candidate |
|---|---:|---:|
| Client TTFT | 25017.426 / 25188.427 | 29836.863 / 30548.698 |
| Native prefill wall | 24939.136 / 25083.699 | 29755.928 / 30457.034 |
| Outside native | 78.290 / 104.728 | 80.935 / 91.664 |
| Outside native minus logged waits | 78.097 / 104.424 | 80.627 / 91.229 |
| HTTP handler start after TCP accept | 0.062 / 0.133 | 0.077 / 0.100 |
| Body receive | 0.091 / 0.173 | 0.122 / 0.136 |
| Parse/validate | 0.217 / 1.013 | 0.193 / 0.463 |
| Chat render | 0.041 / 0.352 | 0.115 / 0.521 |
| Tokenization total | 30.570 / 53.613 | 20.151 / 32.700 |
| Full worker encode / cached suffix path | 18.566 / 34.508 | 20.136 / 32.683 |
| Generation-prefix encode / cached cut | 11.824 / 18.924 | 0.013 / 0.014 |
| Prefix match | 0.000 / 0.000 | 0.000 / 0.000 |
| Prefix take/restore, inclusive | 2.379 / 2.439 | 3.635 / 4.006 |
| Checkpoint capture | 19.411 / 20.713 | 24.395 / 25.761 |
| Prepared mailbox wait | 0.007 / 0.017 | 0.034 / 0.064 |
| Scheduler slot wait | 0.005 / 0.006 | 0.008 / 0.048 |
| Admission work, inclusive | 14.746 / 21.478 | 4.071 / 4.476 |
| Admitted → first slice | 0.003 / 0.017 | 0.003 / 0.019 |
| First sample after finish | 0.453 / 0.459 | 0.484 / 1.008 |
| Sample → raw piece | 0.017 / 0.018 | 0.022 / 0.048 |
| Raw piece → visible SSE flush | 0.886 / 0.980 | 0.872 / 1.245 |
| Flush-return → client read | 0.237 / 0.252 | 0.238 / 0.286 |
| All logged ready waits | 0.286 / 0.304 | 0.436 / 0.777 |
| Wait covered by peer prefill | 0.000 / 0.000 | 0.000 / 0.000 |
| Wait covered by peer decode | 0.000 / 0.000 | 0.000 / 0.000 |

## Interpretation

The short/~1K outside-native p50 fell **105.692→51.365 / 85.773→42.190 ms**. Total tokenization fell **52.086→2.681 / 39.495→3.163 ms** (about 19.4x / 12.5x faster for that component); independent generation-prefix work fell **20.580→0.014 / 14.614→0.013 ms**. All 19 short and eight ~1K visible calls used incremental cache hits. Median reused token fractions were **98.78% / 95.97%**, encoding only **674 / 1,769 suffix tokens** at p50 (p95 3,268 / 3,264). The suffix includes the previous generated reply as well as the new input. Roots are full-encode misses, as intended.

Across all 40 traced requests, including the censored call and ramp, handler start max was **0.160125 ms**, p95 **0.136 ms**. Handler-start latency is fixed at the dispatch point: short p50/p95 **15.191/1082.155→0.081/0.160 ms**, ~1K **6.440/12.353→0.067/0.097 ms**. The CPU work now starts while the peer forward runs. Slot queueing remains about 0.005 ms.

The peer-pre-fill tail is **moved to the ready mailbox, not eliminated**. `a1-t8` starts HTTP after 0.097 ms, then spends 764.344 ms in the mailbox while `a0-t18` prefills (round3551, 764.435 ms overlap) and about 141 ms crossing subsequent peer decode rounds. Total ready wait 905.537 ms; client TTFT 1459.027 ms versus reference 1622.344 ms. `a1-t6` similarly starts in 0.058 ms and waits 545.200 ms in the mailbox behind `a0-t11` (round2157). The next scheduling question remains service ahead of a ready short append and decode credit; it is not addressed by this stage.

Checkpoint capture was untouched and rose **21.649→27.662 / 21.585→26.193 ms p50**. Native append wall also rose slightly. Root p50 worsened **25.017→29.837 s** with native wall **24.939→29.756 s**. Host load/native drift and the unpaired design prevent a causal root regression attribution; they also prevent calling this a root non-regression pass.

Common-label geometric reference/candidate ratios: short client TTFT **1.1309x**, native **0.9836x**; ~1K TTFT **1.0217x**, native **0.9770x**; roots TTFT/native **0.8619/0.8614x**. These descriptive ratios include all matched observed calls, including the output-diverging one; nothing is excluded to improve the result.

## Exactness finding and next gate

**Tokenization's strict witness passes; cross-run observed-output equality did not hold.** All **37 common completed responses** have identical delivered model-output bytes and matching generated/prompt/reuse counts. One common partially delivered response is a matching prefix; one has no candidate visible output at cutoff. The remaining partial response, **`a2-t2`, differs at byte104**, long before cancellation (~10K visible characters). This is not a terminal truncation/UTF-8 artifact and is not relabeled as exact because it was cancelled.

The diverging turn has the same 53,837 prompt / 52,039 reused / 1,798 new-row extents and one native append, but baseline has no prompt-ID/cut digests, so exact input/state attribution is unavailable. Visible prior responses match. Batch composition differs: the preceding `a2-t1` average decode width is **2.78947→2.81053**, `a2-t0` **2.88617→2.89276**. Existing source explicitly limits column-batch bitwise equivalence for skinny Q5_K (`forward/columns.rs:240`) and Q8_0 pair (`:264`) paths; the separate-step bitwise test (`gpu_step/tests.rs:592`) uses toy F32 weights that force the column fallback. Existing batch-dependent rounding is therefore a plausible explanation, **not a demonstrated cause**. No unsupported claim that the cache or scheduler is fully served-output-bitwise is made.

Performance components justify further qualification. Before promotion, capture actual prompt/token/cut digests for matched calls and isolate the differing turn / batch history; then use the planned full paired gate. The permitted paired teacher-forced non-inferiority protocol has not been run; this stage does not establish full served output-bitwise behavior. Both prototypes remain **default-off**, no `ready to land` claim. No approximation, requantization, checkpoint change or new numerical kernel was introduced to work around this finding.

## Reproduction and closure

Raw candidate data: `metagross:~/qwen-runs/2026-11-12-v41-prefill/serve-prepare/measurement/`; reference: `.../ttft-instrumentation/measurement/`. The maintained comparison script reruns the original shared analyzer, annotates cache counts and checks every common model-output stream (control/response timestamps and random response IDs are intentionally not model-output bytes):

```sh
python3 docs/bench/2026-11-14-v41-serve-prepare/compare.py "$REFERENCE/serve" "$CANDIDATE/serve" --out "$OUT"
```

M5 task-process audit found none. The task worktree, compiled target, source bundle and transfer archive were removed; raw data remains. Local task-process audit also found none. Pinned executables are retained under the candidate run's `pins/` because the proposed paired/exactness gate spans stages. Final committed-tree contracts are recorded in the handoff.
