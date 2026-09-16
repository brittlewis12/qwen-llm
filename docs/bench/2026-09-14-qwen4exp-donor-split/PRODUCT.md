# Split Attention Product Opt-In Delivery

Historical packet. September16 production retirement supersedes this opt-in:
`../2026-09-16-flash-defaults/RESULT.md`. The measurements below remain unchanged;
they do not override the later failed default guardrail and finite repair outcome.

Status: **exclusive actual-product qualification and live CLI PASS**. Default
remains off. The earlier unguarded runs below stay provisional; the new exclusive
runs supply delivery authority rather than retroactively validating them.

## Exclusive Completion

The user explicitly approved gracefully stopping PID48728. Its command was
rechecked, SIGINT sent only to that PID, and exit confirmed. No forced termination
or lease bypass was used. The server was not restarted.

After rebuilding the guarded release test and CLI, the actual-product packet
holds the production lease plus real wired-memory check before any Metal work.
It passes in12.96s: all32 full-logit/hyper/state checks, baseline restore replay,
121 persistent tensors and384 split/merge pairs. Numerical maxima match the
earlier observed results below; gates are unchanged. Raw rows are retained at
`target/profiles/qwen4exp-product-split-82130/`.

| Exclusive metric, four forwards | A1 ms | B1 ms | B2 ms | A2 ms | Time saved | A spread |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 232.631500 | 195.117374 | 195.029375 | 232.031583 | 16.036638% | 0.258216% |
| Executor wall | 237.025168 | 199.643584 | 199.426708 | 236.456291 | 15.715751% | 0.240295% |

Both metrics pass the frozen mean/pairwise floors and control-spread gates.
The test uses one fully packed prefix and a common restored state, not a second
reference prefill. This is bounded executor performance, not request throughput
or broad semantic/default-promotion authority.

The separate production CLI known-answer request acquires its normal production
lease, reports `qsa_split_decode=true`, and returns exactly:

```text
FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}
```

It consumes2578 prompt tokens and emits23 tokens with22 model transitions,
finishing at EOS. Observed prefill6367.3ms /404.88 tokens/s, generation1162.3ms /
19.79 reported tokens/s, CLI total7607.9ms. These are one unpaired, warm-filesystem
delivery observation, not an additional speedup or OS-cold claim. The request
stats envelope reports status=ok and output fingerprint
`5cc37f779e7ab0682ae22a010f456cd6b7242493a0bf80ee186941f0dc3095c3`.

Exclusive logs: `target/profiles/2026-09-14-qwen4exp-product-split-exclusive-02.log`
and `2026-09-14-qwen4exp-product-cli-exclusive.{stdout,stderr,stats.jsonl}`.

## Implementation

Implementation commit: `b1310225`.

- Library: `Qwen4ExpDecodeOptions { split_qsa: true }` through
  `Qwen4ExpLoadedModel::load_with_decode_options`. Legacy constructors remain off.
- CLI: strict `QWEN4EXP_QSA_SPLIT_DECODE=1`; unset/0 disables. Empty, malformed,
  and non-Unicode values fail. Packed-load retries retain the typed option in
  scalar fallback. A diagnostic reports the effective configured setting.
- Memory plan adds one logical1,585,152-byte allocation plus normal allocator
  pricing before admission. Private QSA children share retained views; separate
  sessions allocate separately. Root ownership spans all serial layer work.
- Binding validates geometry, dtype, extent, device, aliasing and pipeline
  capability. Production uses singleton24/2/256 F16 attention only at2048-2051
  active IDs. Other ID counts use the incumbent. Packed kernels do not change;
  eligible scalar-prefill calls may also use the singleton opt-in.
- The shader moves unchanged from research to the product metallib. Scratch is
  transient, excluded from checkpoints; the comparison keeps it live in both
  arms. Test binding switches validate every child before changing any binding.

## Earlier Observed Checks

CPU strict-option parsing passes; production CLI check and release builds pass.
Before the lease discovery, Metal API-validation boundary and custody tests
passed0.19s/0.16s: default-off memory identity, exactly one added allocation,
idempotent enable, sibling sharing/independent-session isolation, root pending
and in-flight refusal without mutation, reset retention, size/alias rejection,
and actual singleton routing at short and2048-2051 ID counts. The2052 fallback
boundary is checked as a pure eligibility condition. No shader-validation claim.

The actual-product test uses no TLS override. One current packed SSH2179 prefix
has no split dispatch; baseline32-token replay is bitwise after restoration.
Candidate32 full-vocabulary rows have identical argmax throughout, minimum
cosine0.999999999205, maximum relative RMS3.986687e-5 and absolute6.924868e-4.
All unchanged hyper/F32/new-F16 state gates pass over121 persistent tensors;
F32 maximum absolute0.000515938, new F16 maximum0.00390625. Old prefixes remain
unchanged. Census witnesses384 split and384 merge calls.

Raw full-logit rows, each31,784,960 bytes, were saved before census/numerical
assertions in `target/profiles/qwen4exp-product-split-80500/`. The directory was
reserved before model work. Test wall37.07s.

Observed, **provisional**, four-forward ABBA timings:

| Metric | A1 ms | B1 ms | B2 ms | A2 ms | Observed time reduction |
|---|---:|---:|---:|---:|---:|
| GPU | 231.898792 | 192.997125 | 193.921417 | 231.543041 | 16.51195% |
| Executor wall | 236.368292 | 197.555459 | 198.653834 | 236.157209 | 16.15071% |

Control spreads were0.15353%/0.08934%. These are not accepted exclusive
performance authority, regardless of stable controls. They also exclude
restores/readbacks and are not request-throughput measurements.

## Lease Discovery And Correction

The real CLI known-answer request (2578 prompt tokens, maximum64 generated)
failed before Metal initialization because PID48728 owns
`/tmp/qwen-llm-501/metal.lock`. Inspection identified an existing server in the
main worktree: `qwen serve -m /Users/tito/models/Qwen3.8-27B-Q8_0.gguf
--max-tokens 32684`, running for roughly ten hours. It was not interrupted at
that stage; the approved shutdown occurred afterward.

`metal/context.rs` gives test builds a per-process test-lock directory. The
earlier Flash-Next test runs therefore did not prove production-exclusive GPU
access, and server activity during them was not established. All September14
Flash-Next timing and coarse-attribution promotion claims are downgraded to
provisional. Numerical/state/API checks are retained as observed results.

`acquire_metal_benchmark_lease` now reuses the secured production path and
nonblocking flock, then performs the real wired-memory check. The targeted
Flash-Next test entry points hold its RAII guard before Metal context/resource
creation until after those resources drop. Generic unit-test namespaces remain
unchanged. These exact benchmark tests must run serially, without unguarded
tests, forks, or a child production CLI inside their lease scope.

The guarded actual-product test now refuses the occupied lease in0.00s before
GPU setup. This is an environmental validation block, not a numerical failure.
It deliberately does not wait, including when `QWEN_METAL_LEASE_WAIT=1` is set.

## Validation Boundary

The approved exclusive window completed the guarded actual-product packet,
real CLI request, and guarded API-validation boundary/custody checks (PASS
0.19s/0.15s). The final production CLI `cargo check` also passes. No historical
primitive/native campaign was repeated. Prior unguarded timing and coarse
attribution remain provisional.
Default-on promotion, other quant artifacts, stochastic sampling equivalence,
and a broad far-context quality study are outside this delivery evidence.

Next leverage: HC rank320 up-projection scheduling and a bounded complete-MoE
observation. Avoid further split-body/RMS tuning without new parent-budget
evidence; the earlier GDN-containing block total is not recurrence attribution.

Raw logs in the worktree: `target/profiles/2026-09-14-qwen4exp-product-split-01.log`,
`2026-09-14-qwen4exp-product-cli.stderr`,
`2026-09-14-qwen4exp-production-lease-check.log`, and the product boundary/custody
logs. Reviewer `01a0a13a-f2f8-7673-ace3-4fbfd25a3aef` reviewed ownership, outcomes,
and the lease correction and withdrew the earlier performance-based delivery GO.
After the exclusive checks, final review found no blocker to default-off
experimental delivery; it did not endorse default-on promotion.
