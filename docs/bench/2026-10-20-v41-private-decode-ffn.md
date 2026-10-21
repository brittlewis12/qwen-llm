# Private Focused FFN Decode Pilot

Successor to preserved `3f564bc9`, still based on production `1388793d`.
Private instrumentation only: no shader, precision, selector, backend or runtime
knob changes. Parent owns independent review and worker scheduling.

## Protocol

Schema `v41-private-restored-decode-ffn-v1`, depth 2048, TG32, capacity 2080,
seed 3. One prefix and snapshot; restore outside timing/I/O before model warmup,
instrumented warmup, control, profile, control. The old capacity-8224 observations
motivated this pilot but are NOT matched causal rates for it. Use this pilot's
own controls and FFN enclosing views. Restored diagnostics are not canonical
throughput. Keep one measured profile window, not repeated sampling/retries.

All 32 full-vocabulary outputs must remain finite and bitwise equal to the
same-candidate model-warmup reference. Every token still has one command and one
wait, no trace/capture copies. Present monotonic physical-read/pagein counters,
zero measured I/O, exact position/stage coverage and the 10% observer-overhead
gate remain mandatory. Retain warmup I/O and all failed attempts.

## Focused Boundaries

Only layers `[0,2,5,10,15,20,25,30,35,38]` replace their FFN composite with:

| Stage | Includes |
|---|---|
| `decode_ffn_mhc_pre` | mHC projection/RMS/controls/collapse and FFN norm |
| `decode_ffn_router_shared` | router projection and shared gate/up/SwiGLU/down |
| `decode_ffn_routed` | routing selection, native routed gate/up/SwiGLU, weighting, down, sum with shared output |
| `decode_ffn_mhc_post` | mHC residual/post/combine |

The selected union covers the measured Q2/IQ3 FFN type cohorts. All other coarse
spans remain unchanged. Forty layers with Engram at 1/14 produce 196 spans and
392 samples/token, indices 0..391 within the unchanged 408-sample buffer.
Without Engram: 193 spans/386 samples. Layouts exceeding the budget fail closed.

`ffn_gpu::Workspace::encode` calls four shared helpers in the original order;
the ordinary path remains one encoder. The sampled path calls these same helpers
with serial encoder boundaries only on selected layers. Each starts with a
self-contained binding operation: mHC projection, router projection, route rows,
or mHC post. No encoder bindings carry across boundaries. `encode_routed`, row
prefill encoding, dispatch wrappers and shaders remain unchanged.

## Interpretation

The predecessor's sorted-endpoint geometry and observation/exclusivity separation
are unchanged. Every stage is an **inclusive, nonadditive encoder envelope**.
Overlap is valid; it proves neither concurrent execution nor hazards. Reject
invalid samples without clipping, normalization, tolerance or fallback timings.
Command GPU, CPU encode/wait/wall and window wall remain separate. There is no
command endpoint-containment witness and no command-uncovered-time claim.

`costs.ffn_enclosing_views_nonadditive` adds a CPU-only alternative FFN view per
layer and position: selected layers use minimum start/maximum end across their
four valid samples; unselected layers retain the original sampled composite.
This is NOT a sum and includes internal gaps. It is excluded from primary stage
statistics and union/overlap geometry to avoid double counting. It is not an
additional sampled span or a matched original-single-encoder measurement.

The routed stage is not pure expert cost. Router/shared remains composite.
Neither these bins nor their ratios establish bandwidth saturation or a speedup.
The earlier M4 and M5 overlapping-envelope observations remain valid under this
schema; see the predecessor interpretation document for exact geometry semantics.

## Verification And Handoff

CPU tests cover selected/unselected presence and absence, exact sample budget,
schema/recipe, all strict admission gates, overlapping/disjoint/nested/identical
intervals, corrupt samples, and enclosing-view math independent of encoded order.
The small local GPU fixture checks restored TG32 bitwise parity with and without
Engram at depths 0/3/6, exact full ordered dispatch census (kernel/grid/threadgroup)
between control and profile, one versus sampled serial encoders, one command,
zero trace copies, valid timestamps and reusable bounded sample buffers.
Census is test-only and remains disabled in the real pilot.

Local results: all 29 checks pass (13 family/protocol, five profiler/geometry,
eight GPU-step, three snapshot). Offline locked Cargo used the normal target,
one test thread, and `MTL_DEBUG_LAYER=1`, `QWEN_REQUIRE_METAL_TESTS=1`,
`QWEN_METAL_LEASE_WAIT=1`. In the focused M4 fixture, profiled Engram-enabled
windows had 25/27, 22/27, and 28/24 overlapping positions at depths 0, 3, and 6;
without Engram there were none. These are fixture observations, not performance
results. All windows passed bitwise and geometry checks.

A source-expansion check verifies the four helper bodies reconstruct exactly
the original 138/3f control dispatch statements (modulo function-return syntax
and repeated local aliases). Routed/prefill bodies, timestamp encoder creation,
sample resolution, geometry and command-execution tail remain byte-identical to
the predecessor. No separate original-model run was performed. Runner CPU-only
fixtures accept valid overlap and reject 11 corrupted parity/command/capacity/
I/O/overhead/coverage/sample/geometry/schema cases. Existing compiler warnings
include the unavailable local Metal 4.1 TensorOps tier; no fallback or new backend
was introduced.

Prepared runner: `/tmp/v41-decode-ffn-attribution.sh`. Q2 with its existing Q6
Engram and IQ3 only; normal lease, offline locked Cargo, normal target directory,
clean pinned build identity, create-new evidence directory. Success restores
clean detached `abefb0af`; failure/interruption withholds checkout restoration
until owned descendants and evidence have been inspected. The runner is not
executed by this checkpoint. Local synthetic tests do not qualify real-model
I/O, overhead, quantized bindings or performance; parent review precedes M5 use.
