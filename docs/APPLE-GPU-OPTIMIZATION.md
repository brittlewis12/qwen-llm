# Apple GPU Optimization for qwen-llm

This is the project-level guide for optimizing `qwen-llm` Metal inference on
Apple GPUs. It distills the highest-signal ideas from Apple's profiling and
compute guidance, plus the best AGX reverse-engineering resources, into one
playbook aimed at our actual workloads: token-by-token decode, packed prompt
prefill, long-context attention, dense FFN mat-mat, MoE routed experts, GDN
recurrent state updates, and speculative paths like DFlash and MTP.

This document is intentionally practical:

- It assumes we already know basic Metal.
- It prefers counter-driven workflow over folklore.
- It is biased toward headless and automatable profiling.
- It maps advice onto the current `qwen-llm` bottlenecks and knobs.

The sources fall into three useful layers:

| Layer | Best use |
| --- | --- |
| Apple counter workflow | Decide whether the problem is occupancy, ALU, memory, atomics, MMU / TLB, or timeline gaps. |
| Apple compute and tensor guidance | Decide how to reshape dispatches, specialize shaders, tile GEMM-like work, and use Metal 4 / MPP when available. |
| Independent AGX internals | Generate hypotheses about SIMD groups, register pressure, caches, scalar-lane behavior, and command queues. |

Primary repo context:

- `docs/PERF-ROADMAP.md`
- `docs/PERF-LOG.md`
- `docs/PERF-TOOLS.md`
- `docs/INFERENCE-GRAPH.md`
- `crates/qwen-llm/src/metal.rs`
- `crates/qwen-llm/src/metal_forward.rs`
- `crates/qwen-llm/src/metal_dflash.rs`
- `kernels/`

## Executive view

If we only keep ten ideas in our head, keep these:

1. The first job is not heroic shader math. The first job is keeping the GPU
   busy. Apple repeatedly emphasizes eliminating GPU timeline gaps, reducing
   CPU/GPU synchronization, and giving the hardware enough work to distribute.
2. Decode and other tiny recurrent kernels are often launch-limited or
   submission-limited before they are ALU-limited.
3. Prompt prefill and large FFN / projection surfaces are usually won by better
   arithmetic intensity, fusion, cache locality, and batching, not by random
   threadgroup-size superstition.
4. On Apple GPUs, occupancy matters, but not as a vanity metric. If ALU is
   already saturated, more occupancy may not help. If occupancy is low and ALU
   and bandwidth are also low, launch shape or resource pressure is the problem.
5. Apple GPUs are unusually sensitive to register pressure, threadgroup-memory
   pressure, cache locality, and queue cadence. They are less forgiving of
   desktop-GPU-style "just stuff more into shared memory" habits.
6. For many modern Apple GPU GEMM cases, especially the new M5 / Metal 4 / MPP
   guidance, direct device-memory access plus caching beats automatic
   threadgroup-memory staging.
7. Simdgroup-first reductions are the default. Use threadgroup memory only when
   there is real inter-simdgroup cooperation to do. Avoid global atomics in hot
   paths unless measurement leaves no alternative.
8. Unified memory removes explicit copy pain, but it does not remove residency,
   locality, synchronization, or working-set pain.
9. Headless is good enough for daily iteration: `qwen-bench` + `xcrun xctrace`
   + Metal Counters API + programmatic captures + HUD / GPUDebug logging.
   Xcode remains the best periodic deep-dive tool for occupancy, shader cost,
   heat maps, execution history, and limiter diagnosis.
10. For this repo today, the highest-EV work is still dense prompt prefill,
    MoE grouped expert-major packed prefill, long-context attention quality,
    and only then command-cadence or speculative-path cleanup.

## What qwen-llm is actually doing

The repo has several distinct Metal workload classes, and they do not want the
same optimization strategy.

- Decode: one command buffer per token, one compute encoder, explicit layer loop,
  final norm + lm_head, then either logits readback or GPU argmax readback.
- Prompt prefill: layer-major packed chunks `P`, with batched projections and
  FFN work, but still some sequential work inside the chunk for attention / KV
  handling and recurrent state updates.
- Long-context attention: `attn_v4` is the production attention family, with
  split-K / group-specific shapes and repo knobs for `NWG`, tile size, and
  group-16 tile choice.
- Dense FFN and projections: decode uses fused mat-vec style kernels; prefill is
  increasingly about real mat-mat quality on Q4/Q5/Q6 surfaces.
- MoE: router and expert execution are already GPU-resident, but packed prefill
  is still bottlenecked by token-major routed expert execution.
- GDN / recurrent state: these kernels are structurally different from attention
  and can be dominated by tiny dependent updates, not bulk FLOPs.
- DFlash / MTP: speculative paths add their own attention and verification
  structure and are especially sensitive to repeated long-context attention cost.

Current repo bottleneck map from the docs:

- Dense packed prompt prefill is now mostly FFN / projection mat-mat quality,
  not recurrent glue.
- Long-context dense decode still has meaningful attention cost growth.
- MoE packed prefill is mostly a routed expert scheduling problem.
- Current dense KV-Q8 is a measured negative result on M4 for the existing
  reader shape.
- CPU encode overhead exists, but current roadmap evidence says it is not yet
  the main bottleneck on hot paths.

Implication: we should not treat "Metal optimization" as one task. We should
optimize different pipeline classes with different measurements and different
success criteria.

Current benchmark guardrails:

- Dense: `Qwen3.6-27B-Q4_K_M.gguf`
- MoE A3B: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- MoE A10B: `Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf`
- Fast dense canary: 9B for quick falsification, then re-confirm on 27B.
- Never run performance benchmarks in parallel.
- Always keep dense 27B in the analysis loop while optimizing MoE.

## The Apple GPU mental model that matters for inference

Apple's official talks and the independent AGX work broadly agree on the parts
that matter most for LLM inference:

- The machine is best thought of as a 32-lane simdgroup machine with strong
  hardware scheduling, large effective register capacity, relatively modest
  caches, good simdgroup communication, and limited patience for bad memory
  locality and overgrown live ranges.
- AGX behaves much more like a scalar-lane machine coordinated by simdgroup
  collectives than like a classic wide-vector desktop ISA.
- Register pressure is often destiny. Many "mystery slowdowns" are really
  occupancy collapses, spills, or code-size / live-range blowups.
- Threadgroup memory is useful, but it is not the universal cure. On newer Apple
  families, direct buffer reads can compete with or beat software-managed
  staging, especially when occupancy and caches are healthy.
- Main-memory round trips still hurt, even under unified memory.
- CPU/GPU coherence is convenient, but synchronization still costs.

The most useful practical translation for `qwen-llm` is:

- Think in terms of launch sufficiency, live-register budget, cache locality,
  and queue cadence before thinking in terms of fancy one-off kernel tricks.
- Default to simdgroup-first reductions and compact kernels.
- Fuse memory-bound epilogues only until register pressure or I-cache pressure
  makes the fused kernel slower than two smaller kernels.
- Assume CUDA instincts are suspect until counters agree.

Two reverse-engineering-derived details are especially useful when interpreting
surprising results:

- Source-level vector types are not proof of vector ALU execution. AGX analyses
  describe scalar arithmetic with vectorized I/O, so `half4` / `float4` should be
  treated primarily as packed memory and compiler-visibility tools unless a
  capture or microbench proves otherwise.
- Half precision is not valuable only because of peak FP16 math. Philip Turner's
  M1/M2 measurements suggest FP16 FMA is not universally 2x FP32 peak on those
  chips, but half can still win by reducing bandwidth, register footprint,
  dependency latency, and cache pressure. Keep FP32 where quality requires it,
  especially GDN state and softmax accumulators.

## The strongest Apple-endorsed optimization ideas

### 1. Eliminate GPU timeline gaps first

Apple's best compute-scaling talk is explicit: remove CPU/GPU sync when
possible, pipeline more work, use `MTLSharedEvent` when signaling is necessary,
and consider concurrent dispatches when independent work can interleave cleanly.

Why it matters here:

- Decode is structurally vulnerable because it is token-by-token and latency
  sensitive.
- DFlash / MTP are vulnerable because small verification and bookkeeping passes
  can fragment the command stream.
- MoE can become dispatch-fragmented if expert work stays token-major.

Use this idea to ask:

- Are we blocking on `waitUntilCompleted` where a completion handler or event
  would do?
- Are we creating avoidable per-token idle gaps between command buffers?
- Can independent work be interleaved to hide ramp-up and tail effects?

### 2. Make each dispatch large enough to saturate the machine

Apple's rule of thumb from `Scale compute workloads across Apple GPUs` is that
relatively complex kernels often want on the order of `1K-2K` concurrent threads
per shader core. The exact number is not the point. The point is that many
small kernels simply do not put enough work in flight.

Why it matters here:

- Decode helpers like norm, argmax, small reductions, and recurrent updates can
  look "fast" individually while still leaving most of the GPU idle.
- Packed prefill, grouped MoE work, and large FFN surfaces are the natural ways
  to create enough in-flight work.

### 3. Prefer the smallest useful multiple of `threadExecutionWidth`

Apple repeatedly recommends using the smallest threadgroup size that is a clean
multiple of the SIMD width and still matches the workload well. Bigger is not
automatically better.

Why it matters here:

- `attn_v4` split-K and tile sweeps can accidentally over-commit threadgroup or
  register resources in the name of "more work".
- Recurrent kernels can become less distributable with bloated threadgroups.

### 4. Use occupancy as a diagnostic, not a religion

Apple's newer profiling workflow on M3 / A17 Pro and M5 is especially clear:
low occupancy is not automatically bad, and high occupancy is not automatically
good. Ask what is limiting occupancy, and ask whether more occupancy would help
the actual bottleneck.

Interpretation rule:

- Low occupancy + low ALU + low bandwidth = likely underfill or resource pressure.
- Low occupancy + high ALU = probably fine.
- Low occupancy + high LLC / MMU / L1 pressure = memory / locality problem first.

### 5. Optimize memory locality before chasing exotic math

Apple's counter guidance consistently points to LLC and MMU limiters, TLB miss
rate, and memory-span issues as first-class bottlenecks. This is especially
important for long-context attention and gather-scatter-heavy MoE work.

Why it matters here:

- Attention is fundamentally a locality problem once the algorithm is good.
- Routed expert execution is fundamentally a locality problem once the router is
  correct.
- Dense prompt mat-mat is often won by lowering bytes moved per unit of useful
  work, not by heroic scalar ALU tuning.

### 6. Be skeptical of threadgroup-memory staging by default

The new Metal Performance Primitives guidance for M5 is one of the clearest
Apple statements on modern GEMM design: direct device-memory access plus caches
and enough occupancy can outperform threadgroup-memory staging; explicit
software pipelining is not always needed if occupancy is healthy.

This does not mean "never use threadgroup memory." It means:

- benchmark direct reads vs staged reads on Apple hardware,
- especially for quantized matmul inner loops,
- and especially when staging explodes live state or barrier count.

### 7. Use simdgroup collectives before atomics

Apple's compute-scaling guidance is direct here: reduce global atomic pressure,
aggregate locally first, and use threadgroup-level atomics only when needed.

Why it matters here:

- softmax max/sum,
- MoE routing reductions,
- expert accumulation,
- argmax,
- profiling counters inside kernels.

Default reduction hierarchy:

1. Reduce inside one simdgroup with `simd_sum`, `simd_max`, scans, or shuffles.
2. Write one value per simdgroup to threadgroup memory.
3. Reduce across simdgroups inside the threadgroup.
4. Write or atomic-add one value per threadgroup to device memory.

This is the safe starting point for softmax, RMS / L2 norm, MoE scatter-reduce,
top-k histograms, and future calibration kernels.

### 8. Use smaller types and compiler-friendly code shapes

Apple's shader best-practices talks keep returning to the same ideas:

- prefer `half` / `bfloat` / smaller integer types when correct,
- prefer `constant` data for small shared metadata,
- avoid dynamic stack arrays and spill-heavy code,
- avoid unsigned indexing in tight loops when it harms optimization,
- specialize with function constants when that removes branches and bounds work.

Specialize the shapes that are common in this repo: `head_dim`, GQA `GROUP`,
attention tile `C`, quant type, KV dtype, RoPE mode, and causal / window flags.
Also sweep `max_total_threads_per_threadgroup` / pipeline occupancy hints for
large complex kernels; Apple's own guidance is that the optimum is not always
the device maximum. Use `h` suffixes for half literals where needed, and avoid a
single uber-kernel controlled by many runtime buffer flags.

### 9. Keep unified-memory thinking honest

Apple silicon's unified memory means we do not need CUDA-style copy choreography,
but it does not mean every shared buffer shape is equally good. Working-set
size, residency churn, and locality still matter.

This maps directly onto a current repo question: whether no-copy GGUF-backed
views plus warmup are better than copying many weight tensors into individual
shared buffers.

### 10. Use GUI tools periodically even in a headless workflow

The best daily loop is headless. The best explanation for weird winners and
losers is still often in Xcode's GUI profiling views: shader cost graph, heat
maps, shader execution history, occupancy manager targets, register pressure,
L1 / LLC pressure, and limiter views.

Use the GUI sparingly, but do use it.

## A counter-driven workflow for qwen-llm

This should be the default optimization loop.

### Step 0: establish a clean baseline

Use untraced throughput runs first. Traces explain throughput; they do not set
the regression threshold.

Keep the measurement regime fixed and label it explicitly: throughput,
phase-attribution, queue-timeline, or correctness-gated speculative run.

Primary repo commands are already documented in `docs/PERF-TOOLS.md`.

Preflight for a clean local run:

```sh
cargo build --release -p qwen-cli --bin qwen-bench

: "${MODEL:?set MODEL to a GGUF}"
test -f "$MODEL"
mkdir -p target/profiles
```

Typical starting points:

```sh
./target/release/qwen-bench decode -m "$MODEL" --tokens 128
./target/release/qwen-bench phase -m "$MODEL" --ctx 4096
./target/release/qwen-bench ctx-sweep -m "$MODEL" --checkpoints 1,16,64,4096,16384,32768 --window 2
```

For prompt work:

```sh
./target/release/qwen-bench decode -m "$MODEL" -p "$PROMPT" --tokens 4
./target/release/qwen-bench decode -m "$MODEL" -p "$PROMPT" --tokens 4 --sequential-prefill
./target/release/qwen-bench decode -m "$MODEL" -p "$PROMPT" --tokens 4 --prefill-chunk 128
```

For DFlash:

```sh
./target/release/qwen-bench dflash -m "$TARGET_MODEL" --drafter "$DRAFTER_MODEL" --tokens 64 --profile --n-policy adaptive
```

### Step 1: classify the bottleneck before changing code

Ask these in order:

1. Is the wall time dominated by prompt prefill, decode, attention, FFN, MoE,
   or speculative verification?
2. Is the GPU actually busy, or are there queue gaps and CPU waits?
3. If the GPU is busy, is the hot kernel compute-bound, memory-bound,
   occupancy-limited, or launch-limited?
4. Is the observed pain coming from algorithmic bytes moved, from code shape, or
   from submission topology?

Never start with "let's tune threadgroup size" as the first question.

### Step 2: separate queue problems from kernel problems

Use `Metal System Trace` via `xcrun xctrace` when the question is command-buffer
cadence, queue overlap, idle gaps, or CPU ownership.

Typical flow, using a unique output path because `xctrace` will not overwrite an
existing trace bundle:

```sh
xcrun xctrace list templates

TRACE="target/profiles/metal-decode-$(date +%Y%m%d-%H%M%S).trace"
xcrun xctrace record --no-prompt \
  --template "Metal System Trace" \
  --time-limit 20s \
  --output "$TRACE" \
  --target-stdout - \
  --launch -- ./target/release/qwen-bench decode \
    -m "$MODEL" --tokens 256 --no-warmup

xcrun xctrace export --input "$TRACE" --toc
```

If command-buffer GPU time is materially smaller than token interval, and the
timeline shows gaps, fix cadence first.

### Step 3: instrument hot phases with runtime timestamps and counters

For headless iteration, the best Apple-native path is Metal Counters API plus
repo-local timing output.

Key APIs to know:

- `MTLDevice.counterSets`
- `supportsCounterSampling(_:)`
- `makeCounterSampleBuffer(descriptor:)`
- `sampleCounters(...)`
- `resolveCounterRange(_:)`
- `MTLCounterSamplingPoint.atDispatchBoundary`
- `MTLCounterSamplingPoint.atStageBoundary`

Best use here:

- timestamp start/end around hot compute passes,
- per-phase timing for attention, FFN, MoE, DFlash verify, and logits tail,
- JSON or CSV output per run so sweeps are easy to compare.

Minimal API shape, shown as Swift-style pseudocode because it is clearer than the
Rust FFI surface:

```swift
let sets = device.counterSets
let timestampSet = sets.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }

let desc = MTLCounterSampleBufferDescriptor()
desc.counterSet = timestampSet
desc.storageMode = .shared
desc.sampleCount = 2
let samples = try device.makeCounterSampleBuffer(descriptor: desc)

let enc = commandBuffer.makeComputeCommandEncoder()!
enc.sampleCounters(sampleBuffer: samples, sampleIndex: 0, barrier: true)
enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
enc.sampleCounters(sampleBuffer: samples, sampleIndex: 1, barrier: true)
enc.endEncoding()
```

Use `barrier: true` for diagnostic stability, not final throughput numbers; it
can change the performance being measured.

### Step 4: use limiters and occupancy to decide what kind of change to try

Fast interpretation guide:

- High ALU, decent occupancy, low memory pressure: simplify math, shrink live
  state only if it improves issue rate, otherwise move on.
- High LLC / MMU limiter, low ALU: fix layout, access order, fusion boundaries,
  or tile traversal.
- Low occupancy, low ALU, low bandwidth: increase work per dispatch, reduce
  register pressure, reduce threadgroup memory, merge tiny kernels.
- High atomic pressure: privatize, reduce hierarchically, or change work
  partitioning.
- Clean kernel counters but bad wall time: look for queue gaps, readback, or CPU
  control overhead.

### Step 5: only trust wins that move end-to-end qwen-bench numbers

Microbench wins are not enough.

Require at least:

- correctness on the existing repo gates,
- end-to-end `qwen-bench` improvement on the relevant guardrail model,
- no obvious regression on the dense 27B baseline when optimizing MoE, and vice
  versa where relevant.

## Headless tooling stack

This is the practical stack for daily work.

### 1. qwen-bench

Use it for model-aware throughput and phase attribution first. It already knows
about our real decode, prompt, attention, GDN, MoE, and DFlash surfaces.

### 2. xcrun xctrace

Use it for headless Instruments traces. It is the best CLI for queue timeline
questions and targeted CPU attribution.

Good uses:

- `Metal System Trace` for GPU ownership, queue gaps, idle periods, and command
  cadence.
- `Time Profiler` for CPU encode, host-side waits, load path, tokenizer, and
  readback work.
- `--toc` and targeted export instead of giant raw XML dumps.

### 3. Metal Counters API

Use it when we want automated per-dispatch or per-phase GPU timing and counter
collection inside normal benchmark sweeps.

This is the most important Apple-supported path for runtime / CI-like profiling.

### 4. Programmatic GPU capture

Use `MTLCaptureManager` and `MTLCaptureDescriptor` when we want to capture a
specific bad case for later Xcode inspection.

Enable with either:

- `MetalCaptureEnabled=YES` in `Info.plist`, or
- `MTL_CAPTURE_ENABLED=1` on macOS 14+.

This is ideal for "capture the slow token" workflows.

The capture shape is simple once the app has capture entitlement / environment
enabled:

```swift
let manager = MTLCaptureManager.shared()
let desc = MTLCaptureDescriptor()
desc.captureObject = device
desc.destination = .gpuTraceDocument
desc.outputURL = URL(fileURLWithPath: "/tmp/qwen-prefill.gputrace")

try manager.startCapture(with: desc)
// encode one representative prefill/decode iteration
manager.stopCapture()
```

Remember the file types: `xctrace` produces `.trace` bundles, while Xcode Metal
capture / `MTLCaptureManager` produces `.gputrace` bundles. A CLI trace cannot be
converted into a GPU capture after the fact.

### 5. Metal Performance HUD and logs

Use HUD logging when we want a lightweight sanity check on GPU time, encoder
timing, and run-to-run cadence without a full trace.

Useful env vars from Apple docs:

```sh
MTL_HUD_ENABLED=1
MTL_HUD_LOG_ENABLED=1
MTL_HUD_LOG_SHADER_ENABLED=1
```

Older Apple HUD material also documents `MTL_HUD_LOGGING_ENABLED=1`; verify the
spelling accepted by the local Xcode / macOS pair before building automation.

Apple also documents extra HUD configuration, reporting, and encoder timing
options. The HUD is not a deep profiler, but it is a fast way to verify that a
change really moved GPU time in the direction we expect.

### 6. GPUDebug logs and enhanced command-buffer errors

Use these in correctness and validation runs, not in final performance numbers.

Key pieces:

- `MTLCommandBufferDescriptor.errorOptions = .encoderExecutionStatus`
- `MTLCommandBufferEncoderInfoErrorKey`
- `log stream --predicate "subsystem = 'com.apple.Metal' and category = 'GPUDebug'"`

These are not performance tools first. They are "stop silently lying to
yourself" tools.

## Existing repo instrumentation worth reusing

Before adding new plumbing, use what is already here.

- `qwen-bench phase` for model-aware phase attribution.
- decode token profiling and phase profilers in `metal_forward.rs`.
- intra-block profilers for attention, GDN, and MoE in `metal_forward.rs`.
- DFlash per-phase GPU timing in `metal_dflash.rs`.
- benchmark flags like `--prefill-chunk`, `--sequential-prefill`,
  `--full-logits-decode`, `--oracle-phase`, and `dflash --profile`.

The point is to add only the instrumentation the repo is still missing, not to
duplicate timing paths we already have.

## Workload-by-workload playbook

### Decode

Main risk profile:

- launch underfill,
- too many tiny dependent kernels,
- queue gaps,
- readback or CPU synchronization at the wrong boundary,
- long-context attention growth.

Measure first:

- total ms/token,
- phase share,
- GPU command time vs CPU encode vs CPU wait,
- command buffers per token,
- attention share vs context length.

Try first when decode is slow:

- verify whether attention is the real culprit,
- if not, fuse tiny memory-bound tails or batch helper work,
- if queue gaps show up, pipeline more work or reduce sync points,
- avoid reopening large structural bets like ICB / Metal 4 until traces prove
  cadence is the real issue.

Q4_K mat-vec details worth preserving from the lower-level playbook:

- Keep dequantization fused into the dot product; apply scale / min once per
  sub-block after accumulating raw nibble products, as the existing fast kernel
  does in `kernels/mat_vec_q4_k.metal`.
- Reuse the input activation vector across multiple output rows only while it
  improves cache reuse without collapsing occupancy.
- Report effective GiB/s over real bytes read, including quant metadata, not only
  math throughput.
- Treat inferred 128-byte Apple GPU cache-line behavior from microbenchmarks as a
  layout sanity check, not a vendor guarantee.

### Prompt prefill

Main risk profile:

- insufficient chunk size,
- underpowered mat-mat kernel shape,
- extra read / write round trips between same-input projections,
- sequential leftovers inside the packed path,
- working-set or locality problems on large surfaces.

Measure first:

- prompt tok/s,
- TTFT,
- phase share,
- saturation vs `--prefill-chunk`,
- effective bandwidth / throughput on the real 27B and 9B surfaces.

Try first:

- larger chunk until the plateau is obvious,
- same-input projection fusion where correctness is clean,
- locality-preserving tile or work traversal,
- only then broader mat-mat backend gardening.

For GEMM-like prompt work, use the Metal 4 / MPP mental model even when the
shipping kernel is still custom Metal:

- decompose output into simdgroup tiles, threadgroup tiles, and grid traversal,
- sweep tile sizes per shape because larger tiles improve reuse but can hurt
  occupancy and awkward-size quantization,
- try Morton-like or other locality-preserving threadgroup walk orders for large
  surfaces,
- fuse epilogues while values are still in registers or cooperative tensors,
- on M5 / A19 targets, evaluate MPP TensorOps / cooperative tensors behind a
  device gate while preserving the portable custom path.

### Long-context attention

Main risk profile:

- memory locality,
- LLC / MMU pressure,
- split-K balance,
- tile choice,
- register pressure in the hot path,
- bytes moved by KV handling.

Measure first:

- context slope at 4K / 16K / 32K,
- attention phase share,
- GPU busy vs queue gap,
- occupancy,
- LLC limiter / utilization,
- MMU limiter / TLB miss rate,
- ALU vs memory balance.

Try first:

- the existing env knobs before new architecture work,
- access-order and tile experiments that reduce span,
- only revisit KV-Q8 with a fundamentally different reader shape.

Relevant repo knobs:

- `QWEN_ATTN_V4_NWG`
- `QWEN_ATTN_V4_TILE_C`
- `QWEN_ATTN_V4_G16_TILE`

Attention-specific rules:

- Keep QK scaling, mask, max reduction, exp / sum reduction, and V accumulation
  fused; avoid materializing score matrices.
- Preserve the `attn_v4` GQA-dedup property: read each K/V tile once per KV head
  and share it across sibling Q heads.
- Use `exp2` with prescaled inputs where that remains numerically acceptable.
- For speculative verification, prefer multi-query packed attention that shares
  KV reads across candidate tokens over policy-only tuning.

### Dense FFN and projection mat-mat

Main risk profile:

- bytes moved dominate FLOPs,
- paired same-input projections are split unnecessarily,
- staging or fusion choices damage occupancy,
- quantized-dequantized live ranges get too big.

Measure first:

- end-to-end prefill change on 27B,
- FFN phase share,
- bandwidth / occupancy tradeoff,
- whether fusion improves wall time or only microbench time.

Try first:

- fuse `gate_proj + up_proj` style pairs that read the same activation stream,
- keep fusion compact enough to avoid register explosions,
- benchmark direct memory access vs more elaborate staging on Apple hardware.

### MoE

Main risk profile:

- token-major expert execution,
- atomics or scatter-reduce pain,
- poor expert-major locality,
- tiny expert dispatches,
- gather / scatter overhead consuming the expected grouped-work gain.

Measure first:

- routed FFN share,
- mixer-prep share,
- gather / scatter cost,
- expert batch-size distribution,
- end-to-end packed prefill speedup after grouping.

Try first:

- explicit ledger of `(token_idx, topk_rank, expert_id, weight)`,
- group by expert,
- run grouped expert-major FFN,
- scatter / reduce back in fixed top-k order,
- treat reduced-precision routed-mid storage as a follow-on, not the first bet.

### GDN / recurrent state

Main risk profile:

- many tiny dependent operations,
- launch-limited update kernels,
- state traffic and synchronization overhead,
- over-fusion that destroys occupancy.

Measure first:

- intra-block phase timing,
- whether the kernel is actually GPU-bound or just too small,
- per-token cadence cost from recurrent glue.

Try first:

- batch the time loop where correctness allows,
- keep small reductions simdgroup-local,
- reduce kernel count before over-engineering a single large fused kernel.

GDN-specific rules:

- Keep the 128 x 128 per-head recurrent state row in registers for the recurrence
  body when possible.
- Keep GDN state FP32 unless a separate numerical investigation proves otherwise.
- Precompute or fuse scalar decay / beta chains outside the state inner loop.
- Watch SFU-heavy pieces such as `exp`, `softplus`, and `sigmoid`; move or
  approximate them only when correctness gates allow it.

### DFlash / MTP

Main risk profile:

- repeated long-context verify attention,
- weak acceptance so verification dominates,
- additional queue fragmentation,
- materialization of intermediate K/V ranges that increase traffic.

Measure first:

- `alpha`, mean emitted tokens per step, verify cost share,
- phase timing from `--profile`,
- context sensitivity,
- GPU time vs total wall.

Try first:

- attack repeated verify attention cost before policy micro-tuning,
- prefer multi-query packed verify structures over pure control-flow tweaks,
- revisit cache layout only if measurement shows the extra materialization is a
  real memory bottleneck.

## A measurement matrix for quick diagnosis

| Workload | First metric | First deep signal | Likely first lever |
| --- | --- | --- | --- |
| Decode | ms/token | GPU time vs CPU wait | kernel count, sync points, attention share |
| Prompt prefill | tok/s, TTFT | chunk saturation | batching, fusion, mat-mat quality |
| Long-context attention | attention ms/token | LLC / MMU / occupancy | tile / split-K / access order |
| Dense FFN mat-mat | end-to-end prompt delta | bytes moved vs occupancy | pair fusion, layout, tile walk |
| MoE | packed-prefill tok/s | routed FFN share | expert-major grouping |
| GDN | per-block timing | launch sufficiency | batching and fewer tiny kernels |
| DFlash / MTP | net speedup | verify attention cost | packed verify attention |
| Command cadence | token interval | idle gaps in Metal System Trace | fewer waits, better pipelining |

## Sweep dimensions to record

Every optimization needs a sweep, not a single run. Persist enough metadata to
replay the result later.

| Dimension | qwen-llm examples |
| --- | --- |
| Model | 9B canary, 27B dense guardrail, 35B A3B, 122B A10B |
| Context | 1, 256, 4K, 16K, 32K, plus awkward non-powers-of-two |
| Prompt chunk | `P=32,64,128,256,512,1024` where memory allows |
| Threadgroup shape | multiples of `threadExecutionWidth`; 1, 2, 4, 8 simdgroups per TG |
| Work per thread | rows per lane, independent accumulators, vectorized vs scalar loads |
| Attention | `NWG`, tile `C`, group-specific variants, split-K partitioning |
| Precision | F16 storage, F32 accumulators, Q4_K, Q8_0 KV experiments, mixed epilogues |
| Fusion | separate vs fused gate / up / activation, dequant, reductions, residuals |
| Scheduling | command buffers per token / chunk, dispatch count, waits, future ICB / MTL4 |
| Correctness | logits cosine, hidden capture, KV / GDN state checks, DFlash equivalence |

Always include device, macOS, Xcode, power mode, thermal state if available, git
commit, build flags, model path, prompt length, context length, and active env
knobs.

## A decision tree for interpreting results

Use this before changing code.

- If `phase` says dense prompt is FFN / projection dominated, work on paired
  same-input projection fusion and mat-mat quality first.
- If MoE packed prefill is still routed-FFN dominated, do expert-major grouping
  before any more token-loop cleanup.
- If attention cost rises sharply with context and the GPU is busy, tune
  attention shape or KV layout.
- If attention cost rises with context and LLC / MMU limiters are high while ALU
  is low, treat it as a locality problem first, not an occupancy problem.
- If occupancy is low but ALU or bandwidth is already saturated, leave occupancy
  alone.
- If occupancy is low and ALU and bandwidth are also low, increase work per
  dispatch or reduce resource pressure.
- If traced runs show real idle gaps, then and only then prioritize command
  cadence work like pipelining or event-based overlap.
- If a microbench improves and `qwen-bench` does not, discard the change or
  tighten the experiment.
- If DFlash acceptance is acceptable but speedup is poor at long context, attack
  repeated verify attention cost before draft-policy gardening.

## Repo-specific priorities right now

Given the current roadmap and measurements, the most promising order is:

1. MoE grouped expert-major packed prefill.
2. Dense packed prompt same-input projection fusion, especially `gate + up`.
3. Long-context attention knob sweeps and locality work using the existing
   `attn_v4` control surface.
4. Continue using 9B as a fast falsifier, then re-confirm on 27B.
5. Only revisit KV-Q8 with a new reader structure.
6. Only escalate to ICB / Metal 4 / deeper command-cadence work if traces show
   the GPU is being starved.
7. Keep speculative-path work behind the main prompt and decode bottlenecks
   unless a specific acceptance result changes the order.
8. Prototype no-copy GGUF-backed Metal views and residency warmup as a TTFT,
   memory-object, and cold-start stability bet, not as a presumed steady-decode
   win.
9. Evaluate MPP / TensorOps behind device and build gates for dense packed GEMM
   on M5 / A19, while keeping custom K-quant kernels as the cross-family path.

## Minimal additional instrumentation worth adding

The repo already has substantial timing hooks. The highest-value additions are
small and operational:

- Optional `qwen-bench` JSON or CSV output with model, context, prompt shape,
  env knobs, GPU phase times, CPU encode time, CPU wait time, and total wall.
- p50 / p95 token latency and prefill chunk latency, not only means.
- Stable command-buffer and encoder labels so `Metal System Trace` and HUD logs
  line up cleanly with our repo phase names.
- Optional counter-sample-buffer timestamps around hot compute passes once basic
  export is stable; start with timestamps before richer counters.
- A small table of supported counter sets / counter names emitted once per run,
  because support varies by GPU family, OS, and Xcode.

## Common traps and anti-patterns

- Treating every Metal slowdown as a shader issue when the real problem is queue
  cadence.
- Porting CUDA shared-memory recipes directly without benchmarking direct-memory
  variants.
- Over-fusing until register pressure or code size cancels the memory-traffic
  win.
- Chasing occupancy when the real limiter is already ALU or memory.
- Reopening KV-Q8 without a materially different design after a clear negative
  result.
- Using validation-heavy builds to judge throughput.
- Comparing traced runs as regression numbers instead of using them for
  explanation.
- Optimizing tiny microbench winners that do not move end-to-end `qwen-bench`.
- Building one flexible shader for all shapes instead of specializing common Qwen
  shapes and measuring the variants.
- Trusting reverse-engineered constants as specs instead of using them to choose
  experiments.

## Ranked resource guide

This is the reading order that gives the most signal for our use case.

### Tier 1: the backbone

1. Apple - [Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/)
   - Best single Apple talk for dispatch scaling, GPU saturation, queue gaps,
     memory locality, and atomic strategy.
2. Apple - [Optimize Metal apps and games with GPU counters](https://developer.apple.com/videos/play/wwdc2020/10603/)
   - Best Apple counter-interpretation workflow.
3. Apple - [Explore Live GPU Profiling with Metal Counters](https://developer.apple.com/videos/play/tech-talks/10001/)
   - Best Apple headless/runtime profiling direction.
4. Apple - [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
   - Best Apple guidance for GEMM, arithmetic intensity, fusion, and modern tile
     strategy, especially on M5 / Metal 4.
5. Apple - [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/)
   - Best Apple talk for shader/compiler-facing optimization.

### Tier 2: essential support

6. Apple - [Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/)
   - Best Apple execution-model and unified-memory refresher.
7. Apple - [Discover new Metal profiling tools for M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111374/)
   - Best Apple resource for occupancy-manager and new profiling views.
8. Apple - [Boost your graphics performance with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111431/)
   - Useful for M5-specific occupancy-target signals and new profiling intuition.
9. Apple - [Optimize Metal Performance for Apple silicon Macs](https://developer.apple.com/videos/play/wwdc2020/10632/)
   - Rendering-heavy, but still very useful for dependency management, smaller
     types, and compiler-friendly code shape.
10. Apple docs - [Capturing a Metal workload programmatically](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically)
    and [MTLCaptureManager](https://developer.apple.com/documentation/metal/mtlcapturemanager)
    - Best path for reproducible captures from our own runtime.
11. Apple docs - [Monitoring your Metal app's graphics performance](https://developer.apple.com/documentation/xcode/monitoring-your-metal-apps-graphics-performance)
    plus the HUD customization and metrics docs
    - Best lightweight runtime sanity-check layer.
12. Apple - [Debug GPU-side errors in Metal](https://developer.apple.com/videos/play/wwdc2020/10616/)
    - Best guidance for enhanced command-buffer errors and GPUDebug logs.

Practical docs to keep bookmarked while implementing the headless path:

- Apple docs - [GPU counters and counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
  - Programmatic `MTLCounterSet`, `MTLCounterSampleBuffer`, support queries,
    `sampleCounters`, and resolving samples.
- Apple docs - [Command buffer debugging](https://developer.apple.com/documentation/metal/command-buffer-debugging)
  - `errorOptions`, encoder execution status, command-buffer logs,
    `gpuStartTime`, and `gpuEndTime`.
- Apple docs - [Customizing the Metal Performance HUD](https://developer.apple.com/documentation/xcode/customizing-metal-performance-hud)
  and [Generating performance reports with the Metal Performance HUD](https://developer.apple.com/documentation/xcode/generating-performance-reports-with-metal-performance-hud)
  - HUD logging, encoder timing, and report output.
- Apple docs - [Xcode command-line tool reference](https://developer.apple.com/documentation/xcode/xcode-command-line-tool-reference)
  and the installed `xctrace(1)` man page
  - Template listing, trace recording, `--toc`, and targeted export.
- Apple - [Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/)
  - TensorOps, cooperative tensors, quantized tensor inputs, neural accelerator
    utilization, K synchronization, and GEMM threadgroup traversal.

### Tier 3: independent deeper dives

13. Philip Turner - [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
    - Best independent benchmark-driven view of Apple GPU behavior for kernel
      designers.
14. Dougall Johnson - [applegpu](https://github.com/dougallj/applegpu)
    and [docs](https://dougallj.github.io/applegpu/docs.html)
    - Best ISA-level AGX reverse-engineering resource.
15. Alyssa Rosenzweig - Asahi GPU reverse-engineering series:
    [part I](https://rosenzweig.io/blog/asahi-gpu-part-1.html),
    [part III](https://rosenzweig.io/blog/asahi-gpu-part-3.html), and
    [part IV](https://rosenzweig.io/blog/asahi-gpu-part-4.html)
    - Best readable architectural explanation, including scalar arithmetic,
      register pressure, occupancy behavior, and command-stream context.
16. Asahi Linux - [AGX docs](https://asahilinux.org/docs/hw/soc/agx/)
    and related driver notes
    - Best queue / firmware / memory-system background when official Metal docs
      are too shallow.
17. Mesa - [Asahi driver docs](https://docs.mesa3d.org/drivers/asahi.html)
    - Useful AGX glossary, tiler / renderer vocabulary, image-layout notes, and
      driver tooling context.

## Source notes and caveats

- Apple's best current shader-level and occupancy visuals are still mostly GUI
  first. Xcode remains the richest periodic deep-dive environment.
- The M3 / A17 Pro and M5 profiling talks are architecture-generation-specific.
  Use them for newer-family intuition, not as timeless law.
- The MPP guide is especially important for M5 / Metal 4 tensor-style kernels.
  Treat it as authoritative for that path, not as a blanket rule for every older
  Apple GPU.
- M5 / A19 TensorOps guidance is highly relevant for future Metal 4 paths, but it
  does not replace portable custom kernels on older Apple GPUs.
- Counter names, availability, and GUI views vary by GPU family, macOS, and Xcode.
  Query support at runtime and record tool versions with benchmark output.
- Independent AGX sources are invaluable, but they are still reverse-engineered.
  Treat them as evidence-backed mental models and experiment generators, not as
  substitute vendor specs.
- Thermal state and power mode matter. Record them when possible or run enough
  repeats to detect drift.

## Practical default loop for this repo

When in doubt, do this:

1. Run an untraced `qwen-bench` baseline on the relevant guardrail model.
2. Use `qwen-bench phase` to identify the dominant phase.
3. If cadence is in doubt, record `Metal System Trace` with `xcrun xctrace`.
4. If the hot phase is clear, add or use runtime timestamps / counters there.
5. Sweep one knob family at a time.
6. Confirm on the real target model, not only the fast canary.
7. Keep only changes that move end-to-end numbers and preserve correctness.

That is the Apple-native optimization loop for `qwen-llm`.
