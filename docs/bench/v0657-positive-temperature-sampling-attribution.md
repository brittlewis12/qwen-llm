# v0.657 Positive-Temperature Sampling Attribution

Status: preregistration. No v0.657 instrumentation, model-backed observation,
or timing result exists yet.

## Intent And Authority

Measure the host-side work around sampler-v1 in one exact positive-temperature
A3B product request before building reusable workspaces, borrowed-logit views,
or a different top-k organization.

This packet may authorize one later bounded implementation packet. It cannot
promote a product optimization, GPU sampling, a changed distribution, a new
sampler algorithm version, another model, or a general sampling claim.

The primary question is whether even an optimistic removal of current logits
allocation/copy and candidate-vector work clears both:

- `5%` of complete generation wall; and
- `5 ms` on the named 128-output request.

Greedy GPU-argmax evidence is excluded. Full-head GPU execution, mandatory
probability arithmetic, and prompt prefill receive no avoidable-work credit.

## Frozen Fixture

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`,
  `22,134,528,992` bytes, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt:
  `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  `1,891` bytes, 419 product tokens, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Tokenizer: the native `qwen_llm::tokenizer::Tokenizer` constructed from the
  exact authenticated GGUF metadata, with no chat-template rendering and
  `add_special_tokens=true`. The 419 signed token IDs, encoded consecutively as
  little-endian `i32`, have SHA-256
  `fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f`.
  The first IDs are `[248045,8678,198,2523,513,279,8852,1892]`; the last are
  `[198,248045,74455,198,248068,271,248069,271]`. The model content hash binds
  tokenizer metadata; every child must also report the same runtime tokenizer
  identity as the reference.
- Sampling: sampler-v1, temperature `0.7`, top-k `200`, top-p `1.0`,
  min-p `0.05`, seed `42`.
- Request: 128 output tokens, context 1024, prefill chunk 1024, no prompt
  lookup, no RAM or durable prefix admission, no warm follow-up, and redirected
  stdout.
- Process: one fresh process and one request per observation. The model file may
  be cache-warm; file-cache state does not enter the decode attribution.
- Environment: archive the complete inherited environment and require no
  `QWEN_*` variables. Do not silently force a decode branch.

The direct product command is:

```text
target/release/qwen
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt
  --tokens 128
  --temp 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 42
  --prefill-chunk 1024 --max-context-tokens 1024
  --prefix-cache-max-mib 0 --cache-prefix-auto-min-tokens 0
  --request-timings UNIQUE_PATH
```

Profiled children add only `--sampling-attribution`.

## Instrumentation Contract

`--sampling-attribution` is opt-in and fail-closed. It requires one single-turn
request, `--request-timings`, positive temperature, no prompt lookup, no warm
follow-up, and the current concurrent-MoE full-logit path. It is unavailable to
JSONL and durable-cache modes. Ordinary CLI behavior and timing schemas remain
unchanged when the flag is absent.

The profiled Metal transition must call the same production concurrent-MoE body,
final norm, resident full Q6_K head, command-buffer commit, and wait. It may add
timers only around the existing post-completion `vec![0.0; vocab]` operation and
the existing `copy_nonoverlapping` from Shared logits. It must not split the
command buffer, add a counter sample, change tail ownership, or alter command
status policy.

For each of the 127 target transitions, aggregate:

- CPU command encoding;
- commit plus completion wait;
- GPU command timestamps, labeled as nested inside completion wait;
- full-logit allocation and zero-fill;
- full-logit CPU copy;
- inner profiled transition wall;
- outer generation-loop transition wall; and
- signed inner residual plus signed outer wrapper/`sequence.advance_by`
  residual.

The inner wall reconciles CPU encoding, completion wait, logits allocation,
logits copy, and its signed residual. The outer wall encloses the complete inner
call plus wrapper and sequence-advance work. GPU time is nested in completion
wait and never enters either additive sum twice.

The prompt-tail logits copy remains inside aggregate prefill and receives no
transition-readback attribution.

The profiled sampler must preserve the exact sampler-v1 statement, comparison,
floating-point, and RNG order. Across all 128 selections, aggregate:

- total sample wall;
- shape validation;
- full-vocabulary candidate allocation;
- candidate validation, F32-to-F64 conversion, and fill;
- top-k partial selection and retained-set ordering;
- min-p filtering;
- positive-infinity filtering;
- temperature scaling;
- probability-weight construction;
- top-p filtering;
- final sum, one RNG draw, and categorical scan; and
- additive residual.

Also record call count; total/minimum/maximum input-logit counts; candidate
capacity bytes; retained counts after each filter; probability-weight capacity
bytes; candidate-index counts; and sampler draws. Capacity fields distinguish
aggregate allocation volume across calls from peak live capacity. No complete
logit row is serialized or hashed. Timers must not enter inner arithmetic or
comparison expressions. Tiny phases are diagnostic totals over the request,
not per-call performance claims.

The instrumentation adds exactly 11 timed spans per sampler call and two per
transition, or `128*11 + 127*2 = 1,662` new start/elapsed clock pairs. In the
same binary, run seven batches of 100,000 back-to-back, black-boxed
`Instant::now`/`elapsed` no-op pairs. Record nanoseconds per pair for all seven
batches; define the outward observer bound as the maximum batch value rounded
up to the next whole nanosecond. The request-level observer bound is that value
times 1,662. Subtract the complete bound from every removable-work upper bound;
do not use a paired timing control.

The request-timing row uses schema 11 only when attribution is present and adds
one optional nested object. Existing schema 10 serialization and key sets remain
unchanged without the flag. All time fields below are finite `f64` milliseconds;
all counts and byte fields are nonnegative integers; digests are lowercase
64-character SHA-256 strings.

```text
sampling_attribution: {
  version: u32 = 1,
  prompt_token_ids_sha256: string,
  clock_probe: {
    batches: u32 = 7,
    iterations_per_batch: u64 = 100000,
    pair_ns: [f64; 7],
    upper_pair_ns: f64,
    new_timer_spans: u64 = 1662,
    observer_upper_ms: f64
  },
  sampler: {
    calls, timer_spans, input_logits_total,
    input_logits_min, input_logits_max: u64,
    wall_ms, shape_validation_ms, candidate_alloc_ms,
    candidate_fill_ms, top_k_order_ms, min_p_ms,
    positive_infinity_ms, temperature_scale_ms,
    probability_weights_ms, top_p_ms, categorical_ms,
    residual_ms: f64,
    candidate_capacity_bytes_total,
    candidate_capacity_bytes_peak,
    probability_capacity_bytes_total,
    probability_capacity_bytes_peak: u64,
    after_top_k, after_min_p, after_positive_infinity,
    after_top_p, candidate_index: {
      total, min, max: u64
    }
  },
  transitions: {
    calls, new_timer_spans, logits_bytes_per_call,
    logits_bytes_total: u64,
    outer_wall_ms, inner_wall_ms, cpu_encode_ms,
    completion_wait_ms, gpu_ms_nested,
    logits_alloc_zero_ms, logits_copy_ms,
    inner_residual_ms, outer_wrapper_advance_ms: f64
  },
  bounds: {
    observer_upper_ms,
    workspace_raw_ms, workspace_adjusted_ms,
    workspace_adjusted_fraction,
    borrowed_raw_ms, borrowed_adjusted_ms,
    borrowed_adjusted_fraction,
    combined_raw_ms, combined_adjusted_ms,
    combined_adjusted_fraction,
    structural_raw_ms, structural_adjusted_ms,
    structural_adjusted_fraction: f64
  }
}
```

`CountSummary.min` uses zero only when its call count is zero; this packet
requires nonzero counts. `probability_weights_ms` and its capacity fields remain
attribution only because the phase includes mandatory exponentiation.

## Correctness Before Acquisition

CPU tests compare ordinary and profiled sampling from cloned sampler states for
all existing golden and edge fixtures. Require identical success or error,
token, candidate index, draw count, and the next sample from each resulting RNG
state. Cover empty input, every NaN position, all negative infinity, positive
infinity support, signed zero, finite ties, top-k boundaries, min-p, top-p,
singleton support, and seeds `0`, `1`, `42`, and `u64::MAX`.

Generation-loop tests cover token limit, immediate EOS, later EOS, callback
failure, transition failure, and one-token output. Require identical tokens,
callback order, stop reason, transition count, and draw count.

One model-backed A3B conformance test starts ordinary and profiled transitions
from equivalent state. Require bit-identical full logits, equal logical position,
byte-identical used KV/GDN/conv state, and bit-identical continuation logits.
It emits no phase timing result into the packet decision.

Attribution invariants require:

```text
generated_tokens = selections = sampler_draws = callbacks = 128
transitions = profiled_transitions = 127
stop_reason = token_limit
decode_policy = sampled_cpu
all non-residual durations are finite and nonnegative
inner residual and outer wrapper/advance residual are finite signed values
encode + completion_wait + logits_alloc + logits_copy + inner_residual
  = inner profiled transition wall within 0.001 ms
inner profiled transition wall + outer wrapper/advance residual
  = outer generation transition wall within 0.001 ms
all sampler phase totals + residual = profiled sampler wall
GPU time is nested and is never added to completion_wait
sampler timer_spans = 1408
transition new_timer_spans = 254
total new_timer_spans = 1662
```

## Acquisition

Build one clean release `qwen` binary after implementation and require matching
clean build/runtime/source identity with no overrides or problems.

Run one unprofiled fresh product reference first. It creates the immutable stdout
bytes, generated-token digest, stop reason, counts, sampler telemetry, and request
schema-10 reference. It contributes no performance number.

Then run exactly six profiled fresh processes. Cool down five seconds between
children. Every profiled child must match the reference stdout bytes, token
digest, stop reason, token/transition/callback counts, sampler configuration,
and draw count. Every child must emit schema 11 and exactly one complete nested
attribution record.

Before the reference, before each profiled child, and after the final child,
require AC power, no thermal or performance warning, at least 50% parsed memory
availability, median CPU idle of at least 75% across three one-second samples,
and no competing qwen, llama, or Metal benchmark. The runner additionally
requires an explicit operator attestation that no other user GPU workload is
active. Record swap, compressor, pageout, compression, major-fault, and
process-I/O counters. Fatal occupied-state growth invalidates the packet;
cumulative counters without occupied-state growth remain advisory.

If the reference does not produce exactly 128 tokens with `token_limit`, the
frozen fixture fails and v0.657 is consumed with no authority. Identity,
correctness, schema, digest, count, or reconciliation failure in any profiled
child is `KILL`. A host/environment failure after any profiled child begins is
`INVALID`. A pre-profile environment failure stops with no candidate
observation but still consumes v0.657. No child may be dropped, trimmed,
replaced, repaired, or rerun after the first profiled result is observed.

## Frozen Reductions And Decision

For each profiled child define raw candidate-specific bounds:

```text
W = transition_logits_alloc_zero_ms
  + sampler_candidate_alloc_ms

B = transition_logits_alloc_zero_ms
  + transition_logits_copy_ms

C = B + sampler_candidate_alloc_ms

S = C
  + sampler_candidate_fill_ms
  + sampler_top_k_order_ms

X_adjusted = max(0, X - observer_upper_ms), X in {W,B,C,S}
X_fraction = X_adjusted / generation_ms
```

`W` is the workspace-only upper bound. `B` is the borrowed-logit-only upper
bound: borrowing the synchronized Shared row removes both the host destination
allocation and copy. `C` requires one later candidate to implement both. `S` is
a still more optimistic borrowed-logit plus streaming/direct-top-k bound: it
treats complete candidate fill and current ordering as removable even though an
exact replacement must still inspect every logit and determine the same order.
Probability-weight construction receives no credit because its timer includes
mandatory exponentiation.

Use medians across the six children. Preserve all raw values and report ranges.
The lane is killed if median adjusted `S < 5 ms`, median `S_fraction < 5%`, or
fewer than five of six children clear both thresholds. Passing only `S` may
authorize one borrowed-logit plus streaming/direct-top-k design packet.

A workspace-only packet requires adjusted `W` to clear both thresholds in the
median and in five of six children. A borrowed-logit-only packet applies the
same rule to `B`. If neither clears independently but `C` does, only a combined
workspace-plus-borrowed implementation may proceed. A large phase may never
authorize a different candidate-specific mechanism.

Passing an upper bound does not establish a realizable speedup. Any later
candidate must retain exact sampler-v1 outputs and independently clear the
normal request-level gate. Failure here closes local workspace, readback-copy,
and full-candidate-vector work until a changed sampling contract or measured
implementation premise appears.

## Implementation Order

1. Commit this preregistration before instrumentation exists.
2. Add profiled sampler and Metal-readback APIs plus CPU/model conformance tests.
3. Add opt-in CLI wiring and schema-11 serialization; keep ordinary schemas
   byte-for-byte unchanged.
4. Adversarially review the clean runner and implementation before GPU work.
5. Run the one reference and six profiled children exactly once.
6. Reduce mechanically, record the result, and remove or retain instrumentation
   according to its maintenance value. Build no optimization before the result.
