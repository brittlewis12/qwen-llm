# v0.660 A3B Sampled Structural Product Packet

Status: preregistration. No v0.660 implementation, runner, build, correctness
result, product child, or timing observation exists.

## Intent And Authority

Implement and charge the sole bounded successor authorized by v0.659: exact
bounded top-k candidate preparation combined with scoped consumption of
synchronized Shared logits for decode transitions.

The sealed authority record is:

```text
packet
target/profiles/v0659-positive-temperature-sampling-parser-repair-p1

decision.json SHA-256
ae21f8944eae07d950127961f29b6e2814efbc6970af60994fee175655af8534

disposition=GO_BOUNDED_IMPLEMENTATION
authority=["structural"]
max_authorized_packets=1
```

Freeze the tracked v0.659 inputs:

```text
preregistration SHA-256
89a15f05ff7cc87b8fe22270522350e2c883fd86f45b6f1db5d01f17b9730af1

runner SHA-256
adde8b1ca275f9fa66795959bfe9f5eef72b292d07fa3586227dfd10456e57d5
```

The runner first requires the new root absent, creates and fsyncs it, and
durably writes a reserved attempt record. It then authenticates the tracked
files and canonical sealed decision fail-closed. It may read no other v0.659
artifact and may import no child row, timing, conformance result, output digest,
or VM observation. Any existing root or partial attempt refuses reacquisition.

This packet consumes the sole structural successor authorization. A GO may
retain the hidden force path and support a policy-only review for the complete
frozen fixture: exact A3B asset, prompt/token identity, sampling, 128-output
request, context, chunk, and cache settings. That review may collect no new
performance evidence. This packet cannot authorize broader default admission,
GPU sampling, prompt-logit borrowing, a changed sampler distribution, another
model/configuration, or a second performance packet.

Freeze the new packet identities:

```text
runner
scripts/profile/v0660_a3b_sampled_structural_product.py

packet root
target/profiles/v0660-a3b-sampled-structural-product-p1

success schema
qwen-v0660-a3b-sampled-structural-product/v1

failure schema
qwen-v0660-a3b-sampled-structural-product-failure/v1
```

## Candidate Boundary

Add a hidden, default-off `--sampled-structural` product switch. Its force path
is valid only for a single-turn concurrent-MoE request with positive
temperature and `0 < top_k < vocab`. It conflicts with prompt lookup, sampling
attribution, JSONL, RAM/durable prefix caching, warm follow-up, and any batched
or multi-request mode. CLI-known incompatibilities fail before model load;
architecture, vocabulary, and decode-path incompatibilities fail after model
validation but before prompt prefill. Absence preserves the current product
path byte-for-byte, including full-vector behavior for `top_k=0`,
`top_k=vocab`, and `top_k>vocab`.

The force implementation may accept other request lengths and bounded top-k
values under those eligibility rules, and schema-12 counts are dynamic. v0.660
performance and any later policy review remain restricted to the complete
frozen fixture below. Unsupported flagged requests never fall back silently and
receive no B-path credit.

### Exact Bounded Top-K

Keep `Sampler::sample` unchanged as arm A. Add one arm-B method that:

1. runs the existing shape checks;
2. scans every logit once in ascending token-ID order;
3. returns the same first-NaN token error before any RNG draw;
4. converts every considered `f32` to `f64` exactly as sampler-v1 does;
5. retains exactly `top_k` best candidates in a bounded heap whose worst
   retained candidate is lowest logit, then highest token ID;
6. sorts the retained candidates by descending logit, then ascending token ID;
7. executes the existing min-p, positive-infinity, temperature, probability,
   top-p, categorical, fallback-last, and RNG statements unchanged.

Use `total_cmp` semantics. Heap equality must use logit bits plus token ID so
signed zero satisfies `Eq`/`Ord`. Ties, infinities, all-negative infinity,
candidate index, errors, successful draw count, and subsequent RNG state must
match sampler-v1 exactly. `+0.0` ranks above `-0.0`, as in the current descending
`total_cmp` order. Freeze the heap operation:

```text
better(a, b) :=
  a.logit.total_cmp(b.logit) == Greater
  or (a.logit.total_cmp(b.logit) == Equal and a.token < b.token)

worse(a, b) := better(b, a)
heap Ord: worse candidates compare Greater, so root is the unique worst

for candidate in ascending token order:
  if heap.len < k: push(candidate)
  else if better(candidate, heap.root): pop root; push(candidate)
```

Allocate with `BinaryHeap::with_capacity(k)`. Never push while full, so logical
heap length never exceeds `k`; record actual allocator capacity separately and
require it smaller than vocabulary size. Candidate equality is exact token plus
`f64::to_bits()`. Convert the heap to a length-k vector and apply the existing
descending-total-logit/ascending-token sort before all downstream statements.

Do not bump `SAMPLER_ALGORITHM_VERSION`. The flagged path fails outside
`0 < top_k < vocab`; ordinary unflagged requests retain the current full-vector
method and receive no candidate claim.

Do not add a reusable full-vocabulary workspace. v0.659 closes that premise.

### Scoped Shared-Row Consumption

Keep the production concurrent-MoE body, resident Q6_K head, command buffer,
commit, and completion wait unchanged. Add one narrow higher-ranked scoped
consumer equivalent to:

```text
fn with_completed_logits<R, F>(..., session: &mut MetalSession, consume: F)
  -> Result<(R, TokenProfile, StructuralRowEvidence), MfError>
where
  F: for<'row> FnOnce(&'row [f32]) -> R
```

`R` cannot borrow from `'row`, and `&mut MetalSession` remains exclusively
borrowed through consumption. Only after the existing completion wait returns:

- validates F32 dtype, exact vocabulary element count, owned-writable
  provenance, Shared storage mode, checked offset/length, alignment, and buffer
  bounds;
- creates a read-only `&[f32]` for the completed row;
- invokes the arm-B sampler before the slice can escape or another command can
  overwrite `session.logits`; and
- returns an owned sample result, never a pointer, slice, or general logits
  handle.

The unsafe pointer-to-slice construction is private to `metal_forward.rs`.
Require exact shape `[vocab]`, provenance
`MetalTensorProvenance::OwnedWritable`, Shared storage, non-null host contents,
checked byte range, and `f32` alignment. No raw tensor, buffer, pointer, or slice
may escape.

Here “completed” means only that the unchanged production
`waitUntilCompleted()` returned. The current resident path does not query
command status/error; v0.660 forbids candidate-only status queries and makes no
status-validation claim. Evidence counters increment at the actual resident
head/wait and validated borrow boundaries. The candidate adds zero command
buffers and zero GPU sampling dispatches.

Prompt selection continues to use the existing owned prefill logits row, but
uses bounded top-k in arm B. Only the 127 decode transitions avoid full host
logits allocation and copy.

### Generation And Telemetry

Keep the single authoritative generation loop and its order:

```text
select -> append -> stop check -> callback -> limit check -> transition
```

Generalize only its private closure context as needed so prompt selection and
transition-time borrowed sampling share one request-local sampler. Preserve
callback failures, EOS, token-limit, N-1 transitions, pending terminal token,
sequence advancement, and first-token timing.

Use transactional sampler state and defer a post-transition sample result:

```text
trial = sampler.clone()
outer = forward.with_completed_logits(|row| trial.sample_bounded(row))
if outer is forward error: return error without logical advance or RNG commit
state = outer.owned_inner_sample_result
if sequence.advance_by(1) fails: drop trial; return error; count no transition
sampler = trial
return Selected(state) so the central loop counts the transition
on next select: expose success or sampling error
```

Thus advance failure changes neither RNG nor transition count; successful
advance commits exactly the trial state; sampling failure is surfaced from the
next selection after one logical advance and one counted transition. The device
state on advance failure remains mutated exactly as on arm A. No command,
snapshot, or other session access may intervene between row consumption and
advance. Add a dedicated advance-failure test.

When arm B and request timings are active, emit schema 12 with one
`sampled_structural` object:

```text
version: u32 = 1
algorithm_version: u32 = 1
path: string = "bounded_topk_borrowed_transitions"
prompt_owned_bounded_calls: u64
borrowed_transition_calls: u64
resident_head_wait_calls: u64
validated_shared_row_calls: u64
fallback_calls: u64
input_logits_total, input_logits_min, input_logits_max: u64
retained_top_k_total, retained_top_k_min, retained_top_k_max: u64
max_heap_len, max_heap_capacity: u64
full_candidate_vector_allocations: u64
transition_logits_copy_bytes: u64
extra_command_buffers: u64
gpu_sampling_dispatches: u64
```

For the frozen B request require exactly:

```text
prompt_owned_bounded_calls = 1
borrowed_transition_calls = resident_head_wait_calls = 127
validated_shared_row_calls = 127
fallback_calls = 0
input_logits_total = 31784960
input_logits_min = input_logits_max = 248320
retained_top_k_total = 25600
retained_top_k_min = retained_top_k_max = 200
max_heap_len = 200
200 <= max_heap_capacity < 248320
full_candidate_vector_allocations = 0
transition_logits_copy_bytes = 0
extra_command_buffers = gpu_sampling_dispatches = 0
```

Counters increment at the actual bounded-method, production-head/wait,
validated-borrow, fallback, allocation, copy, command-buffer, and dispatch
boundaries with checked arithmetic. Arm A remains exact schema 10 and has no
object. Schema 12 has exactly the schema-10 key set plus
`sampled_structural`; the object has exactly the keys above, all counts are
nonnegative integers, `sampling_attribution` is absent, `decode_policy` remains
`sampled_cpu`, and sampling reports algorithm version 1 and 128 draws. Strict
duplicate-key JSON parsing is mandatory. Schema-selection precedence rejects
the conflicting attribution flag. The object records path execution, not a
timing estimate; no attribution timer runs in either arm.

## Correctness Before Timing

CPU differential tests compare the bounded method against unchanged
`Sampler::sample` over:

- the existing golden/filter/error fixtures and repeated streams;
- SplitMix64 seeds `0`, `1`, `42`, and `u64::MAX`, lengths
  `[2,17,201,257,1024]`, and 16 finite plus 8 raw-`u32` bit-pattern vectors per
  seed/length;
- direct bounded `k` values from `{1,199,200,201,len-1}` filtered to
  `1 <= k < len`;
- every token position at lengths `[2,17,201]` for NaN payloads `0x7fc00000`,
  `0x7f800001`, and `0xffc00001`, requiring the first ascending NaN token;
- explicit equal logits, `+0.0/-0.0`, both infinities, and all-negative
  infinity; and
- token, candidate index, exact error, draw count, and the next result on one
  frozen finite row after every success or failure.

Finite vectors clear exponent `0xff` before `f32::from_bits`; raw vectors use
the low 32 SplitMix64 bits unchanged. The local generator is frozen to the
sampler-v1 SplitMix64 constants. Ordinary dispatcher tests cover `k=0`,
`k=len`, and `k>len` and must invoke unchanged `Sampler::sample`; they receive
no bounded-path credit. Small lengths simply omit direct `k` values not less
than `len`.

CLI tests compare copied-row and selected-token generation for one token,
token-limit, immediate and later EOS, callback failure, transition failure, and
sampling failure after a committed transition. Require identical tokens,
callbacks, stop reason, transitions, pending-token semantics, and sampler state.
Add a distinct sequence-advance failure after successful borrowed sampling and
require no sampler/RNG commit and no counted transition.

One release test named `metal_sampled_structural_matches_copied_a3b` authenticates
the exact A3B file, creates two fresh equal-capacity sessions and cloned
samplers, and uses an owned initial row for the prompt sample. Across 127
subsequent transitions it executes the same resident full head in both arms,
compares every copied A logit against the scoped B row bit-for-bit before
sampling, and requires equal sample token/index/draw count. At the end require
identical explicit position progression, equal `kv_n_pos`, active KV K/V bytes,
complete GDN conv/state bytes, and bit-identical continuation logits. Unused KV
capacity and padding are excluded. CLI tests own logical-sequence advancement;
range/storage evidence must report 127 validated rows.

Run and archive exactly:

```text
cargo test --locked -p qwen-llm sampling:: --lib
cargo test --locked -p qwen-cli --bin qwen
cargo test --locked --release -p qwen-llm \
  metal_sampled_structural_matches_copied_a3b -- --ignored --nocapture
```

The CPU/CLI suites must report at least 17/58 passing tests respectively after
the new cases land; exact discovered and filtered counts are archived. Every
command is one attempt with captured stdout/stderr, timeout, process-group, and
completion records.

No performance observation may run before these gates and adversarial review
are green.

## Frozen Product Fixture

Use the exact v0.659 model, prompt, tokenizer, and request:

```text
model
/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
bytes=22134528992
SHA-256=ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61

prompt
docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt
bytes=1891
tokens=419
SHA-256=e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474
little-endian signed-i32 token SHA-256=
fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f
runtime tokenizer identity=a4b0b26f8a8c9917

sampling
temperature=0.7 top_k=200 top_p=1.0 min_p=0.05 seed=42

request
outputs=128 context=1024 prefill_chunk=1024 prefix_cache=off
```

Each child is one fresh process, one model load, and one request. Use this exact
ordered argv, replacing only `TIMING` with a unique initially absent path:

```text
target/release/qwen
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt
  --tokens 128
  --temp 0.7 --top-k 200 --top-p 1.0 --min-p 0.05 --seed 42
  --prefill-chunk 1024 --max-context-tokens 1024
  --prefix-cache-max-mib 0 --cache-prefix-auto-min-tokens 0
  --request-timings TIMING
```

Arm B appends only `--sampled-structural`. Capture stdout and stderr into unique
initially absent files with exclusive creation. Require exit zero, no timeout,
no leaked process group, exactly one strict-JSON timing row, and fsync every
artifact before validation.

Children receive the same frozen non-secret allowlist as v0.659. Strip all
inherited `QWEN*`, `MTL*`, `METAL*`, and logging controls; archive parent and
child environment names plus value lengths/SHA-256 commitments, never plaintext
values. Helper failures retain output length/digest only. Process arguments are
stored by length/digest while frozen child argv and executable identities remain
plaintext. Any inherited-control rejection reports variable names only, never
values.

Model conformance must authenticate the frozen prompt token digest. Every
scored child must report 419 prompt tokens and runtime tokenizer identity
`a4b0b26f8a8c9917` before its row may enter correctness or reduction.

## Acquisition

Build one clean release `qwen` and `qwen-bench` after implementation. Require
matching clean build/runtime/source identity, exact model/prompt hashes, no
inherited `QWEN_*` controls, and the v0.659 authority checks above.

The sole acquisition command is:

```text
uv run scripts/profile/v0660_a3b_sampled_structural_product.py \
  --phase acquire --attest-no-other-user-gpu-workload
```

Run six scored fresh-process pairs in exact order:

```text
AB BA BA AB AB BA
```

After root reservation, run an early redacted competitor census before static
work. Capture VM state before CPU conformance and after scoring. Fatal growth is
any positive delta in swap occupancy, compressor stored pages, or compressor
occupied pages; pageouts, compressions, and swapouts are recorded but advisory
without occupied-state growth.

Run 14 full host gates: before model conformance, before each of 12 scored
children, and after scoring. Each requires AC power, no thermal or performance
warning, at least 50% parsed memory availability, median CPU idle at least 75%
over three one-second samples, no detected qwen, llama, or Metal benchmark, and
the explicit operator no-other-user-GPU attestation. Cool down five seconds
between children.

The runner durably journals pair index/orientation, child arm, launch intent,
PID/process group, B-exposure state, completion, exit, timeout, leak, and
artifact identities. Candidate exposure begins immediately before the first B
process spawn attempt, even if that launch or child later fails. No process is
killed by name; timeout cleanup signals only the exact recorded process group
with TERM then KILL and records every action.

All 12 children, not merely each pair, must share one identical stdout byte
string, generated-token digest, 128-token sequence, stop reason, sampling
config, 128 draws, 127 transitions, and terminal semantics. Callback ordering
is pinned by the pre-timing CLI tests. No v0.659 output digest is imported. A
must have the exact schema-10 top-level key set. B must have exactly that set
plus the schema-12 structural object above. All durations must be finite and
milestone ordering valid.

`generation_ms`, `ttft_ms`, and `total_request_ms` retain their schema-10
product boundaries exactly. The runner starts a separate monotonic
`spawn_to_exit_ms` clock immediately before process spawn and stops it after
normal child exit is observed and the child is reaped. Stdout/stderr are
redirected directly to files, so no pipe-drain phase exists. Stop the clock
before parent validation, artifact writing, flushing, or fsync. It must not
reuse a broader runner wall clock.

For each pair define:

```text
generation_fraction = (A.generation_ms - B.generation_ms) / A.generation_ms
request_saving_ms = A.total_request_ms - B.total_request_ms
ttft_delta_ms = B.ttft_ms - A.ttft_ms
spawn_saving_ms = A.spawn_to_exit_ms - B.spawn_to_exit_ms
```

Require `A.generation_ms > 0`. Compute every quantity from unrounded serialized
values; rounding is display-only. A strict win is saving `>0`. For six values,
the median is the arithmetic mean of sorted values 3 and 4. AB and BA strata
contain the three pairs with that literal launch orientation. Missing,
duplicate, nonfinite, or invalid rows never enter a median. Preserve every raw
ordered value and mechanically recompute reductions in the terminal record.

## Decision

`GO` requires every correctness, identity, schema, command, memory, and
environment gate plus all of:

- median paired `generation_fraction >= 0.05`;
- B wins generation wall in at least five of six pairs;
- median paired `request_saving_ms >= 5.0`;
- B wins model-ready request wall in at least five of six pairs;
- AB and BA medians are positive for generation and request saving;
- median paired `ttft_delta_ms <= 10.0`; and
- median paired `spawn_saving_ms >= 0.0`.

Terminal disposition is exhaustive:

- `CONSUMED_NO_AUTHORITY`: the first infrastructure or environment stop after
  root reservation but before B exposure, including readiness, A-child launch,
  timeout, leak, I/O, or capture failure;
- `INVALID`: the first post-exposure host/VM/process-capture contamination not
  attributable to candidate correctness;
- `KILL`: identity/source/build/authority, eligibility, correctness/state,
  schema/path-evidence/structural-memory, output/terminal, command-policy,
  accounting, or reduction failure at any stage, plus post-exposure
  launch/timeout/leak or child I/O/parse/serialization failure, unless a
  chronologically earlier post-exposure host/VM invalidity already sealed
  `INVALID`;
- `NO-GO/PARK`: a correct, complete, valid acquisition missing any performance
  gate; and
- `GO`: every frozen gate passes.

The first chronologically sealed terminal condition wins. Publish atomic
`failure.json` and byte-identical `decision.json` on failure, or one complete
`decision.json` on success, with inventory hashes. No child, pair, or packet is
rerun under any disposition.

Do not subtract mandatory scan/heap work, reuse v0.659 attribution as realized
saving, drop a pair, change the prompt/configuration, or rerun a child or packet.
The first terminal record is final.

## Normative Exclusions

Exclude workspace-only work, GPU sampling, prompt-logit borrowing, lm-head
changes, command-schedule changes, extra command buffers, sampler-v2 or a
version bump, cache-format changes, prompt lookup, broad runtime redesign,
other models/configurations, and another performance packet. A later
policy-only review may use only a valid v0.660 GO and the complete frozen
fixture authority above; it may collect no new timing evidence. Only `GO` may
retain the product force switch. Every other disposition grants no product
admission authority and must remove the switch or retain only unreachable,
test-only primitives.

## Implementation Order

1. Commit this preregistration before candidate code exists.
2. Implement the default-off sampler, scoped row, generation, telemetry, and
   differential tests without timing.
3. Add and adversarially review the one-shot runner.
4. Build and run all correctness gates from a clean committed worktree.
5. Execute the six frozen pairs once and reduce mechanically.
6. Retain the force path only on `GO`; otherwise remove it or make the primitives
   unreachable outside tests. Make no automatic-admission change without the
   exact-fixture policy review above.
