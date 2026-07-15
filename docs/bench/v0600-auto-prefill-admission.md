# v0.600 Query-Capped Auto-Prefill Admission

Status: preregistration. No v0.600 implementation or timing result exists yet.

## Intent

Finish the exact 8K-16K A3B/A10B auto-prefill path already measured in v0.567
and memory-bounded in v0.568. The product candidate changes the outer prefill
chunk only when its complete scratch topology fits a conservative request-local
memory contract. Every unsupported, overridden, cache-bearing, or unadmitted case
uses the existing outer-1024 baseline.

Expected cap-adjusted fresh-TTFT gains are about `1.060x` A3B and `1.206x` A10B.
v0.572 closes 32K, wider chunks, and range expansion.

## Frozen Profiles

- A3B: Qwen3.6 35B A3B, file type 15, no MTP, outer 2048.
- A10B: Qwen3.5 122B A10B, file type 15, no MTP, outer 4096.
- Both require the complete architecture allowlist already implemented by v0.567.
- Prompt range is inclusive `8192..=16384` tokens.
- Candidate matrix query cap is 1024 rows.
- Candidate matrix max position is `max(prompt_tokens, outer_chunk)`.
- Let `M=matrix_max_pos`, `Q=1024`, and `H` be 16 A3B or 32 A10B query
  heads. Candidate attention-overlay bytes are the checked, 256-byte-aligned
  result of `2*Q*H*M + 8*Q*H*ceil(M/64)`. GDN bytes are constant
  `235,405,312` A3B or `807,403,520` A10B. Backing is their maximum; saved
  bytes are their minimum.
- At the 11,287-token confirmation prompt, backing/attention/GDN/saved bytes
  must be `393,052,160/393,052,160/235,405,312/235,405,312` A3B and
  `807,403,520/786,104,320/807,403,520/786,104,320` A10B.

Any `QWEN_PREFILL_*` environment variable makes the automatic candidate
ineligible. The legacy baseline runs with that environment unchanged.

## Branches And Ordering

Use one shared allocator for single-turn and JSONL, backed by pure classification,
plan-validation, and memory-adjudication helpers.

- **Numeric chunk**: preserve the current scratch-then-sequence constructor order,
  environment behavior, schema, and absence of admission telemetry. Its legacy
  matrix max position is `max(prompt_tokens, effective_outer)`.
- **Ineligible auto**: select `min(1024, max(prompt_tokens, 1))`, allocate through
  `fresh_prefill_with_matrix_max_pos` with matrix max position
  `max(prompt_tokens, effective_outer)`, then create the sequence. This includes
  unsupported profiles, prompts outside the frozen range, and environment
  overrides.
- **Eligible auto**: create the sequence first. Record the exact Metal allocation
  delta. Build, validate, price, and adjudicate the candidate plan. If admitted,
  allocate only through `fresh_prefill_from_plan`.
- **Rejected auto**: after sequence creation, allocate the untouched legacy
  outer-1024 constructor with matrix max position
  `max(prompt_tokens, effective_outer)`. Planning, pricing, topology, and
  admission rejection are fail-closed reasons. A candidate allocation failure
  after admission is a hard request error; never attempt a second baseline
  allocation.

The legacy baseline passes no explicit scratch config and does not mutate or
reinterpret process environment.

## JSONL Boundary

Automatic admission is allowed only when the prefix cache is empty at request
start and the request will not create a request-, CLI-, or discovered-auto cache
snapshot. Otherwise use outer 1024 with
`prefix_cache_interaction_unvalidated`.

This avoids claiming a whole-request memory bound for CPU snapshot arenas. It does
not change explicit cache behavior or authorize cache-bearing wide prefill.

## Candidate Plan

Build with `PrefillScratchConfig { matrix_query_cap: Some(1024) }`. Require:

- exact candidate block size, matrix max position, and query rows;
- configured query-cap success plus a nonempty overlay as observable proof of the
  default online matrix path;
- exact checked overlay equations above for the selected profile and prompt;
- no `QWEN_PREFILL_*` control;
- every eager and deferred allocation represented in the plan;
- every priced shared-buffer size nonzero, at least its logical size, and returned
  with nonzero power-of-two placement alignment. Alignment constrains a heap
  offset, not divisibility of the returned resource size;
- `fresh_prefill_from_plan` to revalidate plan/model identity before allocation.

Record eager/deferred allocation counts, logical totals, maximum logical bytes,
priced total, plan geometry, and overlay geometry. No sampled v0.567 delta
participates in admission.

## Memory Contract

On this macOS desktop, both Mach `limit_bytes_remaining` and
`os_proc_available_memory()` report zero because no finite per-process jetsam
budget is exposed. v0.600 must not label that absent limit as a second valid
headroom measurement.

After eligible-auto sequence creation, define:

```text
S = priced eager scratch + priced deferred scratch
delta_seq = currentAllocatedAfterSequence - currentAllocatedBeforeSequence
transient = 512 MiB
reserve = delta_seq + transient
required = S + reserve
```

All arithmetic is checked. Require `currentAfter > currentBefore`; zero or a
decrease is an invalid accounting signal and falls back. The
sequence delta is deliberately double-charged: it is already present in current
Metal allocation and is added again to reserve as a conservative physical-touch
allowance. The fixed 512 MiB is only a transient/TOCTOU cushion; it is not a
sequence, snapshot, or whole-request bound.

Call `evaluate_metal_memory_admission` immediately before candidate allocation.
Require valid positive Metal recommended working set, positive computed Metal
headroom, and `required <= headroom`. A positive finite process limit, if ever
reported, must also fit. For this auto policy, `Some(0)` always means no finite
process limit is exposed and is accepted only as the explicit
`admitted_process_budget_omitted` reason. Authority remains limited to the device
boundary below. `None`, arithmetic overflow, invalid working-set signals, or any
finite insufficiency fall back to 1024.

## Telemetry And Tests

Auto rows become schema 5 and retain existing chunk/query/overlay fields. Add a
nested record containing classification, candidate and selected widths, baseline
identifier, stable fallback reason, plan geometry/counts/logical/priced bytes,
sequence delta, transient and total reserve, required bytes, raw memory signals,
computed headroom, evaluator reason, and admission result. Keep observed Metal
allocation samples in single-turn timing rows; JSONL has none. Pre-planning
fallback rows include classification/selection/baseline reason and omit the nested
plan/admission record. Numeric rows retain their current schema.

The nested plan record includes one row per eager and deferred allocation with
name, logical bytes, priced bytes, and returned alignment. Record
`currentAllocatedBeforeSequence` and `currentAllocatedAfterSequence` separately;
do not serialize only their delta. These raw operands make every admission term
independently recomputable.

Tests must cover:

- both exact profiles, MTP/topology rejection, and all prompt boundaries;
- numeric bypass and current allocation order;
- eligible admitted and rejected ordering;
- no fallback after candidate allocation starts;
- exact overlay/profile validation and plan/allocation identity;
- eager/deferred pricing, overflow, and invalid size/alignment;
- sequence-delta and reserve overflow;
- exact memory boundaries, positive finite limits, zero omitted limit, and `None`;
- environment override fallback without mutation;
- cache-free versus cache-bearing JSONL;
- stable reason taxonomy and schema compatibility.

## Confirmation Packet

Run one clean fresh-process packet after implementation:

- source/build/runtime identity must be clean and equal; hash the executable,
  runner, contract, prompt, all model shards, and imported helpers;
- A3B model SHA-256:
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`;
- A10B shard SHA-256 values in order:
  `467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3`,
  `ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98`,
  `13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97`;
- tracked 11,287-token Mei strip prompt, SHA-256
  `ca924d7a3613ef8a6aa02fdcbfc56f2a7bffb5ec7aa2ddbd11e7efa3be7ad3f6`,
  one output token, redirected stdout;
- numeric 1024 A versus admitted auto B;
- A uses timing schema 3; B uses schema 5. Schema-5 topology/admission requirements
  apply only to B;
- exact common argv is `qwen -m MODEL --prompt-file PROMPT --tokens 1
  --no-special-tokens --prefix-cache-max-mib 16384 --request-timings ROW`, followed
  by `--prefill-chunk 1024` for A or `--prefill-chunk auto` for B. Do not pass
  `--max-context-tokens`, cache-prefix flags, prompt lookup, requests JSONL, or
  warm follow-up. Effective sequence capacity must be 11,304;
- four pairs per profile in order `AB/BA/BA/AB`;
- exact cooldown 30 seconds A3B and 120 seconds A10B before every child;
- construct the child environment by deleting every key whose name starts
  `QWEN_`, `METAL_`, or `MTL_`, plus exact key `RUST_LOG`; preserve all other
  key/value pairs. Hash the complete result and retain the removed-key list and
  surviving control-key map, which must be empty;
- exact output bytes, zero decode transitions, no fallback, and exact schema-5
  admission/topology telemetry;
- A3B calls `60/120` tiled-layer/query-tile with outer/query `2048/1024`;
- A10B calls `36/144` with `4096/1024`;
- bracket cache conditioning and children with AC/thermal/memory and VM state;
  `/usr/bin/time -l` supplies block input and process resources. Zero block input,
  zero pageout/swap growth, and valid host/memory signals are required.

Before each child, read every model shard completely with an 8 MiB buffer, then
wait the exact cooldown. Sample host state before cache read, before spawn, and
after child exit using `pmset -g therm`, `pmset -g batt`, and
`memory_pressure -Q`. Valid means no recorded thermal or performance warning, AC
power, and parsed system memory free percentage `>=50`. Retry that host sampler up
to six times with 30 seconds between attempts. Sample `vm_stat` Pageouts and
`sysctl -n vm.swapusage` immediately before cache read, before spawn, and after
exit. Cache-interval growth stops before launch; child-interval growth invalidates
the complete pair. Any `/usr/bin/time -l` block input also invalidates the pair.

For pair `i`, define `s_i=A_i.ttft_ms/B_i.ttft_ms`. Sort values and define median
as the arithmetic midpoint of the two middle values for an even-size set. Overall
speedup is the four-value median; each order-stratum speedup is the two-value
median; a win means `B_i.ttft_ms < A_i.ttft_ms`. The runner must recompute every
per-allocation and aggregate scratch price,
`delta_seq`, transient, reserve, required, working-set headroom, process-limit
handling, and evaluator reason from raw schema-5 fields.

Performance gates:

- A3B median TTFT speedup `>=1.04x`, both order strata `>=1.03x`, 4/4 wins.
- A10B median `>=1.15x`, both strata `>=1.10x`, 4/4 wins.

Any signal, topology, admission, allocation, output, or validity failure stops that
profile. No reserve tuning, 32K, extra width, extra pair, sibling model, or
environment rescue follows.

## Authority

A passing profile authorizes only a separate profile-specific default decision on
Apple M4 Max, 128 GiB unified memory, macOS 15.x. That later decision must define
and justify any broader device/OS boundary. The packet does not itself change the
default. It grants no authority over cache-bearing JSONL, sibling assets, 32K,
server concurrency, numeric overrides, or the parallel-copied loader pilot.
