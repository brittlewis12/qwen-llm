# v0.656 A3B Integrated Grammar-Row Request

Status: preregistration. No `response-shape-runtime/v1` artifact, integrated
executor tail, benchmark command, GPU conformance observation, timing result,
or decision exists. The source-aligned prompt fixture is frozen with this
preregistration. v0.655 is the sole performance authority for proceeding.

## Question And Authority

Does the v0.655 exact grammar-row organization reduce one complete loaded-model
A3B request by at least 5 ms after charging all request-local setup, exact
grammar state tracking, singleton handling, sampling, callbacks, and teardown?

A GO may authorize only a later product-design decision for:

- the exact A3B profile, response-shape language, tokenizer policy, prompt,
  sampler, and executor organization frozen below;
- exact selected logits over each state's admissible token IDs;
- one state-major branch bank and one state-major singleton bank built from the
  authenticated `output.weight`; and
- loaded-model, source-aligned, single-request execution.

It grants no general grammar API, dense integration, second prompt, fresh-
process first-byte result, default-on policy, full-logit equivalence, arbitrary
schema transfer, or empirical output-distribution claim. The command remains a
dedicated `qwen-bench` experiment. No product CLI flag is allowed under this
preregistration.

## Frozen Inputs

Primary model:

- path `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`;
- file size `22,134,528,992`;
- model SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`;
- MoE, 40 layers, hidden 2,048, vocabulary 248,320; and
- untied Q6_K `output.weight` `[2048,248320]`, `417,177,600` bytes.

The complete `output.weight` tensor-byte SHA-256 is
`122386a599833ffec3a266e48dd8a90aedc65e24c31323ce2388f40d7cc7b30c`.
The prerequisite v0.655 A3B result JSON SHA-256 is
`1e37f4882be856fc8043a98a00bfeaa3a7ff8a726ab0f73a99a9e98d707c689c`;
the v0.655 A3B topology report SHA-256 is
`98c5c51296f4580b3dd6d7d4122c3d6dfd175ae36a90801f728fbc514b57c4fe`.
Before implementation or acquisition, the runner must validate the result's
A3B profile, GO disposition, `one_later_a3b_integrated_packet` authority,
manifest binding, empty `qwen_env`, and clean matching build identity.

That prerequisite result was acquired from commit
`c0b8289630b04390b7688174faac79dffb47f4ab` and source identity
`git-source-sha256-v2:4499acefdf7a548048b2dfb58b3e0f364ce52eaec9137c488a39af5af7dcf334`,
with build/runtime status `match`, both clean, and no overrides or problems.

Grammar inputs:

- `docs/bench/v0654-grammar-response-shape.json`, SHA-256
  `f3a053733738aa8e05c10c2f078ca5a36c26a3bc5746487903463cdcf31b2634`;
- `docs/bench/v0654-grammar-response-shape-traces.json`, SHA-256
  `eb702b6b93fe58255cc16783363150f886dbfd0d6f7c032c1798bae1c41b74b1`;
- `docs/bench/v0655-grammar-branch-manifest.json`, SHA-256
  `2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0`;
- grammar-piece-policy SHA-256
  `58748dfe811c1710c732239ae59053f2888bf32c1c3ee4f8ec10f0ac089fdd60`;
- finite-language SHA-256
  `94dc537ecb3d9209cfbc8e28b502916b20ee7b685393b6cf85ab610c08be0faf`;
- state SHA-256
  `afa64a9a91677309fffbc54833b73d8881d7c4745127ae5ec10eebc87d70be40`;
  and
- canonical-path SHA-256
  `b72213294b55231dd4eaa12a05276be7dfe4e5ce994ff7a9451c258876333401`.

The source-aligned request is
`docs/bench/v0656-response-shape-prompt.json`, complete-file SHA-256
`c668685d855eea9270ab18a5f64a479b25c5efe85e17dc51391b92dfe98a59d9`.
It binds source item `0.5-rb-005`, branch `Q-self-pred`, exact messages,
renderer settings, 684 rendered bytes, and 156 A3B token IDs. The rendered
SHA-256 is
`59f6494df194686aec3493a496ebd970caee488d13a0e706626fe63c9351d945`;
the domain-separated token-ID SHA-256 is
`6e8203c02198324845eb104f5805adcb7eb0de27bcfb597dfcc2b2708a7f6814`.

The token-ID digest is exactly SHA-256 over ASCII
`qwen-token-ids-i32le/v1\0` followed immediately by every token ID as signed
i32 little-endian bytes, with no count prefix. Fixed-width records make the
count unambiguous; the fixture separately requires exactly 156 IDs. The
`qwen_tok_binary_sha256` field is provenance-only: it hashes the complete
release `qwen-tok` executable used for the frozen tokenization. Correctness
depends on reproduced token IDs plus model and grammar-piece-policy identity,
not continued availability of that binary.

Every arm must parse the prompt fixture, render the two messages with
`preserve_thinking=false` and an appended generation prompt, tokenize with
`add_special=false`, and reproduce both hashes and all 156 token IDs. Using the
stored IDs as a pretokenized request is forbidden.

Absolute paths in the fixture are provenance only. The committed messages,
render settings, token IDs, and hashes are the durable request definition.

## Runtime Artifact Before Executor Work

The first implementation stage may only add optional deterministic
`response-shape-runtime/v1` emission to `qwen-grammar-oracle`. Ordinary oracle
output and v0.655 branch-manifest output must remain byte-identical when the new
option is absent. When all outputs are requested together, the old two outputs
must still match their committed complete-file hashes.

The runtime artifact must contain one compiled finite-state table, not a parser
or general grammar framework. Freeze:

- 616 reachable productive states in raw-prefix lexicographic order;
- state kinds: 451 branch, 129 singleton, and 36 terminal;
- root state index and all terminal state indices;
- 1,597 edges, sorted by ascending token ID within each nonterminal state;
- every edge's token ID and exact successor state index;
- the exact token piece bytes or their checked prefix delta needed to validate
  that successor;
- v0.655 branch-bank indices for branch states;
- separate contiguous singleton-bank indices for singleton states;
- no bank index and no outgoing edge for terminal states;
- vocabulary size 248,320 and declared stop-token set `[248046]`;
- all 36 canonical paths and exact root-to-terminal minimum and maximum token
  counts;
- the branch-manifest complete-file SHA-256 and all fingerprints above; and
- a domain-separated semantic runtime-table SHA-256.

The semantic digest begins with ASCII
`response-shape-runtime/v1\0`. Its binary stream uses only these encodings:

- unsigned values are u32 or u64 little-endian exactly as named below;
- token IDs are signed i32 little-endian;
- a byte string is `u64_le(length) || bytes`;
- a vector is `u64_le(count)` followed by its records;
- a SHA-256 is its 32 raw bytes decoded from lowercase hexadecimal; and
- optional bank indices are one u8 tag: `0` for none or
  `1 || u32_le(index)` for some.

After the domain, append this exact sequence:

1. `u32_le(vocab_size)` and `u32_le(root_state_index)`.
2. Stop IDs as one vector, strictly ascending.
3. Seven named fingerprints in this order: `grammar_file_sha256`,
   `trace_file_sha256`, `branch_manifest_file_sha256`,
   `grammar_piece_policy_sha256`, `finite_language_sha256`, `state_sha256`,
   and `canonical_path_sha256`. Each record is that ASCII name as a byte string
   followed by the raw 32-byte digest.
4. States as one vector in state-index order. Each record is
   `u32 index`, `u8 kind`, prefix byte string, optional bank index,
   `u32 edge_start`, and `u32 edge_count`. Kind tags are branch `1`, singleton
   `2`, and terminal `3`. Branch and singleton indices inhabit separate
   contiguous bank namespaces implied by kind; terminal uses none.
5. Edges as one vector in source-state order and then ascending token ID. Each
   record is `u32 source`, `i32 token`, piece byte string, and `u32 successor`.
6. Terminal state indices as one strictly ascending u32 vector.
7. Canonical paths as one vector in path-index order. Each record is
   `u32 path_index`, target byte string, raw target SHA-256, token-ID vector,
   complete u32 state-sequence vector including root and terminal, and
   `u32 terminal_state`.
8. `u32 minimum_root_to_terminal_tokens` and
   `u32 maximum_root_to_terminal_tokens`.

There are no alignment bytes or implementation-selected fields in this stream.
The emitter and parser independently recompute it. The parser verifies every
canonical token step against the edge table, reproduces the target bytes, and
requires the final state to equal the recorded terminal.

The artifact is generated CPU-only, reviewed, and committed with its complete-
file SHA-256 before any executor-tail implementation. Its exact counts, index
continuity, edge closure, prefix transitions, terminal reachability, and path
bounds become hard implementation assertions. Runtime execution uses the
successor table; it does not rescan the grammar or retokenize generated text.

Runtime-table construction itself occurs only when the new output option is
present. The no-option path must not execute or fail any new v0.656 assertion.
When topology, branch manifest, and runtime outputs are requested together, the
first two complete files must still equal respectively
`98c5c51296f4580b3dd6d7d4122c3d6dfd175ae36a90801f728fbc514b57c4fe`
and
`2a349e612d9cbec271b25c2afdc82f28d79b29015f755b5efc4a98af7f3846d0`.

## Physical Candidate Banks

The Q6_K row payload is `(2048 / 256) * 210 = 1,680` bytes. Every state span is
aligned independently to 32 bytes. Rows remain duplicated by state incidence;
deduplication is forbidden.

Branch bank, inherited from v0.655:

- 451 states and 1,468 copied row incidences;
- payload `2,466,240` bytes;
- padding `2,720` bytes; and
- physical span `2,468,960` bytes.

Singleton bank:

- 129 states and 129 copied row incidences;
- one 1,680-byte row in each 1,696-byte state span;
- payload `216,720` bytes;
- padding `2,064` bytes; and
- physical span `218,784` bytes.

The combined candidate setup therefore copies 1,597 row incidences with
`2,682,960` payload bytes and `2,687,744` physical bytes. Internal and trailing
padding use inherited byte poison `0xa5`. Candidate setup validates the exact
model profile, Q6_K geometry, tensor bounds, row IDs, source-row bytes, offsets,
complete bank hashes, read-only provenance, and all per-state views before
model execution.

The candidate rebuilds and uploads both banks for every request, including
every scored B arm. No bank, parsed candidate manifest, state view, or compact
output survives into the next arm. Setup begins before reading the v0.655
branch manifest and ends only when all candidate views and guarded outputs are
usable. It includes manifest read, SHA-256, parse, validation, row copies,
padding, hashes, Metal allocation, upload, and view binding.

Compact output inherits the minimal v0.655 organization exactly. Allocate one
request-local F32 backing of `64 + 17 + 64 = 145` elements, or 580 bytes. The
first and last 64 elements are fixed guards; the middle 17 elements are the
maximum-width payload. Bind one aliased output view for each of the 580
nonterminal states over the same payload prefix at that state's exact width.
The guard value is F32 `-1234.5`; correctness-only NaN poison has bits
`0x7fc01234`. Do not allocate per-state output storage. Poisoning never enters
warmup or scored timing.

## One Production Body, Two Tails

Do not copy the A3B model body. Refactor one existing production A3B body so its
final normalized hidden state can select exactly one tail:

- full resident `output.weight` into the existing vocabulary logits buffer; or
- a supplied read-only Q6_K state view into a compact guarded output.

Both tails execute in the same command buffer as the body. Existing production
wrappers must continue through the full tail with unchanged behavior. The same
tail choice must be added to the final-row packed-prefill path so B never runs a
full root head. No Metal kernel or model-body arithmetic change is authorized.

The benchmark's full packed-prefill tail executes the unchanged full head into
`session.logits`, waits for and checks that command, then reads only the current
state's ascending selected F32 positions. It never materializes a host
full-vocabulary `Vec<f32>`. Existing ordinary full-logit prefill wrappers remain
unchanged. Later full tails follow the same sparse selected-position readback;
B reads only its contiguous compact payload. Both feed the same ascending
original token IDs and selected F32 logits into sampler version 1.

## Grammar And Terminal Semantics

Grammar state follows emitted token bytes. At each nonterminal state:

1. Run exactly one full or compact head for that state.
2. Sample once using `SamplingConfig::qwen_chat(0)`.
3. Map the local result back to the original token ID.
4. Validate the selected edge and advance to its successor immediately.
5. Decode the exact token piece and invoke the callback exactly once.

If the successor is terminal, generation commits with no model transition, no
head at the terminal, and no additional sampler draw. Otherwise, consume the
selected token through the shared A3B body and the successor state's tail.
Thus the final visible content token remains emitted but unconsumed, and
`transitions = generated_tokens - 1`. Every successful request must also satisfy
`nonterminal_heads = generated_tokens` and `sampler_draws = generated_tokens`.

Declared stop tokens are absent from every runtime edge. Selecting a stop token
at a nonterminal state, missing an edge, reaching a terminal with outgoing
edges, remaining nonterminal at the maximum path bound, attempting another
token beyond that bound, or producing invalid exact token bytes is a
correctness failure. Taking the maximum-th edge into a terminal is valid. No
lossy UTF-8 conversion participates in state advance. Debug validation may use
`NativeTokenizer::try_decode_piece_bytes_exact`; timed execution uses the frozen
edge table.

## Control And Candidate

Both arms start one request after the exact model is fully loaded and stable.
Construct the target tokenizer once after model load and exclude that
construction from arm timing. Rendering and `encode(rendered, false)` remain
request-local and charged. The effective packed-prefill chunk is exactly 156,
so the prompt executes as one packed chunk. Default-on concurrent shared-MoE
A3B decode remains enabled. Every declared or undeclared `QWEN_*` override is
forbidden; acquisition archives the exact environment and requires an empty
`qwen_env`.

Each arm executes this exact order:

1. Read, validate, render, and tokenize the prompt fixture.
2. Read, hash, parse, and validate the common runtime artifact.
3. Allocate fresh sequence, packed-prefill scratch, grammar state, callback
   sink, sampler, and raw result storage.
4. For B only, read and validate the branch manifest, then construct both banks,
   all weight and output views, and compact guards.
5. Prefill, generate through terminal, and commit the result with its final
   visible token pending and unconsumed.
6. Complete the frozen teardown endpoint below.

For every arm, charge:

- prompt-fixture read and validation;
- message rendering and tokenization;
- runtime-artifact read, SHA-256, parse, and complete validation;
- a fresh sequence, prefill scratch, sampler, callback sink, and state tracker;
- packed prefill of all 156 prompt tokens;
- exact constrained generation through terminal;
- sparse or compact selected-logit readback, sampling, mapping, and callbacks;
- command completion and status checking; and
- request-local teardown.

Sequence capacity is exactly
`prompt_token_count + runtime_max_root_to_terminal_tokens`, after checked
arithmetic. Both arms use the same fixed packed-prefill chunk selection and the
same production A3B body organization.

`A`, full-head control:

- parse only the common runtime artifact;
- use the resident full Q6_K head at root and every later nonterminal state;
- gather only the current state's admissible logits; and
- use the common grammar and sampler path.

`B`, compact candidate:

- after the common runtime parse, read, hash, parse, and validate the candidate-
  only v0.655 branch manifest;
- authenticate `output.weight`, build, hash, upload, and bind both banks;
- use the compact branch or singleton tail at every nonterminal state; and
- execute no full-head dispatch, including at the prefill root.

Model loading, model hashing, Metal initialization, PSO construction, and model
residency happen once before arm acquisition. Full model and full
`output.weight` tensor hashing also happen exactly once there. They are recorded
but excluded from the loaded-model request wall. B obtains source rows only
through the retained authenticated `LoadedModel::gguf()` mapping and bound
descriptor; it does not reopen the model or rehash 417 MB per request. B's
request-local authentication is limited to the frozen profile and descriptor,
copied source rows, padding, and complete candidate banks. Its setup remains
fully included.

### Frozen Teardown Endpoint

The primary request clock ends only after all submitted commands complete, raw
output token IDs and bytes are copied into the result record, and request-local
objects are explicitly dropped: sequence, scratch, prompt and runtime parses,
grammar state, sampler, callback sink, candidate manifest, banks, weight views,
output views, and compact storage. One identical arm-local autorelease pool
encloses each request and is drained after those drops. Record post-drop Metal
`currentAllocatedSize`, process physical footprint, and RSS immediately after
the clock stops. The loaded model, tokenizer, and common Metal context remain
alive.

## Correctness Before Scored GPU Work

CPU tests must exhaust all 1,597 edges and all 36 canonical paths. They must
cover deterministic ordering and digest reproduction, every state kind and
index map, malformed counts and ranges, duplicate or unsorted edges, bad
successors, unreachable or cyclic terminal structure, path bounds, stop-token
exclusion, checked Q6_K row geometry, bank padding, and fixture rendering and
tokenization hashes.

Untimed GPU bank validation must compare every one of the 451 branch views and
129 singleton views with the corresponding selected values from the full head
on the exact four v0.655 LCG hidden vectors with seeds `{1,2,3,4}`. Require
bit-identical selected logits, complete compact-output overwrite, intact output
guards and bank padding, unchanged source and bank hashes, completed command
status, and no Metal error.

CPU-injected branch and singleton cases must cover NaN at every local position,
all logits negative infinity, finite and positive-infinity ties, and the exact
local-to-original error-token remap. Real seeded streams do not substitute for
these error-path tests.

Before timing, run fresh-sequence A/B lockstep requests for seeds
`{0,1,42,u64::MAX}`. At every generated position require exact agreement on:

- selected F32 logits and ascending token IDs;
- bit-identical post-norm `session.h` at every real tail;
- sampled token, candidate index, sampler draw count, and error mapping;
- runtime state and successor edge;
- exact token bytes, callback bytes, and accumulated output;
- terminal status, generated count, transition count, and pending final token;
- logical KV positions plus KV, GDN-state, and convolution contents; and
- command status, tail type, and per-tail dispatch counts.

B must report zero full-head dispatches; both arms must report zero terminal
heads. Raw kernel and tail records must identify head weights, dimensions, and
state kind well enough to prove those facts. Full A/B model state is compared
immediately after prefill, after every consumed transition, and at commit. Hash
only each attention layer's used KV prefix and logical positions, plus every
complete recurrent-state and convolution tensor; exclude unused KV capacity,
scratch, guard, and padding bytes. These four seeded streams are structural
deterministic conformance, not an empirical distribution test.

## Acquisition And Metrics

Use one clean release binary and one authenticated loaded A3B model in one
process. Build and runtime identity must both report `status=match`, clean,
without overrides or problems. Archive the complete exact `QWEN_*` environment
and require it to be empty. Every arm receives fresh request-local state and a
fresh `SamplingConfig::qwen_chat(0)` sampler as defined above.

Run exactly five unscored warmup pairs. Warmup round zero is `AB`; later rounds
alternate `BA`, `AB`, `BA`, `AB`. Then run exactly 12 scored pairs with scored
round zero `AB` and every later round alternating, producing ABBA over each
adjacent pair. Seeds `1`, `42`, and `u64::MAX` beyond seed zero are untimed
conformance only. No artificial head-conditioning dispatch occurs before or
between integrated arms.

The primary measurement starts before prompt-fixture read and ends after
terminal commit, final callback accounting, command checks, and request-local
teardown. Record separately:

- prompt parse, render, and tokenization;
- common runtime parse and validation;
- B-only manifest and bank setup phases;
- sequence and scratch allocation;
- prefill body, tail, selected-logit readback, sampling, and first callback;
- later transition body, tails, readback, sampling, callbacks, and grammar work;
- teardown;
- total request wall;
- TTFT from request start through the first completed callback; and
- generation wall after the first callback through terminal commit.

For every arm also report output bytes, output token IDs, path states and kinds,
full/branch/singleton/terminal head counts, transitions, sampler draws, command
status, CPU wall, GPU timestamps, Metal `currentAllocatedSize`, process physical
footprint, RSS, and bank bytes. Preserve all raw pair values and order. Head or
body timing inferred inside a shared command buffer is diagnostic and grants no
isolated-head authority.

Primary reductions are:

```text
request_saving_i = A_total_request_i - B_total_request_i
ttft_delta_i     = B_ttft_i - A_ttft_i
decode_saving_i  = A_generation_i - B_generation_i
```

Use medians of the 12 paired differences. Report AB and BA strata, acquisition
halves, and leave-one-pair-out reductions as diagnostics only. Do not convert a
diagnostic into a decision gate after observation.

## Environment And Validity

Before warmup, before scored acquisition, and immediately after scored
acquisition, require AC power, no thermal or performance warning, at least 50%
parsed memory availability, and median CPU idle of at least 75% across three
one-second samples. Require no competing qwen, llama, Metal benchmark, or user
GPU workload. Record persistent system daemons; do not manipulate them merely
to improve a packet.

Record swap occupancy, compressor stored and occupied gauges, pageout,
compression, major-fault, and thermal counters before warmup and after scored
acquisition. Any positive before-warmup to post-scored delta in swap occupancy
or compressor stored/occupied gauges, a thermal/performance warning, or a
competing benchmark invalidates acquisition. A nonzero baseline alone does not.
Cumulative global counters that do not show occupied-state growth remain
advisory and cannot rescue or kill a result after observation.

If preflight fails before any scored B, stop with no scored candidate
observation. Any host or environment validity failure after a scored B yields
`INVALID` with no reacquisition under v0.656. A schema or parser failure is a
packet failure, not permission to repair and rerun after seeing B.

## Gates And Decision

GO requires all of:

- every identity, runtime, fixture, bank, bounds, exact-logit, sampler, state,
  callback, terminal, guard, command, memory, and environment check passes;
- B executes zero full-head dispatches and neither arm executes a terminal head;
- median paired total-request saving is at least `5.0 ms`;
- at least 10 of 12 paired total-request savings are strictly positive;
- median paired generation/decode saving is strictly positive; and
- median paired B TTFT regression is no greater than `+10.0 ms`.

Identity, correctness, state, command, structural memory/accounting, or full-
head-in-B failure is `KILL`. A post-start host or environment failure is
`INVALID` as defined above. Median total-request saving below zero or median
generation saving below zero is `KILL`. If either median is exactly zero, the
result is `NO-GO/PARK`. A correct positive result that misses `5.0 ms`, wins
fewer than 10 pairs, or exceeds the TTFT bound is also `NO-GO/PARK`. It cannot
be promoted by subtracting setup, excluding singletons, changing the generated
path, changing the prompt, or invoking the v0.655 projection.

One valid acquisition is the complete packet. No second prompt, dense arm,
fresh-process arm, repeat seed sweep, or broad grammar workload may be added
after observation. A GO remains scoped exactly to the authority at the top of
this document.

## Implementation Order

1. Commit this preregistration and source-aligned prompt fixture.
2. Add only optional runtime-artifact emission; generate, review, and commit the
   frozen artifact before executor work.
3. Add CPU parser, bank, grammar, and terminal tests.
4. Refactor the single shared A3B body and packed-prefill tails; preserve all
   existing full-tail wrappers.
5. Add the dedicated integrated benchmark and complete untimed conformance.
6. Commit the clean implementation before any scored GPU acquisition.
7. Run the one frozen packet and reduce it mechanically.

Independent design review continues in `cx` session
`019fb694-8a53-7482-b65b-7592f729be32`.
