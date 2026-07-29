# v0.652 final independent A3B one-shot GPU-greedy admission

Status: preregistration. No v0.652 candidate implementation, runner, or
observation exists. This is the one independent successor permitted after the
consumed v0.651 packet.

## Question and authority

May exact GPU greedy become automatic when `QWEN_GREEDY_GPU_ARGMAX` is absent
for the exact disposable A3B single-turn CLI profile?

A GO may authorize only:

- realized profile
  `DisposableA3bQ4kmV1PreadLogicalExactRetainedPlanV1`;
- ordinary fresh single-turn request index zero;
- temperature-zero generation without prompt lookup requesting `128..=512`
  output tokens;
- no reusable model path, JSONL, warm follow-up, durable checkpoint store,
  active RAM-prefix-cache restore/admission, concurrency, speculation, or
  sampling; and
- explicit `QWEN_GREEDY_GPU_ARGMAX=0` rollback.

The measured performance claim is limited to the frozen M4 Max host below. The
selector scope is the exact realized structural profile, not the named fixture's
content bytes: a structurally matching load that produces this same marker is in
policy scope, while every other marker/profile remains off. The marker is not
model-content or tokenizer authentication. A GO grants no other load profile,
provider, or serving authority. Explicit `=1` remains independently supported
for otherwise eligible exact-greedy requests.

Any non-GO, including `INVALID`, ends this automatic-admission branch and
requires a committed restoration of absent-environment default-off before any
further product or GPU experiment use. The temporary candidate commit grants no
authority.

## Independence from v0.651

v0.652 is not a repair, retry, continuation, or completion of v0.651:

- use a new preregistration, temporary candidate commit, runner-only commit, and
  artifact root;
- never read the v0.651 artifact root;
- generate every response reference and digest inside v0.652;
- use no v0.648-v0.651 row, output, estimate, variance, gate, or result in any
  v0.652 calculation or decision; and
- treat the fixed `n=8` and `n=4` counts as a new finite decision budget, with no
  prospective-power claim.

Reintroducing the same narrowly audited candidate semantics is allowed only
after this preregistration. Reserving the v0.652 artifact root consumes the sole
successor even if acquisition stops before a scored child.

## Candidate contract

Propagate the same versioned marker only after successful
`PreparedAutoSelection::Selected` loading with:

- profile `A3bQ4kmV1`;
- population `Pread`;
- destination `LogicalExact`; and
- proof `A3bRetainedPlan`.

Derive the candidate marker before the prepared selection is consumed, but
attach it to `LoadedModel` only after `MetalModel::load_prepared` succeeds.
Forced physically similar loads, `NoMatch`, `NotEligible`, future variants, and
failed realization expose no marker.

The bounded candidate commit may modify only:

- `crates/qwen-cli/src/main.rs`;
- `crates/qwen-llm/src/metal_forward.rs`; and
- `crates/qwen-llm/src/runtime.rs`.

It may reintroduce only the marker propagation, absent-environment selector,
telemetry, and pure tests required by this contract.

Absent-environment automatic selection requires every authorized condition
above plus exact marker equality. Decision precedence is:

1. falsy, invalid, or non-Unicode environment value: explicit rollback;
2. sampling or prompt lookup: ineligible;
3. truthy value: explicit force;
4. absent value plus every narrow Auto condition: enabled; and
5. absent value otherwise: scoped default-off.

An inactive RAM cache budget may exist, but any populated/restored RAM-prefix
state or selected insertion is out of scope and fails closed before Auto
selection. Freeze `default_off_ram_prefix_cache_active` and cover it in the pure
scope matrix.

JSONL receives explicit reusable scope and remains default-off when the variable
is absent. The candidate must not repeat the
`metadata_compatibility_v1` tokenizer/metadata walk for policy, add identity I/O,
or change timing schema solely for this packet. Freeze these reason strings:

- `disabled_by_explicit_rollback`;
- `auto_disposable_a3b_q4km_v1`;
- `default_off_requested_tokens_outside_128_512`;
- `default_off_reusable_path`;
- `default_off_no_disposable_profile`;
- `default_off_warm_followup_requested`;
- `default_off_durable_store_configured`;
- `default_off_request_index_not_zero`;
- `default_off_ram_prefix_cache_active`;
- `ineligible_request`; and
- `force_enabled`.

Pure tests cover rollback, force, absence, invalid and non-Unicode values;
127/128 and 512/513 boundaries; JSONL, warm-follow-up, durable-store, and request
index scope; active RAM-prefix state; missing and future markers; sampling;
prompt lookup; and exact marker matching.

## Timing-domain contract

The runner must use an explicit field-type/domain table. It must test this table
in both `--check-only` and the post-reservation acquisition path.

Finite duration-like numbers `>=0`, accepting integer zero and `0.0`:

- prompt acquisition, tokenizer initialization, tokenization, and capacity
  validation;
- scratch and sequence allocation;
- first-token selection and callback duration;
- rusage CPU durations;
- external teardown `X`.

Nonnegative integer fields, accepting integer zero but rejecting floats and
Booleans:

- PSO misses, miss/compiler walls in nanoseconds, and other counters;
- loader byte/resource/count fields and allocation/source/copy/binding
  microseconds;
- Metal allocation currents and sampled maxima;
- rusage counters and page/block-fault fields.

Positive integer fields, rejecting every zero or float:

- PIDs, launch/read/wait and request-start timestamps, and read-event byte
  counts;
- model device/inode/size/mtime identity fields; and
- loader `ready_us`.

Metal allocation deltas are signed integers, may be negative, and must reconcile
exactly against model-ready and request-start currents.

Strictly positive for every live N128/N512 process:

- runtime/model load, prefill, first-token-ready, TTFT, generation,
  inference-complete, and total-request walls;
- transition wall and TPS, because every live process has a positive transition
  count; and
- external `F`, `L`, and `E`.

Also require:

```text
0 < F <= L <= E
X = E - L >= 0
0 < prefill <= first_token_ready <= TTFT
TTFT <= inference_complete <= total_request
```

Loader components may be zero, but integer `ready_us>0`; it differs from the sum
of allocation/source/copy/binding by at most four microseconds. Allocation deltas
and sampled maxima reconcile exactly.

Pure validator fixtures must prove:

- every nonnegative duration accepts integer zero and `0.0`, individually and
  together;
- every nonnegative integer field accepts integer zero but rejects `0.0`;
- every positive integer field rejects zero and floats;
- signed allocation deltas accept negative, zero, and positive integers;
- every positive endpoint rejects zero;
- negative values are rejected outside signed deltas; Booleans, NaN, and
  infinity are rejected everywhere;
- zero transition time/TPS is accepted only when transition count is zero;
- paired-log arithmetic works at `n=8` and `n=4`; and
- every exact statistical boundary and decision-precedence branch behaves as
  frozen below.

## Frozen fixture and environment

- Model: `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,134,528,992
  bytes, SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Prompt: `docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt`,
  1,891 bytes, 419 tokens, SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
- Host: exact device line
  `device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes`,
  `hw.memsize=137438953472`, macOS `15.6.1`, build `24G90`.
- Warm-filesystem-cache process-cold contract; one fresh process and one request
  per scored observation.
- Fixed context 1024, prefill chunk 1024, temperature zero, zero RAM/durable
  prefix admission, no prompt lookup, and no warm follow-up.
- Pin `QWEN_DECODE_GDN_FUSED_BETA_PROJ=1`, `QWEN_DECODE_ROPE_PAIR=1`,
  `QWEN_DECODE_MOE_GROUPED_FINALIZER=1`, and
  `QWEN_DECODE_MOE_FUSED_FINALIZER=1` in both arms.
- Archive then remove every other inherited `QWEN_*` variable.
- A sets `QWEN_GREEDY_GPU_ARGMAX=0`; B leaves it absent. No other environment,
  argument, fixture, load policy, or process rule differs.

The positional competitor executable set is exactly `qwen`, `qwen-bench`,
`llama-cli`, `llama-server`, `llama-bench`, `ollama`, `mlx_lm`, and names
beginning with `mlx_lm.`, recognized only in `comm` or `argv[0]`. Python
interpreters match `python`, `python3`, or `python3.<digits>` exactly. Their
option region recognizes `-m mlx_lm` or a module beginning with `mlx_lm.` only
before `-c`, `--`, `-`, or the script operand;
`-W` and `-X` consume their following operand. Arbitrary argument basenames are
never classified. Pure fixtures include the prior
`opencode recall search qwen ...` query as a negative case and direct
qwen/llama/ollama/MLX launches as positive cases.

Each host capture runs exactly `memory_pressure -Q`, `pmset -g therm`, and
`ps -axo pid=,comm=,args=`. Require zero return code and no timeout for all three.

Each scored command is the direct release `qwen` binary with:

```text
--model MODEL --prompt-file PROMPT --tokens N --temp 0
--prefill-chunk 1024 --max-context-tokens 1024
--prefix-cache-max-mib 0 --cache-prefix-auto-min-tokens 0
--request-timings UNIQUE_PATH
```

## Pre-acquisition gates

Only non-consuming `--check-only` validation may precede acquisition. Acquisition
exclusively creates a previously nonexistent, non-symlink artifact root first,
writes raw precreation host evidence, then evaluates it. Every later failure
seals `INVALID`.

After reservation and all static/domain self-tests:

Run exactly this archived charged build first:

```text
cargo build --release -p qwen-cli --bin qwen --bin qwen-bench
```

Require it to exit zero, then verify clean matching build/runtime/source identity
and binary hashes. Complete the initial model identity/hash before any
model-backed gate.

1. Run the complete CLI policy suite and the individually targeted pure
   load-intent, prefetch-advice, and realized-marker coupling tests. Require each
   test below to pass exactly once; do not freeze the unrelated total suite count
   or cargo summary wording:
   - `greedy_gpu_env_parsing_preserves_absent_force_and_rollback_states`;
   - `absent_gpu_greedy_requires_exact_realized_marker`;
   - `absent_gpu_greedy_is_narrowly_scoped`;
   - `absent_gpu_greedy_enforces_requested_token_envelope`;
   - `gpu_greedy_precedence_is_rollback_then_request_eligibility_then_force`;
   - `model_load_intent_scopes_parallel_copy_auto_admission`;
   - `prefetch_action_selector_table_is_fail_closed`;
   - `realized_auto_load_marker_candidate_is_exact_and_versioned`; and
   - `prepared_auto_prefetch_advice_selector_table_is_fail_closed`.
2. Run the exact model-backed A3B greedy-chain state test and require exact token
   IDs, bit-identical continuation logits, and byte-identical active KV, GDN
   convolution, and recurrent state. Require exactly one literal
   `[greedy-chain-moe-a3b] exact-state PASS` marker and no skip.
3. Run one fresh N128 A/B conformance pair. Require identical response bytes,
   token digest, stop/terminal semantics, 128 tokens, and 127 transitions. A must
   report rollback/CPU greedy; B must report automatic GPU greedy. Both must
   realize the exact A3B pread/logical-exact marker set. Both require zero block
   input and major faults. Exclusively create the immutable N128 reference from
   conformance A, recording response bytes, stdout/response hashes, and token
   digest before conformance B launches.
4. Run one absent-environment N128 JSONL smoke and require reusable-path
   default-off with no disposable Auto marker. Require exactly one
   newline-terminated stdout object and one stats object, matching response/token
   digest, token counts, stop/terminal fields, zero cache, and CPU-greedy labels.
   Extract `generated_text` as UTF-8 bytes and match the immutable N128 response.
5. Run one explicit-force N128 JSONL smoke and require GPU greedy without
   disposable Auto authority, with the same exact framing, output, digest,
   terminal, and cache contract as the absent case. The absent and forced JSONL
   outputs/digests must match each other and the immutable N128 reference.
6. Run one absent-environment N128 ordinary smoke with
   `QWEN_GGUF_PARALLEL_COPY=0` and require no-profile default-off. Its response,
   digest, stop, and terminal fields must equal the fresh v0.652 N128 A reference.

These are conformance gates, not timing observations. None enters a scored
estimate. Any failure seals `INVALID`. N1/N16 are removed: pure tests cover every
outside-envelope and zero-transition policy boundary. There is no separate N512
reference process.

Cool down five seconds between every model-backed pre-gate child and between the
final pre-gate child and the first scored process.

## Scored acquisition

Run only the two authority boundaries:

1. N128: exactly eight pairs in order
   `AB, BA, AB, BA, AB, BA, AB, BA`.
2. N512: exactly four pairs in order `AB, BA, AB, BA`.

The packet has 12 pairs and 24 fresh scored processes. Every N128 row must match
the immutable conformance-A N128 reference, including the first scored A. The
first scored N512 A exclusively creates `reference-n512.json` and archives its
response bytes, stdout/response hashes, and token digest before paired B launches;
every later N512 row matches it. The common stdout parser removes exactly one
CLI-added newline before every conformance/reference comparison. Cool down five
seconds between every scored arm process. There is no retry, replacement,
adaptive count, continuation, trimming, early stop, or extra row.

For this packet charged children are only the archived release build, cargo test
gates, and qwen model/request children. Host-evidence commands, Git/identity
probes, and in-process file hashing are not charged children and cannot recurse.
Before and after each charged child, require at least 85% free memory, no
structurally identified inference competitor, and no thermal or performance
warning. Archive raw evidence before evaluating it. Every scored child and
same-profile N128 conformance child reports zero block-input operations and zero
major faults. Every non-workload identity probe must also return zero.

Hash the complete model after successful build and before model-backed gates.
Bind every model-backed child's pre/post device, inode, size, and mtime to that
initial fully hashed identity. After successful acquisition, hash it again before
a normal decision. An early `INVALID` preserves the last completed child
post-identity and does not require another 22 GB read.

Freeze exact prompt and binary pre/post identity. Require clean commit topology:
the preregistration commit, one bounded candidate-only commit, then exactly one
runner-only commit at clean HEAD. Its full delta is exactly the added path
`scripts/profile/v0652_a3b_one_shot_gpu_greedy_final.py`. The runner verifies its
worktree and index blob against HEAD; build/runtime commit and complete
source-state hashes must match.
The manifest and any GO authority name the exact candidate and runner commits.
Freeze argv, normalized environment, preregistration, runner, source, prompt,
binary, and documentation hashes through acquisition.

The runner source must contain no v0.651 artifact path or imported v0.651
reference/result constant. The v0.652 root must not preexist or resolve through a
symlink; acquisition refuses any existing entry and never reads the v0.651 root.

## Endpoints and validity

Launch `qwen` directly with piped stdout. Use one monotonic clock from immediately
before spawn through:

- `F`: first nonempty stdout read;
- `L`: last nonempty stdout byte-read;
- `E`: successful exact-PID wait; and
- `X=E-L`: post-response teardown.

Drain stdout and stderr concurrently. Timestamp every nonempty stdout read; EOF
is not `L`. Concatenated stdout equals the response plus exactly one CLI newline.
`E` and `X` are diagnostic and cannot reject admission except through process or
timing validity. Launch, read-event, wait, and request-start timestamps are
positive integers; read-event byte counts are positive integers; all events are
monotonic and reconcile exactly with F/L/E/X.

Every scored row preserves:

- exact source/build/runtime, fixture, model, device, and environment identity;
- exact disposable profile and complete pread/logical-exact loader markers;
- 419 prompt tokens and exact requested/generated counts;
- 127 or 511 transitions for N128 or N512;
- exact response bytes and generated-token digest within each cell;
- `token_limit` and
  `terminal_token_target_transition_consumed=false`;
- A rollback telemetry and B exact Auto telemetry;
- no cache, durable restore/publication, lookup, sampling, warm follow-up, or
  extra request surface;
- complete host evidence and zero scored block input/major faults; and
- the frozen timing domains and orderings.

Archive PSO, allocation, loader phase, rusage, and E/X telemetry. They are
diagnostic unless a frozen validity rule above names them.

## Statistics and gates

For each inferential endpoint define `y_i=log(M_A/M_B)`. Positive logs, ratios
above one, and exponentiated estimates above one favor B. Report raw A/B values,
every ratio and log, geometric estimate, log standard deviation, order strata,
and separate one-sided bounds. Upper bounds are report-only and never decide
admission.

N128 uses eight pairs and `t7=1.894579`:

```text
S128   = exp(mean(y_i))
LCB128 = exp(mean(y_i) - 1.894579 * sd(y_i) / sqrt(8))
UCB128 = exp(mean(y_i) + 1.894579 * sd(y_i) / sqrt(8))
```

Admission requires:

- external `L`: `S>=1.02` and `LCB>1.00`;
- internal generation per selected token: `S>=1.05` and `LCB>1.03`;
- internal total request: `S>=1.02` and `LCB>1.00`;
- process-cold first byte `F`: `LCB>=0.97`; and
- runtime/load, prefill, and model-ready TTFT point estimates each inside the
  inclusive band `[0.97,1.03]`.

N512 uses four pairs and `t3=2.353363`:

```text
S512   = exp(mean(y_i))
LCB512 = exp(mean(y_i) - 2.353363 * sd(y_i) / sqrt(4))
UCB512 = exp(mean(y_i) + 2.353363 * sd(y_i) / sqrt(4))
```

Its external `L`, generation per token, and internal total request each require
`S>=1.02` and `LCB>1.00`; first byte requires `LCB>=0.97`. Load, prefill, and
TTFT are descriptive at N512. The cell guards the upper requested-token boundary
but does not prove monotonic performance at every intermediate length.

No prior or conformance observation enters any estimate or variance. Claim no
prospective power or simultaneous 95% coverage. Unresolved fixed-n bounds are
non-GO by construction.

## Mechanical decision and one-shot rule

Decision precedence:

1. `INVALID` for any validity, identity, state, process, output, profile, timing,
   artifact, or completion failure;
2. `INCONCLUSIVE_CONTAMINATION` for an N128 load/prefill/TTFT band miss;
3. `KEEP_DEFAULT_OFF_COLD_REGRESSION` for an N128 F miss;
4. `KEEP_DEFAULT_OFF_EFFECT_MISS` for an N128 L/generation/total miss;
5. `KEEP_DEFAULT_OFF_ENVELOPE_GUARD_MISS` for any N512 gate miss; and
6. `ADMIT_A3B_ONE_SHOT_AUTO_V1` only if every prior gate passes.

Only GO carries the narrow authority at the start of this document. Every
non-GO has `authority=none` and requires the committed default-off rollback.

After artifact-root reservation, every exception writes failure and decision
artifacts, inventories available evidence, and seals `INVALID`. The packet is
never repaired, retried, replaced, extended, or followed by another admission
packet. The inventory excludes itself and `packet-complete.json`; completion
records the decision and inventory hashes. Independently verify both hashes and
every inventory entry. Completion does not and cannot self-hash.

For every non-GO, seal the packet first, then commit the exact inverse candidate
delta so the three candidate source blobs equal the preregistration state. Run
the default-off CLI tests and workspace check, rebuild `qwen` and `qwen-bench`
from the clean rollback head, and require matching build/runtime/source identity.
The branch is not closed until the rollback and result record are committed. No
further product or GPU experiment may run between the non-GO and that closure.

Planned artifact root:
`target/profiles/v0652-a3b-one-shot-gpu-greedy-final-p1/`.
