# v0.635 A3B Page-Rounded Copy Compatibility

Status: preregistration. No v0.635 GPU, model, correctness, or timing work has run.

## Question And Scope

Compare two authenticated parallel mmap-copy populations for one loaded A3B asset:

- A: `QWEN_GGUF_PARALLEL_COPY=1`, 733 exact-sized independent MTLBuffers;
- B: `QWEN_GGUF_PARALLEL_COPY=page-rounded-copy`, the same 733 independent,
  offset-zero resources and exact logical tensor views, with each exposed
  `MTLBuffer.length` rounded to 16 KiB.

This packet asks only whether B preserves A's bit-exact state and loaded
prefill/decode/request/memory compatibility. There is no fresh or cold timing and
no imported performance observation. It cannot authorize a product mode,
converter, default, no-copy conclusion, or production sidecar.

## Frozen Source And Host

The base is `0619c925d488a9f4b47b64c509b6231f4ab6bfb1`. R is its clean,
single-parent child and adds exactly this document and
`scripts/profile/v0635_a3b_page_rounded_copy_compatibility.py`. H is R's clean,
single-parent child and modifies only `crates/qwen-llm/src/metal_forward.rs`.

```text
final metal_forward.rs SHA-256  0c7fa44527365eb791fd0edf8e581d88c5a7de849059b8aefd2c251fa3d24ce8
R..H file-only binary diff SHA  e14a5a08a105d7cbaeb76e789a097a238e2b37127d07de73c10ec5d02bf05dab
gguf commit                      c7369fd4868a6f613459fff355477f53bf4ee2f1
llama-cpp-rs commit              fe4fb533d1ed2855b6ac5492e56c42007d410409
```

The main worktree and tracked sibling state must be clean. GGUF permits no
untracked file. llama-cpp-rs permits exactly `.claude/settings.local.json` in the
all-untracked porcelain view. The runner freezes the final file and R..H binary
patch digests, requires `HEAD=H`, matching clean build/runtime identity H, and
hashes the v0.602 gate-definition source plus the imported v0.630 infrastructure
helper and its v0.593 dependency without monkeypatching any module.

Host identity is macOS `15.6.1` build `24G90`, Apple M4 Max, 128 GiB. The runner
requires the exact Metal identity, source topology, sibling identities, release
binary identity, normalized environments, and parser dialects.

## Asset And Protocol

```text
model  /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
size   22134528992
SHA    ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61
v0.602-observed runtime model ID      e6024ce53109fdf7
v0.602-observed runtime tokenizer ID  a4b0b26f8a8c9917
prompt docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt
prompt size/tokens    1891 / 419
prompt SHA e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474
```

The prompt is read at runtime and is not embedded. Before correctness/timing, the
runner hashes the complete model. Before every timed child it rereads and hashes
the complete model with one 8 MiB buffer. The historical runtime IDs are recorded
for continuity but are not independently remeasured or used as v0.635 gates; the
complete model/prompt hashes, authenticated load contract, and token trace are.

Both arms retain 733 direct-copy ledger entries over 22,123,538,944 logical bytes:

```text
[metal-load-ledger] source=733/22123538944 direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 tail_fallback=0/0 converted=0/0/0 derived=0/0
```

Every successful load emits the Q8_0 native-embedding line, its arm marker, then
that ledger. Unknown, duplicate, malformed, or reordered load/storage lines are
contract defects.

A uses the canonical v0.602 schema-1 marker. Its immutable prefix is:

```text
[metal-gguf-parallel-copied] schema=1 resources=733 bytes=22123538944 workers=4 cuts=155,359,539 tasks=155,204,180,194 worker_bytes=5532746240,5462315776,5595522304,5532954624 first_offsets=10990048,5543736288,11006052064,16601574368 last_offsets=5392741344,11004937952,16450579424,22134520800 create=shared,default_cache,default observed=shared,default_cache,tracked page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 layout=0x5ae645df5cf7d568 inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af
```

B replaces the opening accounting only:

```text
[metal-gguf-parallel-page-rounded] schema=2 resources=733 logical_bytes=22123538944 allocated_bytes=22126297088 padding_bytes=2758144 padded_resources=232
```

It then has byte-identical workers/cuts/tasks/offsets/modes/page/layout/inventory/
plan fields. Both append exactly, in order:

```text
allocation_us=<u64> source_us=<u64> copy_us=<u64> binding_us=<u64> ready_us=<u64>
```

Values are canonical unsigned decimal and
`abs(ready_us - sum(first four fields)) <= 4`.
The `allocated_bytes` field sums requested/exposed `MTLBuffer.length` values. It
makes no claim about the driver's underlying VM or allocator granularity.

## Fresh Correctness Gate

Before any timed child, run exactly:

```text
cargo test --release -p qwen-llm \
  metal_forward::tests::gguf_parallel_page_rounded_a3b_q4_is_bit_exact \
  -- --ignored --exact --nocapture --test-threads=1
```

Require a unique passing Cargo result and the six recognized lines in exact order:
A native policy, exact marker, ledger; B native policy, rounded marker, ledger.
The fresh test directly compares exact-parallel A with rounded-parallel B and must
establish bit-exact full state plus its internal storage/topology assertions.
Correctness is not imported from v0.602.

Before correctness, run the CPU-only `qwen-bench decode --help` protocol check.
Require `--generated-token-trace` and its exact description once.

## Loaded Cell

Run six independent pairs in this exact order:

```text
AB BA BA AB AB BA
```

Each child runs once, serially, with no retry:

```text
/usr/bin/time -l target/release/qwen-bench decode
  --model <frozen model>
  --prompt <exact runtime prompt text>
  --tokens 127 --runs 5
  --prefill-chunk 1024 --kv-capacity 1024
  --full-logits-decode --generated-token-trace
```

The normal untimed warmup remains enabled. Each timed repetition uses a fresh
session. Score repetitions 3-5 only; repetitions 1-2 are always excluded. The
trace is the initial prefill argmax plus all 127 timed transition results: exactly
128 canonical IDs in `[0,248320)`. All 12 exact arrays and generated-output hashes
must match. The first valid A trace is durably sealed as the golden event.

For pair i, using each arm's repetition-3-5 median wall:

```text
P[i] = median(prefill_A) / median(prefill_B)
D[i] = median(decode_A)  / median(decode_B)
R[i] = median(request_B) / median(request_A)
```

Reuse exactly the v0.602 loaded gates:

- every `P >= 0.99`, `D >= 0.99`, and `R <= 1.01`;
- each child repetition-5/repetition-3 decode TPS is in `[0.98,1.02]`;
- each child's late decode-TPS `(max-min)/median <= 0.03`;
- every pair has `RSS_B/RSS_A <= 1.05` and footprint `B/A <= 1.05`.

All 12 children must be complete and valid before any performance calculation.
That yields exactly 36 scored repetitions. Stability is evaluated first. A
stability-gate failure is inconclusive; stable P/D/R/memory failure is kill; the
complete conjunction is go. Parsed and derived values must be finite with positive
ratios.

## Conditioning And Validity

Immediately before each child:

1. Verify immutable non-model identity and capture host/VM state.
2. Read and SHA-256 exactly the full model using one 8 MiB buffer.
3. Cool down exactly 30 seconds.
4. Sample host state at most six times, 30 seconds apart.
5. Require AC power, no thermal/performance warning, and memory availability >=50%.
6. Capture pre-spawn VM state, durably seal conditioning, and launch once.
7. Capture and durably seal post-exit host, VM, and process resources.

Pageouts and Compressions growth and compressor-gauge motion are advisory.
Counter regression, capture/parser failure, positive Swapouts growth, positive
interval growth in swap occupancy, process swaps, child block input, invalid host,
or a safely reaped spawn/exit/I/O failure makes the packet inconclusive. Loaded
major faults are recorded but advisory. Raw durability failure or inability to
reap the exact PID aborts without a decision, inventory, completion, or authority.
No child, pair, or packet retry is authorized.

## Decision Precedence And Authority

Apply exclusive precedence:

1. Source/build/correctness/parser/marker/ledger/trace/artifact defect:
   `implementation_or_contract_defect`.
2. Invalid execution, host, pressure, capture, process, or incomplete population:
   `inconclusive`.
3. Complete stability failure: `inconclusive`.
4. Complete stable performance or memory miss: `kill`.
5. Complete conjunction: `go`.

Every status records `authority=no-production-authority` and
`force_authorized=false`. Only GO grants
`successor_authorization=preregister-one-bench-only-page-aligned-a3b-sidecar-image`.
That permits one separately preregistered bench-only sidecar image. It grants no
converter, product, default, no-copy, or production authority. KILL closes only
this 733-independent-resource page-rounded image construction. Inconclusive or
defect grants nothing. There is no `cx` authority.

Claim scope is explicit:

```text
v0602_performance_observations_imported=0
v0602_timed_children_imported=0
correctness_imported_as_gate=false
loaded_gate_definition_imported=true
```

## Durable Packet

On complete execution, the pre-inventory set has nine packet members: manifest,
token-protocol out/json, correctness out/json, attempts, launch seal, final
identity, and decision; plus four files per child (out, err, conditioning, post
exit). Thus `9 + 12*4 = 57` inventory members. The SHA-256 inventory and completion
binding produce exactly 59 final files. JSON/JSONL, raw output, launch/completion,
one golden-trace event, final identity, decision, inventory, and completion are
fsynced. `--preflight-only` does not reserve the artifact; `--self-test` is CPU-only.
SIGINT is deferred from reservation through the pre-publication decision boundary;
timed and correctness children run in separate sessions and are reaped exactly.
Any signal observed before publication makes a non-defect result inconclusive.
Publication runs with SIGINT blocked; a later signal cannot retroactively change a
durably completed packet.
