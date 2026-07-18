# v0.605 Dense-27B Parallel-Copied Loader Successor

Status: preregistered harness-only successor; no product observation exists yet.

## Intent

Run the exact v0.604 dense-27B force-only product experiment under one
prospectively corrected pressure classifier. v0.604 remains a sealed prelaunch
inconclusive. This packet neither retries nor rescues it: v0.605 uses a new
artifact path, manifest, decision, completion seal, and complete set of sole
attempts.

The only authorized experimental change is removal of the unilateral compressor
stored/occupied gauge vetoes. Pageouts and Compressions remain advisory exactly
as in v0.604. All product implementation, correctness, command, order,
performance, CPU, memory, remaining validity, decision, and authority contracts
remain frozen.

## Authorization Evidence

The immutable runner must authenticate these exact v0.604 artifacts before
reserving the v0.605 packet directory:

- `docs/bench/v0604-dense27b-parallel-copied-loader.md`
  `7841ac41580db81b6274b0002b4c4cbd4348ef5e29ef5eb48f6006e37371a8e7`
- `target/profiles/v0604-dense27b-parallel-copied-loader-p1/decision.json`
  `094a98e061d3e46be4f14733cd30a4bd269ea7999844e42d831f06b7049d52b6`
- v0.604 `artifact-inventory.sha256`
  `0a553cbeb6ae5ebe03c931c0ada74b37633c7e320359e587c208626741d71415`
- v0.604 `packet-complete.json`
  `b6eca51ee5ff910a4d8dadc96227554b0961f2a85275c314fa84de995da1a04a`

Require the decision to be schema 1, `status=inconclusive`, `authority=none`,
`stopped_after=prelaunch`, failed child `loaded-p01-ab-r1-a`, sole reason
`cache_compressor_occupied_pages_growth`, null attempts hash, clean source
`040e4d77e8f048e3599d58ffa94d5408b4a5721a`, and passed correctness. Require
the completion seal to bind the exact decision and inventory hashes.

Verify every path/digest member named by the sealed v0.604 artifact inventory
against its current payload. Parse the prelaunch evidence and require the exact
causal record: compressor occupied `+2`, stored `-14`, zero Pageouts,
Compressions, Swapouts, and swap-occupancy delta, 95% available memory, valid AC,
and valid thermal/performance state. Top-level inventory and completion hashes
without member verification are insufficient.

No production build input may differ from v0.604. The runner must reject any
diff from source `040e4d7` under `Cargo.lock`, root `Cargo.toml`, `.cargo/`,
`crates/`, or `kernels/`. This closure includes workspace/dependency resolution,
build scripts, qwen-llm, qwen-cli, tokenizer, loader, benchmark commands, and
Metal sources. Documentation and immutable-runner changes lie outside the
production build-input closure.

Follow the existing packet workflow: commit the preregistration and immutable
runner, build `qwen` and `qwen-bench` normally from that clean commit, and require
their source/build/runtime identity to match it exactly. Both arms use those same
binaries, so the rebuild is common-mode within v0.605. No v0.604 product row is
reused or compared because none exists. Any production-tree, dirty-state,
binary, build-info, command, working-directory, or environment drift invalidates
preflight.

## Frozen Cell

- Hardware: local Apple M4 Max, 128 GiB unified memory, exact device identity.
- OS: exact macOS product/build identity captured in the manifest and rechecked
  before publication.
- Model: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`, exact size and SHA-256
  imported from v0.604.
- Prompt: the exact 419-token Reva fixture and SHA-256 imported from v0.604.
- Arm A: normalized environment plus `QWEN_GGUF_PARALLEL_COPY=0`.
- Arm B: normalized environment plus `QWEN_GGUF_PARALLEL_COPY=1`.
- Fixed pair order: `AB/BA/BA/AB/AB/BA` for loaded, then the same order for
  fresh output-128 only if loaded passes.
- No retries, replacement children, parallel benchmarks, serving, or concurrent
  loading. Every launched arm consumes its sole attempt.

The runner, preregistration, common helper, exact model and prompt, both release
binaries, v0.603 authorization chain, and v0.604 authorization chain are hashed
into the manifest. Current source/build/runtime, working directory, environment,
device, OS, model, and non-model identity are rechecked before publication.

## Correctness

Rerun the exact v0.604 release full-state gate before product timing. Arm B must
authenticate the dense schema-2 profile, 851 independent exact-sized offset-zero
Shared/DefaultCache/Tracked resources, all `16,806,250,496` source bytes, the
literal four-worker schedule, ordinary copied ledger, checked compute/blit write
rejection, bit-exact packed-prefill logits, complete KV/GDN/conv state, argmax,
one forced transition, continuation logits, and continuation state.

Any malformed or missing correctness evidence is an implementation/contract
defect. v0.604 correctness is authorization evidence, not a substitute for this
gate.

## Loaded Stage

Run the exact v0.604 loaded command for every child:

```text
qwen-bench decode
prompt=current-reva-n8-interactive-qwen36.txt (419 tokens)
prefill_chunk=1024 kv_capacity=1024 full_logits=true
decode_calls=127 runs=5
```

Score repetitions 3-5. Every pair must satisfy:

- prefill `median_A/median_B >=0.99`;
- decode `median_A/median_B >=0.99`;
- request `median_B/median_A <=1.01`;
- each arm repetition-5/repetition-3 decode TPS in `[0.98,1.02]`;
- each arm repetition-3-5 decode-TPS range at most 3% of its median;
- paired RSS and footprint B/A each `<=1.05`;
- exact run count, output, marker, ledger, and load identity.

Complete all 12 children before inspection. Instability is inconclusive. A
stable performance or memory miss kills before fresh. The six B endpoint
total-CPU values must have overall median `<=2.45 s` and AB/BA medians each
`<=2.55 s`. Loaded complete-process CPU, instructions, and cycles are
diagnostic after the endpoint cap.

## Fresh Output-128 Stage

Run only after loaded passes. Use the same prompt, exactly 128 emitted greedy
tokens and 127 target transitions, chunk/context 1024, disabled prefix cache,
one schema-3 timing row, and exact stdout/timing/load identity.

For paired metrics imported from v0.604, require:

- first-byte median `F >=1.25x`, AB/BA each `>=1.20x`, with at least 5/6
  overall and 2/3 per-stratum wins;
- exit median `E >=1.08x`, AB/BA each `>=1.05x`, with the same win counts;
- runtime/load saving median `L >=750 ms`, AB/BA each `>=600 ms`;
- runtime/load B/A median `Q <=0.70`, AB/BA each `<=0.80`;
- first-prefill and model-ready-TTFT B/A medians each `<=1.02`, AB/BA each
  `<=1.03`;
- generation and model-ready-request B/A medians each `<=1.01`, AB/BA each
  `<=1.02`;
- B endpoint total-CPU median `<=2.45 s`, AB/BA each `<=2.55 s`;
- complete-process CPU B-A median `<=0.90 s`, AB/BA each `<=1.00 s`;
- maximum paired RSS and footprint B/A each `<=1.05`.

Use paired ratios/differences before medians. No cold, CPU, or memory win may
erase a model-ready miss.

## Revised Pressure Classifier

Capture raw Pageouts, Compressions, compressor stored/occupied pages, Swapouts,
swap occupancy, block I/O, major faults, and every cache/child interval delta.

Stop inconclusive on:

- any host/VM capture or parse failure;
- regression of cumulative Pageouts, Compressions, or Swapouts counters;
- any positive Swapouts or swap-occupancy delta;
- child block input;
- fresh child major faults;
- invalid AC, thermal, performance, or memory-availability state;
- spawn/nonzero exit, child-attributed I/O, or operator interruption.

Positive Pageouts and Compressions remain advisory under the existing
v0.600/v0.601 causal boundary. Compressor stored and occupied gauges are
required, recorded, and reported in both directions, but neither gauge
independently vetoes an interval. No materiality threshold is inferred from
v0.604's `+2` pages. Loaded timer-local major faults remain recorded but advisory
under the existing harness floor.

This is the sole protocol change. Existing swap, I/O, major-fault, host-state,
and execution gates continue to reject actual invalidity without treating a
host-wide allocation gauge as an activity counter.

## Sealing And Decision

Preserve v0.604 durability semantics: deferred child interrupts, complete child
cleanup, fsynced raw/timing/launch/completion/attempt evidence, exact frozen launch
prefix validation, final model rehash, artifact inventory, decision hash, and
completion seal. Incomplete durability grants no decision authority.

Decision precedence remains:

1. correctness, identity, schema, output, or accounting defect;
2. revised pressure/host/execution inconclusive;
3. loaded instability inconclusive;
4. stable loaded miss: kill before fresh;
5. valid fresh miss: kill;
6. complete conjunction: GO.

Any GO authorizes only explicit `QWEN_GGUF_PARALLEL_COPY=1` for the exact frozen
dense structural profile. It does not authorize default selection, MTP variants,
other dense assets, MoE, storage-cold claims, serving, concurrent loading, or
energy claims. Any kill or inconclusive has no authority.

Do not reuse, pool, rescore, or reinterpret v0.604 product observations; none
exist. Do not launch another successor after any valid v0.605 product child.
