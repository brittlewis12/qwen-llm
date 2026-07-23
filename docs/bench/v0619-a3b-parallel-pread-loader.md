# v0.619 A3B Direct-Pread Loader Confirmation

Status: preregistered; implementation is present and this v0.619 packet remains
unrun.

## Intent

Confirm whether direct retained-descriptor `pread` preserves the exact loaded
behavior of the authenticated A3B copied loader while transferring a material
process-cold gain into the complete output-128 product path.

v0.617 is strong primitive evidence: direct `pread` populated the same 733
exact-sized, offset-zero Metal resources 214.743 ms faster in 6/6 pairs and
removed 1.350 million minor faults. v0.618 is evidence-only because its
one-repetition output-1 packet replaced the retained v0.602 per-pair loaded
contract with an aggregate prefill/TTFT check. This successor does not use
v0.618 results as authority or as a loaded-parity substitute. It starts a new
sealed packet and restores the complete loaded stage before any fresh timing.

## Frozen Scope

All v0.602 asset, prompt, architecture, topology, schedule, environment,
conditioning, pressure, process-lifecycle, output, and no-retry rules remain in
force except for the explicit deltas below.

- A is the existing copied path with `QWEN_GGUF_PARALLEL_COPY=0`.
- B is direct retained-descriptor population with
  `QWEN_GGUF_PARALLEL_COPY=pread`.
- B must emit exactly one `[metal-gguf-parallel-pread]` schema-1 marker with
  the same 733-resource geometry, schedule, options, inventory, plan, timing
  reconciliation, and copied logical ledger as v0.602.
- Any other `[metal-gguf-*]` marker in either product arm or the correctness
  gate is an implementation/contract defect.
- The ignored release correctness gate is
  `gguf_parallel_pread_a3b_q4_is_bit_exact`. It must audit every destination
  byte and prove bit-exact prefill logits, complete KV/GDN/conv state, argmax,
  one transition, continuation logits/state, topology, ledger, and checked
  write rejection.
- The packet uses the fixed `AB BA BA AB AB BA` order, six pairs per stage,
  no overlapping processes, no child or packet retry, and no performance-loss
  retry.
- The packet hashes its source commit, build/runtime identity, runner,
  inherited protocol, common helper, binaries, model, prompt, and this
  contract before launching a product child.

## Stage 1: Loaded Noninferiority

Use the exact v0.602 loaded shape and analysis:

```text
qwen-bench decode
prompt=current-reva-n8-interactive-qwen36.txt (419 tokens)
prefill_chunk=1024 kv_capacity=1024 full_logits=true
decode_calls=127 runs=5
```

Each child performs the existing untimed warmup and five fresh-session
repetitions. Score repetition 3-5 medians. For every pair require:

- `median(prefill_A) / median(prefill_B) >= 0.99`;
- `median(decode_A) / median(decode_B) >= 0.99`;
- `median(request_B) / median(request_A) <= 1.01`;
- repetition-5/repetition-3 decode TPS in `[0.98, 1.02]` for both arms;
- repetition-3-5 decode-TPS range at most 3% of its median for both arms;
- `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`;
- exact run count, marker, ledger, request wall, and globally identical output.

Instability is inconclusive. A stable noninferiority miss kills B and stops
before fresh timing. An aggregate median cannot rescue any per-pair miss.

## Stage 2: Fresh Output-128 Product Packet

Run only after Stage 1 passes. Use the exact v0.602 fresh shape: 419 prompt
tokens, 128 requested and emitted greedy tokens, 127 target transitions,
prefill chunk 1024, context 1024, disabled prefix cache, request timing schema
3, and external timestamps through first stdout byte and process exit.

For every pair record:

```text
L = runtime_and_model_load_A - runtime_and_model_load_B
Q = runtime_and_model_load_B / runtime_and_model_load_A
F = spawn_to_first_byte_A - spawn_to_first_byte_B
E = spawn_to_exit_A - spawn_to_exit_B
```

Require:

- median `L >= 112 ms`, AB median `L >= 112 ms`, BA median `L >= 112 ms`;
- median `Q <= 0.85`;
- B wins `L` in at least 5/6 pairs and at least 2/3 per order stratum;
- median `F >= 112 ms`, AB median `F >= 112 ms`, BA median `F >= 112 ms`;
- B wins first byte in at least 5/6 pairs and at least 2/3 per stratum;
- median `E >= 112 ms`, AB median `E >= 112 ms`, BA median `E >= 112 ms`;
- B wins exit wall in at least 5/6 pairs and at least 2/3 per stratum;
- maximum paired RSS and footprint ratios at most `1.05`;
- exact global output identity, token counts, timing schema, marker, ledger,
  pressure, zero block input, and zero fresh major faults.

Fresh model-ready prefill and TTFT ratios remain recorded diagnostics. They are
not independent single-observation noninferiority gates: Stage 1 is the
preregistered loaded parity test with five repetitions and a complete request
wall. Complete first-request first-byte and exit endpoints remain authoritative,
and fresh gains cannot rescue a failed Stage 1.

The 112 ms threshold is frozen confirmation of at least roughly half the
v0.617 primitive saving; it is not selected from this packet.

## Decision And Authority

Apply the v0.602 precedence: contract defect, invalid/inconclusive conditions,
loaded instability, loaded kill, fresh kill, then GO only on the complete
conjunction.

A GO authorizes only explicit `QWEN_GGUF_PARALLEL_COPY=pread` on the exact
single-shard A3B asset and geometry. It does not authorize default selection,
other assets, storage-cold claims, physical-memory savings, serving or
concurrent-load behavior. v0.618 remains non-authoritative regardless of this
packet's outcome.
