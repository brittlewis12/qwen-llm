# v0.598 A3B Force-Only Owned-Arena Loader Pilot

Status: preregistered before implementation, correctness execution, and timed
work. v0.597 authorizes only the exact-layout, four-worker A3B pilot described
here.

## Intent

Determine whether v0.597's `1365.044 ms` materialization saving survives the
complete loader and fresh-request path while preserving copied-storage prefill
and decode speed. Separate process-cold first byte, loaded request wall, loaded
prefill, and loaded decode. File-backed retained storage remains a diagnostic
control, not a candidate.

This is an exact-layout, warm-filesystem-cache, fresh-process and loaded-process
A3B experiment. It is not storage-cold, default-policy, broad-family, alias,
conversion, MTP, split-shard, serving, or memory-reduction authority.

## Implementation Contract

Add default-off `QWEN_GGUF_OWNED_ARENA=1`. False values disable it. Invalid or
non-Unicode values fail closed. Owned arena and `QWEN_GGUF_NO_COPY=1` are mutually
exclusive. Any explicit `QWEN_GGUF_NO_COPY_PREFAULT`, including `0`, is rejected
when owned arena is enabled.

The owned path accepts only this exact-layout sentinel:

- single shard; mapped length `22,134,528,992`;
- descriptor digest `0x5ae645df5cf7d568`;
- ordered inventory digest
  `f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5`;
- planner digest
  `fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af`;
- page 16,384, alignment 32, and max buffer `77,309,411,328`;
- 733 all-direct requests over `22,123,538,944` logical bytes;
- 732 unique views over `22,123,530,752` bytes, zero aliases/conversions;
- one `22,123,544,576`-byte window with 13,824 gap bytes;
- one 8,192-byte `FinalPartialPage` fallback;
- non-MTP, untied model with the literal architecture tuple below;
- production-auto Q8 embedding selection, not a forced override.

```text
kind=Moe layers=40 hidden=2048 intermediate=0 vocab=248320
full_interval=4 q_heads=16 kv_heads=2 attn_head_dim=256
rope_theta=10000000 partial_rotary=0.25
gdn_v_heads=32 gdn_k_heads=16 gdn_head_dim=128 conv=4
experts=256 topk=8 routed_ffn=512 shared_ffn=512 mtp_layers=0
```

The runtime predicate authenticates layout, not weight contents. The packet
runner hashes the complete model before correctness and timing. Hashing inside a
child would pre-touch the payload and is forbidden.

`PlannedOwnedStorage` owns ordinary shared Metal buffers, never
`MetalGgufBacking`. Construction performs exactly one materialization:

1. Allocate one anonymous shared window and one anonymous shared fallback.
2. Resolve the complete source window with `try_shard_range`.
3. Partition its pages at `16384 * floor(k*N/4)`, `k in 0..=4`.
4. Copy with exactly four nonempty scoped workers and join them.
5. Copy the fallback serially.
6. Construct typed views lazily during ordered loader consumption; never recopy.

Do not substitute byte-balanced chunks, per-tensor queues, a worker pool,
asynchronous promotion, Metal blits, or serial copy. Worker creation and join are
part of model load.

Introduce `OwnedWeightReadOnly` tensor provenance. A checked constructor validates
shape/dtype bytes, alignment, and `offset + n_bytes <= buffer.length()`. Both arena
views and fallback use it. Typed write paths reject both owned and retained
read-only weights with provenance-neutral diagnostics.

## Accounting

The ordered logical source ledger must report:

```text
source          733 / 22,123,538,944
direct_copy     732 / 22,123,530,752
tail_fallback     1 /          8,192
direct_view       0 /              0
direct_alias      0 /              0
converted         0 /              0 / 0
```

The separate `[metal-gguf-owned]` physical ledger must report:

```text
windows             1 / 22,123,544,576
planner gaps            13,824
fallback resources  1 /          8,192
physical resources  2 / 22,123,552,768
workers             4
```

Validate exact resource lengths/storage mode, complete cursor/realization, every
ordered request identity, and both ledgers at loader finish. Existing copied and
retained behavior and log formats must remain unchanged.

## Correctness Gate

Before timing, run one ignored local-fixture test in the frozen build. Compare
copied against owned for:

- packed-prefill logits and complete KV/GDN prefill state;
- identical greedy argmax;
- one forced transition;
- continuation logits and complete continuation state.

Every float and state byte must be bit exact. Also require exact logical/physical
ledgers, two-resource identity, `OwnedWeightReadOnly` on the window and fallback,
and rejection by typed write guards. Failure terminates before product timing.

## Product Arms

- **A — copied**: `QWEN_GGUF_OWNED_ARENA=0`, `QWEN_GGUF_NO_COPY=0`, no prefault
  variable.
- **B — owned**: `QWEN_GGUF_OWNED_ARENA=1`, `QWEN_GGUF_NO_COPY=0`, no prefault
  variable.
- **C — retained diagnostic**: `QWEN_GGUF_OWNED_ARENA=0`,
  `QWEN_GGUF_NO_COPY=1`, `QWEN_GGUF_NO_COPY_PREFAULT=0`.

Every packet uses all six blocks in this order:

```text
ABC
ACB
BAC
BCA
CAB
CBA
```

Each arm appears twice in every position and every directed predecessor occurs
twice. Processes never overlap. Only host/thermal/I/O-invalid complete blocks may
retry, in the same order, up to three attempts. A valid slow or losing block is
not retryable.

Before every child, read the complete model outside the endpoint and cool down at
least 30 seconds. Require AC power, no thermal/performance warning, and at least
50% memory availability. Bracket cache and child intervals with pageout/swap
state. Hash source, binary, runner, contract, model, prompt, and imported helpers;
require clean matching source/build identity and normalized QWEN/Metal controls.

## Fresh Packets

Run output length 1 first. Continue to output 128 only after the complete output-1
packet passes. Both use the v0.595 CLI shape: frozen 419-token Reva prompt, chunk
1024, context 1024, prefix cache disabled, and process spawn through first byte
and exit measured externally.

Preserve v0.595 fault policy: child hard faults and block input must be zero.
Child pageout/swap growth, invalid host state, output mismatch, early EOS, identity,
environment, command, timing, or ledger mismatch invalidates or terminates exactly
as classified in the runner.

For length `L`, block `i`, and first-byte wall `F`, define:

```text
S[L,i] = F[A,L,i] / F[B,L,i]
```

Require independently at output 1 and output 128:

- median `S[L,i] >= 1.20x`;
- median `S[L,i] >= 1.20x` for the three `B before A` blocks;
- median `S[L,i] >= 1.20x` for the three `A before B` blocks;
- B wins first byte in at least five of six blocks;
- every paired `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`.

Record spawn-to-exit, internal model load, TTFT, transition rate, PSO accounting,
outer residual, and C attribution. They cannot rescue a failed first-byte gate.

## Loaded Packet

Run only after both fresh packets pass. Reuse v0.596's 419-token, chunk-1024,
context-1024, full-logits, 127-transition `qwen-bench decode --runs 5` shape. Each
process performs the existing untimed prompt-plus-one-transition warmup, then five
fresh-session repetitions.

Add an explicit per-repetition request wall from session/reset start through
prefill, generation, and termination. Model load and untimed warmup remain outside.
Do not substitute `prefill_ms + decode_ms` for this endpoint.

For each block, use repetition 3-5 medians:

```text
P[i] = median(prefill_A) / median(prefill_B)
D[i] = median(decode_A)  / median(decode_B)
R[i] = median(request_B) / median(request_A)
```

Require in every block:

- `P[i] >= 0.99`;
- `D[i] >= 0.99`;
- `R[i] <= 1.00`;
- B and A repetition-5/repetition-3 TPS inside `[0.98, 1.02]`;
- B and A repetition-3-5 TPS range at most 3% of its median;
- paired `RSS_B/RSS_A <= 1.05` and `footprint_B/footprint_A <= 1.05`.

Preserve v0.596 fault policy: process faults are recorded but never gated because
they are unlocalized. Block input, cache/child pageout or swap growth, and host
invalidity remain block-invalidating. Exact final generated output, request count,
load contracts, and timing schema are mandatory.

Instability yields `inconclusive`, not retry authority. C remains attribution
only: owned parity with copied while retained stays slow implicates file provenance;
owned and retained loss implicates giant-resource topology or offsets.

## Decision And Authority

Promotion to a force-only A3B optimization requires the correctness gate plus
both fresh packets and the loaded packet. Any failed performance gate leaves the
flag diagnostic-only and closes this pilot premise. An inconclusive packet may be
repeated only under a separately preregistered protocol repair.

A full pass authorizes only explicit `QWEN_GGUF_OWNED_ARENA=1` on this exact-layout
A3B asset. It does not authorize default-on selection, other assets, aliases,
conversions, MTP, split shards, asynchronous promotion, storage-cold claims,
memory savings, persistent serving, or broad loader integration.
