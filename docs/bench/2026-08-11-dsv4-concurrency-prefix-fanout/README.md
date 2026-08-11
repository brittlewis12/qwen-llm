# DeepSeek V4 Concurrent Prefix Fanout

Update (2026-08-11): a shared bounded affinity planner exists, but DeepSeek
keeps input-order pairing by default until a memory-safe model-backed run
qualifies reordered worker assignment.

Date: 2026-08-11

Status: product `GO` for qualifying DeepSeek V4 concurrency pairs.

## Scope

DeepSeek `--concurrency 2` already retained one immutable model residency and
two worker-local sessions, serialized their packed prefills, then overlapped
decode on independent queues. Serial prefill is correct but repeats the largest
piece of work when requests share a long system prompt or document.

The candidate computes the exact pairwise token LCP, selects a boundary shared
by both prompts' real packed-chunk schedules, prefills it once, captures a
causal snapshot, restores the second session, and evaluates only private
suffixes. Identical prompts may use the complete prompt boundary.

Rollback:

```bash
QWEN_CONCURRENCY_PREFIX_FANOUT=0
```

## DeepSeek Contract

DeepSeek causal snapshots contain exact committed tokens, raw chronological
cache rows, compressor state, and published rows. They intentionally omit the
source observation and logits. The source worker therefore copies prefix logits
before capture. A restored exact prompt reuses that immutable copy; a partial
prompt evaluates its suffix and produces a new observation normally.

Both sessions bind the same pair-local transient content identity over one
`Arc<DeepSeekV4MetalResidency>`. This avoids hashing roughly 89 GB of model
content for an ephemeral in-process transfer without weakening durable snapshot
identity. The snapshot never enters a cache or store and is dropped before
decode begins.

## Fixture

- Model: DeepSeek V4 Flash 0731 REAP K216 UD IQ3_XXS
- Device: Apple M4 Max
- Decode: greedy, eight requested tokens per lane
- Packed prefill chunk cap: 4,096
- Whole-model residency set: disabled
- GGUF residency: cache-warm except the first control's load
- Control/candidate: separate processes using one source-matched release binary

The identical fixture tokenizes the ring0 prompt to 6,219 tokens per lane. The
partial fixture appends distinct notebook suffixes; each prompt has 6,228 tokens
and their exact LCP is 6,225 tokens.

## Results

| Pair | Arm | Selected prefix | Model prefill, ms | Preparation, ms | Pair wall, ms |
|---|---|---:|---:|---:|---:|
| Identical | rollback | 0 | 57,458.312 | 57,464.606 | 58,106.922 |
| Identical | candidate | 6,219 | 28,145.338 | 28,762.131 | 29,248.723 |
| Partial | rollback | 0 | 55,860.968 | 55,866.299 | 56,376.329 |
| Partial | candidate | 6,144 | 30,418.412 | 30,994.501 | 31,511.527 |

Identical preparation improves 1.998x and pair wall improves 1.987x. Partial
preparation improves 1.802x and pair wall improves 1.789x. The endpoint excludes
model loading, so the first control's colder 10.76-second load does not confound
the claim.

| Pair | Payload bytes | Capture, ms | Restore, ms | Restore image bytes |
|---|---:|---:|---:|---:|
| Identical | 60,621,612 | 410.532 | 201.229 | 5,636,096 |
| Partial | 60,137,472 | 379.165 | 191.436 | 5,636,096 |

Decode remains the existing independent-queue path. Its short timing windows
carry no new throughput claim.

## Correctness

Every complete JSON output row is byte-identical between rollback and candidate
for both fixtures. This includes generated-token SHA-256, decoded text, token
counts, stop reason, and the unconsumed-terminal-token marker. The partial pair
also preserves its two intentionally distinct continuations.

The runtime validates snapshot identity, compatibility digest, exact prefix
tokens, session capacity, and fresh destination state before mutation. Worker
failure signals abort and join both scoped workers. Snapshot and prefix-logit
arcs are explicitly released after the restore worker is ready and before either
lane begins decode.

## Memory And Lifecycle

Each session is priced at 4,390,109,184 bytes. Fanout admission adds both
sessions, the encoded-record upper, the full raw-ring restore image, and the
existing 512 MiB transient reserve. Required bytes are 9,383,363,404 for the
identical pair and 9,382,879,264 for the partial pair. Observed two-session Metal
allocation is 8,771,600,384 bytes.

After each large-model process exited normally, no qwen process remained and
`memory_pressure -Q` reported 90% free memory. Pre-existing swap did not grow
across the packet. Whole-model residency sets remained disabled throughout.

## Decision

Promote pair-local prefix fanout for DeepSeek under the same 256-token,
default-on capability policy as Qwen. Preserve family-specific snapshot and
observation handling, real packed-boundary intersection, explicit memory
admission, and the common rollback. Unrelated, short, disabled, or denied pairs
retain the established two-serial-prefill path.
