# Qwen B2 File-Scoped Root Fanout

Date: 2026-08-11

Status: `GO` for the bounded Qwen `--concurrency 2` path.

## Question

Can one immutable checkpoint for the common root of a regular JSONL file avoid
re-evaluating a large system or document prefix once per B2 pair while retaining
the pair planner's deeper local fanout and concurrent decode?

The motivating 14-request ring0 evaluation corpus shares 6,476 exact prompt
tokens. The current pair path evaluates a chunk-aligned 6,144-token boundary
once for each pair. A file-scoped checkpoint can evaluate that root once, then
restore it as the base for every pair.

## Scope

The implementation remains deliberately bounded:

- Qwen `--concurrency 2` over a regular, pre-tokenized JSONL file;
- fixed prefill chunks and at least two planned pairs;
- a root of at least 1,024 tokens, aligned to the requested chunk;
- paired requests only; odd serial tails neither constrain nor consume the root;
- one immutable CPU snapshot for the file and at most one deeper transient
  snapshot for the active pair;
- no durable store, trie, paged KV, or prefix-cache insertion.

`QWEN_CONCURRENCY_FILE_ROOT_FANOUT=0` restores pair-local behavior.

Memory admission prices two Metal sessions and prefill scratch against Metal
working-set headroom. The retained CPU root plus the largest possible deeper
pair snapshot are separately included in the process budget before allocation.
If the root extension is denied, execution falls back to ordinary B2 admission.

## Product Packet

Model and host:

- `Qwen3.5-0.8B-Q4_K_M.gguf`
- Apple M4 Max
- four realistic prompts copied from the ring0 qualitative corpus
- prompt lengths `6,482`, `6,561`, `6,561`, and `6,561` tokens
- greedy generation, eight requested tokens, chunk 1,024

The control set `QWEN_CONCURRENCY_FILE_ROOT_FANOUT=0`; the candidate used the
default. Two serialized comparisons reversed the execution order. Completion
JSONL was byte-identical across controls, candidates, and repeats.

| Order | Control wall | Candidate wall | Speedup |
|---|---:|---:|---:|
| control then candidate | `2.59 s` | `2.09 s` | `1.239x` |
| candidate then control | `2.59 s` | `2.10 s` | `1.233x` |
| reviewed final code | `2.59 s` | `2.11 s` | `1.227x` |

Representative internal totals from the first pair:

| Metric | Control | Candidate |
|---|---:|---:|
| Pair preparation | `2,256.542 ms` | `952.991 ms` |
| Pair-local prefix tokens evaluated | `12,705` | `417` |
| Pair-local prefix prefill | `1,604.786 ms` | `314.356 ms` |
| Private suffix prefill | `606.807 ms` | `610.295 ms` |
| File-root restores | `0` | `3` |
| File-root restore wall | `0 ms` | `15.845 ms` |

The candidate evaluates and captures the 6,144-token root once before the pair
loop (`775.039 ms` prefill, `25.245 ms` snapshot), then avoids one complete root
evaluation across the two pairs. The second pair retains its deeper 6,561-token
boundary by replaying only 417 tokens above the root and capturing a transient
pair checkpoint.

## Correctness And Review

- completion output is byte-identical with the rollback arm;
- model-derived snapshot geometry is checked against allocated session geometry
  in debug builds and uses the same architecture and KV-dtype policy;
- snapshot estimates include root logits and charge the simultaneous root plus
  deepest pair payload;
- source/target restore frontiers and exact logits are validated before suffix
  prefill;
- actual savings telemetry is emitted only after successful file completion;
- 16 concurrency tests, runtime sizing tests, the split CPU/Metal admission test,
  strict clippy, formatting, and diff hygiene pass;
- the complete release workspace suite passes every implementation-related test;
  its sole failure is the pre-existing fixture census requiring a base GGUF with
  EOS 248044, while the current `~/models` inventory contains only instruct
  fixtures declaring 248046;
- final adversarial review verdict: `GO`.

## Decision

Promote bounded file-root fanout by default for Qwen B2. This removes repeated
prompt computation rather than merely reshaping it, composes with pair affinity
and concurrent decode, and degrades to the existing path when admission or
qualification rejects it.

Do not generalize this into a durable radix cache inside the B2 executor. The
next reuse layer should share the same checkpoint substrate across execution
widths rather than turning this product-local root into a second cache index.
