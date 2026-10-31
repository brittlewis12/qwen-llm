# Routed Gate/Up Qualification Tranche

This tranche changes tests and generated caller documentation only. It does
not change production kernels, arithmetic, defaults, allocation, decode or
native benchmark code. The parent owns independent review and all GPU runs.

## Source Dependencies

- Strict A metadata follow-up `07aacf6a` is imported with provenance, retaining
  all twelve KDA CPU contracts and the shared `normalized_half` adaptation.
- Maintained-main `266c2825` is merged non-destructively: the stale IQ1_M CPU
  expectations are repaired using real native capability evidence; no
  production changes enter through this dependency.
- The parent's forthcoming A-owned heated latency module must be imported
  before the single shared release test-binary build. This tranche does not
  duplicate A's latency implementation or rebuild the native benchmark.
- Historical `tensor_m5_transfers.rs` and the old packet bodies are unchanged.
  The parent quality file gains only module declarations.

## Common Controls And Admission

Both routed tests require a canonical release library test binary, actual M5
tensor support, normal production lease, the pinned actual six-shard Q4 model,
and a create-new mode-0600 JSONL report. Set:

```text
GLM53_GGUF=<standard first-shard path>
QWEN_METAL_LEASE_WAIT=1
QWEN_EXPERT_TENSOR_TILE=1
QWEN_GLM53_TENSOR_GATE_UP=1
QWEN_GLM53_KDA_HALF_INPUTS=0
QWEN_GLM53_DENSE_Q8_WIDE=1
QWEN_GLM53_TENSOR_Q5_DOWN=1
QWEN_MATMAT_Q6_K_N64=1
QWEN_MATMAT_Q6_K_N64_MIN_N=1024
```

Global routed ON owns identical admitted 16 MiB `normalized_half` storage in
both arms, with workspace R2048. Only the routed gate/up TLS false/true value
differs during prefill. Core dense/wide/Q5 controls stay ON and KDA execution
stays OFF. Ordinary forward continuation is outside all prefill scopes.

Before weight loading, both tests verify primary metadata SHA, shard names and
sizes, exact named expert assignments (Q6 down 11/12/44, Q5 down elsewhere,
Q5 gate/up only at 11), all 24 serialized Q8 MLA k/v banks including MTP 45,
and every KDA projection matrix. The latter are all Q8_0: q/k/v `[4096,8192]`,
f_a/g_a `[4096,128]`, beta `[4096,64]`, f_b/g_b `[128,8192]`, and output
`[8192,4096]`, across 34 KDA blocks. Rank expansions must not be inferred as
Q6 from synthetic fixture defaults. The existing model binder also validates
the remaining KDA convolution/vector roles. Retained shard stamps are checked
again before completion; weight payload hashes are not reverified.

Normal full-model/session preflight precedes weight loading and uses
`evaluate_metal_memory_admission_with_cpu_bytes(ledger.peak_bytes(), cpu, 0,
signals, true)`. CPU availability is checked separately as well as overall
physical/process/working-set capacity. Every fresh session repeats normal
CPU-aware admission with the same reserve. No admission exception, expanded
weight bank, or full-weight F32 oracle is introduced.

The private reports contain hashes, metric counts, parsed tool call metadata
and sanitized shard stamps, not home directories, hostnames, prompts or raw
token IDs. Do not publish private reports as public results. Source, fixture,
strict-scorer and binary hashes identify the exact protocol.

## Extended Quality

Exact test:

```text
glm5_next_metal::tests::quality::tensor_prefill::routed_qualification::m5_routed_extended_quality_and_strict_tools
```

Use `MTL_DEBUG_LAYER=1` and a fresh `GLM53_ROUTED_EXTENDED_REPORT`. The frozen
historical selection is all 22 documents at their maximum registered prefix:
8 prose, 8 code, 6 long, each with 64 teacher-forced targets. That is 1,408
targets per arm, 2,816 full-vocabulary finite rows, and 44 document sessions.
One immutable CURRENT 64-row F32 baseline is retained per document, not all
documents at once. Per-document and aggregate NLL, KL(CURRENT || arm), top1
correctness and agreement require every expected row before completion.

The existing T09-T12 tool fixtures run greedy and sampled seed 11, producing
eight sequences per arm (16 tool sessions), with identical frozen prompt IDs
and seed policy in both arms. Prompt lengths are 1455/1455/5008/5008; the
1455-row controls do not select routed gate/up, whereas long tasks contain
two eligible 2048-row chunks. The strict Rust scorer matches the historical
Python scorer's T09-T12 semantics: exact one-call and observation stop,
successful parsing, normalized exact city or numeric amount/currency checks.
It does not use the old lenient first-call diagnostic. Serialized parse/stop/
call bytes, decoded UTF-8 bytes and sampled-ID equality are compared separately,
descriptively; raw IDs and decoded prompt text are not emitted.

All sessions have capacity 5264, workspace 2048, and equal ledgers. CPU reserve
includes one 64-row full-vocabulary baseline, live logits/readback, two F64 KL
vectors, conservative inherited census/report allowances, parsed fixture and
metadata records, tokenizer construction/retention/decode, sampler vectors
and histogram, and two retained tool sequences with parsing/serialization.
The tokenizer's actual maximum decoded piece size bounds tool decode scratch
before weights are loaded. There is no census or profiler in this test.

Completion requires 44 document arms, 2,816 finite rows, 16 tool generations,
60 sessions, clear observers and unchanged stamps. No measured quality
threshold, noninferiority claim or automatic promotion is encoded.

The parent plans a fresh external 15-minute outer limit. It must be enforced
by the runner: a synchronous GPU call cannot be interrupted by checks placed
between calls. Incomplete/terminated reports have no valid complete event.

## Heated Latency

Exact test:

```text
glm5_next_metal::tests::quality::tensor_prefill::routed_qualification::latency::m5_routed_heated_plain_latency
```

Use `MTL_DEBUG_LAYER=0` and a fresh `GLM53_ROUTED_HEATED_REPORT`. One model load,
capacity 4416 and workspace 2048 are shared across shapes 512/1024/2048/4096.
Each shape first warms both arms, then runs eight fresh-session samples:
CURRENT/GATEUP/GATEUP/CURRENT followed by GATEUP/CURRENT/CURRENT/GATEUP.
That is 32 measured samples, eight excluded warms and 40 fresh sessions.

Each timer includes ordinary prefill with full GPU wait and normal logits
readback, not session allocation/drop. During each continuous eight-sample
block there is no validation/oracle work, hashing, JSON, census, profiling,
diagnostic capture or other intervention. Fresh admitted allocation and
session drop are outside individual timers. Only returned logits and passive
sample metadata are retained; all checks, hashes and report work follow the
whole eight-sample block. CPU reserve prices eight retained full-vocabulary
rows plus two live/readback rows and metadata/report space. Equal ledgers and
observed buffer charges are verified after each block. The unchanged 512-row
dispatch control must remain bitwise equal; measured latency has no threshold.

Pooled and ABBA/BAAB contrasts and within-arm first/second-block drift are
descriptive, not a gate based on prior observed performance. No oracle work
or diagnostic instrumentation is inserted between timed samples. The parent
plans a fresh external four-minute outer limit, with the same synchronous-call
limitation as the quality test.

## Earlier Evidence And Replay Hygiene

The parent reports successful tiny-composition and three-document numerical
execution for frozen `6991cbde`, followed by a separate native conversion
experiment. Its original owned C/A model preloads charged Metal+CPU to total
budgets, and fresh sessions checked CPU-aware admission, but preload itself
did not separately validate CPU availability. This tranche fixes those owned
preloads and both new qualification paths before any new weight load. This
is a replay-hygiene correction, not a reinterpretation of already observed
numerical rows or a claim that the healthy M5 encountered an actual failure.
The historical transfer helper remains unchanged.

Commit source and complete CPU/type/format/generator checks before the one
authorized `cargo test --locked --offline --release -p qwen-llm --lib --no-run`
build, and only after importing the A-owned latency dependency and receiving
the parent's independent review. No new native build, GPU run, remote change,
main merge or default promotion is delegated here.
