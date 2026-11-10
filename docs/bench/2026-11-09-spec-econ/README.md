# Speculative-decoding economics on GLM-5.3-Flash, 2026-11-09

Read-only research lane on the M5 Ultra. No engine, kernel, serve, or drafter
change. The question: for coding-agent traffic, does any available speculative
decoding method pay on these machines, at the sampling agents actually use and
at the batch width they actually produce?

Raw probe output: the four `*.jsonl` files beside this note, the batch-path
`batch-rows.jsonl`, and the NextN artifacts `nextn-summary.json`,
`nextn-quality.json` and `nextn-probe-report.json`. Probe sources:
`crates/qwen-llm/examples/spec_verify_cost.rs`,
`spec_dflash_accept.rs`, `spec_batch_rows.rs` and `spec_nextn_accept.rs`;
reductions: `scripts/research/spec-econ/analyze.py`, `nextn_quality.py` and
`nextn_project.py`; driver: `scripts/research/spec-econ/glm_campaign.sh`.

## Bottom line

For GLM-5.3-Flash at the real three-agent workload, **none of MTP/NextN,
DFlash2, DSpark, or DSpine pays with the current verifier.** This is not mainly
an acceptance problem:

- The packed all-row verifier costs **1.76-6.6x one serial token** at 48K and
  never gets cheaper than **12.6 ms/row**; the three-member batch already runs at
  **9.37 ms/row**.
- Partial acceptance adds a serial replay in `commit_retained_inputs`; measured
  DFlash2 packets spend **59-80 ms** there, as much as half the verify cost.
- At B=3, even a **free drafter with 100% acceptance on an 8-row block** cannot
  beat the measured batch round, because verifying 8 rows (123 ms) emits at most
  8 tokens while a 28.1 ms round emits 3.
- Measured GLM DFlash2 is **0.36-0.45x** serial on frozen code at 48K, even
  though code acceptance (39-56%) is far above the October gameplay 14%.
- The native NextN one-step break-even acceptance is **0.84** against the
  **0.716** overlap recorded on an 11K agent prompt. On 48K code the measured
  one-step acceptance is **0.799**, so acceptance is not the blocker.
- With that measured acceptance, a measured **1.72 ms** head and a measured
  **1.20 ms/row** state ring, the batch-path verifier gives **1.20-1.31x at
  B=1** (K=1/K=2, including the draft draw), but still **0.65-0.69x at B=3**.
  The call is **no-build for aggregate throughput**; a solo-only K=1 proof is
  the optional product case (final section).

DSpine has no GLM drafter. DSpark has only third-party GLM previews, and its
scheduler idea does not change the verifier cost. The highest-value next step is
a cheaper verifier (and cheap partial commit), not another drafter.

## What agents actually send

The wire path accepts `temperature`, `top_p`, `max_output_tokens`, and
`x_qwen.{seed,top_k,min_p}`; there are no repetition penalties. opencode's
captured request bodies contain **no sampling fields**, so `qwen serve` applies
the GLM release defaults from `generation_config.json`: **temperature 1.0,
top_k 0, top_p 0.95, min_p 0, fresh seed** (`crates/qwen-llm/src/sampling.rs:70`,
`crates/qwen-llm/src/glm5_next_chat.rs:37-38`). The repo's own throughput
harness sends **temperature 0, top_p 1** (`scripts/bench/serve_realistic.py`),
so the existing greedy standings are not the opencode default. This note
measures both.

## Method

Two examples, run under one canonical Metal lease, no API validation:

- `spec_verify_cost`: prefills frozen GLM quality-v1 code tokens
  (`scripts/reference/glm53/quality-v1.json`, document 8) to a depth, captures
  one session, then times `begin_append` + `verify_packed` + all-row CPU
  readback and one full transaction rollback. It also times an independent
  `Glm5NextBatch` round over three prefilled sessions (distinct code documents).
- `spec_dflash_accept`: runs the production `Glm5NextDFlashRunner` with the
  incoai `GLM-5.3-Flash-DFlash2-Q8_0` head (block 8, target layers
  5/14/24/33/42) over 48K code prefixes, 96 tokens, at the agent default
  (T=1, top_p 0.95) and greedy, on two code documents.

Depth 49152, four timed reps for native costs, excluded warm transaction.
Thermal pressure could not be read on this host (`notifyutil` and `pmset -g
therm` both fail with OS errors); the host reports level `9` only because the
probe parses the failure text. Sample spread was small and the serial controls
were stable, but this is an unverified-thermal qualification.

## Measured native costs, 48K, M5 Ultra

One row = one ordinary forward including full-logit readback. Verify includes
checkpoint copy and all-row readback.

| rows m | V(m) ms | V/S | begin ms | forward ms | ms/row |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (serial) | 18.60 | 1.00 | — | — | 18.60 |
| 2 | 32.67 | 1.76 | 1.16 | 31.51 | 16.34 |
| 3 | 77.74 | 4.18 | 1.29 | 76.44 | 25.91 |
| 4 | 50.39 | 2.71 | 1.26 | 49.14 | **12.60** |
| 5 | 107.91 | 5.80 | 1.28 | 106.63 | 21.58 |
| 6 | 114.37 | 6.15 | 1.26 | 113.11 | 19.06 |
| 7 | 120.45 | 6.48 | 1.26 | 119.19 | 17.21 |
| 8 | 123.17 | 6.62 | 1.30 | 121.88 | 15.40 |
| 12 | 220.20 | 11.84 | 1.30 | 218.90 | 18.35 |
| 16 | 259.35 | 13.94 | 1.25 | 258.11 | 16.21 |

The curve is **not monotonic**: odd widths (3, 5, 7) and widths above 8 hit
expensive shapes. Four rows is the cheapest per-row verify shape. Three
independent sequences in one batch round cost **28.12 ms** = **9.37 ms/row**,
so the verifier is 1.34x the batch per row even at its best shape, before
acceptance waste.

An 8K control (`cost-8k.jsonl`) matches the 2026-11-06 diagnostic
(8 rows 118.7 vs 120.7 ms): the November batched/few-row work improved the
*independent batch*, **not** the packed verifier.

## Measured DFlash2 end to end, 48K

`emitted` includes the initial token. Decode excludes the 46.7-46.9 s prefill.
`B=3 ratio` is aggregate tokens/ms relative to the 28.12 ms three-sequence
round when a verify blocks batching (the conservative model).

| arm | block | alpha/draft | emitted/packet | decode tok/s | vs serial | B=3 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T=1, doc 8 | 8 | 0.387 | 3.10 | 19.5 | 0.36x | 0.18x |
| T=0, doc 8 | 8 | 0.405 | 3.56 | 19.8 | 0.37x | 0.19x |
| T=1, doc 9 | 8 | 0.563 | 3.84 | 24.0 | 0.45x | 0.22x |

Per speculative packet (mean): verify **117.6-121.9 ms**, commit replay
**59.1-79.7 ms**, draft **6.6-7.4 ms**. The verifier processes the full 8-row
block regardless of how many drafts survive; the retained-prefix replay then
re-runs the accepted prefix serially after rollback. Both are large and mostly
independent of acceptance. Code acceptance is 2.8-4.0x the October gameplay
14.1% and the result is still a large slowdown.

## Economics

Let S be serial ms/token, V(m) verify ms for m = k+1 rows, D draft cost, and
E+1 expected emitted tokens. At B=1 the speedup is `(E+1)S/(D+V(m))`. At B=n,
with warm-up/cleanup ignored and a verify serialized against the batch, it is
`(E+1)C(n)/(n(D+V(m)))`.

- **NextN one-step** (recorded alpha 0.715891, one-step only; head cost excluded
  but budgeted at 1.5 ms): break-even alpha = `(V2 + 1.5)/S - 1 = 0.837`.
  At alpha 0.716: **0.93x at B=1**, **0.47x at B=3** before replay. Adding the
  expected rejection replay makes B=1 worse.
- **DFlash2**: measured directly, **0.36-0.45x at B=1** and **0.18-0.22x at B=3**.
- **Perfect-acceptance bound**: a free 8-row block at 48K still emits at most
  8 tokens per 123 ms; a three-agent batch emits 3 per 28.1 ms. At B=3 the
  speculative route is **0.61x even at 100% acceptance**.
- **B=3 bar**: to beat 9.37 ms/row of emitted tokens a block must emit more
  tokens than rows verified. NextN's one-step already emits 86% of its rows but
  has only 2 rows; DFlash2 emits 39-48% of its 8 rows. Either way the fixed
  verifier cost wins.

These are native-cost economics. The served ledger shows only 65-85% of the
window is native decode and GLM has the largest unexplained residual, so any
delivered gain would be smaller still.

## Availability and requirements

| method | GLM-5.3-Flash | Flash-Next | DeepSeek V4.1 |
| --- | --- | --- | --- |
| native MTP/NextN | head present (block 45, 29 tensors, ~2.6 GiB); detected/inventoried but **never loaded or executed** (`glm5_next.rs:773-779`, `glm5_next_metal.rs:385-428`). Needs residency, a Metal MLA+MoE head graph, draft state, runner wiring. Verify/accept/rollback already exist. | `nextn` tensors are inventoried as `ignored_mtp` (`qwen4exp_loader.rs:102,145-159`); no execution. | not applicable |
| DFlash2 | **fully wired**; incoai Q8_0 head exists (1.25 GB). Measured negative above. | wired, opt-in; measured 0.289x at T=1 gameplay (21.4% acceptance). | not the shipped drafter |
| DSpark | **no official drafter**; third-party `RedHatAI/...dspark-preview` and a community artifact whose own card says it does not compose. Needs a new Markov/RNN + confidence-head runtime. | community drafter only; reported weaker than built-in MTP. | **inside the official checkpoint** as `mtp.0..2`, but this repo explicitly excludes `mtp` as unused (`deepseek_v41/gguf/resident_storage.rs`), and only the reference Python fixture contains a (never-called) DSpark forward. |
| DSpine | **no checkpoint** for GLM. | none. | none. |

DSpine is a DFlash-lineage parallel block drafter (block-causal attention plus
depth-wise predecessor-feature injection and a transition cache) with official
Qwen3-8B and Qwen3.8-27B heads. It is not a new way to avoid verification cost.
For GLM it has no drafter at all, so it cannot be explored without training one.
Its natural home here is the dense 27B, where DFlash2 already delivers 1.8x and
DSpine's published +10-28% acceptance over DFlash2 would be the next increment;
that needs a GGUF/Metal port of a new drafter architecture.

DSpark is a semi-autoregressive DFlash backbone plus a Markov/RNN head and a
confidence-scheduled variable verify length. Its transferable idea is the
scheduler, not the drafter. That scheduler can avoid verifying low-confidence
rows, but it cannot make V(4)=50 ms or V(8)=123 ms cheap, and the measured
alpha is not high enough for a shorter block to cross the B=3 bar either.

## Recommendation

1. **Do not build GLM NextN or port DSpine/DSpark for GLM now.** The measured
   economics are negative at both B=1 and B=3; the recorded NextN break-even is
   above its measured acceptance; DSpine has no GLM head; DSpark has no
   reliable GLM head.
2. **If speculation is pursued, build the verifier first.** The 2026-11-06
   proposal (dedicated 2-4-row Q6_K weight-decode reuse) targets the right
   thing, and the cheapest observed shape is 4 rows at 12.6 ms/row. The target
   is <9.37 ms/row, i.e. at least a 1.34x verifier improvement, just to reach
   batch parity. Also remove the serial retained-prefix replay: per-row
   checkpoints or a state splice would save the measured 59-80 ms/packet.
3. **Re-price after those two changes, and price against B=3, not B=1.** B=1 is
   <0.3% of GLM wall in the steady ledger; a B=1-only win is not an aggregate
   win. At B=3, even a perfect-acceptance 8-row block loses today; a viable
   policy needs both a cheap verifier and >~75% block acceptance, and should
   batch multi-row members rather than serialize per-agent verifies.
4. **If the goal is aggregate throughput for three GLM agents, keep reducing
   the B=3 round.** The measured GLM B=3/B=4 kernel cuts have near-1.0
   pass-through (the 32K B=4 stack gave +13.2% delivered), unlike speculation.

## Limits

- One model, one GPU, one depth, two frozen code documents for acceptance.
  Acceptance is text-dependent (0.387 vs 0.563 at the same T=1 setting); the
  three-point picture bounds direction, not a precise mean.
- NextN acceptance is the stored 2026-10-22 one-step overlap on an 11,105-token
  agent prompt, not a recursive block measurement; recursive acceptance would
  be lower, so the 0.93x B=1 estimate is an upper bound.
- The B=3 ratio assumes a verify blocks the batch. The current engine cannot
  merge multi-row members into one batch, so this is the relevant regime;
  a future multi-row batched verify would need its own measurement.
- Native-cost comparisons exclude prefill, transport, scheduler and the served
  GLM residual (11-18% of the window).
- Thermal state could not be read; timings are retained as measured on a host
  shared with other lanes, with stable sample spread.

## Follow-up: batch-path verify cost and replay design (measured)

The main result prices today's *packed* verifier. This follow-up prices the
same K+1 rows through the independent-batch kernel path, and prices the cheapest
way to remove the serial retained-prefix replay. Raw output:
`batch-rows.jsonl` (probe `crates/qwen-llm/examples/spec_batch_rows.rs`).

### Batch-path row cost, 48K

One 48K sequence was prefilled and snapshotted, then restored into independent
members; a `Glm5NextBatch` round priced 2/3/4 rows with distinct next tokens.

| rows | round ms | ms/row | vs packed V |
| ---: | ---: | ---: | ---: |
| 2 | 23.69 | 11.85 | 32.67 |
| 3 | 27.81 | 9.27 | 77.74 |
| 4 | 32.26 | 8.06 | 50.39 |

The distinct-document B=3 control was 28.12 ms, so the shared-snapshot proxy is
within 1.1% and the premise is confirmed: the batch kernels process K+1 rows at
23.7-32.3 ms, not 32.7-77.7 ms. This is a **conservative upper bound** on a
batch-kernel verifier: the members update separate KDA/MLA state and read the
prefix independently, while a causal verify serializes the KDA recurrence and
in-block attention. Add roughly 0.5-1 ms for that serialization.

### Replay design and price

`Glm5NextAppendCheckpoint` already copies exactly the state `rollback` restores:
KDA convolution/recurrent state plus the MLA pending ring. The append-only
KV/latent caches are visibility-gated, so they need no copy.

Measured compact state: **152,813,568 priced bytes (145.7 MiB)**, copy median
**1.20 ms** (about 127 GB/s). That is context- and width-independent.

**Cheapest design: a per-row state ring.** After each verify row, copy that
compact state into slot *i*. On partial acceptance *c*, restore slot *c* and
publish position instead of replaying *c* tokens.

| K drafts | extra copies | extra state | cost |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 153 MB | 1.2 ms |
| 2 | 2 | 306 MB | 2.4 ms |
| 7 | 7 | 1.07 GB | 8.4 ms |

If the KDA scan can store each row's state to the ring directly, the cost is
only the write bandwidth (same ~1.2 ms/row) with no extra command buffers. The
ring doubles the existing checkpoint for K=1. To compare: the current serial
replay is 59-80 ms/packet for DFlash2 and an expected 0.284 x 18.6 = **5.3 ms**
per NextN one-step round when the draft is rejected.

### Corrected single-agent projections

These use S=18.60 ms, the measured batch-path V(2)/V(3), the measured 1.20 ms
replay checkpoint per row, and a head cost of 1.5-2.36 ms. `alpha` is the
recorded 0.716 one-step overlap; K=2 is shown with that held constant and with
a skeptical 0.5 second-step acceptance.

| case | emitted/round | round ms | speedup |
| --- | ---: | ---: | ---: |
| K=1, head 1.5, checkpoint | 1.716 | 26.39 | **1.21x** |
| K=1, head 2.36, checkpoint | 1.716 | 27.25 | **1.17x** |
| K=1, replay free, head 1.5 | 1.716 | 25.19 | 1.27x |
| K=2, const alpha, head 1.5, checkpoint | 2.229 | 33.21 | **1.25x** |
| K=2, alpha2 0.5, head 1.5, checkpoint | 2.074 | 33.21 | **1.16x** |
| K=2, const alpha, head 2.36, checkpoint | 2.229 | 34.93 | 1.19x |
| K=2, const alpha, replay free, head 1.5 | 2.229 | 30.81 | 1.35x |

**Arithmetic verdict.** The B=K+1 premise is correct, and 1.27-1.35x is the
optimistic edge (no replay, 1.5 ms head, constant recursive acceptance). The
central estimate is **1.15-1.25x**, and the 1.4x end requires all of those
assumptions plus V at the very bottom of the measured range. The K=2 case is the
least secure: the 0.716 is a one-step conditional overlap, not a measured
second-step acceptance, and the recorded value was on an 11K agent prompt rather
than 48K code. Add the ~0.5-1 ms causal serialization penalty and the central
range drops another 3-4%.

### Adaptive concurrency

Speculate only at **B=1**. At B>=2 the batch is already 8.06-11.85 ms/row and a
verify emits fewer tokens than rows verified (NextN K=1 emits 1.716 per 2 rows),
so it is less token-efficient than batching. The steady ledger puts B=1 at
**0.27% of GLM wall**, so the aggregate gain is under 0.1%; the value is
solo-window responsiveness, where inter-token latency falls from 18.6 ms to
about 15.5-16 ms (~1.2x). TTFT remains prefill-dominated.

No production path was built. The prototype and records stay on
`research/spec-econ`.

## Follow-up: measured NextN head cost and T=1 acceptance (measured)

This section replaces the two unmeasured inputs in the preceding projection.
The head is the native block-45 NextN block extracted from the stock GGUF
(29 head tensors, 3.84 GB payload) and executed through the llama.cpp MTP
probe (`load_mtp=true`, `LLAMA_CONTEXT_TYPE_MTP`, history batch 512). The
capture is one 47,697-token quality-v1 code prefix (document 8) with 128
on-policy T=1 / top_p 0.95 / seed-42 scored rows; the probe emits one full
154,880-vocabulary row per position conditioned on the captured prefix. Raw
evidence: `nextn-probe-report.json`, `nextn-summary.json`,
`nextn-quality.json`; reductions `scripts/research/spec-econ/nextn_quality.py`
and `nextn_project.py`. The raw logits (~76 MiB draft + ~76 MiB target) and
the head-only GGUF stay in
`metagross:~/qwen-runs/2026-11-09-spec-econ/nextn/`.

`pmset -g therm` on this host reports no thermal warning level and no
performance warning level recorded; the probe did not enter level 2. Sample
spread was small. This is still a single host, single model, single document
measurement.

### Head cost at 48K

`full_head_step_ms` is process + synchronize + full-logit retrieval, excluding
input setup and output-file IO.

| step | ms |
| --- | ---: |
| first scored step (warm-up) | 214.81 |
| steps 2-6 (shape transition) | 2.75, 2.61, 2.67, 2.70, 2.65 |
| steady-state median (steps 2-128) | **1.722** |
| steady-state mean (steps 2-128) | **1.750** |
| 47,697-row head history init | **3,881.4** |

The head step alone is inside the old 1.5-2.36 ms budget. The draft draw is
separate: one `Sampler::diagnose` at T=1 / top_p 0.95 over the 154K row is
**0.953 ms** (target row 0.953, draft row 0.956; 64 timed iterations). The
production cost of one draft step is therefore **~2.68 ms** (head + draw), not
1.72 ms. The head also needs its own 48K history: **3.88 s** in a separate
512-row pass, plus its own KV/state. That cost is not in the per-step number
and must be amortized over the session.

### Acceptance at T=1 / top_p 0.95, 48K code

Filtered acceptance is `sum min(P,Q)` over the sampler-v1 top-p support; TV is
`1 - total variation` over the full softmax. Errors are row SEMs (128 rows).

| quantity | value |
| --- | ---: |
| one-step filtered | **0.7992 ± 0.0196** |
| one-step full-vocabulary TV | 0.7982 ± 0.0192 |
| greedy agreement | 0.7969 |
| second step, on-trajectory subset (90 rows) | **0.8142 ± 0.0214** |
| second step, chain all rows (127) | 0.7976 ± 0.0196 |
| per-row distribution: median / p10 / p90 | 0.889 / 0.438 / 1.000 |

The old **0.716** was a one-step filtered overlap on an 11K agent prompt. On
48K code the one-step is **0.80**, and the second step is also ~0.80. The
second-step on-trajectory value is the closest available conditional estimate:
it uses rows where the head's greedy first draft equals the captured token, so
the next row is conditioned on the accepted first token. The chain value holds
the first draft on the captured trajectory for all rows. Both bracket the
central estimate near 0.80, though the exact T=1 expectation over arbitrary
draft samples is not directly captured (the probe only conditions on the
captured trajectory).

**Accuracy.** Signed teacher-forced dNLL (head minus target) is **+0.337 nats**
mean (SEM 0.110), **+0.0034** median; KL(target || head) is **0.378** mean,
0.108 median; TV is 0.203. The mean is a heavy tail: most rows are near-exact
and a few are poor. This is drafter quality, not a correctness failure.
Speculative output is distribution-exact under standard rejection sampling
regardless of head quality; the exactness test must compare the **final
sampler** at T=1 / top_p 0.95 against the non-speculative sampler, not the head
against the target. The head's dNLL does not gate the method, it only caps
acceptance.

### Economics with measured inputs

`S=18.60 ms` serial native forward, `s=0.953 ms` sampler draw, `head=1.722 ms`,
ring `1.201 ms/row`, `V(2)/V(3)=23.69/27.81 ms` on the batch-path proxy,
`a1=0.7992`, `a2=0.8142`. The spec round charges one target draw and one draft
draw + head per draft step; the non-spec baseline charges one target draw per
token. `emitted = 1 + a1` for K=1 and `1 + a1 + a1*a2` for K=2.

| case | round ms | emitted | speedup |
| --- | ---: | ---: | ---: |
| K=1, batch path | 28.52 | 1.799 | **1.233x** |
| K=1, batch path +0.75 ms causal | 29.27 | 1.799 | **1.202x** |
| K=2, batch path | 36.52 | 2.450 | **1.312x** |
| K=2, batch path +0.75 ms causal | 37.27 | 2.450 | **1.285x** |
| K=1, today's packed verifier | 37.50 | 1.799 | **0.938x** |
| K=2, today's packed verifier | 86.45 | 2.450 | **0.554x** |

The measured acceptance makes the method viable **only** through the
batch-path verifier; the packed verifier still loses because `V(2)=32.67` and
`V(3)=77.74`. The single-agent central estimate is **1.20-1.31x**, with K=2 at
the top and the causal-serialization penalty at the bottom. This is below the
old 1.35-1.4x optimistic edge because the draft draw is now charged and the
second-step acceptance is measured at 0.81, not assumed at 0.716.

At B=3, with a speculative verify blocking the batch, the ratio is
**0.645x (K=1)** and **0.686x (K=2)** including sampling. Subtracting 0.8 or
2.5 ms of batched fixed cost from both the verifier and the B=3 round does not
change the direction (0.646-0.683x). Speculation is still solo-only.

The head-history pass adds a break-even length. The non-spec token costs
`S+s=19.55 ms`. Before the causal penalty, K=1 spec costs 15.85 ms/token
(saving 3.70 ms, **~1,050 emitted tokens**) and K=2 costs 14.91 ms/token
(saving 4.65 ms, **~835 tokens**). With the +0.75 ms estimate, those become
**~1,180** and **~895 tokens**. If the head state is fused into the target
prefill or reused across turns, the break-even is lower; if it is a separate
pass per request, short turns lose.

### Build/no-build

**No-build for aggregate throughput.** The acceptance objection is gone
(0.80 vs the old 0.72), but the measured best case is 1.23-1.31x for a single
agent, the aggregate value is under 0.1% because B=1 is 0.27% of GLM wall, and
the same policy is 0.65-0.69x at B=3. The build is not small: head residency
and a native block-45/nextn graph (3.84 GB payload, about 2.6 GiB incremental
over the already-resident output/embedding tensors), a causal 2-row
verifier through the batch kernels (23.69 ms is a proxy, not a prototype), a
per-row target-state ring (1.20 ms and 153 MB per row) plus a head-state
rollback ring or replay, the adaptive `active_sequences==1` gate, and head
history integration (3.88 s per 48K context). Rough cost is **2-3
engineer-weeks**. The same effort on the B=3/B=4 GLM kernels has near-1.0
delivered pass-through and helps all three agents.

**Optional solo proof.** If Britt wants solo responsiveness, build the
smallest thing that can prove it: K=1 only, B=1 only, native head, a causal
2-row verify through the batch-path kernels, a per-row target-state ring and a
head-state ring (replay is the fallback if either ring is not exact), and the
`active_sequences==1` gate. Success bar: **>=1.15x delivered inter-token at
48K on frozen code**, distribution-exact by fixed-seed replay against the
non-speculative sampler at the repo's dNLL/KL noise floor, with the 3.88 s
head prefill amortized in the serve session. Do not build K=2 or any
multi-agent speculation until K=1 clears. If the head prefill remains a
separate 3.88 s pass, the proof must also show the expected generation length
is above the ~1,050-token break-even.

**Sensitivity to the batched fixed cost.** If the other lane's **2-3 ms**
batched fixed cost is removed, subtract it from the verifier and from the
non-speculative batch round. With the same +0.75 ms causal penalty, solo K=1
becomes **1.29-1.34x**, K=2 **1.36-1.40x**, and the head-history break-even
falls to roughly **700-900 emitted tokens**. That flips the solo call to
**build the K=1 proof** if solo latency is a product goal. It does **not**
change B=3: the non-spec batch gets the same saving, so the adaptive ratio
stays 0.65-0.68x. One caveat: GLM's own
batch fit implies only **~0.8 ms** of removable B=1 overhead
(`15.13 + 4.28 = 19.41` vs 18.60 solo), so the 2-3 ms result may not transfer
1:1; with 0.8 ms, solo is 1.24-1.31x including the causal penalty.

### Where the other methods sit

- **DFlash2:** measured **0.36-0.45x** solo on the same 48K code, because an
  8-row block emits only 3-4 tokens and pays a 59-80 ms retained-prefix replay.
  NextN's measured 0.80 one-step over 2 rows is the better GLM drafter if any
  spec path is built; DFlash2 remains a 27B tool, not a GLM one.
- **DSpark:** no official GLM head; V4.1 ships `mtp.0..2` but this repo
  excludes `mtp`, and Flash-Next inventories `nextn` as `ignored_mtp`. Its
  variable-length confidence scheduler could shorten NextN's K, but it does
  not change the verifier cost; not a near-term GLM option.
- **DSpine:** no GLM checkpoint at all (Qwen3-8B / Qwen3.8-27B only). It is a
  DFlash-lineage block drafter, so on GLM it cannot run, and on the dense 27B
  it is an incremental acceptance gain over the already-banked DFlash2, not a
  NextN competitor.

### Limits

- One document, 128 rows; acceptance is text-dependent (per-row p10 0.438,
  p90 1.000). The mean is 0.80, but a different code document could move it by
  a few points.
- The head ran through llama.cpp's MTP path, not the Rust engine. The 1.72 ms
  and 3.88 s are native-library numbers; a Rust implementation must reproduce
  them, and the head has no engine residency yet.
- The 23.69/27.81 ms verifier is the independent-batch proxy. A causal verify
  serializes KDA and in-block attention; the 0.75 ms penalty is an estimate,
  not a measurement.
- The draft draw is measured with the current CPU sampler (0.953 ms). A fused
  or faster sampler would recover a few percent.
- The head-state rollback ring is not priced; the target ring is 1.20 ms/row.
  Head history is a separate 3.88 s pass in this probe.
- No production path was built; all of the above is measurement and arithmetic.
