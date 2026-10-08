# GLM MLA grouped-tail screen

Decision: **retain the experiment, leave production selection off**. Whole
target latency improves at useful ragged widths, but deterministic output
differences on the code corpus need numerical attribution before promotion.
No new bit-equality requirement or numerical threshold is introduced, and the
incumbent Fast path is not assumed to be ground truth.

## Hypothesis and implementation

GLM MLA key absorption and value expansion use grouped Q8_0 matrices at each
of 11 layers. Shapes are K256/M512/G64 and K512/M256/G64, token-major F32
activations. Fast executes whole 128-token blocks with an F32 grouped matrix
kernel, then one grouped GEMV per remaining token. Exact uses GEMV throughout.

Candidate B preserves the old full-128 prefix, handles complete eight-token
tail fragments in one guarded matrix dispatch, then retains GEMV for 0-7 rows.
It preserves the existing matrix kernel's F32 dequantization, activations,
accumulation and K traversal. This is not a half-staging optimization. The
SIMD-uniform guards cover loads, MMA and stores; all staging barriers remain
unconditional. The old full-128 implementation is unchanged.

No padding buffers, production scratch or context limits are added. Checked
bindings cover physical ranges, offsets, alignment, aliasing, narrowing, device
identity and SIMD32/TG128/4096-byte dynamic TGM capacity. The primitive and shader
are compiled; only a cfg(test) GLM override selects it. `None` and explicit false
retain production; Exact and the Exact-stage ablation remain excluded.

At N127 this replaces 2,794 grouped GEMV dispatches with 176 total dispatches
across both directions and all layers. The incumbent sequence already shares
one command buffer: this does not remove thousands of command submissions.

## Retained measurement

Release tests, normal production Metal lease and ordinary memory admission;
no API validation during timing. Production router and IQ3_S expert-down
policies stay enabled. Warm A/B then B1/A1/A2/B2. All attempts, source/metallib
hashes, artifact descriptors/stamps, exact token streams and errors are flushed
to the JSONLs. Allocation, diagnostic readback, hashing and continuations are
outside timing. Whole-prefill wall time includes ordinary status checks and
logit readback; GPU time uses completed-command timestamps.

`leaf-screen.jsonl`: five ordinary setup prefills, then 60 leaf commands.
Actual block43 inputs survive block44 KDA in separate packed scratch; both
directions are replayed independently without capture copies. One CPU output
reference is admitted/touched before warmup, with bounded 1 MiB readback.

At N32/64/127, the key path saves 81.69/89.71/86.03% leaf GPU time and value
63.88/80.48/82.22%. N8 is mixed: key saves 32.77%, value regresses 29.34%.
These are hot single-layer replays, not all-layer attribution. Finite relative
L2 differences are roughly 3e-7 to 7e-7; this only covers qualification inputs
at block43, not code inputs or earlier layers. N128 has unchanged arithmetic.

`whole-screen.jsonl`: two fixed streams (qualification and Rust code review),
108 target prefills including warmups plus 12 separately timed incumbent
4096-token prefix setups. Every timing/topology/admission check succeeds.

| Target rows | Qualification GPU saving | Code-review GPU saving |
| ---: | ---: | ---: |
| 8 | 0.43% / 1.04 ms | 0.86% / 2.15 ms |
| 32 | 3.07% / 15.14 ms | 3.36% / 18.00 ms |
| 64 | 4.21% / 30.75 ms | 6.02% / 50.12 ms |
| 127 | 5.53% / 64.36 ms | 5.51% / 64.28 ms |
| 255 | 3.98% / 71.26 ms | 4.22% / 70.63 ms |
| 127 after 4096 | 6.76% / 77.96 ms | 6.14% / 83.21 ms |

Both measured pairs and ordinary wall support N32/64/127/255 and deep127.
N8 is not a convincing promotion target. N128/129/512 have zero substitutions
and are unchanged-path controls, with sub-1% mean timing variation. This
candidate does not fix the N129 cliff. Deep percentages describe the suffix
only, excluding its 4096-token setup. Aligned 512-row and fresh4K execution
have no eligible tails and therefore no claimed direct speedup.

## Numerical result and limits

All outputs are finite. Every A1/A2 output hash matches, as does every B1/B2
hash; the differences between variants are reproducible, not repeat noise.
Deep-prefix endpoint hashes agree across all six arms. This is not full
persistent-state equivalence. Four common teacher-forced tokens are checked
at N32/127/255 and deep127, not every width.

Code N127 reaches KL(B||A)=0.087470 and KL(A||B)=0.075322 at the prefill
endpoint. Its first continuation changes the top token from A=5588 to B=18154:
A-side regret for B's choice is 0.017616; B-side regret for A's choice is
0.353165. Neither is ground-truth quality. The summary's two flips count the
same event twice through opposite comparisons, not two independent positions.

Code N255 reaches uncentered logit relative L2 0.40209 and max absolute 4.2605
on continuation1; that event's bidirectional KL is about 0.00232. The cell's
maximum KL, 0.01901, occurs at continuation2. Uncentered logit L2 is sensitive
to common shifts and must not be presented as a probability-error percentage.

Comparison orientation is recorded: A1 compares against retained B1; A2 and
B2 compare against A1. The summarizer reports maximum bidirectional KL and
maximum regret on either side, not a single incumbent-oriented quality score.

GEMV applies block scale after quantized-product accumulation and uses SIMD
reductions; matrix execution scales reconstructed weights first and changes
reduction order. Rounding amplification and downstream route changes are
plausible, **not established** by this packet. In particular, route changes
could contribute to whole timing differences, so those savings cannot yet be
assigned entirely to the changed MLA dispatches.

## Next bounded work

Keep this candidate off. For code N127, compare surviving per-layer routes
before continuation and locate the first amplified activation difference.
Replay the implicated direction on identical code inputs against a small F64
dot-product reference; compare both arithmetic paths rather than treating A
as the oracle. Native Exact on the same case can add numerical context, but
is not itself a task-quality verdict. Do not run another broad timing campaign
before localizing this.

A smaller alternative is a token-axis sibling of existing grouped GEMV:
keep the same per-dot body, scale placement and reductions, and consolidate
the token loop into one grid. That adds no explicit weight reuse; it probes
dispatch consolidation and scheduling/cache effects. One representative leaf
is enough to decide whether to pursue it. Arithmetic preservation would be a
specific implementation contract, not a general bitwise model-quality gate.

Main next workstream: native IQ capacity in `docs/IQ-QUANT-NATIVE-CAPACITY.md`.
Flash's existing indexed down kernels versus existing IQ4_NL N16 retile remain
a separate short-prefill candidate; neither has been measured by this packet.

## Verification and reproduction

- Four primitive tests pass under Metal API validation, including every
  multiple of eight through128 at both real geometries, CPU/GEMV reference,
  full128 output-prefix comparison, exact physical tails, offset/subview
  invariance, guards and rejected unsafe bindings.
- Three scoped-policy tests pass: default/Exact exclusion, full-prefix and
  remainder dispatch counts, nested/unwind/thread restoration.
- Timestamp/expected-topology CPU test passes. No numerical promotion
  verdict is implied by successful diagnostic completion.
- Workspace check including tests, release non-test check, release test build,
  formatting and whitespace checks pass.
- Independent adversarial review finds no concrete kernel/harness blocker
  and recommends retaining the experiment off pending numerical attribution.

Build with `cargo test -p qwen-llm --release --offline -j2 --lib --no-run`.
Select `glm5_next_metal::packed::mla_tail::mla_tail_packet` with
`--ignored --exact --nocapture --test-threads=1`, `QWEN_METAL_LEASE_WAIT=1`
and a new `GLM53_MLA_TAIL_OUT`. Default is leaf; `GLM53_MLA_TAIL_FULL=1`
selects whole mode. Unset `MTL_DEBUG_LAYER` for timing. The test acquires
the normal lease; the fixture optionally uses `GLM53_GGUF`.
Summarize with `uv run summarize.py <packet.jsonl> ...`.

Implementation jams: cx `01a11528-973c-7e40-ba8c-d4bece18c93b` and
`01a11893-3352-79d3-8d8d-8343c0d8f293`; independent reviewer
`01a118ca-c0f6-7c00-a0dd-23ae962883a5`. Review added all-fragment/full-prefix
coverage before measurements and corrected interpretation of KL/regret and
duplicate flip counts afterward. Raw attempts remain unchanged.
