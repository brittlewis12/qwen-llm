# Native short-packet cost: GLM-Flash and Flash-Next

Target-only follow-up to the [negative native DFlash rollout](../2026-10-20-dflash-native-targets/README.md).
GLM-5.3-Flash now has a bounded production small-packet policy. Flash-Next HC
changes remain test-only. Neither result establishes an end-to-end speculative
decoding speedup. [results.json](results.json) retains measured pairs, ablations,
correctness summaries, source hashes, and input-transform provenance without
private text, paths, or token IDs.

## Measurement contract

- Targets: GLM-5.3-Flash UD-IQ3_XXS and Qwen3.8-Flash-Next Q3_K_XL.
- Apple M4 Max, 128 GiB; Metal API validation enabled, production lease and memory admission.
- Native target only: no drafter. Depths 64/2179/8192, widths 1/2/4/8;
  variant timing runs use depth 2179. Native tokenization without special tokens;
  first 14 saved message contents joined by two newlines, not a chat rendering.
- A/B/C/D are serial N forwards, packed-last, all-row verification, and all-row
  verification plus five DFlash feature captures. Each arm warms once, then runs
  forward/reverse order twice total. Tables show arithmetic mean wall milliseconds.
- Identical base snapshots restore outside timing; prefill, validation and held
  continuation are excluded. Serial and verification include all requested logits
  readback; packed-last includes only one head. Feature readback is not charged.
- GLM production correctness compares against same-width legacy execution in the
  same executable, using legacy-prefix snapshots. Older timings are cross-build;
  final depth2179 runs share a binary but are sequential processes, not ABBA.
- Separate packed-last profiles perturb encoder boundaries. Flash detailed labels
  cover layers 5 and 7 only; nested spans overlap and are not full-model sums.
  Two measured repeats on one source are not a statistical performance qualification.

## GLM: bounded production improvement

Final same-binary depth 2179, all-row verification versus serial N:

| Width | Baseline verify | Production verify | Baseline serial N | Production serial N |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.96 | 46.07 | 38.24 | 39.50 |
| 2 | 145.30 | 80.23 | 76.23 | 83.86 |
| 4 | 212.49 | 143.73 | 152.79 | 180.06 |
| 8 | 269.16 | 204.40 | 312.90 | 378.97 |

Later production serial controls drift **+3..21%**: this supports a downward cost
trend, not a fixed 2.05x speedup. Earlier production verify2/4/8 at depth2179 was
70.99/124.45/185.21 ms; at depths64/8192 it was 65.33/114.27/168.46 and
72.29/128.09/182.94. Thus observed verify2 is 71..80 ms at depth2179, only **4..9%**
below its same-run serial-N control. Target cost remains the gate; width-one
packed still loses to scalar. JSON preserves both timing series and their provenance.

Only supported **Fast/Half** packed paths change: half-operand experts and router
at rows 1..8, Q8 few-row projections at rows 2..4. Scalar decode and chunks larger
than eight retain their dispatch. Small ragged prefill tails intentionally change.
Production: `ce075a09`; half primitives: `27c4a5ad`; Q8 preflight: `2e1bfb7e`.

Earlier width-two profiles put routed experts at 74.49 -> 26.72 ms,
router at 11.30 -> 0.98, KDA input projections at 23.04 -> 10.98, and MLA
projection at 15.95 -> 10.22. KDA recurrence stays about 0.67 ms.
Experts and projections, rather than recurrent attention math, explain the gain.

Selected diagnostic ablations at depth 2179 (verify ms, widths 2/4/8):

| Diagnostic | 2 | 4 | 8 |
| --- | ---: | ---: | ---: |
| Q8 few-row only | 126.90 | 186.10 | 243.97 |
| Broad narrow-projection substitution | 160.26 | 209.21 | 291.62 |
| Half experts + router | 87.61 | 137.89 | 173.96 |
| Half experts + router + Q8 at all tested widths | 71.23 | 126.76 | 185.14 |

The broad substitution loses; extending Q8 few-row to width eight also loses in
this screen. Faster exact-stage expert/router experiments are retained in JSON,
but change expert arithmetic and are not the precision-preserving production policy.
Across the earlier three depths and four widths, production versus same-width
legacy has maximum KL(reference || candidate) **3.0668014e-7**, zero top-1 flips
over 45 readout rows, and zero flips over 12 held continuations (maximum KL
2.0977598e-7). This is bounded fixture evidence, not bit identity or gameplay quality.

## Flash: HC cost isolated, promotion withheld

Depth 2179, all-row verification milliseconds:

| Width | Older baseline | Inject rows | Mix few-row | Combined | Combined-run serial N |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 214.27 | 150.62 | 189.15 | 122.48 | 78.87 |
| 4 | 228.73 | 167.52 | 203.31 | 138.71 | 157.02 |
| 8 | 214.19 | 157.41 | 208.15 | 149.17 | 314.38 |

The two-row case remains slower than serial; four and eight rows cross over in
this combined run. At depth 8192 the older baseline already shows the structural
floor: serial2 78.10, packed-last2 208.26, verify2 209.67 ms; serial8 318.62,
packed-last8 199.59, verify8 208.08 ms.

HC injection reduces K=10240 to four outputs: the generic packed matrix path
underutilizes the GPU at small N. Per-row existing F32 matvec removes most of that
cost. HC Q8 down/up use the existing few-row primitive, including K=320 supported
by its K%32 contract but excluded by the ordinary K%256 gate. Injection stays F32.
At N=2, sampled attention/FFN combine labels fall from 1.429/1.441 ms to
0.0568/0.0563; HC-pre labels fall from 0.633/0.638 to 0.0675/0.0657 ms.
These are sampled labels, not totals extrapolated over every layer.

Baseline A/A/A, after serial-N, after packed-last, and candidate A/B/A controls
are bit-stable for baseline logits, continuation and snapshot-owned persistent
state. The candidate difference is real, not a first-use or restore artifact.
At width two, route histograms agree through layer 19; row-one feature relative
RMS jumps 0.000364 -> 0.048787 at layer 20. Row zero jumps 0.000735 -> 0.057170
at layer 31. Width four reproduces these early crossings.
Raw router capture confirms top-k ordering against the actual scores: the first
switch has baseline cutoff margin **0.0002412796**, candidate margin 0.0000672340,
and maximum score change 0.0005250. Selected-softmax weight error versus F64 is
at most 2.89e-8. This supports routing sensitivity at observed cutoffs, not a
universal near-tie rule or a claim that the downstream difference is harmless.

### Continuing 64-token characterization

One prefix restore per arm; identical held corpus inputs, no per-packet reset.
Native scalar mean NLL is **3.2434875186**, a precision reference, not author truth.

| Width | Baseline NLL | Candidate NLL | KL(scalar || baseline) | KL(scalar || candidate) | Top-1 agreements baseline/candidate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.2550961692 | 3.2639408728 | 0.0108085092 | 0.0042575339 | 61/64, 62/64 |
| 4 | 3.2550961692 | 3.2639408728 | 0.0108085092 | 0.0042575339 | 61/64, 62/64 |
| 8 | 3.2571990448 | 3.2858727478 | 0.0071561540 | 0.0055425864 | 60/64, 62/64 |

Candidate mean KL is closer to scalar, but mean held-token NLL is higher at every
width. Continuation NLL changes are mixed; all final continuation argmaxes agree
with scalar. Layer20/31 selected-route sets differ in 5/5/2 of 128 layer-token
observations at widths 2/4/8. Exact summaries are in JSON; no quality threshold
or production-promotion conclusion is inferred from this short teacher-forced run.

## GLM NextN: provisional economics, not native head timing

One 11105-token prompt, 128 correlated target-on-policy rows, post-norm hidden
capture: filtered one-step overlap **0.7158909865**, greedy agreement **96/128**
(temperature 1, top-p 0.95, top-k disabled). The external reference head median is
2.362791 ms excluding its first 200.030625 ms compilation step; batch-512 history
initialization takes 1.881680 s and startup 12.355781 s. These are not native costs.
History batch1/batch512 witnesses agree on 128/128 argmaxes, worst KL 2.3590375e-6,
not bit-identical. Fresh batch1 post-norm overlap is 0.7158919254, confirming batch512.
Measured raw-hidden batch512 overlap is 0.6935570781, greedy agreement 90/128
(0.703125), and full KL(target || head) 0.80469563 versus post-norm 0.65784053.
Even a free head gives `(1 + 0.715891) * 39.17 = 67.21 ms`, below verify2's
70.99 ms. This cross-context optimistic screen does not justify single-token
NextN integration; it is not recursive acceptance, gameplay quality, or measured speedup.

## Verification and use

Recorded passes: half primitives 8, policy 7, native dispatch **22**, Flash HC 3,
Q8 GPU tile-edge 1, and Flash routing/quality 1. GLM same-width capture at 1/2/8,
greedy/sampled seeded native DFlash, and native NextN 128-row capture each pass.
Additional final validation reports Flash CPU 5, NextN CPU 2, Q8 CPU 2 plus tiling;
the full `cargo test -p qwen-cli` suite passes. JSON distinguishes reported checks from hashed logs.
Probe suffix: `packet_cost::native_short_packet_cost_and_stage_attribution` (ignored).
JSON columns define sample order; hashes identify private evidence without its location.
