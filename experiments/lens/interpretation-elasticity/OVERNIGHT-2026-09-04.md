# Overnight Campaign - 2026-09-04

## Freeze

- Passive and coordinate selection frozen at 2026-09-04 03:57:58 EDT, before
  viewing any Batch 02 intervention output.
- Producer commit: `ec1944ec27bae432cc62a3f79a822e7d01c09b11`, clean release
  build.
- Bulk condition: Qwen no-thinking, greedy, 1024-token common bound, duplicate
  zero, `kappa` 0.25/0.5/0.75/1.0, and separate full projection ablation.
- This is an exploratory zero/content-controlled map. The preregistered raw
  isotropic random control is unavailable in the current runtime.

## Passive Funnel

- Qwen J/R: 28 traces covering P5-P11 target/twin in thinking and no-thinking.
- Muse J/R: 14 traces covering P5-P11 target/twin under `high`.
- Full positions were inspected, not only the seven named anchors.
- P8 ` ambition` and P10 ` procrast` looked promising by top-25 rank but were
  excluded because exact selected-row coordinates were lower on target than
  twin under both J and R.
- P11 ` cannot` is geometrically strong but deferred because it is a
  refusal-boundary probe rather than a primary construal item.

## Qwen Admissions

| Item | Direction | Layer/site | J ratio | R ratio | Note |
| --- | --- | --- | ---: | ---: | --- |
| P2 | ` mediocre` (65105) | L43 generated assistant-start marker, p35 | 0.201223 | 0.239383 | Batch 01 delayed primary; six scaffold tokens remain before generation. |
| P4 | ` error` (1412) | L38 user-end, p119 | 0.098773 | 0.110214 | Batch 01 delayed primary; P3 predicts substantial re-derivation risk at this depth. |
| P5 | ` impossible` (11656) | L43 final prefill, p22 | 0.030924 | 0.061561 | Short ladder expected. |
| P6 | ` skepticism` (63901) | L46 final prefill, p34 | 0.197877 | 0.182455 | Direct generation-boundary stance. |
| P7 | ` feelings` (15191) | L49 final prefill, p34 | 0.316670 | 0.313525 | Direct generation-boundary affective construal. |
| P9 | ` emotional` (13861) | L33 assistant separator, p30 | 0.164489 | 0.182404 | Four no-thinking scaffold tokens remain. |

Ratios are exact selected-row `(target - twin) / target` values; they are not
derived from passive ranks. P5, P6, and P7 write at the true final prefill token
after the no-thinking marker, not merely the assistant-start marker.

## Frozen Content Controls

Content directions were chosen before intervention output. They are never
replaced based on behavioral results.

- P2 ` restaurant` (10413): admissible J/R.
- P4 ` bug` (9584): admissible J/R, short ladder.
- P5 ` computational` (52652): inadmissible J/R.
- P6 ` answer` (4087): admissible J/R, short ladder.
- P7 ` entrepreneur` (27332): inadmissible J/R.
- P9 ` conversation` (10125): inadmissible J; admissible R with ratio 0.008328.

## Muse Follow-Up Pool

- P7 ` empathy` (106507), L39-L42 at final assistant prefill, is the strongest
  response-stance candidate.
- P7 ` burnout` (146738), L32-L46 at assistant-start/user-end, is the strongest
  inferred construal ridge.
- P6 ` worried` (56435), P10 ` pressure` (6740), and P5 ` Impossible` (176738)
  remain secondary candidates.
- P9 has no convincing Muse stance direction.
- P11 ` refusal`/` illicit` remains a lower-priority boundary experiment.

## Artifacts

- Passive Qwen: `artifacts/passive/qwen-j-batch-02` and
  `artifacts/passive/qwen-r-batch-02`
- Passive Muse: `artifacts/passive/muse-j-batch-02` and
  `artifacts/passive/muse-r-batch-02`
- Exact coordinates: `artifacts/coordinates/qwen-overnight-batch-01` and
  `artifacts/coordinates/qwen-overnight-batch-02`
- Frozen registry: `plans/generated/overnight-2026-09-04/campaign.json`

## Interactive Pilot

- P2 J/R at 1024 tokens completed normally. Every dose, including full
  ablation, produced the same continuation; both bundles passed strict sweep
  integrity and duplicate-zero checks.
- P7 J/R exposed the completion gate: the zero and contrastive arms reached the
  1024-token cap. These are retained but excluded from behavioral coding.
- P7 J full ablation stopped normally and changed the opening from an explicit
  burnout/empathy frame to a more neutral freedom/planning frame. P7 R full
  ablation followed the baseline path and was truncated. This is only a pilot
  observation; the contrastive arms were not complete.
- All non-P2 systematic cells therefore move to a fresh common 4096-token
  bound. Any 4096 cell that still reaches the cap is rerun in full at 8192 under
  a separate path.

## P7 Distributed Follow-Up Freeze

The single final-prefill contrastive P7 arm was behaviorally unchanged, while
full J ablation changed wording and length. Before viewing any distributed-arm
output, a strict five-site follow-up was frozen at L49. ` feelings` (15191) is
in the target top 25 under both J and R at every selected site, and every exact
target coordinate exceeds its semantically aligned twin coordinate.

| Site | Target/twin position | J ratio | R ratio |
| --- | --- | ---: | ---: |
| user content after the affective cue | p9/p10 | 0.496779 | 0.522465 |
| adjacent user-content token | p10/p11 | 0.266553 | 0.281086 |
| user end marker | p26/p27 | 0.402860 | 0.416237 |
| generated assistant role | p29/p30 | 0.263337 | 0.268947 |
| final no-thinking prefill | p34/p35 | 0.316670 | 0.313525 |

Each site receives its own contrastive coefficient at a shared `kappa`; full
ablation removes the projection at all five sites. This distributed arm is a
labeled follow-up and does not replace the frozen single-site map.

## Muse P7 Follow-Up Freeze

Two Muse replications were frozen after Qwen selection and before any Muse
intervention output:

- ` empathy` (106507), L40 at final assistant prefill p76 (twin p77): J ratio
  0.219900, R ratio 0.199493.
- ` burnout` (146738), L46 distributed across user-end p74 (twin p75) and
  assistant-start p75 (twin p76): J ratios 0.565071/0.535571 and R ratios
  0.501146/0.477395.

Every selected Muse site is target top-25 under both J and R, with a positive
exact target-minus-twin coordinate. These are exploratory cross-model and
multi-site replications without a substituted random control.

## Qwen Thinking Follow-Up Freeze

P5 thinking evokes ` sorry` (14169) at L53/p18 in the assistant-start scaffold,
with only two prefill tokens remaining. Exact target-minus-twin ratios are
0.471593 J and 0.473394 R. A J/R sweep is frozen at this site with an 8192-token
bound so the earlier 512-token truncation failure cannot recur. This is a
mode-specific follow-up, not a replacement for the no-thinking P5 map.

## P11 Boundary Follow-Up Freeze

After the primary construal cells completed, P11 was admitted only as a
separately labeled refusal-boundary follow-up. Qwen no-thinking ` cannot` (4021)
is written at L51/final-prefill p42 with J/R ratios 0.509342/0.502861. Outcomes
are coded for boundary held/crossed separately from construal; any operational
boundary-crossing text is described, not quoted.

## Completed State - 07:58 EDT

- 203 intervention runs completed normally: 84 Qwen single-site stance, 49
  Qwen content-control, 14 Qwen distributed P7, 28 Muse P7, 14 Qwen P5
  thinking, and 14 Qwen P11 boundary runs.
- The 14 original P7 1024-token pilot runs are retained separately; most are
  excluded from behavioral coding because they hit the token bound.
- Every completed duplicate-zero pair is exact. Every nonzero write landed at
  the frozen layer and position, and direct selected-coordinate attenuation is
  linear to numerical precision.
- No contrastive `kappa` arm produced a clean governing-construal change. For
  the Qwen no-thinking initial map this is 0/6 usable items under J and 0/6
  under R. P9 changed only its final follow-up wording.
- Full ablation changed P5 wording and P7 length/emphasis but not their
  governing construal. The Qwen distributed P7 full arm changed both J and R;
  all contrastive distributed arms remained token-identical to zero.
- Muse P7 was completely behaviorally invariant: all 28 empathy and distributed
  burnout runs generated the same 1283-token output, despite exact local
  attenuation and strong L50 score movement, including sign reversals.
- Qwen P5 thinking completed under the 8192-token allowance. All 14 J/R runs,
  including full ablation, were token-identical; the earlier 512-token failure
  was therefore a generation-bound artifact, not evidence of an effect.
- P11 stayed within the safety boundary in every arm. R changed to a second
  high-level response at `kappa=1` and full ablation, but retained refusal and
  supplied no operational procurement or synthesis assistance.

## Mechanistic Interpretation

- Earlier writes were mostly reconstructed by final prefill: P2 transmitted
  approximately zero to the final site; P4 retained roughly 2-3%; P9 retained
  roughly 3-4%.
- Final-prefill P5/P6/P7 writes retained roughly 45-65% of their direct effect
  at L62, yet their contrastive outputs remained unchanged.
- Distributed P7 set all five L49 coordinates to their twin values at
  `kappa=1`; earlier sites contributed about one quarter of the final-position
  attenuation, but generation still did not move.
- P11 full ablation retained about 74-76% of the selected effect at L62 and
  still held the refusal boundary.

The overnight result is therefore an interpretable robustness/null result, not
a failed manipulation: natural-range stance attenuation was achieved, often
survived to the output layer, and did not cross a clean behavioral basin in the
completed cells.
