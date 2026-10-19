# Depth And Sequence Order: Exploratory Findings

After integrating native observer controls into main at `14e4ce68`, this pass
addresses two omissions in the broad archive survey: layers 0-15 were absent,
and episode occurrence fingerprints discarded sequence order. No new model
completions, activation interventions, external judge, or precision qualification
were run. These are hypothesis-generating measurements, not a basin atlas.

## Coverage And Reader Identity

- Full-depth R replay of all 24 existing unique prefills: 4,598 input tokens,
  289,674 layer-position cells, top eight at all 63 source blocks, plus 1,512
  transported vectors at final-prefill positions. No generated tokens consumed.
- Four diverse, short, completed archive trajectories at seed 17: greeting,
  cartoon conversation, roasting advice, and word-history curiosity. Two fixed
  sites each, reasoning32 and answer-after-close32; eight exact causal prefixes
  shared by J, R, and plain. Selection favors short conversations and is not
  representative of the archive or the original survey's conceptual range.
- Scalar full scores and vectors at every layer: J/R 0-62, plain 0-63;
  24 bundles, 1,520 rows, 1,509,785,600 raw F32 bytes. Total depth artifacts
  approximately 2.17 GiB. Every shape/hash/finite/order/compact-score check passes.
- At layer62, all 24 observer-pair checks have bit-equal vectors and logits
  across separate scalar invocations. This is measured capture consistency at
  the shared identity anchor, not independent observer corroboration. Plain63
  is the actual final-block readout under this reader at the archived prefix.

The depth reader was frozen before collection from a clean integrated tree:
SHA256 `dedadc401c23d665ea118ef67270e154f3b070b1c7d2a15a7f6e4d7b7d383cbc`.
Source head and input hashes are in the private depth `provenance.json`.
Lease contention and failed/interrupted attempts are retained, not erased or
counted as successful runs; the shared lease was never bypassed.

Sequence-order analysis instead uses the original fixed-reader J/R traces at
layers32/46. That reader differs from the depth reader; no old/new numerical
equivalence is claimed. The original generator also changed builds during
collection, so same-prefix samples are not a controlled seed-only comparison.

The matched n25 J/R target62 assets declare transfer unvalidated. This pass adds
no transfer qualification; it does not negate prior n1000 J target63 passive and
intervention BF16/Q8 qualification. Those are distinct fitted artifacts/scopes.

## Depth Findings

### Early R contains context-associated variation

All 24 final-prefill sites consume token198, a newline. R top1 has 9, 10, and
12 distinct IDs at layers7, 8, and 9, respectively. By layer15, a single
variation-selector token is top1 in 21/24 cases. The earlier five-layer survey
missed this variation and its subsequent convergence entirely.

At layers7-9, examples include ` weird` for radiator noise, `-ish` for computing
convention history, and ` badass` for film enthusiasm. These are readout labels,
not endorsed descriptions or evidence of model attitude. Matching one consumed
token does not match prior content, style, length, or position. Early R deserves
content-controlled inspection; agreement with early J is not an admission gate.

### Intermediate sharpness is not conviction

Full-distribution softmax is computed at temperature one in F64 from stored F32
logits. Its entropy measures this observer distribution, not calibrated model
belief, response correctness, or a causal cost of changing a construal.

- At the roasting reasoning site, J/R/plain all favor ` evenly` at layer55,
  with entropies about 0.012/0.023/0.012 nats. All switch to `5` at layer62;
  `5` is the archived sampled next token. Agreement and sharpness at one depth
  do not imply persistence through remaining computation.
- The cartoon answer site falls inside an emoji's UTF-8 token sequence. R11
  entropy is 0.093 nats and top-eight mass is 0.99935, yet the archived next
  byte ranks 1,459. This is a useful format/tokenization warning, not proof of
  an answer-phase psychological state or evidence against R's intended use.

The continuation is observational, not gold: seven of eight archived next tokens
rank first at plain63; one ranks second, consistent with sampled generation.
Intermediate lenses do not execute the remaining blocks. The test is not that
every early observer should predict the final sampled token.

### Top-eight set geometry is often underdetermined

Across the eight scalar sites, excluding both sites of the query's own case
leaves six candidates. For R32-47, a lexical tie-break gives only 19/128 nearest
matches between full-distribution Jensen-Shannon divergence and top-eight set
distance. Allowing every tied set neighbor raises inclusion to 101/128; those
ties admit an average 72.27% of all candidates already.

Thus 27/128 comparisons genuinely disagree with every best set neighbor, while
most apparent disagreement from the single-neighbor statistic was arbitrary
tie-breaking. Set features discard ranking and probability weights as well as
tails; the discrepancy cannot be attributed to tail censoring alone.

At layer62 all cross-case top-eight sets are disjoint: every candidate ties at
distance one. Full JS still orders the distributions, but near its ln(2) ceiling.
Being nearest among near-disjoint distributions does not make two sites similar.
Report absolute divergences, neighbor margins, and tie sets, not just a graph.

The apparent early-R same-phase neighbor association reverses when one of the
four cases is omitted. Depth rows are not independent samples; no phase claim
is banked from this eight-site panel. Native geometry and each transported
pullback metric `T_l^T T_l` also remain separate measurement objects.

## Order Findings And Controls

The order analysis reads all 96 original traces, validates replay IDs, removes
32 positions at each observed phase edge, and makes nonoverlapping 16-position
windows. A minimum-length rule excludes the greeting, leaving 23 prefixes,
46 trajectories, and 253 unordered cross-prefix pairs. Each prefix has equal
weight. Direct readouts of the consumed token ID are removed.

Window features are IDF-weighted top-eight token-ID occurrences, not probabilities
or native vectors. Two statistics preserve the distinction between occupancy
and order:

- Local recurrence: adjacent-window cosine minus cosine four windows apart.
- Coarse progression: signed window means weighted linearly from -1 to +1,
  followed by direction comparison. Constant occupancy cancels; vocabulary
  support, IDF, syntax and topic schedules can still influence the result.

Window permutations and permutations of contiguous four-window blocks preserve
retained within-phase occurrence marginals. Each uses 64 draws per view; tokens,
pairs, layers and correlated observers are not independent hypothesis tests.
Local recurrence excess over the block control ranges from -0.016 to +0.010:
preserving short contiguous structure reproduces most of it, without identifying
the mechanism responsible for that structure.

Subtracting a leave-pair-out phase-common progression reduces broad R32 reasoning
alignment from 0.2145 to 0.0485/0.0382 under raw/unit reference normalization.
The common reference averages prefixes equally; raw versus unit slopes differ
in magnitude weighting. Subtracting the same finite reference from both samples
can itself induce positive alignment, so zero is not a calibrated null.

More selective patterns survive at layer46:

| Pair and phase | Evidence after common-trend subtraction |
| --- | --- |
| Euler / linear algebra, reasoning | Rank1 among253 in J/R, both reference normalizations |
| Bundles / Planck units, reasoning | J rank3; R rank5, weaker than before subtraction |
| Radiator / outlet voltage, answer | Rank1 in J/R; survives the complete-answer panel |
| Radiator / roasting, answer | Near the bottom despite earlier shared occurrence fingerprints |

These are coarse early-to-late readout changes, not matched complete routes or
identified operators. Negative slope agreement can miss cycles, reordered steps,
and different timing. Removing punctuation-only/special displays is a sensitivity
check, not matching emitted syntax, headings, length, or discourse structure.

### Censoring correction

Six original trajectories end at the 4,096-token generation cap: Euler seed29,
speculative cosmology seed17, and both linear-algebra and hiking samples. Their
reasoning phases close naturally; their answers are incomplete. Trimming the
observed answer end cannot restore its missing ending.

Excluding an entire prefix if either answer is capped leaves 19 prefixes,
38 trajectories, and 171 answer pairs. IDF and all pair scores are recomputed
on that panel. At J/R46, radiator/outlet stays rank1 under both reference
normalizations and both display filters; radiator/roasting stays ranks169-170.
Euler/linear algebra is absent from this answer panel: no absent-answer-operation
claim follows. Bundles/Planck answer alignment is normalization-sensitive and
is not treated as a robust finding. Residual norms do not approach zero under
the recorded checks, but that alone does not establish signal-to-noise quality.

## Adversarial Review And Reproduction

A separate reviewer independently reproduced all 1,520 scalar entropy/mass/rank
rows, site JS matrices, target62 bit-equality checks, and early-R final-prefill
variation. It identified the omitted cap qualification and a checkpoint-provenance
weakness: historical cached views were not bound to their producing code and
boundary/filter inputs.

The final order bank binds producing scripts/helper, parameters, trajectories,
survey, all 96 traces, all 48 generation records, and the full token-filter label
map. All 16 views and both original permutation controls were recomputed fresh;
the numerical results exactly reproduce the historical bank. Mutation tests
reject changed code and inputs. Historical files are preserved separately.

On follow-up, the reviewer checked those content bindings, cap exclusions,
coverage/ranks, and independently re-extracted all 171 complete-answer R46 pairs
under both normalizations and filters (maximum score difference 2.8e-17).
It found no reporting blockers. It did not independently rerun every permutation
bank, historical precision qualification, or model inference.

Private, Git-ignored artifact roots in the observer-controls worktree:

- `target/observer-controls/depth-panel/summary.json`: depth results, linked
  scripts, selection, provenance, validations, per-layer matrices and attempts.
- `target/observer-controls/order-analysis/summary-final.json`: corrected order
  bank; SHA256 `b5822c35baf241c4c89a2abe84e81c4942bbeb63884cc974542507d0bdb7b297`.
- `target/observer-controls/order-analysis/residual-final.json`: full and
  complete-answer panels; SHA256
  `476bd49b0fc876ee1988ed5f20b29da5493abf705fe53568ba7989abb5d873e2`.
- `target/observer-controls/order-analysis/final-notes.md`: exact methods,
  reproduction commands, verification records and exclusions.

Archive-derived text and raw outputs remain private; these locations are local
evidence, not a public dataset or portable replication package. Preserve them
before cleaning the worktree's target directory.

## Decision

The next discriminating move is a small crossed content/task experiment, not
more clustering or another large archive generation pass: same subjects under
explanation versus diagnosis, and the same task across subjects, with matched
response scaffold and token/format-aware checkpoints. Ask whether depth bands
and progression follow the requested task rather than its topic or prose form.

Early R7-9 and later layer46 are inspection targets, not assumed causal sites.
Use full distributions where set ties or omitted probability weights matter;
keep native residual geometry distinct from observer geometry. A robust task
contrast would then motivate short-pulse versus sustained intervention using
the earlier error-arbitration foothold, rather than equating low entropy with
conviction or selecting steering directions from evocative readout labels.
