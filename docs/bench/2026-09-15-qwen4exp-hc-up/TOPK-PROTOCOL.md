# Finite N512/K10 Parallel Selector Screen

Frozen before GPU execution. V2 stage intervals valid, absolute timing INCONCLUSIVE
(18.76% unsampled drift). Routing group includes projection/selection/shared sigmoid;
its large sample is a lead, not top-k attribution or predicted gain. Source shows
one-thread insertion selection of10 from512, common to48 singleton layers.

Test the existing local parallel selector directly at512 threads,6144 dynamic
shared bytes after capability preflight. Keep production host n<=256 and product
routes unchanged. Shader comparisons and selected-score softmax preserve local
arithmetic; do not port donor full-softmax-then-renormalize or shared-dot reduction.
Donor portable router does parallel selection, supporting the mechanism, not math
equivalence. Existing phase-only256 route history is not Flash512 qualification.

Finite-only research: existing shader's -infinity removal can repeat selected IDs
on nonfinite inputs. No product delivery until nonfinite compatibility or safe
fallback is explicitly proven. Do not silently assume the research scope suffices.

13 fixed512-element fixtures: four deterministic random, ascending/descending,
all equal, signed zeros, ties across reduction halves, rank10/11 tie, ULP cutoff,
finite extremes/underflow, actual saved-layer2 router logits. Nonzero buffer offsets,
guards, poisoned outputs and immutable logits. CPU ordering descending numeric
value then ascending ID (signed zeros equal); F64 softmax absolute tolerance
2e-6+1e-5*reference. Both GPU arms exact IDs and bitwise weights; finite nonnegative
weights. Save outputs before gates.

Then qualify leaf, complete router and complete saved-layer2 MoE: all produced
intermediates/output/routes bitwise, census differing only in top-k name/TG512.
Router/shared-router weights and native input/logits immutable. Per scope fixed
warm16-chain ABBA then one measured16-chain ABBA; same single encoder/command,
scratch poison, no sampled-stage instrument. Save times/census/raw outputs before
gates. GPU mean/both pairs>=10% savings, executor wall>=5%, A spread<=5%; otherwise
HOLD or INCONCLUSIVE, never retry. Complete-MoE is the advancement gate; leaf alone
does not earn native work. Any numerical failure stops before timing.

Production lease/real wired gate before Metal, API validation on, existing artifact,
no prefix/native forwards. Passing screen earns bounded native qualification only,
not all-layer/end-to-end/default-on claims. HC performance remains HOLD.
