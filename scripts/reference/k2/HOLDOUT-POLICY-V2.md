# Guarded-256 ranking-indeterminate candidate policy, version 2

V1 remains failed. Its two distinct teacher-forced flips occurred between nearly
tied logits; no conclusion about reciprocal native-side regret can be reconstructed
from its retained aggregates. V2 is a new experiment with new sources, not a rerating
of v1. It retains v1's numerical, probability, residual, and execution controls.

For a non-generative row, permit differing argmax IDs only when both regrets are
finite, nonnegative, and strictly below 0.001 logit:

- reference maximum minus reference logit at the native choice;
- native maximum minus native logit at the reference choice.

Within each distribution this bounds the chosen probability relative to that
distribution's maximum by `exp(-0.001)`. It is not an absolute cross-distribution
probability-error bound; the existing KL/TV gates still apply independently. Record
both IDs, all four witness logits, both F64 regrets, every exact-ID mismatch, and
every policy-accepted ranking-indeterminate row. Validate witnesses from live vectors
before discarding them. Invalid or contradictory evidence is a failure, not a tie.

For each reference trajectory, require exact top-1 on predictor visible lengths
241 through 256 inclusive: all 15 generated choices plus the terminal prediction.
Do not apply the near-tie rule to those rows. Highest-ID mathematical ties include
signed zeros; EOS remains part of the fixed mathematical path, not a serving test.
The original strict 42-row regression remains unchanged.

Freeze the complete JSON, selected artifact/tokenizer identities, exact token counts
and prefix hashes, and semantic classifier tests before any v2 target forward pass.
The four new sources are assistant-authored synthetic coverage probes selected
after v1, not a random or independently audited dataset. No fixture replacements,
gate changes, or alternative-kernel reruns may turn this frozen experiment green.

A passing result would support a guarded final-Q8 research envelope, not cross-
backend token identity in every teacher-forced context. Public capacity, attention
dispatch, and performance claims still require separate reviewed surface checks.
