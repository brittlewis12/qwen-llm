# v0.664 Generic LM-Head Screening Result

Status: sealed `KILL_PRUNING_AND_BYTES`, authority `none`. The sole acquisition
is consumed. Do not rerun it or build a production screener from this bound.

## Identity

- Clean build/runtime commit:
  `9f14e5ccab928b40a93b5cc26cb1f0fc0ea6e0ed`.
- Root:
  `target/profiles/v0664-generic-lm-head-screening-oracle-a3b-p1/`.
- Decision SHA-256:
  `97320e1ff959c364683dcd7d2efa949504f754f4c9fae51726b97786c8bc82fa`.
- Artifact-manifest SHA-256:
  `c674a207e2e6af8fe58ee9b0d00acd54d1894ea52af5b804a8d900d173e1f00a`.
- Screening-result SHA-256:
  `bc4b7c4a8418aa507688953d121e17c0af3bcf1e5e31df0f5129c6d7bf3155d2`.
- Exact-validation SHA-256:
  `ab465aac5465f6e0a873157b7528f3d54cd47036f92b5bbcb826e08c55714635`.

All 32 final manifest entries authenticate. The packet uses the frozen A3B
model, 419-token prompt, six generated positions, complete Q6_K output head,
and repaired shared raw-i32le prompt identity.

## Result

| Call | Position | Winner | Bound pruned | Survivors | Charged 128 B |
|---:|---:|---:|---:|---:|---:|
| 0 | 419 | 760 | 0 | 248,319 | 431,109,248 |
| 1 | 420 | 3,594 | 0 | 248,319 | 431,109,376 |
| 7 | 426 | 30,630 | 0 | 248,319 | 431,109,376 |
| 31 | 450 | 38,895 | 0 | 248,319 | 431,109,248 |
| 63 | 482 | 35,101 | 0 | 248,319 | 431,109,376 |
| 127 | 546 | 279 | 0 | 248,319 | 431,109,248 |

Every exact comparison confirms a unique ideal winner and all 248,319
competitors below it. The certificate is correct but entirely nonselective:
block-norm upper bounds eliminate no row at any capture.

The full output head is 417,177,600 bytes and the frozen 30% gate is
125,153,280 bytes. Screening charges about 431.1 MB, or 103.3% of the complete
head, before any production integration. Both pruning and byte gates fail.

## Decision

KILL this generic norm-certified screening mechanism. Its failure is geometric,
not an implementation shortfall: exact containment holds, but the bound is too
loose to reject even one row. Reopen only for a materially tighter certificate
whose own charged traffic remains below the head bytes it avoids, or for a
different head/model geometry. Do not retune alignment, compactors, or survivor
execution around an all-survivor set.
