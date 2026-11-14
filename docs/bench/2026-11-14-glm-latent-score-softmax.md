# GLM latent score→softmax fusion: closed at exact-design gate

**Status: design gate closed; no implementation, build, or GPU timing for this
candidate.** I found no credible single-pass mapping that removes the global
score surface while keeping the current tensor-MMA and softmax reduction order
bitwise. This is not a proof that no different per-head tensor mapping could
ever work; such a mapping would need a new exactness witness. No NI comparison
was started. The routed gate/up epilogue is a separate candidate and record.

## Constraint

`kernel_glm_latent_score_tensor` launches one workgroup per 32-key panel and
query. Each workgroup computes all 64 heads with a 64×32×128 TensorOps tile,
repeated over the four 128-wide K panels, then writes F32 scores. Its declared
threadgroup scratch is 32,768 bytes: 16 KiB for Q, 8 KiB for K, and 8 KiB for
the score tile. `kernel_glm_latent_softmax` then launches one SIMD group per
head and query and reduces all 2,051 selected slots.

A single workgroup per query that retained every score until its per-head
softmax would need at least `64 * 2051 * sizeof(float) = 525,056` bytes
(512.75 KiB), over 16× the current score workgroup's whole scratch allocation.
The existing 32-key workgroups cannot globally synchronize within one Metal
dispatch. Retaining the surface in device memory still performs the score
write/read round trip. Online/partial softmax changes the current floating
point reduction sequence. Recomputing scores in exact order avoids that
reordering but adds at least one complete QK pass. A per-head workgroup could
fit one head's score vector, but the existing M64 TensorOps mapping computes
all heads together; replacing it with a per-head tile is a different score
kernel whose bitwise identity and speed are unknown.

## Cost evidence and decision

The waste audit counted 352 score→softmax pairs at the 48K + 2,048-row cell
and 176 at 1K@80K. The score producer accounts for 177.8604 ms and 12.0 GB at
root, and 89.6068 ms and 6.0 GB at append. The modeled memory-round-trip price
is 29.9893 ms / 14.9946 ms; this is an upper bound on removable cost, not a
measured saving. An exact recompute path therefore adds roughly
178 ms / 90 ms of score production to pursue a 30 ms / 15 ms traffic estimate,
before its additional dispatch and scratch costs. This is not a measured
candidate result; it rejects exact score recomputation as a plausible route to
the requested gate without first disproving the byte-price floor.

**Gate disposition:** no candidate exists to run the requested warmup plus
three interleaved stage pairs, so the ≥1.25× stage gate is not reached. No
native-cell A/B was run. This is a design-stage no-go, not a measured speedup
or regression. A future candidate would need a new per-head tensor mapping
that avoids the full score surface and repeated QK work, then prove bitwise
scores and softmax output. The ordinary reduction-order-changing online
softmax route is outside this decision.

Sources: `kernels/tensor/glm_latent_attention.metal`,
`crates/qwen-llm/src/metal/latent/tensor.rs`,
`docs/bench/2026-11-13-waste-audit.md`.
