# Complete MoE Observation Protocol

Frozen before GPU execution. QSA on, HC off. Existing artifact, one2179-token
prefix, two one-token forwards from the same checkpoint at2179. Scoped test-only
observer captures layers2/4/5 by bound router buffer identity and checks layer
ordinal. Ordered GPU copies preserve input before reuse, final output and selected
IDs/weights. Full ordinary/observed logits, hyper and121 persistent tensors must
be bitwise. No CPU read while work is pending. Serial production lease plus real
wired-memory check; Metal API validation stays on.

Each layer gets private replay scratch and retained original weights/options.
Actual complete native census must equal isolated census including kernel names,
grids and threadgroups:8 dispatches for IQ4_XS/Q8 and IQ3_XXS/Q8,9 for IQ3_XXS/IQ4_NL.
Keep fused Q8 down/weighted-sum intact. Reuse the same six encode helpers, without
changing product arithmetic or dispatch order. Poison scratch before each replay
command; complete output and selected IDs/weights must equal native bitwise.

Fixed per-layer schedule: one census qualification plus three warm replays, then
16 unsampled complete chains,16 complete chains with six ordered sampled encoders,
then16 unsampled chains. One command buffer per packet, no interstage waits or
readbacks. Stages: router/topk/shared-router, routed gate/up, routed down/sum,
shared gate/up, shared down, accumulation. Retain raw timestamps; scale stage
ticks by sampled command GPU time / outer sampled span, as existing diagnostics
do. Report normalized stage costs and their sum beside unsampled before/after
totals, not as direct per-kernel timestamp milliseconds. No adaptive timing retry.

Interpretation: warm isolated replay of three captured native MoE inputs. Cache
reuse and extra encoder boundaries differ from a full forward. Diagnostic budget
only, no all-layer attribution, route-distribution average or speedup claim.
HC performance HOLD remains unchanged. Independent read-only review approved
this bounded approach; choose the next investigation only after seeing the budget.

Pre-execution review adds finite input/output/top-k/logit/hyper checks, unique
in-range selected IDs and poisoned replay IDs. Reject missing, zero, sentinel or
nonpositive stage timestamps without retry. Capture the already-planned sampled
packet census and require each repeated sequence to match native. Report scale,
stage sum and residual explicitly; no extra GPU packet for these checks.
