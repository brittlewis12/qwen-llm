# W1A: Chunked (WY/UT-transform) formulation of the Qwen3.5/3.6 gated delta rule

Status: DERIVED, hand-verified at C=1 and C=2; adversarially reviewed
(cx 019f7fcb-38ab-7d31-9af3-60c09a427ea1, verdict SOUND-WITH-FIXES; all
fixes applied below — core algebra confirmed unchanged, corrections were
in numerical rules, cost model, and the kernel sketch); numerical
validation is W1b's job (oracle-before-kernel).
Date: 2026-07-20. Sources: fla-org/flash-linear-attention
`chunk_gated_delta_rule` path (chunk.py, chunk_fwd.py, wy_fast.py,
common/chunk_delta_h.py, common/chunk_o.py, utils/solve_tril.py), Yang et
al. 2024 (arXiv:2406.06484, exact chunkwise delta rule) and Gated DeltaNet
(arXiv:2412.06464), mapped onto THIS engine's conventions and verified
against them. cx research session captured in /tmp/cx-w1-research-out.md
(session 019f7f-class, 2026-07-20); algebra independently re-derived at
C=1,2 in-session before acceptance.

## 1. Serial reference (ground truth, this engine's conventions)

Per V-head, state S in R^{d_v x d_k}, d_v = d_k = d = 128, layout
S[dv][dk] row-major (forward.rs:1091-1092). Per token t
(forward.rs:1104-1131 == kernels/gated_delta_net.metal, all variants):

    g_t = -exp(A_log) * softplus(a_t + dt_bias)   in (-inf, 0)
    alpha_t = exp(g_t)                            in (0,1)  (fp32 can
                round alpha to exactly 0 or 1 at the extremes)
    beta_t  = sigmoid(b_t)                        in (0,1)  (same caveat)
    k_t, q_t: per-K-head L2-normalized (see eps semantics below), tiled
              to V heads by hv % n_k; v_t: raw conv+SiLU output.

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T
        = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t / sqrt(d)          # POST-update: includes own (v_t,k_t)

(S k)[dv] = sum_dk S[dv,dk] k[dk]. GPU kernel computes sk on the
post-decay state (s *= exp(g) then sk = S.k), identical algebra since
alpha*(S k) = (alpha*S) k.

L2-norm semantics (BINDING for any reimplementation): ggml-style
`x / max(sqrt(sum x^2), eps)`, eps = 1e-6 (forward.rs:1307-1313,
kernels/elementwise.metal:342 comment). NOT fla's `x/sqrt(sum+eps)`.

Dims: 27B n_v=48/n_k=16, A3B n_v=32/n_k=16, 0.8B 16/16; d=128 everywhere;
state fp32 non-negotiable.

## 2. Chunk-level quantities

Chunk of C tokens, 1-based i in [1,C]. Per head:

    l_i = log(alpha_i) = g_i  (ALREADY have g_i pre-exp — use it directly,
                               do NOT recompute log(exp(g)))
    G_i = sum_{r<=i} l_r          (cumulative log-decay, G_0 = 0)
    D_i = e^{G_i}                 (cumulative decay product)

    K, Q in R^{C x d}   rows k_i^T, q_i^T (post-L2-norm, post-tiling —
                        per K-HEAD storage is fine; see sec. 7)
    V in R^{C x d}      rows v_i^T
    S_0 in R^{d_v x d_k}  chunk-initial state

Strictly-lower-triangular A in R^{C x C}:

    A_ij = beta_i * e^{G_i - G_j} * (k_i . k_j)   for i > j, else 0

(beta on the ROW/current-token index.) Unit-lower-triangular inverse:

    T = (I + A)^{-1}      (exact; A nilpotent, A^C = 0)

computed by forward substitution:

    T_ii = 1
    T_ij = -A_ij - sum_{j<r<i} A_ir T_rj      for j < i     — O(C^3)

Pseudo-values (fla wy_fast.py `recompute_w_u_fwd`):

    W = T diag(beta_i * D_i) K        [C x d]
    U = T diag(beta_i) V              [C x d]
    Z = U - W S_0^T                   [C x d]   (state-dependent; fla's
                                       `b_v = b_u - b_w @ b_h`, h = S_0^T)

Row form (equivalent, shows the sequential meaning):

    w_i = beta_i D_i k_i - sum_{j<i} A_ij w_j
    u_i = beta_i v_i     - sum_{j<i} A_ij u_j
    z_i = beta_i (v_i - D_i S_0 k_i) - sum_{j<i} A_ij z_j

z_i is exactly the serial delta coefficient of token i (the
`beta*(v - alpha*S k)` vector) after accounting for all earlier
in-chunk writes and the decayed initial state.

## 3. End-of-chunk state and outputs (the two GEMM formulas)

Terminal state (exact):

    S_C = D_C * S_0 + Z^T diag(e^{G_C - G_i}) K
        = D_C * S_0 + sum_i e^{G_C - G_i} z_i k_i^T

Outputs, all C rows at once, with INCLUSIVE-diagonal causal mask
(because output is post-update):

    L_ij = e^{G_i - G_j} * (q_i . k_j)   for j <= i, else 0
    O    = [ diag(D_i) Q S_0^T + L Z ] / sqrt(d)        [C x d]

Prefix-state formula (needed for continuation/partial-accept semantics
and the induction):

    S_t = D_t S_0 + sum_{i<=t} e^{G_t - G_i} z_i k_i^T    for any t <= C

## 4. Verification record

- C=1: T=I, z_1 = beta_1(v_1 - alpha_1 S_0 k_1); S_1 and o_1 match the
  serial update and post-update output exactly. VERIFIED by hand.
- C=2: expanding z_2 = beta_2(v_2 - D_2 S_0 k_2) - A_21 z_1 and
  S_2 = D_2 S_0 + alpha_2 z_1 k_1^T + z_2 k_2^T reproduces
  S_2 = alpha_2 S_1 (I - beta_2 k_2 k_2^T) + beta_2 v_2 k_2^T with
  S_1 = alpha_1 S_0 + z_1 k_1^T, including the cross term
  A_21 z_1 = beta_2 alpha_2 (k_1.k_2) z_1. o_2 likewise. VERIFIED by hand.
- General C: induction via the prefix formula (substitute into the serial
  update; the triangular system defining Z is precisely the condition
  that each z_i corrects for all earlier writes). Sketch verified;
  NUMERICAL validation at C in {8,16,32,64} on real streams is W1b.
- Exactness: no approximation anywhere — WY/UT is an algebraic rewrite.
  fp32 equivalence is NOT implied (different reduction orders); that gap
  is exactly what W1b's preregistered contract must measure.

## 5. Numerical-stability rules (binding for W1b oracle and any kernel)

R1. Every decay ratio used has non-positive exponent: e^{G_i - G_j} only
    for i >= j; e^{G_C - G_i}, i <= C; e^{G_i} = e^{G_i - G_0}. NEVER
    form D_i / D_j (0/0 after underflow). NEVER evaluate exp on unmasked
    upper-triangle arguments (G_i - G_j > 0 there can overflow). The
    mask must be applied to the ARGUMENT, not the result —
    select(0, exp(delta), m) still EVALUATES exp(delta):
        bool  m          = (j <= i);
        float safe_delta = select(0.0f, G_i - G_j, m);   // arg masked
        float factor     = select(0.0f, exp(safe_delta), m);
    Kernels build with -ffast-math (build.rs): NaN/Inf semantics are
    relaxed, so no code path may produce Inf/NaN even transiently.
    (Review fix: my original phrasing was itself the unsafe form.)
R2. Work in natural-log G, single exp path (no exp2/log2 conversion à la
    fla). Integration reality check (review finding): the PRODUCTION
    gate kernel emits alpha = exp(g), not g, at the kernel boundary
    (elementwise.metal ~:820 decay chain). The W1b CPU oracle computes g
    directly (fine). A W1c kernel needs g itself — emit g alongside (or
    instead of) alpha from the gate stage; recovering it as log(alpha)
    is PROHIBITED (reintroduces the underflow this rule exists to
    avoid).
R3. fp32 for EVERYTHING: G cumsum, Gram dots, A, T, W, U, Z, both GEMM
    stages, state. No half staging of any operand that feeds state (the
    mma8 half-staged mat-mat family is NOT admissible for this).
R4. alpha == 0 exactly (g = -inf or far below fp32 range) gives G = -inf
    and (-inf) - (-inf) = NaN downstream. NO CLAMP: a g >= -80 clamp is
    NOT serial-equivalent (e^-80 ~ 1.8e-35 is a NORMAL fp32 value; fp32
    normals reach ~e^-87.3, subnormals ~e^-103.3, so the serial path
    does NOT underflow at -80 — review caught this). Correct handling:
    compute production alpha_i = exp(g_i); if alpha_i == 0.0 exactly,
    that token is an EXACT RESET boundary — segment the chunk there
    (state before it contributes nothing) and restart the cumulative G
    at the segment head. Any clamp ever adopted must be applied to BOTH
    serial and chunked paths and declared a model change. (Softplus in
    the engine is branch-stabilized, elementwise.metal ~:824, so g is
    finite for finite inputs; the reset branch guards the corner, and
    W1b's edge battery must include synthetic alpha == 0 tokens.)
R5. L2-norm semantics per sec. 1 — max(norm, eps), not sqrt(sum+eps).
R6. Same-invocation comparisons only when measuring; the CONTRACT
    comparison in W1b is chunked-vs-serial in the SAME process, both
    fp32 CPU, plus a widened fp64 serial oracle as the truth anchor.

## 6. Cost model (corrected per adversarial review findings 9-12)

GEMMs per chunk per V-head: T(bDK), T(bV), L Z ([C,C]@[C,d]);
W S_0^T and (DQ) S_0^T ([C,d]@[d,d]); Z^T (R_C K) ([d,C]@[C,d]).
Shareable per K-HEAD (amortized over r = n_v/n_k V-heads): K K^T and
Q K^T ([C,d]@[d,C]). Elementwise O(C^2 + C d); T-solve (1/3)C^3 flops;
G scan O(C).

Serial baseline per token is 7 d^2 flops (S k: 2d^2; decay+rank-1
update: 3d^2; S q: 2d^2), so F_serial ~ 7 C d^2, not 6.

Per V-head with Gram sharing, layer-level leading model:

    F_chunk(layer) ~ 6 n_v C d^2 + (6 n_v + 4 n_k) C^2 d + (1/3) n_v C^3

Per-V-head ratio vs serial (d=128, r = n_v/n_k):

    ratio ~ 6/7 + (6 + 4/r) C / (7d) + C^2 / (21 d^2)

    r=3 (27B):  C=8: 0.93   C=16: 1.00   C=32: 1.15   C=64: 1.44
    r=2 (A3B):  C=8: 0.93   C=16: 1.01   C=32: 1.16   C=64: 1.47
    (unshared worst case, r=1: C=8: 0.95  C=16: 1.04  C=32: 1.22  C=64: 1.58)

Corrected conclusion: at C <= 16 the chunked form is FLOP-NEUTRAL or
slightly cheaper than serial (chunk consolidates the per-token
whole-state decay); the earlier claim that any win "must" come from
shape alone was too strong. At C >= 32 the flop premium is real and the
win must come from GEMM utilization / parallelism across C. v0.483's
host-loop chunk16 (238->124 t/s) collapsed on execution shape; the
unfalsified claim is one-dispatch-per-layer with in-kernel chunk loop.
C=16 or 32 is the opening bid; C=64 only if utilization dominates.

Per-token serial kernel state I/O: the packed serial kernel re-reads and
re-writes NOTHING between tokens (state lives in registers across the
in-kernel token loop) — the serial baseline is stronger than naive
dispatch-per-token math suggests; the chunked win case is about
PARALLELISM across the C dimension (mat-mat units, more threads per
head), not about state I/O.

## 7. Head-count reality (sizing honesty)

Q/K live per K-head (n_k=16); V/state per V-head (27B 48, A3B 32).
K K^T, Q K^T, A, T depend only on (K, alpha, beta...) — beta and alpha
are PER V-HEAD, so A and T are per V-head even though the k-vectors
repeat across the n_v/n_k tiling. Only the Gram matrices K K^T and
Q K^T are shareable across the tiling (compute n_k of them, reuse
n_v/n_k times). W, U, Z, state GEMMs are per V-head irreducibly.

## 8. Kernel-primitive constraints (W1c open problems — corrected per
review findings 14-16; deliberately NOT a design)

Hard facts that any W1c design must respect:
  - TGM budget 32 KB. Staging K+Q+V for one chunk costs 3*C*d*4 B:
    C=16 -> 24 KB (fits, barely, before A/T/scratch); C=32 -> 48 KB
    (does NOT fit); C=64 -> 96 KB. Larger C forces streaming or
    partial staging.
  - The full 128x128 fp32 state is 64 KB — never TGM-resident.
  - The serial kernel does NOT prove whole-state register residency in
    one threadgroup: it proves one state ROW per SIMDGROUP (4 floats
    per lane). Whole-state-in-one-threadgroup at 128 threads means 128
    floats/thread before tiles — feasibility and spill behavior
    entirely unproven.
  - kernels/mat_mat_mm_tile.h is HALF-STAGED (simdgroup_half8x8
    operands, float accumulate) — violates R3, CANNOT be cited as the
    primitive. A full-fp32 simdgroup_float8x8 tile path (Metal supports
    fp32 loads directly) must be written and validated first.
  - fla's head-group indexing is contiguous-block; ours is modulo
    (hv % n_k) — do not copy fla indexing.

Candidate organizations to price IN W1c (not now): (a) C=16,
row-distributed state exactly like the serial kernel (one simdgroup per
state row) with chunk math cooperatively computed across the 128-row
simdgroups of a head — keeps state residency story proven, makes the
[C,C] work redundant-or-shared across simdgroups; (b) state re-streamed
from device per chunk (adds 2*64KB*M I/O per head per layer — cost
against measured serial); (c) split-k across d with device-scoped
reduction. Register pressure and occupancy are the twin risks; none of
this is assessable on paper. NOT starting before W1b passes and the
post-W0 economics packet prices the verifier lane.

## 9. Contract plan (W1b preregisters the numbers)

Compare per layer/head over real captured streams (0.8B F32 CPU path
first; A3B captured streams after):
  serial-fp32 (bit-exact vs production CPU reference) vs chunked-fp32
  (C in {8,16,32,64}) vs serial-fp64 anchor.
Metrics: terminal-state and per-chunk-boundary state cosine + max-abs
(fp64-anchored relative), plus a 16-step continuation in the v0.554/
v0.556 audit shape. Edge batteries: alpha near 0 / near 1 runs, beta
near 0/1, 128+ token drift, near-parallel k collisions (adversarial
conditioning for T). Tolerances frozen in the W1b prereg BEFORE runs;
the fla test floor (~5e-3 RMS-relative) is NOT a precedent for a
1e-7-class contract — nobody has published one; W1b measures whether it
exists at fp32 at all.
