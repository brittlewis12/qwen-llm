# v0.644 Production-Q4 Mat-Mat Attribution

Status: **preregistered; no B/C GPU observation exists**.

This packet asks which terms separate the exact production Q4_K N64 kernel from the
synthetic MMA ceiling on the dense-27B `[5120,17408]`, `N=1024` projection. It is a
five-arm attribution packet, not production implementation authority.

## Pilot Disclosure

An implementation-validation smoke reportedly observed A/E before this preregistration.
It was explicitly non-authoritative and cannot be pooled with this packet:

- base commit `533e49c47def84a85339c9d3691c94b8d48bfe3a`;
- dirty source state
  `git-source-sha256-v2:5c3992d31f651a36df0271a7c34cd931dd60141b11f0cc6953070b2cf68e5e37`;
- 60 samples per arm;
- A `13.21229 TFLOP/s`, E0 `15.70277 TFLOP/s`, E8 `15.89244 TFLOP/s` means;
- its operator reported that analytic output, nonce, guard, command-status, and
  optimized-LLVM checks passed.

The functional A/E implementation was subsequently committed as `2a7d0a8`. The smoke
also contained unrelated formatting-only dirty-tree differences and is not a clean
`2a7d0a8` result. No raw artifact exists in this packet, so those observations are not
independently auditable packet evidence.

Therefore this packet cannot freshly decide whether E exceeds `14.7 TFLOP/s`, nor infer
meaning from E8 versus E0 direction. Fresh A/E rows are contemporaneous replication
controls only. B/C correctness, generated code, contrasts, and design-lane decisions
remain prospective.

## Grounding

v0.606 measured production Q4_K N64 at:

- P1024: `13.80553 ms`, `13.22195 TFLOP/s`;
- P4096: `55.06145 ms`, `13.26054 TFLOP/s`;
- gate+up: `1767.70 ms`, or `40.75395%` of the cited `4337.494 ms` prefill.

For gate+up share `f=0.4075395`, a candidate primitive ratio `r` projects whole-prefill
speedup `1 / ((1-f) + f/r)`. Solving for `1.05x` gives `r=1.132304x`. The roadmap uses
`1.133x` as the conservative upward implementation gate.

The pilot makes the old A/E `14.7 TFLOP/s` early-close question retrospective. The fresh
packet instead asks whether a production-shaped B or C retains enough of that synthetic
gap to justify a named semantic design.

## Identity

The only fixture is:

- path: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`;
- bytes: `16817244384`;
- SHA-256: `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`;
- architecture: dense, 64 layers, hidden 5120, intermediate 17408, vocab 248320;
- tensor: `blk.0.ffn_gate.weight`, Q4_K `[5120,17408]`, 50135040 bytes;
- tensor SHA-256:
  `966f5cd8316a4f8f4baf141be13465747440c0f97846816ff4c5351980cc59d9`.

The runner requires a clean worktree and a release binary whose build/runtime commit is
the packet HEAD, whose identity status is `match`, and whose override list is empty. It
scrubs `QWEN_*`, `MTL_*`, `METAL_*`, `DYLD_*`, and inherited `RUST_LOG`, then sets
`RUST_LOG=warn`. It also removes inherited Rust/Cargo compiler flags, target overrides,
SDK/developer overrides, deployment target, and C/C++ tool overrides. The runner archives
the effective PATH, Rust/Cargo/Xcode/SDK/Metal identities, and pre/post binary hashes.

## Arms

All arms use `M=17408`, `N=1024`, `K=5120`, grid `16x272x1`, 256 threads, and eight
simdgroups per threadgroup.

1. **A -- production**: untouched `kernel_mat_mat_q4_K_f32_n64`, real Q4_K source,
   real all-ones F32 activations, 8192 bytes dynamic TGM.
2. **B -- source segments live, no dequant**: retains production activation traffic,
   TGM staging, barriers, matrix loads, MMA, accumulators, and stores. Each active A
   loader performs two aligned volatile 16-byte reads: the Q4_K header/scale segment and
   the selected q segment. Loaded values do not feed output. A runtime-derived complete
   finite A tile replaces dequantization.
3. **C -- no source, no dequant**: source-identical to B before specialization, but the
   source-A reads and dead address work are compiled away. It stages the same A tile and
   must be bit-exact with B.
4. **E0 -- pure MMA ceiling**: no source traffic, activation traffic, TGM, or barriers;
   retains the production grid, K-loop semantics, half-input/FP32-accumulator MMA shape,
   accumulators, and stores.
5. **E8 -- TGM-cap control**: E0's shared MMA core plus one live endpoint roundtrip and
   barrier across an 8192-byte dynamic TGM allocation. It is 8-KiB-cap-matched, not
   occupancy-equivalent.

B's volatile segments are transaction proxies, not exact production load timing. They do
not guarantee DRAM service, production CSE, scheduling, or critical-path placement.
A-B, B-C, C-E0, and C-E8 are non-isomorphic bounds and must not be added.

## Static Gates

Before any model dispatch, the runner emits optimized LLVM using the production
`-O3 -ffast-math` flags and requires:

- B has exactly two aligned volatile `<4 x i32>` source-A loads, textually after the
  optimized load-A predicate and source-A derivation and before the first TG barrier;
- both volatile results are dead and B has no dequant call;
- C's specific source-A argument is `readnone` and unused in the function body, with no
  volatile or dequant load;
- B and C each retain two threadgroup-barrier callsites, three rolled simdgroup-barrier
  callsites, two rolled matrix-load callsites, one rolled half-half-to-float MMA callsite,
  and one rolled final-store callsite;
- the linked one-file metallib and reflection are archived with compiler output and
  hashes.

Callsite counts describe rolled optimized LLVM. Their dynamic semantics remain 160 K
iterations, four substeps, 12 simdgroup barriers, 24 matrix loads, and 32 MMAs per K
iteration.

## Correctness Gates

All command buffers must complete without an error. The output is NaN-poisoned before
each validation dispatch and surrounded by 4096 F32 guard elements on each side.

- A must be finite, nonzero, and bit-exact across two full-output runs.
- With activation `1.0`, B and C must be full-output bit-exact to
  `84480 * a_base`: `330` or `660`, depending on nonce bit zero.
- B and C must be repeat-bit-exact and mutually bit-exact.
- Flipping nonce bit zero must change every B/C output as analytically predicted.
- With activation `0.5`, B/C must be exactly `165` or `330`; activation must then be
  restored to `1.0` before warmup or timing.
- E0/E8 must pass their complete analytic mapping, mutual bit equality, repeatability,
  and both meaningful nonce-bit checks.
- Guards must remain intact after validation and all scored dispatches.

Any static, identity, command, correctness, or guard failure is terminal inconclusive
with no retry or successor authority.

## Measurement

The exact command is:

```sh
target/release/qwen-bench q4-mma-ceiling \
  -m /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --warmups 12 \
  --sequence-repeats 6 \
  --nonce 1
```

The harness runs 12 cyclically rotated warmups per arm. Scoring uses the frozen
ten-sequence Williams design embedded in the result. Each sequence receives one unscored
dispatch of its first arm, then five scored dispatches. Across one ten-sequence cycle
every arm occurs twice in every position, every unequal ordered predecessor pair occurs
twice, and each self predecessor occurs twice through wash-in. Six rotated repetitions
yield 60 scored samples per arm, 300 scored dispatches, and 60 wash-ins.

Every scored sample uses one command buffer and the command buffer's
`GPUStartTime/GPUEndTime`. Wall time spans commit through `waitUntilCompleted`; encoding
is excluded. No batching, parallel queue, sample extension, outlier deletion, optional
stopping, retry, or replacement is allowed.

Before launch and immediately after the benchmark child returns:

- `pmset -g therm` must report no thermal or performance warning;
- `memory_pressure -Q` must report at least 90% system-wide free memory;
- no `qwen-bench`, qwen inference, llama, or MLX competitor may be live.

## Analysis

Each five-arm sequence is one paired block. The packet therefore has 60 paired blocks.
The five prospective log-speedup contrasts are:

- `R_AB = t_A / t_B`;
- `R_BC = t_B / t_C`;
- `R_CE0 = t_C / t_E0`;
- `R_CE8 = t_C / t_E8`;
- `R_AC = t_A / t_C`.

The runner reports geometric means and simultaneous intervals as
`exp(mean(log R) +/- 2.70 * SE)`. The fixed `2.70` multiplier is a conservative
Bonferroni-style approximation for five two-sided contrasts under approximately normal,
independent block log ratios. Achieved relative resolution is
`exp((2.70 + 0.842) * SE) - 1` and must be at most `0.5%` for every contrast. Fresh A
nominal TFLOP/s, recomputed from the arithmetic mean of raw A GPU times, must be within
1% of v0.606's `13.22195`.

A discriminator is material only when its simultaneous lower bound is at least `1.01x`.
The charged necessary gate is `1.132304x`; `1.133x` remains the upward implementation
gate.

## Decisions And Authority

The packet may produce only one of these bounded decisions:

1. **OPEN_SEMANTIC_DEQUANT_DESIGN** if `LCB(R_AB) >= 1.132304`. This authorizes design,
   not implementation, of one named semantic dequant replacement.
2. **PRIORITIZE_SOURCE_PROXY_DESIGN** if `LCB(R_AC) >= 1.132304` and
   `LCB(R_BC) >= 1.01`. This says the combined synthetic gap is charged and the source
   proxy is material. It does not claim that source removal alone clears the charged gate.
3. **CLOSE_TESTED_BC_ABLATION_LANE** if both `UCB(R_AB)` and `UCB(R_AC)` are below
   `1.132304`. This closes only semantic designs represented by the tested B/C
   organization. It does not close changed work units, topology, exact source timing, or
   representation changes.
4. **ATTRIBUTION_ONLY_NO_IMPLEMENTATION_AUTHORITY** otherwise.
5. **INCONCLUSIVE** if any identity, static, correctness, environment, A-health, or
   achieved-resolution gate fails.

Material C-to-E residuals re-rank any later work toward activation staging, TGM/barriers,
simdgroup-load organization, or topology. They do not authorize such work by themselves.

No outcome changes production dispatch, defaults, the family board, or whole-prefill
performance. Any successor needs a named race-free semantic arm, complete charged costs,
fresh correctness, and a conservative `>=1.133x` gate/up prediction before implementation.

## Execution And Artifacts

Run exactly once:

```sh
uv run scripts/profile/v0644_q4_matmat_attribution.py
```

Artifacts seal under `target/profiles/v0644-q4-matmat-attribution-p1/`. Existing packet
state forbids rerun. The runner creates the packet before preflight, archives every
critical command's argv/stdout/stderr/status, checks clean source, binary, and model
identity at launch and after timing, and recomputes every authoritative summary from raw
samples. It captures the post-child quiet state before writing benchmark artifacts and
archives optimized LLVM, AIR, metallib/reflection, analysis, decision, and a SHA-256
inventory. Any environment or other failure seals a terminal, no-authority decision and
failure inventory; no retry or successor is authorized by this packet.

Adversarial design review: cx session
`019fa4c7-bace-7e11-9d19-10df8ac262aa`.
