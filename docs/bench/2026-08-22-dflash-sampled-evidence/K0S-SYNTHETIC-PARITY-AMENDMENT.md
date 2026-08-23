# DFlash2 K0-S Synthetic Parity Amendment - 2026-08-23

Status: prospective append-only amendment, frozen after static implementation
research and before the K0-S synthetic parity seam or any synthetic Metal
execution. It changes only the asset-free tooling non-perturbation test in
`K0S-AUTHORIZATION.md` and `K0S-IMPLEMENTATION-AMENDMENT.md`. The later
real-model diagnostic-off/on acquisition gate remains mandatory and unchanged.

This amendment grants no real-model execution, acquisition result, projection
parity, K0-S result, RNG, acceptance/correction, K0-L, E1b, verifier, product,
serve, default, or production authority. Every Metal command authorized here
must literally set `QWEN_METAL_LEASE_WAIT=1`.

## Static Feasibility Finding

Within the authorized source surface, an asset-free full call to
`DFlashDecoder::draft_block` cannot be constructed without either changing
`metal_forward.rs` or manufacturing a production-scale target model:

- `DFlashDecoder` requires `MetalForward` and `MetalModel`; private target-model
  state prevents direct construction from `metal_dflash.rs` tests;
- the fixed K0-S geometry `H=5120`, `V=248320` alone requires an H-by-V target
  embedding/lm-head, about 5.09 GB as F32 or about 682 MiB as Q4_K, before target
  layers and drafter weights; and
- existing full DFlash Metal tests use external model assets, which this source
  phase explicitly forbids opening.

Adding a test-only target-model constructor outside the authorized surface would
not remove the fixed H-by-V allocation unless it also replaced production target
execution with a callback. That would test a seam rather than the production
draft for substantially greater source and memory cost.

## Decomposed Non-Perturbation Claim

The diagnostic wrapper has two logically separable parts:

```text
unchanged production draft and completion wait
  -> read-only post-sync K0-S extraction
```

Source and unit tests must prove the wrapper invokes the unchanged production
draft exactly once and that its tail contains no Metal encoder, command buffer,
commit, wait, model forward, generation, state write, or RNG operation. The
asset-free Metal test may therefore isolate the only new behavior: reading and
reducing synchronized selector/session buffers after completion.

This decomposition grants only a tooling non-perturbation claim. It does not
show that a synthetic selector is a faithful model or that `H(h_t)` matches
another backend. The exact authority label remains conditional on authenticated
`z_t` and explicitly denies projection parity.

## Authorized Synthetic Seam

Refactor the feature-gated wrapper inside `metal_dflash.rs` into:

1. an observation stage that owns the existing census/trace guard and exactly
   one unchanged production draft; and
2. a private read-only extraction function over an already synchronized head,
   session, draft-token vector, dispatch census, and kernel counters.

The public wrapper remains behaviorally identical. The extraction function may
perform only checked CPU reads, copies, hashes, codebook dequantization, lattice
construction, and validation. It may not expose a product call site.

The fixed-geometry synthetic fixture must use no file-backed model or GGUF. It
constructs in the existing feature-gated test module:

- `N=8`, `K=16`, `R=256`, `H=5120`, `V=248320`;
- two zero-valued Q4_K A/B codebooks with exact 256-element rows;
- a synthetic selector-hidden tensor and a minimal DFlash head/session whose
  synchronized full logits, top-16 IDs/unary values, projected `z_t`, caches,
  conv buffers, and scratch buffers contain deterministic canaries; and
- one tagged synthetic selector-hidden Metal dispatch plus one completion wait,
  identically executed in every arm before optional extraction.

Both off and on arms activate dispatch census and kernel tracing identically.
The off arm discards the synchronized observation after hashing state. The on arm
passes the same observation to the extraction function. No arm calls a target or
drafter asset.

## Required Parity Matrix

Use four fresh equal-capacity sessions from one immutable synthetic head:

```text
off-A -> on-A
on-B  -> off-B
```

The test must compare both orders and cross-order identities. It requires:

- each on arm to return `Ok` with a complete capture, followed by independent
  assertions of all seven depths, all 97-by-16 lattice rows, exact synthetic
  logits/top-16/unary/`z_t` canaries, raw A/B rows, draft and production-chain
  tokens, dispatch rows, kernel counters, provenance, and analytically determined
  finite scores. Every synchronized input class must appear in and match the
  capture; equal state digests, a successful return, or a self-consistent capture
  hash alone cannot pass;
- exact synthetic draft tokens and complete synchronized selector inputs;
- exact dispatch-census rows/encoder ordinals and kernel-trace counters, with one
  tagged selector-hidden dispatch and observer counts restored to baseline;
- exact complete DFlash diagnostic-state digest before and after extraction;
- exact target-state canary digest before and after extraction;
- zero RNG state/draws by construction; and
- an identical second tagged synthetic continuation dispatch, continuation
  output/token, and final DFlash/target state digest after each off/on arm.

Any mismatch fails the tooling gate. The test must run serially because observer
activation combines process-global counters with thread-local records.

The required command is:

```text
QWEN_METAL_LEASE_WAIT=1 QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --release --features dflash-k0s-diagnostics k0s_synthetic_metal_diagnostic_parity_both_orders -- --nocapture --test-threads=1
```

## Residual Real-Model Gate

The synthetic seam cannot prove integration with a real target/drafter, real
cache contents, or real dispatch graph. The separate prospective acquisition
plan must still freeze and execute diagnostic-off/on fresh real-model sessions in
both orders on the exact acquisition input before its evidence row. It must
compare every observable production output/state, complete dispatch census,
kernel counters, continuation, and retained partial/adverse artifacts.

Synthetic success cannot waive, tune, or replace that acquisition gate. Tooling
remains ineligible for acquisition review if the synthetic test is skipped,
fails, uses an external model asset, or executes without the literal Metal lease
environment variable.
