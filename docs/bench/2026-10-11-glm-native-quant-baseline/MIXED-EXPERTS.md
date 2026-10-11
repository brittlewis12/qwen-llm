# Mixed Expert Baseline

Follow-up to main checkpoint `b056515b`.

## Contract

- Gate and up no longer need the same storage dtype. Native all-slot execution
  accepts independent types from 18 implemented formats; down accepts the same
  native formats plus existing specialized paths.
- Packed Half execution composes batched gate projection and a clamped up
  epilogue for all 17-by-17 generic grouped combinations. Existing fused IQ2_S
  and IQ3_S paths remain preferred.
- Exact and explicit F32 operand modes retain native F32 operands, falling back
  to row execution where an operand-preserving grouped tile is unavailable.
- Two-pass gate/up reuses the existing inner activation buffer. No extra session
  allocation, weight expansion, route-ID readback, or model-sized repacking.
- Latent and expert kernels share the extracted F32 decoder header. This removes
  a second copy of those decoders without changing latent arithmetic.

The usual shape, byte-range, quant-block, route and clamp constraints still
apply. Grouped buckets rely on the router's distinct-experts-per-token contract;
these changes do not make arbitrary externally supplied routes valid. Missing
codecs and metadata normalization are separate work, not hidden quant whitelists.

## Validation

Release library build, Metal validation, required GPU execution, and serial test
execution. Coverage includes independent decoded CPU oracles, mixed types,
clamp boundaries, token/channel tails, nonzero offsets, canaries, invalid IDs and
statuses, binding admission, and actual precision-path selection.

The first GPU pass found a dispatch-labeling defect and a fixture error: the
ordered census mislabeled preflighted pipelines by the last lookup, and changing
a status tensor to shape `[1]` was incorrectly expected to make it invalid.
The initial census workaround checked launch totals while independent numerical
oracles verified the two banks. The root attribution defect is now fixed and the
ordered-name assertion restored; see `DIAGNOSTICS.md`. Numerical comparisons and
runtime validation were not loosened.

Final results: 5 native-expert, 48 generic-grouped, 73 GLM, 11 optimized-expert,
and 5 grouped-latent tests passed. The 44 existing opt-in tests in those suites
remain opt-in. Workspace release checking passed (one unrelated existing CLI
dead-code warning). Independently enabled UD and Maya 200-token prefill plus
three-step continuation tests both passed with the same reported KL values as
the preceding checkpoint. Logs are the `mixed-*.log` files in this directory.
No new full-model performance claim is made by this compatibility checkpoint.

## Retrospective

Integrate a validated baseline before extending it. Compatibility grows through
independent primitive composition; optimized fused pairs remain an optional
choice rather than an admission requirement. Small real GPU fixtures exercise
new compositions without waiting for a full artifact for every dtype pair.
