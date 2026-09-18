# Muse decode FFN: source-first decision packet

No GPU work is authorized by this document. The shared server is in use. Explicit
coordination precedes the production exclusive lease, real wired-memory gate,
MTL_DEBUG_LAYER=1 and serial execution. An apparently idle lock is not permission.
First close the pending default-delivery/alignment repair packet; do not obscure
that failure with a new performance experiment.

## Source scope

Released Muse has52 layers,H6656,F19968. Generated-token FFN dispatches in
`muse_glimmer_text_session.rs` are RMS norm, gate GEMV, up GEMV, SiLU/product, down
GEMV, post norm and residual add. These are in the ordinary token's existing
encoder/command buffer, not separate command buffers per projection.

The default Q8 dispatcher uses `kernel_mat_vec_q8_0_f32_lcpp`:NR0=2,NSG=4,NQ=8,
128 threads,256B dynamic scratch. Q8_0 has34 bytes per32 input elements. Both gate
and up matrices are [H,F]; down is [F,H]. Each matrix contains141213696 logical
bytes, total423641088 perlayer/22029336576 pertoken. These are unique serialized
Q8_0 weight payloads, NOT measured DRAM traffic or bandwidth limits.

## Independent candidate: reuse existing fused Q8 SwiGLU

`encode_shared_swiglu_q8_0_f32` and `kernel_shared_swiglu_q8_0_f32_lcpp` already
exist for a separate MoE shared-FFN lane. Muse does not call them. Their gate/up
per-lane Q8 accumulation, reduction layout and SiLU expression match the source
structure of the default separate primitives. Compiler/register/memory changes
still require actual bitwise and performance checks; source resemblance is not proof.

- Gate/up/product plus down changes4->2 dispatches, removing104 pertoken. Norms
  and residual remain; this does not halve all FFN dispatches or remove commands.
- Weight deletion is zero. Both matrices and the unchanged down matrix remain.
- Intermediate traffic: gatewrite+upwrite+product2reads+productwrite=5F floats;
  fusedproductwrite=1F. Logical intermediate saving4F*4=319488B perlayer.
- The baseline x load is NOT just one vector per GEMV. Each two-output-row
  threadgroup loads H values once for each projection. Fusing shares that load,
  eliminating(F/2)*H*4=265814016 source-issued load bytes perlayer. Unique x is
  only26624B. Reuse/cache hits/compiled loads and DRAM service remain unmeasured;
  neither the large issued count nor the small unique footprint predicts latency.
- Threads stay128; per-threadgroup scratch doubles256->512B and gate/up accumulator
  banks coexist. Register pressure, occupancy, two weight streams and cache
  contention can erase launch/load gains. No blanket bandwidth-bound dismissal.

This is a cheap existing-primitive reuse question, independent of approximate
sparsity. Do not condition its screen on a successful sparsity census, add a new
shader, change default math now, or create a production opt-in.

After coordination/default closure, first acquire actual normalized inputs from
the saved32K state, with source/model/prefix identity fields and a token hash
recorded by the producer (not authenticated by this scorer). Freeze real weight layers
0/25/51 and the first/last inputs of an8-transition teacher-forced packet before
screening. Compare unfused default lcpp gate/up/product versus the existing fused
primitive, same weights/inputs, bitwise fullF output and finite checks first.
One warmAB then measuredABBA percell; no census/readback comparisons inside timing.
Advance only with >=10% mean and both-pair GPU savings in every frozen cell,
<=5% control spread and no mean wall regression. A failed/noisy cell holds this
screen; no rerun selection or rescoring. One source-justified repair timebox may
start a separately recorded protocol. No model-wide speedup follows from layers
0/25/51. All-layer native continuation/correctness and controlled executor wall
qualification are required before any default-on integration.

## Structural census: separate, no runtime selector

Extend the existing `scripts/profile/ffn_inner_census.py`, not a second framework.
Schema1 inner-only captures retain block256 and their missing-capture zero guard.
Schema2 requires gate/up/inner F32LE streams [samples,layers,F], complete ordered
layer coverage, declared Q8_0/lcpp_nr0_2_nsg_4, H/F/layer_count, source_commit,
model_content_identity, prefix_identity, captured_token_ids and captured_positions.
Each tensor descriptor retains name/dtype/shape/byte_length/sha256. Length/hash/shape
and finiteness are checked. Producer provenance strings are preserved, NOT
independently authenticated by this scorer. F64 CPU SiLU(rawgate) is a labelled
diagnostic proxy, not a reproduction of Metal arithmetic; product is captured,
never reconstructed. Paired thresholds are not a gate-to-product implication.

Future capture:8 consecutive teacher-forced generated transitions on the saved
32K state, all52 layers. Capture gate and up after their GEMVs but BEFORE in-place
SiLU overwrites gate; capture product afterward. Three streams total99680256B;
normalized inputs add11075584B. Price observer storage in aggregate admission.
NaN-poison every destination before capture; require all samples/layers written
and finite, exact positions/token identity. Retained-prefix and whole-logit/KV
identity versus unobserved replay must pass before accepting observations. Captures
are not timing data; preserve observer cost separately. This producer is not yet
implemented; no synthetic input is represented as Muse activation evidence.

Physical down blocks contain32 adjacent intermediate columns across allH rows.
One block is226304 serialized Q8_0 down-weight bytes. After gate evaluation, skipping its
corresponding32 up rows could remove another226304, but gate cost stays paid.
The scorer separates:

1. Ideal post-product lowest-energy block oracle: only down can be removed;
   removable all-FFN weight fraction is blockfraction/3, NOT blockfraction.
2. Gate-proxy masks: up+down removal is only a hypothesis (2*blockfraction/3),
   alongside discarded observed product energy and paired-threshold violations.
3. Raw gate versus SiLU proxy counts, scalar versus full-block occupancy, perlayer
   and aggregate distributions. Valid allzero vectors are explicitly included in
   v2 denominators, never silently dropped; poison/write proof is the producer's job.

Activation-energy error does not bound down-output error, especially through post
RMS norm. Even exactzero skipping can alter reduction/sign-zero behavior. Selection,
index/gather/repacking, issued-load granularity and quality costs are unpriced.

Frozen census continuation floor: at the1e-3 activation-energy budget, require the
post-product oracle's median all-FFN serialized Q8_0 payload removal >=10% (thus >=30% of
down blocks), and independently require the same for each half of the8-token trace.
Missing that floor parks this fixedblock/down-only family before any sparse kernel
or weight-norm scan. Passing only earns further output-error/cost investigation,
not an optimization. Gate masks cannot inherit the oracle's certificate or claim
saved gate work. No layer0 extrapolation or whole-token timing projection.

## Reproduction without GPU

```
PYTHONDONTWRITEBYTECODE=1 uv run --with 'numpy>=2.0' python -m unittest discover \
  -s scripts/profile -p test_ffn_inner_census.py -v
uv run scripts/profile/ffn_inner_census.py /path/to/manifest.json
```

Only CPU tests/source accounting run in this packet. No activation distribution,
sparsity, isolated-kernel timing, native decode gain, or new production path is claimed.
