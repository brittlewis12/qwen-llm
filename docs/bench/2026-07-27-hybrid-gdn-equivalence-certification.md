# Hybrid GDN equivalence certification

This protocol separates algebraic correctness, kernel speed, dispatch savings,
and whole-model bulk. Do not use one cumulative before/after number to claim an
individual optimization.

## Reproducibility

Pin the model, prompt or token seed, context, build, GPU power state, and all
unrelated `QWEN_*` variables. Record the model hash, build identity, device,
macOS version, and dispatch census. Run GPU arms serially; randomize arm order
between rounds. Each arm must be a fresh process because the engine latches
environment flags on first read.

The current rollback levers are:

| Change | Flag | Default |
| --- | --- | --- |
| F32 beta matvec + sigmoid | `QWEN_DECODE_GDN_FUSED_BETA_PROJ` | on |
| Paired Q/K RoPE dispatch | `QWEN_DECODE_ROPE_PAIR` | on |
| MoE grouped finalizer | `QWEN_DECODE_MOE_GROUPED_FINALIZER` | on |

Keep `QWEN_DECODE_MOE_FUSED_FINALIZER=1` fixed during this comparison; it is
the older shared-output/residual fusion and is intentionally separate from the
grouped routed-finalizer switch.

The exploratory raw-Q qscale variant was deleted after review: it retained the
same L2 dispatch and Q norm read, so it only removed the normalized-Q write and
did not contain the K-fold needed to eliminate the preparation work. A future
raw-Q experiment must implement that K-fold as a new variant rather than revive
the removed flag.

Run the helper for a serial paired matrix:

```sh
MODEL=/Users/tito/models/Qwen3.5-0.8B.F32.gguf
cargo build --release -p qwen-cli --bin qwen-bench
SURFACE=gdn ROUNDS=3 scripts/bench/hybrid-gdn-ab.sh "$MODEL"
SURFACE=decode ROUNDS=3 RUNS=3 TOKENS=128 scripts/bench/hybrid-gdn-ab.sh "$MODEL"
# Narrow the matrix when a model does not exercise every surface.
SURFACE=decode ARMS="control moe" ROUNDS=10 RUNS=3 scripts/bench/hybrid-gdn-ab.sh "$MODEL"
```

The helper accepts `ARMS` to narrow the matrix. `BASE_BETA`, `BASE_ROPE`, and
`BASE_GROUPED_MOE` hold unrelated switches fixed when an arm is intended to
isolate one change. For example, the MoE comparison should
keep beta and RoPE enabled while toggling only the grouped finalizer:

```sh
SURFACE=decode ARMS="control moe" BASE_BETA=1 BASE_ROPE=1 ROUNDS=10 \
  RUNS=3 scripts/bench/hybrid-gdn-ab.sh "$MODEL"
```

The GDN surface compares each arm's `baseline_seq` and replay rows against the
same row in the control arm. The decode surface measures whole-model bulk. Use
the GDN surface for beta attribution; use a MoE model for the finalizer arm.

## Correctness gates

Run these before accepting any timing result:

```sh
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib gdn_step_matches_cpu -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib gdn_output_scale_folds_into_rms_epsilon -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib metal_gdn_block_matches_cpu -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib metal_attn_block_matches_cpu -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib attn_v4_matches_naive_f16kv -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib rope_neox_pair_matches_cpu -- --nocapture
QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --lib moe_grouped_finalizer_matches_cpu -- --nocapture
```

Use adversarial GDN inputs with nonzero recurrent state, repeated Q-head
mappings, long multi-token state chains, and saturated sigmoid inputs; use
attention contexts at each tile/NWG boundary. Compare state, intermediate
output, final hidden, logits, and generated token IDs; cosine alone is not
sufficient.

## Attribution ladder

1. **Kernel:** replay fixed tensors through the old and new kernel sequence and
   collect Metal GPU timestamps. Keep input/state buffers identical.
2. **Layer:** use `decode-gdn-layer-replay` or
   `decode-gdn-chain-replay`; compare the same row across flag arms and report
   GPU time per token, dispatch count, and encoder count.
3. **Phase:** use `qwen-bench phase` or
   `decode-window --stage-timestamps --stage-split-gdn-after` to verify that the
   expected GDN, attention, or MoE bucket moved.
4. **Model:** use `tg`, `decode`, or `ctx-sweep` at short, medium, and long
   contexts. Require the same token trace and route fingerprint before timing.

For each arm, use at least 20 untimed warmups for a kernel, 10 for a layer
replay, and 20 paired timed blocks for a model run. Report median paired delta,
MAD or p10/p90, and a 95% bootstrap interval. Repeat after cooldown. A claim
requires a repeatable interval excluding zero, not merely a lower single-run
mean.

Attribute bulk with:

```text
predicted model delta = applicable calls/token * isolated delta/call
unexplained delta = observed model delta - predicted model delta
```

Use GPU timestamps for arithmetic/memory savings, wall time for dispatch and
encoder savings, and `dispatch-census` to confirm that the expected dispatch
shape actually changed. A large unexplained delta means the optimization is
interacting with neighboring work or a different fallback path.

The output-scale fold, legacy attention normalization fold, `attn_v4`
exponential cache, and test-only softmax cache currently have no runtime
rollback flag. Certify those with a paired kernel/source baseline or add a
separate pipeline variant before using them in a one-variable model-level A/B.

## Measured data (2026-07-28)

Host: Apple M4 Max, macOS 15.6.1, AC power, 128 GiB. Build source state was
`git-source-sha256-v2:b2a60186cca58f41431286160a599d663bca6b0aed81857dc7f3b09998e5e124`
from dirty commit `18ef0e975`. Model hashes:

- 0.8B F32: `53d5c025769385c3c189cba43be43af53c4c8325dcf29497b44a6cadd8437c23`
- 35B A3B Q4_K_S: `2bee952b218e4a481430c59d8d3bdc7bae20bed0eb501326340c5fca7ae95d42`

The acceptance model runs used 20 balanced randomized process-order rounds,
3 `tg128` reps per arm, and fresh processes. Intervals below are
100,000-resample bootstrap intervals over the 20 round means. The corrected
harness records all inherited `QWEN_*` settings and uses `BASE_*` flags to pin
unrelated transforms.

### 0.8B whole-model decode

The beta run held paired RoPE on; the RoPE run held beta fusion on. This keeps
the two acceptance comparisons one-variable:

- Beta fused: control `963.700 ms` wall / `915.665 ms` GPU versus fused
  `961.250 ms` / `913.140 ms`; delta `2.450 ms` wall (`0.255%`, CI
  `[0.077, 0.435]`) and `2.525 ms` GPU, CI `[1.285, 3.995]`.
- Paired RoPE: control `954.290 ms` wall / `906.320 ms` GPU versus paired
  `952.215 ms` / `905.045 ms`; delta `2.075 ms` wall (`0.217%`, CI
  `[0.033, 0.401]`) and `1.275 ms` GPU, CI `[-0.060, 2.535]`.

Paired spread was substantial relative to the small wins; the acceptance
packets are archived in the balanced log files under
`target/profiles/hybrid-gdn-ab/`.

The 8-round GDN chain replay did not separate these small transforms from
noise at 16 tokens: control was `0.342313 ms/GPU-token`, beta `0.341987`,
paired RoPE `0.343025`, raw-Q `0.343338`, and all `0.342775`.

### 35B A3B grouped finalizer

With beta and RoPE held on and raw-Q held off, grouped finalizer control versus
grouped used 20 rounds of `tg128`:

- Control: `1186.480 ms` wall, `1107.350 ms` GPU, `107.886 t/s`.
- Grouped: `1178.465 ms` wall, `1100.305 ms` GPU, `108.618 t/s`.
- Delta: `8.015 ms` wall (`0.674%`, CI `[0.483, 0.857]`), `7.045 ms` GPU,
  CI `[5.485, 8.605]`.

Corrected dispatch census at context 1024 attributes the topology change:
`851 -> 814` dispatches/token, with the finalizer stage `0.295 -> 0.232 ms`.
The old path
used 40 shared-accumulation and 37 weighted-sum dispatches; the grouped path
used 37 grouped-finalizer dispatches plus 3 shared-accumulation dispatches.
The archived corrected phase files are noisier than the census topology:
`phase_sum` is `9.41 -> 9.47 ms`, and `moe ffn apply` is `2.49 -> 2.52 ms`
for control versus grouped. Treat phase timing as non-confirmatory here; the
stable result is the `851 -> 814` dispatch topology and the balanced model
packet.

The archived 4-block, 8-slot route traces for each arm report
`route_order_mismatches=0`, `route_set_mismatches=0`, and identical summary
hidden deltas (`min_h_cos=0.999999553`). The trace compares baseline versus
replay within each process, not grouped versus control, so this is useful
partial evidence rather than a full 128-token production trace.

The corrected beta census changes `881 -> 851` dispatches and the beta
alpha family from `0.132 -> 0.069 ms`; corrected paired-RoPE census changes
`861 -> 851` dispatches. These are structural/single-token attribution data,
not substitutes for a cooled production trace.

The historical A3B raw-Q run did not show a win: control versus raw-Q had
median wall `1200.05 -> 1201.45 ms` and GPU `1111.0 -> 1112.9 ms`. Late-run
timing excursions made its 10-round mean interval span zero. The experiment is
closed; the old flag and its middle-state implementation are no longer in the
tree.

After a 300-second cooldown, the grouped A3B repeat remained directionally
positive but was noisy: control versus grouped was `1207.415 -> 1196.760 ms`
wall and `1134.355 -> 1124.785 ms` GPU, with mean deltas `10.655 ms` (CI
`[3.585, 18.720]`) and `9.570 ms` (CI `[2.585, 17.620]`). The paired wall
median was only `7.90 ms` with MAD `6.05 ms`, including large negative and
positive outliers. This repeat supports the direction but does not close the
certification gate.

The corrected post-fix dense GDN replay also failed to close beta attribution:
control `0.343415` versus fused `0.341970 ms/GPU-token`, delta `0.001600`
with CI `[-0.000180, 0.003385]`.

### 27B dense scale check

Fixture: `Qwen3.5-27B-UD-Q4_K_XL.gguf`, SHA-256
`13cb6228344898afa50d963c02ae0d991ae25094eea8837db8d0e452e91c5888`. These
were balanced 20-round packets with the same `tg128` protocol and final build
source state `git-source-sha256-v2:2b259888c0bbe4f39f8d6a4d0fd572291e16a94f2c287753418a7693dc5bc613`.

- Beta: control `5666.970 ms` wall / `5601.475 ms` GPU versus fused
  `5638.540` / `5571.935`; delta `28.430 ms` wall (`0.435%`, CI
  `[-0.887, 1.956]`) and `29.540 ms` GPU, CI `[-47.515, 120.040]`.
- Paired RoPE: control `5660.995 ms` / `5594.670 ms` versus paired
  `5694.845` / `5629.220`; delta `-33.850 ms` wall (`-0.624%`, CI
  `[-2.179, 0.599]`) and `-34.550 ms` GPU, CI `[-121.165, 33.940]`.
- Raw-Q after cooldown: control `5826.155 ms` / `5760.185 ms` versus raw-Q
  `5890.910` / `5823.860`; delta `-64.755 ms` wall (`-1.284%`, CI
  `[-4.172, 0.998]`) and `-63.675 ms` GPU, CI `[-224.425, 67.900]`.

The 27B blocks take about 5.6 seconds each and show substantially more timing
spread than 0.8B/A3B. None of the three effects is statistically separated
from zero at this scale; the beefier fixture does not reveal a hidden material
win.

### Decisions

- **Promising, not fully certified:** grouped MoE finalizer on A3B and beta
  fusion on dense 0.8B. Their balanced intervals are positive, but the full
  gate still requires a cooldown repeat and archived production equivalence
  traces.
- **Promising, not fully certified:** paired RoPE. Its topology change is
  real, but the GPU interval includes zero and the effect is near the noise
  floor.
- **Closed:** raw-Q. The A3B and 27B experiments had no positive interval, and
  the implementation retained the L2 dispatch. Reopen only with the K-fold
  that removes the preparation work, under a new flag and certification packet.
- **Not performance-certified:** source-only output-scale folding, legacy
  attention normalization folding, and cached softmax exponentials. They pass
  correctness tests but still need rollback-capable source/pipeline variants.

Artifacts are under `target/profiles/hybrid-gdn-ab/` and
`target/profiles/dispatch-census/`; the harness supports `ARMS` plus
`BASE_*` flags for one-variable isolation.

### Adversarial review status

The first `cx` review found two measurement bugs, now fixed: the stage profiler
was applying a second beta sigmoid after fused production projection, and the
GDN replay ignored fused beta while carrying recurrent state across samples.
The harness now resets replay state, records inherited flags, and balances
randomized arm order. Corrected census artifacts use the `*-corrected.json`
names, and corrected phase snapshots are archived as
`target/profiles/phase-a3b-*-corrected.log`. The second `cx` review confirms
the topology and balanced-arm fixes, but also confirms that no optimization is
fully certified yet: production arm-to-arm hidden/logit/token traces and a
clean post-cooldown repeat remain outstanding. It also identified the raw-Q
implementation as a non-winning middle state; that branch was removed rather
than shipped permanently off.
