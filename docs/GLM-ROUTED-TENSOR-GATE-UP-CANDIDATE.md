# GLM Routed Tensor Gate/Up Candidate

Base: exactly `4ace096b47cb1a9e9a5a48c45ededfe599023e1c`. Isolated branch:
`perf/glm-routed-tensor-up`. This is an unqualified, default-off candidate;
there is no performance or quality verdict and no default promotion.

## Contract

- `QWEN_GLM53_TENSOR_GATE_UP=1` requests the existing N32 Q4_K/Q5_K
  F16-activation tensor gate and clamped-up kernels. Two dispatches remain;
  this is not a launch or weight-reread fusion. No shaders, expanded weights,
  codecs, decode paths, V4 paths, or precision modes change.
- Production selection requires resolved Fast/Half routed Fast, H4096,
  F2048, E288, top8, equal actual Q4_K or Q5_K gate/up storage, actual
  R1024/R2048, and the selected M5 tensor tier. Exact, strict F32, other
  storage, short packets, M4 and global tensor rollback retain the old route.
- Convert the freshly produced FFN `normed` input every time. The single
  admitted `normalized_half` allocation is shared with KDA, never its old
  contents. KDA execution control remains independent. Either eligible owner
  allocates it, both owners still allocate only once, and TLS/short views
  never discount the full admitted capacity (16 MiB at 2048 rows).
- Unweighted F32 `[2048, 8R]` inner -> incumbent F32-input Q5 tensor or Q6
  generic down -> post-down routing-weighted sum. The unchanged epilogue is
  `silu(min(gate, limit)) * clamp(up, -limit, limit)`. Q5-down rollback
  explicitly declines the new gate/up candidate rather than overriding down.
- All physical extents, devices, alignment, overflow, serial encoder and
  writable-output disjointness are checked before conversion/bucketing.
  Converter, gate, up, effective down, bucket and weighted-sum pipelines are
  resolved before any capacity refusal returns. Missing eligible pipelines
  and malformed bindings error without mutation; capacity refusal uses the
  old full composition with the current down policy. No extra barriers,
  command buffers or waits are introduced.

Half operands do not make the new route bitwise equal to the current generic
route: tensor reduction order differs. Bitwise comparison is only between
F32-input tensor and copy-inclusive F16-input tensor. Sampled Half/F64 and
full-F32/F64 oracles plus the model screen assess the generic-to-tensor change.

The resource specification supports a routed owner without any KDA metadata.
Full-session validation separately pins the release KDA/MLA block layout and
rejects a synthetic all-MLA config. Tests verify the no-KDA case at the shared
buffer-list/pricing boundary; this candidate does not broaden model support.

## Deferred Qualification

No GPU execution, remote work, full release/LTO build, benchmark, installation,
publication, or merge is part of this handoff. Local offline typechecks cannot
qualify the Metal tensor tier: the local compiler probe rejects `metal4.1`.
The parent must authorize builds after the concurrent builders finish.

Build from this worktree, using the established toolchain and a tensor-capable
Metal SDK for the later M5 qualification binary:

```sh
cargo test --offline --release -p qwen-llm --lib --no-run --message-format=json
cargo build --offline --release -p qwen-cli --bin qwen-bench
```

The first command reports the exact native test executable in its
`compiler-artifact.executable` JSON field. The second produces
`target/release/qwen-bench`. Neither command has been run here; do not substitute
a dev/test-profile binary for the release model screen. Preserve binary hashes
when transferring through the parent's established workflow.

Exact ignored primitive tests in that binary:

```text
glm5_next_metal::tensor_down::tests::tensor_gate_up_gpu::m5_gate_up_native_compositions_tensor_f32_f16_bitwise_and_sampled_f64
glm5_next_metal::tensor_down::tests::tensor_gate_up_gpu::m5_gate_up_whole_composition_preflight_failures
```

These reuse the down fixture's native-byte generator, guarded buffers, lease
and no-mutation helpers. H256/F256/E32 tiny compositions cover Q4/Q4 -> Q5 tensor,
Q4/Q4 -> Q6 generic, Q5/Q5 -> Q6 generic; production eligibility rejects E16/E32.
The conservative GPU+CPU+reserve peak must fit 1 GiB before allocation. They
cover bucket edges around 16/32/64, empty/hot buckets, distinct banks, clamp
sides, immutable inputs, slot/backing guards, whole-preflight refusals and
copy-inclusive F32/F16 tensor bit equality. They are not performance tests.

Exact ignored model test:

```text
glm5_next_metal::tests::quality::tensor_prefill::tensor_gate_up::m5_gate_up_actual_q4_three_document_smoke_profile_plain_abba
```

After separate GPU authorization, run each named test directly with
`<test-executable> <exact-name> --exact --ignored --test-threads=1` under the
normal production GPU lease. Primitive tests require `MTL_DEBUG_LAYER=1` and
`QWEN_METAL_LEASE_WAIT=1`. The model screen additionally requires:

```text
GLM53_GGUF=<standard first-shard path for the pinned actual six-shard Q4 model>
GLM53_TENSOR_GATE_UP_REPORT=<new private JSONL path, must not exist>
QWEN_GLM53_TENSOR_GATE_UP=1
QWEN_GLM53_KDA_HALF_INPUTS=0
QWEN_GLM53_DENSE_Q8_WIDE=1
QWEN_GLM53_TENSOR_Q5_DOWN=1
QWEN_EXPERT_TENSOR_TILE=1
QWEN_MATMAT_Q6_K_N64=1
QWEN_MATMAT_Q6_K_N64_MIN_N=1024
```

Global gate/up ON owns identical physical Half storage in both model arms;
only its TLS false/true override changes during prefill. Continuation uses
ordinary forward outside all prefill scopes. No tools or 22-document expansion:
three frozen documents, 64 teacher-forced targets each, 384 full-vocabulary
rows across both arms; finite NLL/KL/top1 counts; two warm censuses, two
diagnostic profiles, and eight ordinary ABBA/BAAB samples at 2048 tokens.
Profiles are observer-perturbed diagnostics, not recoverable wall-time shares.
Normal full model/session GPU+CPU admission precedes weights/sessions. The
create-new mode-0600 report records source/binary/fixture hashes, model pin and
retained stamps; no complete event is emitted without all expected rows/calls
and final state/stamp checks. Payload hashes are not reverified: identity scope
is explicitly recorded, as in the existing actual-Q4 screen.

The historical `tensor_m5_transfers.rs` body is unchanged. The KDA GPU fixture
is also untouched; the parent's forthcoming test-only `view_bytes` fix and
lean KDA model smoke must be preserved when integrating the shared scratch
rename. No overlap with that fixture is introduced here.

## Local Verification

- `cargo check --offline --workspace --tests` passes.
- `cargo check --offline --release --workspace --tests` passes; metadata/typecheck
  only, no release executable or LTO build.
- `cargo fmt --all -- --check` and `git diff --check` pass.
- Dev-profile CPU execution passes 19 focused contracts: four new gate/up,
  seven KDA, two Q5-down policy, two tensor-down geometry/overflow, three
  incumbent down policy/geometry, and the existing packet expert selector.
- The four new contracts and seven KDA contracts also pass in a fresh process
  with `QWEN_GLM53_TENSOR_GATE_UP=1`.
- All GPU numerical, refusal, census, model-quality and timing witnesses remain
  unexecuted. The local SDK's tensor compiler probe is unavailable as noted
  earlier; CPU/typecheck success is not shader or numerical qualification.
