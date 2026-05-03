# qwen-llm: v1 plan

A from-scratch inference engine for the Qwen 3.5 / 3.6 hybrid Gated DeltaNet
family (architecture name `qwen3_5` in HF transformers; `LLM_ARCH_QWEN35` in
llama.cpp), targeting Apple Silicon (M4 Max top-bin, 546 GB/s, 128 GB unified).

The single goal: **maximum tok/sec for prompt processing and token generation**,
single-stream and batched, on the dense 27B variant.

## Baseline to beat

llama.cpp `b8995` on this M4 Max, via `llama-bench` on a **quiet box** (no
competing workloads):

| model                  | size      | pp512          | tg128          | decode BW | % of 546 GB/s |
| ---------------------- | --------- | -------------- | -------------- | --------- | ------------- |
| Qwen3.6-27B Q4_K_M     | 15.65 GiB | 234.62 ± 3.72  | 21.21 ± 0.12   | 357 GB/s  | **62%**       |

> Earlier runs reported lower numbers (197 pp / 11 tg = 32% peak) — that was
> a contended-box measurement. Re-baselined cleanly on 2025-05-02.

**The honest framing**: llama.cpp's full-decode is already at 62% of peak,
much closer to the bandwidth ceiling than the prior 32% suggested. Our
**isolated** Q4_K/Q6_K kernels run at 77-88% peak, but capturing that
headroom end-to-end requires beating llama.cpp's already-good dispatch +
graph-walk overhead, not just writing fast kernels.

**v1 target:** match llama.cpp's tg128 (~21 t/s) end-to-end. The bar is
already in "high single-digit % gains" territory, not 2×.

**v2 target:** ≥26 t/s tg128 (≈30% over llama.cpp baseline), ≥280 t/s pp512.
Beyond requires NEXTN/MTP speculative decoding (1.5-2× effective decode
multiplier per the architecture spec).

**v3 stretch:** ≥35 t/s tg128 with NEXTN, ≥400 t/s pp512.

### Per-kernel headroom (cargo bench, persistent buffers, chained64)

| kernel                          | shape (n_in × n_out) | size   | GiB/s  | % peak |
| ------------------------------- | -------------------- | ------ | ------ | ------ |
| Q4_K mat-vec / attn_gate        | 5120 × 6144          | 17 MB  | 449    | **88%** |
| Q4_K mat-vec / ffn_gate         | 5120 × 17408         | 47 MB  | 413    | **81%** |
| Q4_K mat-vec / ffn_up           | 5120 × 17408         | 47 MB  | 415    | **82%** |
| Q4_K mat-vec / embed            | 5120 × 248320        | 715 MB | 393    | **77%** |
| Q6_K mat-vec / attn_qkv         | 5120 × 10240         | 41 MB  | 438    | **86%** |
| Q6_K mat-vec / ffn_down         | 17408 × 5120         | 70 MB  | 404    | **79%** |
| Q6_K mat-vec / output           | 5120 × 248320        | 1 GB   | 447    | **88%** |
| Q6_K mat-vec / attn_v (skinny)  | 5120 × 1024          | 4 MB   | 270    | 53%    |

The skinny attn_v shape is the only outlier — GQA's n_kv=4 limits parallel
work. On 27B it represents <1% of weight bytes touched per token, so the
end-to-end impact is small.

## Decisions

### Language: Rust + `objc2-metal` 0.3.2

`metal-rs` is officially deprecated in favor of `objc2-metal`, which exposes the
full Metal 4 surface (`MTL4CommandBuffer`/`MTL4CommandQueue`) behind cargo
features auto-generated from Apple headers. `Retained<T>` Drop-based ARC removes
the manual-release class of bug that `metal-cpp` is known for. Per-dispatch
overhead is identical to C/C++/Obj-C — `objc_msgSend` cost is the same regardless
of host language; the only thing that matters is what runs inside the kernel.

Second choice is `metal-cpp` (Apple's official C++ headers). Pick that only if
we ever want to share host code with a Swift app target. We don't.

### Compute backend: 100% custom Metal compute shaders

Rejecting MPSGraph: cannot keep the 128×128 fp32 GDN state in registers /
threadgroup memory across kernel invocations — every MPSGraph executable
boundary is a device-memory checkpoint. For 48 GDN layers × decode steps, this
is the path; ceding it forfeits the bandwidth ceiling on three-quarters of the
model. Every serious Apple-Silicon LLM stack rolls its own kernels (llama.cpp,
MLX, mistral.rs, vllm-metal, candle-metal-kernels). MPSGraph is for application-
layer ML, not hot-path inference.

Rejecting ANE-for-GDN at v1: the most rigorous public attempt at this exact
workload (`thebasedcapital/ane-infer`, March 2026 — Rust + Obj-C + 13 Metal
shaders, Qwen3.5 hybrid) explicitly concludes "we match llama.cpp decode speed,
not exceed it." Stateful CoreML on ANE is broken for `seq_len > 1`
(coremltools #2553). Recurrent state must round-trip CPU↔ANE every layer,
every token, until Apple ships a public stateful-ANE API. **Defer to v3
research spike.**

Rejecting CPU-only Accelerate/AMX: bandwidth-bound on the wrong fabric
(~110 GB/s vs 546 on the GPU). Use Accelerate for tokenizer prep and sampling.
Not the inference path.

### Weight format: three roles, decided independently

The original collapsed answer ("GGUF good") smushed three orthogonal questions.
Restating cleanly:

| role                                          | decision                                  | argument                                                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| user-facing distribution + persistent storage | **GGUF**                                  | mmap-native single-file aligns with unified memory; embedded tokenizer + KV metadata removes the "did the user grab all the sidecar files" failure; predictable K-quant degradation curves (Q4_K_M ≈ 1% MMLU loss vs FP16) are a real shipping virtue; ecosystem lingua franca for a local app.     |
| internal kernel tile layout (GPU hot path)    | **K-quants consumed in place at v1**      | Preserve mmap-from-cold-start as the v1 invariant. K-quant blocks are CPU-SIMD-shaped (32-element groups, shared 6-bit scales + 4-bit mins); Metal wants larger coalesced tiles. We pay the 5–15% efficiency cost on Q4_K mat-vec/mat-mat. Profile-driven re-pack-on-load becomes a v2 question, **not** an a-priori commitment. |
| numerical oracle for kernel validation        | **F32 GGUF + `llama-cli` byte-for-byte**  | Overdetermined convenience choice, labeled as such. llama.cpp is the de facto `qwen3_5` reference today; if a better oracle appears (HF transformers v4.57.1 dump, mlx-lm) we swap it without touching anything else. Not laundered into the principled case for the format.                       |
| codec for tooling / non-hot-path dequant      | **`llama-cpp-sys-2::ggml_get_type_traits().to_float`** | Universal quant coverage (Q*_K, IQ*, Q*_0/1, BF16, MXFP4, F16, F32) via a single function pointer per format. ResponseForest already validates this seam end-to-end on every quant on disk. Use it for: GGUF metadata + tensor table parse, ad-hoc CPU verification, debug dumps, the v2 re-pack-on-load path, any tensor we want to keep in fp32 host memory (e.g. embedding rows for diagnostic runs). **Not** for the GPU hot path. |

The mmap invariant is binding for the GPU hot path: the v1 loader path is
`mmap → metadata parse → tensor descriptor table → done`. No H2D copy, no
allocator pressure at startup. Metal kernels accept K-quant block layouts
directly; this is what `kernel_mul_mv_q4_K_f32_impl` (llama.cpp
`ggml-metal.metal:7716`) already does, and lifting it costs us nothing extra.

The temptation to layer on top of `llama-cpp-2` as a whole engine (use ggml's
graph executor, write custom Metal only for GDN-step + NEXTN) is real and
explicitly rejected. The throughput ceiling we're trying to capture is in
*graph walk + per-dispatch encoding overhead*, not in any single kernel — the
moment we cede the graph executor to ggml we've ceded the thing we're trying
to optimize. The seam is for codec, not for compute.

### Numerical correctness

Byte-for-byte logits comparison vs `~/code/llama.cpp/build/bin/llama-cli` on
`~/models/Qwen3.5-0.8B.F32.gguf` (3 GB, 24 layers, identical kernel ops to 27B
modulo head counts and FFN width). F32 means no quant ambiguity — any
divergence is a bug in our kernels. This runs on every commit pre-merge.

## Architecture facts pinned

From the architecture investigation (sources: HF config.json, transformers
`qwen3_next/modeling_qwen3_next.py` v4.57, llama.cpp `src/models/qwen35.cpp`,
vLLM PR #37975, exo PR #1644):

- 64 layers, hidden 5120, FFN intermediate 17408, vocab 248,320 (padded), `tie_word_embeddings: false`
- Layer pattern: `[linear×3, full×1] × 16` → 48 GDN + 16 full-attn. Indices of full-attn: 3, 7, 11, …, 63.
- Full-attn ("Gated Attention"): 24 Q heads / 4 KV heads (GQA 6:1), `head_dim=256`, partial RoPE (64 of 256 dims), `rope_theta=10_000_000`, MRoPE sections `[11,11,10]`, `qk_norm=True` (per-head RMSNorm), `attn_output_gate=True` (q_proj outputs 2× — second half is sigmoid → multiplied with attn output before o_proj), no sliding window.
- GDN: 48 V heads, 16 K heads (V/K=3), `head_dim=128`, conv1d kernel=4 depthwise with SiLU, scalar gate per head (`g = -exp(A_log) * softplus(a + dt_bias)`, β = sigmoid(b)), L2-norm Q/K inside the kernel (eps=1e-6), `mamba_ssm_dtype=float32` (state in fp32 — non-negotiable, see drift evidence in mlx-lm PR #1066 and ollama #15865).
- FFN: standard SwiGLU (gate_proj, up_proj, down_proj), no MoE on the 27B (`mlp_only_layers: []`).
- MTP: `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=False`. Weights *are* in the released safetensors / GGUFs (under `mtp.*` prefix); HF transformers ignores them.
- GDN projection layout: **separated** (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) for Qwen3.5/3.6, *not* fused (`in_proj_qkvz`, `in_proj_ba`) like Qwen3-Next-80B. This is the most common loader bug source — verify against the safetensors / GGUF tensor list, do not assume.

KV cache (16 attn layers, BF16): 64 KB per token. At 256K context = 16 GiB.
GDN state: 48 layers × 48 V heads × 128² × 4 B fp32 = 144 MiB **constant** —
context-invariant. This is the architecture's headline feature.

## Reference quarry

Kernels to lift verbatim from `~/code/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (MIT):

- `kernel_gated_delta_net_impl` (line 2537) — fused, register-resident GDN step with NSG=1/2/4 templating. The kernel.
- `kernel_ssm_conv_f32_*` (line 2077) — conv1d kernel=4 depthwise + SiLU.
- `kernel_l2_norm_impl` (line 3062).
- `kernel_mul_mm_q4_K_f32` template (line 10077) — Q4_K mat-mat for prefill.
- `kernel_mul_mv_q4_K_f32_impl` (line 7716) — Q4_K mat-vec for decode.
- `kernel_set_rows_*` (line 9224+) — paged KV writes; `kernel_set_rows_q8_0` (10044) for KV-Q8.
- `kernel_rms_norm`, `kernel_silu`, `kernel_mul`, `kernel_add`, `kernel_rope*` — boring but needed.

Design ideas to lift from `~/code/vllm-metal/vllm_metal/metal/kernels_v2/` (MIT via mistral.rs):

- `gdn_linear_attention.metal` — `cu_seqlens` + `slot_mapping` argument layout for varlen continuous batching (better than llama.cpp's fixed-shape layout for our v2 batched path).
- `pagedattention.metal` — paged KV reference for the 16 full-attn layers.

Reference for the fp32 state pool layout: `vllm_metal/mlx_backend/gdn_cache.py`
(`[max_seqs, n_gdn_layers, num_v_heads, value_head_dim, key_head_dim]`).

## v1 scope (target: match llama.cpp tg128 ≈ 21 t/s on 27B Q4_K_M)

> Reordered after a comprehensive code review that surfaced two architectural
> issues blocking forward motion: the test-shaped Metal API and the
> position-blind `KvCache`. Both will become tech debt within hours of
> writing more kernels on top, so we fix them first.

### v1.0 — foundation (current)

1. ✅ Project skeleton, GGUF loader, model binder, tokenizer.
2. ✅ CPU reference forward bit-tight against `llm`/`llama_core` snapshot
   dump (cosine = 1.000000 on Qwen3.5-0.8B-F32 single + multi-token, and
   on Qwen3.6-27B-Q4_K_M single token).
3. ✅ Metal kernels for RMSNorm, F32 mat-vec, Q4_K mat-vec (fast lifted),
   Q6_K mat-vec — all validated against the CPU oracle.
4. ✅ Criterion bench suite. K-quant kernels at 77-88% peak BW chained64.

### v1.1 — production Metal API + foundational fixes

5. **Fix `KvCache` semantics**: explicit `(layer, seq_slot, position)`
   addressing instead of inferred `len/n_pos`. CPU oracle should assert
   `position == n_pos` for now; GPU equivalent should be paged from day 1.
6. **27B multi-token oracle test** (5-9 tokens). Currently the n_v=48 /
   n_k=16 head-repeat path is exercised against zero SSM state only.
7. **Production Metal types**: `MetalTensor`, `MetalModel`, `MetalSession`,
   `ActivationArena`, `ScratchArena`. Encode-only kernel API (caller owns
   command buffer; kernels never commit/wait internally). Test wrappers
   keep the `Vec<f32>`-readback API behind `*_readback_for_test` names.
8. **MetalModel loader**: walk every `TensorDesc`, allocate one
   `MTLBuffer` per tensor at load (one-time copy from mmap is acceptable;
   true zero-copy via `newBufferWithBytesNoCopy` is a v2 question).
9. **Migrate existing dispatchers** (Q4_K, Q6_K, F32, RMSNorm) to the
   encode-only API. Update tests + benches.

### v1.2 — end-to-end Metal forward

10. **Metal kernels lifted from llama.cpp** (correctness-first, final
    state layout — *do not* CPU-bridge GDN):
    - `get_rows` (embedding lookup)
    - `ssm_conv` + SiLU (GDN front-end)
    - `l2_norm` (Q/K normalization in GDN)
    - `gated_delta_net` (the recurrence; FP32 state non-negotiable)
    - `rmsnorm_gated` (post-recurrence norm)
    - elementwise: silu, sigmoid, softplus, add, residual_add (fuse where possible)
    - **Fused full-attention block**: q/k norm + RoPE + KV append + attention + gate + output
      (a standalone softmax kernel would explode dispatch count)
11. **End-to-end single-token Metal decode** on Qwen3.5-0.8B-F32. One
    command buffer per token, one logits readback. Cosine = 1.0 vs CPU oracle.
12. **27B Q4_K_M Metal decode**, validated vs `llm`/`llama.cpp`.

### v1.3 — beat the baseline

13. **`qwen-bench` end-to-end vs `llama-bench`** via `hyperfine`. Same prompt
    corpus, same model file. Target: ≥21 t/s tg128 (match), stretch ≥26 t/s.
14. **Q4_K/Q6_K mat-mat for prefill** — pp512 path. Decode mat-vec doesn't
    buy prompt throughput. llama.cpp `kernel_mul_mm_*` template is the lift.
15. **Profile dispatch count and CPU encode time per token.** At 5-20 µs
    per dispatch and ~1100-1400 dispatches/token naive (~700-900 with
    obvious fusions), dispatch overhead alone could consume 5-20 ms of
    the ~45 ms/token budget. This is the *real* ceiling on tg, not kernel
    perf. Measure before tuning kernels further.

## v2 scope (target: ≥26 t/s tg128 with stretch toward NEXTN-amplified 35+)

1. **NEXTN multi-token-prediction speculative decode** with the native
   head (1.5-1.7× effective tg). Largest single tg win on the table.
2. **Indirect Command Buffers (ICB)** — encode once, reuse every step.
   Designed for from v1: stable buffers, fixed pipeline sequence, dynamic
   scalar args only.
3. **Paged KV on the 16 attn layers**. Lift mistral.rs's PagedAttention
   v2 layout. Required for prefix reuse, multi-request batching.
4. **KV-Q8 on the 16 attn layers**. Halves attn-layer BW.
5. **GPU top-k / argmax**. Avoids full logits readback for sampling.
6. **MTL4** (`MTL4CommandBuffer`) — `objc2-metal` 0.3.2 already exposes
   the bindings.
7. **Adversarial Q4_K/Q6_K block fixtures** — synthetic shapes covering
   nibble-edge values, sign handling, sub-block boundaries. Codex's
   prior review flagged that single-real-tensor validation is necessary
   but not sufficient.
8. **Profile-driven re-pack-on-load decision**: only if the K-quant block
   walk shows up dominant after (1)–(7).

## v3+ research spikes (non-blocking)

- **Beat MLX too** — `mlx_vlm` is the other strong Apple-Silicon Qwen3.5 path; `hyperfine` comparison once v1 lands.
- ANE for FFN-only prefill on the 16 full-attn layers (1×1 conv via `_ANEInMemoryModel`, lifting `ane-infer`'s playbook). Not GDN.
- Continuous batching scheduler (the `cu_seqlens` / `slot_mapping` layout in v2 is already designed for this).
- Cross-request KV reuse / prefix caching. Cheap once paging exists.
- Speculative draft via `Qwen3.5-0.8B-BF16.gguf` for non-NEXTN paths.

## Risk register

| risk                                                                          | mitigation                                                                                                                                                |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GDN projection layout mismatch (fused vs separated)                           | Read `model.safetensors.index.json` / GGUF tensor list before writing the loader. llama.cpp's `convert_hf_to_gguf.py:5254` `_LinearAttentionVReorderBase` is the reference. |
| State drift from BF16 storage                                                 | FP32 state buffer, full stop. `mamba_ssm_dtype=float32` per config. ollama #15865 is the cautionary tale.                                                 |
| GDN order-of-operations bug (decay before vs after retrieve)                  | Numerical oracle test catches this on first divergence. F32 path means no quant noise to mask it.                                                         |
| llama.cpp Metal kernel API churn                                              | Pin the lifted kernels to a known-good `~/code/llama.cpp` commit. Re-port quarterly with diff review.                                                     |
| Apple Silicon Metal compiler regression across OS versions                    | `MTLBinaryArchive` ships compiled pipelines; CI runs on every supported OS.                                                                               |
| Qwen3.5/3.6 tokenizer differs from earlier Qwen (vocab 151,936 → 248,320)     | Hardcode 248,320 path. Tokenizer tests on canonical strings vs `llama-tokenize`.                                                                          |
| MTP head present in GGUF, our v1 loader doesn't expect it                     | v1 loader ignores `mtp.*` tensors explicitly. v2 turns them on.                                                                                           |
