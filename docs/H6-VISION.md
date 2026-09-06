# H6: Native vision for Qwen 3.5 / 3.6

Status: implementation brief, 2026-08-06.

## 1. Decision

Add native image input to the existing Qwen 3.5 / 3.6 runtime by loading a
separate Qwen3-VL-compatible `mmproj` GGUF, encoding images into language-model
embedding rows, and feeding those rows through the existing hybrid decoder.

This is one implementation for both releases:

- Qwen3.5 dense uses HF `Qwen3_5ForConditionalGeneration` and GGUF `qwen35`.
- Qwen3.5 MoE uses HF `Qwen3_5MoeForConditionalGeneration` and GGUF
  `qwen35moe`.
- Released Qwen3.6 checkpoints use the same HF classes, model types, vision
  ABI, and decoder MRoPE contract. This is runtime architecture compatibility,
  not evidence that projector weights are interchangeable across checkpoints.
  Require a projector distributed or converted for the selected checkpoint.

The language-model implementation is mostly reusable. The new work is a
vision front end plus a mixed-embedding/position-aware prefill seam:

```text
image bytes
  -> bounded RGB decode
  -> Qwen smart resize and normalization
  -> Conv3D-equivalent patch projection
  -> Qwen3.5 vision transformer
  -> 2x2 spatial merger into LM-width rows
  -> splice rows between vision_start / vision_end token embeddings
  -> existing Qwen35 or Qwen35MoE decoder
```

There is no cross-attention block. Image embeddings enter the residual stream
where ordinary token embeddings normally enter.

## 2. Initial scope

H6 ships in this order:

1. One local image with a dense Qwen3.5 model.
2. Multiple local images in message encounter order.
3. The same path with Qwen3.5 MoE and Qwen3.6 dense/MoE checkpoints.
4. Multimodal-safe snapshots and prefix caching.
5. Performance work after numerical parity.

V1 includes:

- PNG and JPEG input from local files or caller-provided bytes.
- Dynamic native-resolution preprocessing with explicit pixel limits.
- F32/BF16 Qwen3.5/3.6 mmproj files.
- Dense and MoE language backbones through one shared input path.
- Raw prompt media markers and structured message content parts.
- Correct image MRoPE, post-image logical positions, and cached decode.
- Text-only behavior with no mmproj loaded remains unchanged.

V1 explicitly excludes:

- Video, frame sampling, and timestamp prompt rewriting.
- Audio.
- Remote URL fetching.
- Animated images.
- Standalone Qwen3-VL DeepStack injection.
- A Qwen3.5/3.6 projector with non-empty `deepstack_visual_indexes`.
- Quantized projector kernels until BF16/F32 is oracle-qualified.
- Prompt lookup, MTP, and DFlash on multimodal requests.
- Multimodal prefix-cache reuse until image identity is part of the key.

Unsupported combinations must fail closed with a precise error. They must not
silently fall back to text-only behavior or ignore projector outputs.

## 3. Pinned bring-up fixture

Use OvisOCR2 as the first small, high-signal fixture:

```text
text model: ~/models/OvisOCR2/OvisOCR2-Q8_0.gguf
mmproj:     ~/models/OvisOCR2/mmproj-BF16.gguf
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `OvisOCR2-Q8_0.gguf` | 811,843,498 | `3fba6d94312e550575a92d55cffb8d75997bf68a9f133c92ab6ad7f0bf2bc93e` |
| `mmproj-BF16.gguf` | 207,346,376 | `19040aa7e90af72567f397c9398f0203b13555c3adbc625d01992f77735e2d8f` |

The fixture exercises the same Qwen35 Gated DeltaNet/full-attention decoder as
larger models while keeping iteration cheap. Its vision configuration is:

```text
vision depth             12
vision width             768
vision FFN               3072
vision heads             12 (head_dim 64)
learned position grid    48 x 48 (2304 rows)
patch size               16 x 16
temporal patch size      2
spatial merge            2 x 2
projected LM width       1024
deepstack                disabled
```

The mmproj contains 154 tensors. Matrix weights are mostly BF16; patch weights,
position embeddings, biases, and LayerNorm parameters are F32. The current
GGUF parser and Metal tensor loader already support both storage types.

Current llama.cpp oracle on the formula-heavy paper page 7:

```text
image encode             372 ms
decoder prompt eval      629 ms / 2026 rows
decode                   247.29 tokens/s
end-to-end wall          5.35 s
peak RSS                 about 1.44 GiB
```

Benchmark input and provenance:

```text
source image             page-07.png, 1224 x 1584, 214,771 bytes
source SHA-256           92e61f9eb31a931e4dcdd0dd78f05e90ca257192403b103d2b5a07625be05789
resized grid             1216 x 1600
patch / merged grid      76 x 100 / 38 x 50 (1900 image rows)
llama.cpp revision       6a32c29a746a2e44de463de647f9f6661eb5086b
host                     Apple M4 Max, 40 GPU cores, 128 GiB
OS                       macOS 15.6.1 (24G90)
```

The numbers above are one observed `--no-warmup` run, not a statistical
performance gate. H6.0 must add a fixed warmup/run-count harness and report
median and p95 for image encode, mixed prefill, TTFT, RSS, and decode.

The Q8 output was byte-identical to BF16 on tested pages 2, 7, and 13. Preserve
these output hashes as coarse end-to-end regression fixtures:

```text
page 02  d77d68a9c79771ffbb73edf16c156ef8770d5e217a2cf59c2fb296814670d6b9
page 07  db147039c536f65974650a5035629ba183e3534899bc348a10d07b751a8c916d
page 13  2e1b3c8352cb408f5fc4b11c236b7f02d3633612327060b05c745526190c20e4
```

Generated Markdown is not the numerical oracle, especially because current
llama.cpp and Transformers use different resize filters. It is a useful
product tripwire after layer-level parity passes, not authority over tensor and
logit parity.

## 4. Architecture contract

### 4.1 Model-family compatibility

Detect compatibility from metadata, not filenames or marketing versions.

Accepted language architectures:

```text
qwen35
qwen35moe
```

Accepted HF parent architectures during conversion:

```text
Qwen3_5ForConditionalGeneration
Qwen3_5MoeForConditionalGeneration
```

Qwen3.6-27B and Qwen3.6-35B-A3B use those same contracts. Do not introduce a
parallel `qwen36` vision implementation.

Released tower families are metadata-driven:

| Models | Depth | Width | FFN | Heads |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 12 | 768 | 3072 | 12 |
| Qwen3.5-2B / 4B | 24 | 1024 | 4096 | 16 |
| Qwen3.5-9B / 27B and released MoE/3.6 | 27 | 1152 | 4304 | 16 |

All currently released Qwen3.5/3.6 configs use:

```text
patch_size               16
temporal_patch_size      2
spatial_merge_size       2
num_position_embeddings  2304
hidden_act               gelu_pytorch_tanh
deepstack_visual_indexes []
```

The implementation must still derive dimensions from GGUF metadata and tensor
shapes. The table is an oracle matrix, not a hardcoded switch.

### 4.2 Image preprocessing

The image path must match Qwen's processor, including details that are easy to
approximate incorrectly.

1. Decode into RGB8 with bounded allocation.
2. Reject zero dimensions and aspect ratios greater than 200.
3. Let `factor = patch_size * spatial_merge_size`, currently 32.
4. Preserve aspect ratio and round target dimensions to multiples of `factor`.
5. If area exceeds `max_pixels`, scale down and floor to `factor`.
6. If area is below `min_pixels`, scale up and ceil to `factor`.
7. Resize with the pinned Transformers processor's bicubic resampler,
   including its coordinate, antialiasing, and conversion semantics.
8. Convert channels to float and apply `(pixel / 255 - mean) / std`.
9. Duplicate a still image across the two-frame temporal patch depth.

The official general Qwen image profile uses area bounds 65,536 through
16,777,216 pixels. OvisOCR2's official OCR runner uses 448^2 through 2880^2.
Current mmproj files do not reliably carry these policy values, so expose them
as runtime options with a named model/profile default. Include the selected
limits and preprocessing ABI in cache identity.

Current llama.cpp selects bilinear resizing for this projector. Treat that as
a known behavioral difference and latency baseline, not the preprocessing
numerical oracle.

One merged LM image token covers a 32 x 32 source-pixel region at the current
patch and merge sizes. For a resized image `(W, H)`:

```text
patch grid       = (W / 16, H / 16)
merged grid      = (W / 32, H / 32)
LM image rows    = (W / 32) * (H / 32)
```

For the initial image-only path, both temporal Conv3D slices see the same
frame. The converted mmproj stores them separately as
`v.patch_embd.weight` and `v.patch_embd.weight.1`; their projections are
summed before adding `v.patch_embd.bias`.

V1 uses Transformers-style merge-tile-major flattened patch rows, ordered as
`(block_h, block_w, intra_h, intra_w, channel, temporal, patch_h, patch_w)`.
Do not perform a second post-projection reorder. A future raster-input GPU path
may instead reorder once after projection, but the two strategies are mutually
exclusive and must share an ordering oracle.

### 4.3 Vision tower

The tower is a native Qwen3.5 vision encoder, represented in GGUF as a
Qwen3VL-style projector. It is not CLIP and must not be forced into CLIP
semantics beyond the existing `clip.*` GGUF namespace.

Graph order:

```text
merge-tile-major flattened patch rows
two temporal patch projections, summed
patch bias
bilinear learned-position interpolation (align_corners = true)
add learned positions

repeat for every vision block:
  LayerNorm (mean subtraction, variance, weight, bias)
  fused QKV projection + bias
  2D vision RoPE
  non-causal full self-attention
  output projection + bias
  residual add
  LayerNorm
  FC1 + bias
  GELU tanh approximation
  FC2 + bias
  residual add

merger LayerNorm on each unmerged patch
concatenate each 2x2 patch group
merger FC1 + bias
exact/erf GELU
merger FC2 + bias -> language-model width
```

Hard correctness requirements:

- Use ordinary LayerNorm, not the decoder's RMSNorm.
- Learned position interpolation is bilinear with `align_corners=true`.
- Vision RoPE is a separate 2D operation from decoder interleaved MRoPE.
- Vision attention is non-causal.
- Preserve Qwen's tile-major patch ordering through attention and merger.
- The merger emits exactly `T * H * W / 4` rows.
- A still image uses both temporal patch weights.
- Vision-block MLPs use `gelu_pytorch_tanh`; the merger uses exact/erf GELU.
- Block and merger LayerNorm epsilon is exactly `1e-6` for the supported ABI.
- Bind the two GELU modes from the Qwen3.5 projector ABI; `clip.use_gelu` alone
  does not encode their distinction.

### 4.4 Language-model integration

The prompt contains ordinary embeddings and externally produced image rows:

```text
<|vision_start|> [projected image rows] <|vision_end|>
```

The source template may contain one `<|image_pad|>` marker, but processing
expands it logically to one `image_token_id` position per merger row. Those
token embeddings are replaced by image features. An optimized span plan may
avoid materializing repeated IDs only if token types, positions, masks, and
placeholder-count validation remain equivalent to Transformers. Count/order
mismatches are fatal.

Represent decoder position explicitly:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderPosition {
    pub cache_row: u32,
    pub causal: u32,
    pub mrope: [u32; 3], // temporal, height, width
}
```

Track three concepts independently: physical cache append row, causal/text
position, and rotary `(T,H,W)`. Transformers exposes three rotary planes plus
separate cache/mask state. llama.cpp transports `(T,H,W,reserved)` while cache
storage remains separate. Do not infer one coordinate class from another.
In V1 single-sequence execution, `causal` equals the sequence-relative physical
row ordinal. It may differ from `cache_row` only for cache allocation or future
batching, never because MRoPE compresses the logical position range.

For text at logical position `p`:

```text
cache_row = next physical row
causal = next causal sequence row
mrope = [p, p, p]
```

Let `<|vision_start|>` be a normal text-classified token at logical position
`s`, and let `p = s + 1` be the first expanded image-token position. Merged-grid
row `(h, w)` gets:

```text
cache_row = next physical row
causal = next causal sequence row
mrope = [p, p + h, p + w]
```

After the image, physical cache position advances by every image row while the
logical text cursor advances only by `max(1, merged_height, merged_width)`.
Assign `<|vision_end|>` the text MRoPE position
`p + max(1, merged_height, merged_width)`, then increment once before following
text. This is the rope-delta effect. Cached decode must continue from the
logical cursor, not from the larger physical row count. Pin a complete
two-image position-vector fixture to catch marker-level off-by-one errors.

The decoder's full-attention layers rotate 64 of each 256-dimensional head.
They use theta 10,000,000 and interleaved sections `[11, 11, 10]`: frequency
pairs are assigned in T/H/W interleaved order, not three contiguous blocks.
The GGUF converter serializes this as `[11, 11, 10, 0]` under
`qwen35.rope.dimension_sections` or
`qwen35moe.rope.dimension_sections`. Require four entries, require the first
three to sum to `n_rot / 2`, and require the reserved fourth entry to be zero.

Gated DeltaNet layers do not consume MRoPE positions. They process image rows
strictly in physical sequence order using the existing convolution and
recurrent state. Dense/MoE branching remains confined to the FFN after the
shared mixer, so vision must not duplicate the decoder implementation.

## 5. Runtime design

### 5.1 Ownership boundary

Keep text and vision files independently mmap-backed and independently
resident. Do not add vision tensors to the existing text `Model` lifetime.

```rust
pub struct LoadedMultimodalModel {
    pub text: LoadedModel,
    pub vision: LoadedVisionModel,
    pub compatibility: MultimodalCompatibility,
}

pub struct LoadedVisionModel {
    pub path: PathBuf,
    pub gguf: GgufFile,
    pub model: MetalVisionModel,
    pub identity: VisionModelIdentity,
}
```

This preserves text-only startup, allows independent projector replacement,
and gives residency/cache accounting an unambiguous identity.

### 5.2 Proposed modules

Add three focused library modules:

```text
src/media.rs       bounded decode, smart resize, normalization, media digest
src/vision.rs      mmproj metadata/tensor binding and Metal vision forward
src/multimodal.rs  media markers, span plan, positions, identities, mixed prefill
```

Keep generic GGUF parsing in `src/gguf.rs` and generic Metal lifecycle in
`src/metal.rs`. Avoid a second graph executor.

### 5.3 Projector loader

The current GGUF parser already supports typed metadata and independent mmap
ownership. Add a strict `VisionArch`/`VisionModel` binder rather than changing
the wire parser.

Required metadata for current Qwen projectors:

```text
clip.has_vision_encoder
clip.vision.projection_dim
clip.vision.image_size
clip.vision.patch_size
clip.vision.embedding_length
clip.vision.feed_forward_length
clip.vision.block_count
clip.vision.attention.head_count
clip.vision.image_mean
clip.vision.image_std
clip.projector_type
clip.use_gelu
clip.vision.spatial_merge_size
clip.vision.attention.layer_norm_epsilon
clip.vision.is_deepstack_layers
```

Current tensor families:

```text
v.patch_embd.weight
v.patch_embd.weight.1
v.patch_embd.bias
v.position_embd.weight

v.blk.{i}.attn_qkv.weight / bias
v.blk.{i}.attn_out.weight / bias
v.blk.{i}.ln1.weight / bias
v.blk.{i}.ln2.weight / bias
v.blk.{i}.ffn_up.weight / bias
v.blk.{i}.ffn_down.weight / bias

v.post_ln.weight / bias
mm.0.weight / bias
mm.2.weight / bias
```

`clip.vision.image_size` describes the square learned-position basis
(`48 * patch_size` for current models); it is not a fixed input resolution.

Load-time validation:

- Projector type is Qwen3VL merger.
- `clip.has_vision_encoder` is true.
- `clip.has_audio_encoder` is absent/false and no audio tensor namespace is
  present in V1.
- Text architecture is `qwen35` or `qwen35moe`.
- `projection_dim == text hidden_size`.
- All dimensions and tensor shapes agree.
- Temporal patch size is represented by both split weights.
- Mean/std contain three finite values.
- Patch and spatial merge sizes are nonzero.
- Every required block tensor exists.
- `clip.vision.attention.layer_norm_epsilon` is exactly `1e-6`.
- `clip.vision.is_deepstack_layers` has exactly `block_count` false entries.
- No `v.deepstack.*` or equivalent DeepStack tensor is present.
- Special vision token strings resolve uniquely in the text tokenizer.

A missing DeepStack array may be accepted only as an explicitly versioned
legacy-mmproj ABI after proving that no DeepStack tensor exists.

### 5.4 Mixed-embedding prefill

The current production prefill starts from token IDs and always gathers
`token_embd`. Refactor the population of the packed residual-stream input, not
the per-layer decoder body.

```rust
pub enum EmbeddingSpan<'a> {
    Tokens(&'a [i32]),
    ExternalF32 {
        rows: &'a MetalTensor,
        row_offset: usize,
        row_count: usize,
    },
}

pub struct PrefillInput<'a> {
    pub spans: &'a [EmbeddingSpan<'a>],
    pub positions: &'a [DecoderPosition],
    pub next_causal_position: u32,
    pub next_logical_position: u32,
}
```

Add an external-row copy/scatter kernel or a contiguous Metal blit where
layout permits. Keep the existing token-only prefill as a wrapper that creates
one token span and consecutive text positions. Text-only logits and recurrent
state must remain bit-identical after this refactor.

Full-attention APIs must receive `cache_row`, `causal`, and `mrope` separately:

- KV append uses `cache_row`.
- Causal visibility uses `causal`.
- Q/K rotation uses `mrope`.
- GDN APIs remain position-free.

Packed prefill must accept arbitrary per-row MRoPE coordinates. A
`start_position + row_index` shortcut is invalid for image rows.

### 5.5 Media prompt planner

Use a structural marker such as `<__media__>` between tokenized text spans.
The marker itself is never sent to the tokenizer.

```rust
pub enum PreparedInputSpan {
    Tokens(Vec<i32>),
    Image {
        embeddings: MetalTensor,
        grid_width: u32,
        grid_height: u32,
        identity: MediaEncodingIdentity,
    },
}

pub struct PreparedMultimodalPrompt {
    pub spans: Vec<PreparedInputSpan>,
    pub positions: Vec<DecoderPosition>,
    pub physical_rows: usize,
    pub next_causal_position: u32,
    pub next_logical_position: u32,
    pub identity: MultimodalPrefixIdentity,
}
```

Structured message content must preserve part order. Do not move images ahead
of text as an implementation convenience.

CLI contract:

```text
--mmproj PATH
--image PATH       repeatable; consumed in marker order
```

Validation:

- Every marker consumes exactly one image.
- Images require an mmproj.
- The text model and projector must pass compatibility checks before decode.
- Structured message images and raw markers produce the same span plan.
- Network URLs are rejected in V1.

### 5.6 Cache and snapshot identity

The current prefix cache is token-only. Reusing that key for image prompts
would allow different images to share causal state.

Disable prefix-cache and durable-snapshot reuse for multimodal requests until
the key includes:

```text
text model strong identity
mmproj strong identity
ordered token-span digests
ordered source-image byte digests
preprocessing ABI and selected pixel limits
resized dimensions and merged grid
position-layout ABI
physical row count
next causal position
next logical text position
```

Snapshots need to persist the logical text cursor in addition to physical KV,
convolution, and GDN state. They do not need to persist source pixels or image
embeddings after prefill has been committed.

Replace the current sequence's single position invariant with explicit state:

```rust
pub struct SequencePosition {
    pub physical_rows: usize,
    pub next_causal_position: u32,
    pub next_logical_position: u32,
}
```

Capacity and KV/GDN storage use `physical_rows`; masks use the causal cursor;
rotary continuation uses the logical cursor. Checkpoint publication validates
the canonical multimodal row-plan identity rather than
`prefix_tokens.len() == position`. The token-only wrapper retains the simpler
equality invariant.

Bump the snapshot/numerics ABI when multimodal positions enter session state.
Old text-only snapshots should either remain explicitly versioned or fail
closed; never reinterpret them as multimodal state.

## 6. Metal work

Reusable today:

- Persistent GGUF-backed Metal tensor residency.
- F32, BF16, and quantized matmul dispatch.
- Residual add and generic elementwise kernels.
- Existing decoder full-attention and GDN layer bodies.
- Existing KV, convolution, and recurrent state stores.

New kernels or modes:

1. Standard LayerNorm with mean, variance, weight, and bias.
2. Both `gelu_pytorch_tanh` for vision blocks and exact/erf GELU for merger.
3. Row-wise bias broadcast for vision projections.
4. Patch extraction/reorder or a flattened-patch preparation path.
5. Bilinear learned-position interpolation with aligned corners.
6. Vision 2D RoPE.
7. Non-causal multi-token vision attention, including head dimensions 64 and
   72.
8. Spatial 2x2 merger regrouping.
9. External embedding-row copy/scatter into packed decoder input.
10. Decoder interleaved MRoPE with arbitrary per-row T/H/W coordinates.

Correctness-first patch projection can flatten patches on CPU and use existing
matrix multiplication twice, one per temporal slice. Only add a dedicated
patch kernel if profiling shows meaningful cost.

Vision and decoder RoPE must be separate APIs. They have different dimensions,
axis layouts, theta values, and position semantics.

## 7. Code map

Primary existing seams:

- `crates/qwen-llm/src/gguf.rs`: typed metadata/tensor table; no format change.
- `crates/qwen-llm/src/loader.rs`: text binder and compatibility checks.
- `crates/qwen-llm/src/model.rs`: add MRoPE sections and position contract.
- `crates/qwen-llm/src/metal.rs`: host encoders for new kernels.
- `crates/qwen-llm/src/metal_forward.rs`: mixed-embedding packed prefill,
  session cursor, snapshots.
- `crates/qwen-llm/src/metal_dflash.rs`: production packed prefill seam; keep
  speculative multimodal use disabled initially.
- `crates/qwen-llm/src/runtime.rs`: independently resident composite model and
  physical/causal/logical position advancement.
- `crates/qwen-llm/src/prefix_cache.rs`: multimodal identity or fail-closed
  bypass.
- `crates/qwen-cli/src/messages.rs`: string-or-parts message content and media
  marker rendering.
- `crates/qwen-cli/src/main.rs`: `--mmproj`, repeatable `--image`, JSONL request
  validation, and timing fields.
- `kernels/rope.metal`: add decoder interleaved MRoPE without changing the
  text-only wrapper.
- New `kernels/vision.metal`: vision-specific LayerNorm, GELU, position
  interpolation, patch/merge utilities, and non-causal attention as needed.

Reference implementation quarry:

- `~/code/llama.cpp/conversion/qwen3vl.py`
- `~/code/llama.cpp/tools/mtmd/models/qwen3vl.cpp`
- `~/code/llama.cpp/tools/mtmd/mtmd-image.cpp`
- `~/code/llama.cpp/tools/mtmd/mtmd.cpp`
- `~/code/llama.cpp/src/models/qwen35.cpp`
- `~/code/llama.cpp/src/models/qwen35moe.cpp`

## 8. Delivery phases and gates

### H6.0 - Freeze the contract and oracles

- Record metadata, tensor names/shapes/types, resized dimensions, normalized
  pixels, patch rows, vision positions, merged embeddings, decoder positions,
  and logits for OvisOCR2.
- Pin current llama.cpp HEAD and a known Transformers/model revision.
- Pin source bytes, preprocessing profile, resized grid, hardware/OS, power
  mode, warmup count, run count, and summary statistics for every performance
  comparison.
- Freeze concrete per-stage `max_abs`, relative-L2, cosine, and decoder-logit
  tolerances from the oracle artifacts before implementation qualification;
  prose such as "close" or "bounded" is not an acceptance gate.
- Add one dense larger-model fixture and one MoE fixture when storage allows.

Gate:

- The fixture proves DeepStack is empty.
- Qwen3.6 resolves through the Qwen3.5 architecture path.
- Oracle artifacts can be regenerated deterministically.

### H6.1 - Position plumbing with no vision

- Load interleaved MRoPE sections from GGUF.
- Introduce `DecoderPosition` and split cache position from rotary coordinates.
- Add CPU and Metal interleaved MRoPE.
- Keep text-only wrappers and current public APIs working.

Gate:

- Text-only logits, KV bytes, convolution state, and GDN state are unchanged.
- CPU/Metal interleaved MRoPE matches llama.cpp F32.
- GDN output is independent of MRoPE coordinates.

### H6.2 - Projector loading and preprocessing

- Add independent mmproj residency and strict compatibility validation.
- Add bounded image decode, smart resize, normalization, patch ordering, and
  content/preprocessing identity.
- Implement a slow CPU patch/vision reference or dump every stage from the
  pinned oracle.

Gate:

- Valid 0.8B, mid-size, and 27B-class projector schemas load.
- Shape/type/deepstack mismatches fail before Metal allocation.
- Resize dimensions, normalized samples, patch rows, and learned-position
  interpolation match the reference.

### H6.3 - Metal vision tower

- Add LayerNorm, GELU, bias, learned-position interpolation, vision RoPE,
  non-causal attention, FFN, and merger execution.
- Validate block by block before end-to-end generation.

Gate:

- F32 layer outputs have cosine similarity at least 0.99999 and satisfy the
  concrete `max_abs`/relative-L2 limits frozen in H6.0.
- BF16 final projected embeddings have cosine similarity at least 0.9999 and
  satisfy the concrete relative-L2 limit frozen in H6.0.
- Projected row count and ordering are exact.
- On the pinned page/grid, report image encode median and p95 over a fixed run
  count and match or beat the equivalently measured llama.cpp baseline.

### H6.4 - Mixed prefill and single-image generation

- Add embedding spans and external-row population.
- Build image MRoPE positions and logical cursor advancement.
- Run projected rows through existing GDN/full-attention decoder state.
- Support one raw marker and one local image.

Gate:

- Mixed prefill logits satisfy the concrete quant/dtype-specific limits frozen
  in H6.0.
- Cached decode after the image matches cold continuation.
- A page 7 Markdown hash mismatch is investigated and explained; it does not
  override a passing Transformers tensor/logit oracle.
- Gate mixed decoder prefill rows/s and TTFT separately from image encode.
- Q8 decode remains within 3% of the text-only engine rate.

### H6.5 - Multiple images, messages, and identity

- Add ordered structured message parts and repeatable CLI images.
- Add multimodal-safe prefix/snapshot identity and logical cursor persistence.
- Add JSONL timing and error reporting.

Gate:

- Marker and structured-message forms produce identical spans and logits.
- Changing image bytes, grid, preprocessing limits, or mmproj never hits the
  same cache key.
- Multi-image continuation matches the oracle.
- Oversized/malformed images fail before large allocation.

### H6.6 - Family matrix and optimization

- Qualify Qwen3.5 dense sizes, Qwen3.5 MoE, Qwen3.6-27B, and
  Qwen3.6-35B-A3B.
- Profile tower attention, position interpolation, patch preparation, and
  mixed prefill.
- Add quantized projector support only with a quality gate.

Gate:

- Dense and MoE share exactly one media/position planner and mixed-prefill
  implementation.
- No text-only benchmark regresses beyond noise.
- OvisOCR2 full-document latency matches or beats the equivalently measured
  llama.cpp baseline while preserving the qualified Transformers tensor/logit
  oracle and qwen-llm's pre-optimization output.

## 9. Required tests

Unit and fixture tests:

```text
vision_loader_accepts_qwen35_projector
vision_loader_accepts_qwen35moe_projector
vision_loader_rejects_projection_width_mismatch
vision_loader_rejects_enabled_deepstack
vision_loader_requires_both_temporal_patch_weights
smart_resize_matches_qwen_reference_cases
image_limits_are_area_not_side_lengths
normalization_matches_reference_pixels
still_image_uses_both_temporal_slices
patch_tile_order_matches_reference
position_interpolation_matches_align_corners_reference
vision_rope_matches_reference
vision_attention_is_non_causal
merger_row_count_matches_grid
decoder_imrope_11_11_10_matches_reference
text_positions_reduce_to_existing_rope
gdn_ignores_mrope_coordinates
mixed_prefill_matches_row_by_row_reference
text_only_prefill_is_bit_identical_after_refactor
image_cache_rows_advance_by_embedding_count
image_logical_position_advances_by_max_grid_axis
post_image_cached_decode_matches_cold_decode
different_image_bytes_never_prefix_cache_hit
different_mmproj_never_snapshot_restore
structured_and_marker_prompts_are_equivalent
malformed_or_oversized_image_fails_before_allocation
```

Real-model matrix:

| Case | Purpose |
| --- | --- |
| OvisOCR2 Q8 + BF16 projector | Fast dense image/OCR bring-up |
| Qwen3.5 2B or 4B + matching projector | Second tower shape |
| Qwen3.5 27B + matching projector | Large dense projection width |
| Qwen3.6-27B + matching projector | 3.6 compatibility proof |
| Qwen3.6-35B-A3B + matching projector | MoE compatibility proof |

## 10. Primary risks

1. Decoder MRoPE is interleaved. Implementing contiguous T/H/W sections will
   look plausible and be wrong.
2. Physical KV position and logical rotary position diverge after an image.
3. Vision RoPE and decoder MRoPE are different operations.
4. Learned-position interpolation must use aligned corners.
5. Qwen patch/tile ordering is part of the learned model contract.
6. Standard LayerNorm cannot reuse decoder RMSNorm.
7. A still image must exercise both temporal Conv3D slices.
8. A token-only cache key can silently reuse state for the wrong image.
9. Existing speculative and prompt-lookup paths assume token-only consecutive
   positions.
10. Future projectors may enable DeepStack. H6 V1 must reject them, not discard
    extra embeddings.
11. Image decoders require explicit compressed-byte, dimension, and pixel
    limits to prevent decompression bombs.
12. Large native-resolution images make vision attention and LM prefill the
    dominant latency even when token decode is fast.

## 11. Completion criterion

H6 is complete when one dense and one MoE Qwen3.5/3.6 checkpoint can load a
separate Qwen3VL-style mmproj, process bounded local images, produce
oracle-matching projected embeddings and decoder logits, continue generation
with correct interleaved MRoPE and GDN state, and safely snapshot/cache the
multimodal prefix without regressing text-only numerics or throughput.

## 12. Authoritative references

- Qwen3.5/3.6 repository: <https://github.com/QwenLM/Qwen3.6>
- Qwen3.5 Transformers model: <https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5>
- Qwen3.6-27B config: <https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json>
- Qwen3.6-35B-A3B config: <https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json>
- OvisOCR2 config: <https://huggingface.co/ATH-MaaS/OvisOCR2/raw/main/config.json>
- llama.cpp Qwen projector conversion:
  <https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen3vl.py>
- llama.cpp Qwen vision graph:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/models/qwen3vl.cpp>
