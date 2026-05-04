# H5: DFlash speculative decode — block-diffusion drafter

DFlash is the production speculative-decode path. A separately trained
drafter (~1.7 B params on Qwen3.6-27B) produces a block of `N-1` candidate
tokens **in one parallel forward pass**, conditioned on hidden features
extracted from the target model. The target model verifies the whole
block in one packed forward; greedy accept-prefix logic emits anywhere
from 1 to `N` tokens per outer step.

**Target speedup on M4 Max (Qwen3.6-27B-Q4_K_M target + Q8_0 drafter):**
~2× tg per spiritbuun's tweet (M-series specifically). Public HF cards
document only CUDA + RTX 3090 numbers; M-series verification is part
of H5.5.

This plan builds on the validated H4 infrastructure: `SpeculativeDecoder`
wrapper, drafter loader pattern, GPU-resident hidden carry, cursor
state machine (`processed_pos` + `carry_tok` from H4), EOS /
max_new_tokens edge cases, `MetalForward::single_token_with_hidden`.
New work: drafter forward graph (5 layers, cross+self attention, **SWA**),
multi-layer target hidden capture, packed-N=16 base verify (mat-mat path),
per-token GDN + conv state checkpointing for hybrid rollback, and
optional probabilistic rejection sampling.

**Non-goals (deferred):**
- Continuous batching (single-stream first).
- Tree verify (DDTree) — different recipe lineage.
- Approximate / pollution-tolerant mode.
- Multi-block drafting (N>16).

`--dflash` fails closed if the drafter GGUF isn't present; pass
`--dflash-drafter <path>`.

## 1. Architecture

The DFlash drafter is structurally a small autoregressive transformer
(GQA, RoPE, RMSNorm, SwiGLU FFN) **conditioned on multi-layer target
hidden features**. Two coupled mechanisms:

1. **Target-feature fusion** — at prompt prefill, hidden states from
   `K` fixed target layers are stacked, projected via
   `Linear(K·H_target → H_drafter)` (`dflash_fc`), RMSNorm'd
   (`dflash_hidden_norm`), and stored as **persistent cross-context K/V**
   read by every drafter layer. After each verify, we extend this
   cross-context with hiddens from the newly-committed positions.

2. **Asymmetric attention in the drafter** — Q comes only from the
   noise tokens (the block being decoded). K/V come from
   `concat(target_ctx, noise)`. So each noise position attends to
   the entire fused target context plus all `N` noise positions. The
   noise-side attention is **block-causal** (block diffusion is
   bidirectional within already-emitted positions, but masked for
   strictly-future ones).

For Qwen3.6-DFlash, **most layers use causal sliding-window attention
(SWA)**, per-layer pattern from GGUF `attention.sliding_window_pattern =
[True, True, True, True, False]`. Layers 0-3 are SWA (window=2048);
layer 4 is full-attention. SWA layers use a per-layer mask combining
sliding-window-over-context + block-causal-over-noise (see §1.7).

Fact-checked against `~/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf`:

### 1.1 GGUF tensor inventory (real numbers)

Architecture string: `dflash-draft`. KV namespace: `dflash-draft.*`.

| GGUF tensor                       | Shape           | dtype  | Role                                |
| --------------------------------- | --------------- | ------ | ----------------------------------- |
| `dflash_fc.weight`                | `[25600, 5120]` | Q8_0   | `[K·H_target, H_drafter]` fc fusion |
| `dflash_hidden_norm.weight`       | `[5120]`        | F32    | RMSNorm of cross-context post-fc    |
| `output_norm.weight`              | `[5120]`        | F32    | Drafter's own final RMSNorm         |
| `blk.{i}.attn_norm.weight`        | `[5120]`        | F32    | Pre-attn RMSNorm                    |
| `blk.{i}.attn_q.weight`           | `[5120, 4096]`  | Q8_0   | Q (NOT gated; `n_q·head_dim`)       |
| `blk.{i}.attn_k.weight`           | `[5120, 1024]`  | Q8_0   | K (`n_kv·head_dim`)                 |
| `blk.{i}.attn_v.weight`           | `[5120, 1024]`  | Q8_0   | V                                   |
| `blk.{i}.attn_output.weight`      | `[4096, 5120]`  | Q8_0   | O                                   |
| `blk.{i}.attn_q_norm.weight`      | `[128]`         | F32    | Per-head Q RMSNorm                  |
| `blk.{i}.attn_k_norm.weight`      | `[128]`         | F32    | Per-head K RMSNorm                  |
| `blk.{i}.post_attention_norm.weight` | `[5120]`     | F32    | Pre-FFN RMSNorm (post-attn-residual)|
| `blk.{i}.ffn_gate.weight`         | `[5120, 17408]` | Q8_0   | SwiGLU gate                         |
| `blk.{i}.ffn_up.weight`           | `[5120, 17408]` | Q8_0   | SwiGLU up                           |
| `blk.{i}.ffn_down.weight`         | `[17408, 5120]` | Q8_0   | SwiGLU down                         |

NOT in the drafter GGUF (loaded from target):
- `token_embd.weight` — drafter shares the target's vocab embedding.
  Only works because `H_drafter == H_target == 5120`.
- `output.weight` (lm_head) — drafter shares the target's lm_head.
  Only works because both write into vocab space (248320) from H=5120.

If we ever load a drafter where `H_drafter != H_target`, the loader
must reject with a precise error.

### 1.2 GGUF KV metadata (real values)

```
dflash-draft.attention.causal              = false           # bidirectional within noise
dflash-draft.attention.head_count          = 32              # n_q
dflash-draft.attention.head_count_kv       = 8               # n_kv (GQA 4:1)
dflash-draft.attention.key_length          = 128             # head_dim
dflash-draft.attention.value_length        = 128
dflash-draft.attention.sliding_window      = 2048            # window for SWA layers
dflash-draft.attention.sliding_window_pattern = [T,T,T,T,F]  # per-layer SWA flag
dflash-draft.attention.layer_norm_rms_epsilon = 1e-6
dflash-draft.block_count                   = 5               # n_layer
dflash-draft.context_length                = 262144
dflash-draft.embedding_length              = 5120            # H_drafter
dflash-draft.feed_forward_length           = 17408           # F_drafter
dflash-draft.rope.dimension_count          = 128
dflash-draft.rope.freq_base                = 10000000.0
dflash-draft.dflash.block_size             = 16              # N
dflash-draft.dflash.mask_token_id          = 248070
dflash-draft.dflash.n_target_features      = 25600           # K·H_target
dflash-draft.dflash.target_layer_ids       = [1, 16, 31, 46, 61]   # K=5 layers
```

Every dim must be GGUF-derived in our loader; no hardcoded constants.

### 1.3 Per-step algorithm

State invariants (carried across outer steps; reuse H4's
`processed_pos`/`carry_tok` shape):

```
processed_pos:    u32      // target session has run through this position;
                          //   target_ctx has K hiddens at positions [0..=processed_pos].
target_ctx:       Tensor [K·H_target, processed_pos+1]
                          //   Persistent cross-context. Append-only across outer steps;
                          //   on partial accept, append accepted-prefix hiddens then
                          //   stop. Bounded ring possible (see §1.6).
carry_tok:        i32     // next token to emit/process. Selected by previous outer
                          //   step's bonus or by prompt's first generated token.
                          //   NOT yet emitted to caller; NOT yet processed by target.
                          //   Position when processed = processed_pos + 1.
```

Define `N = block_size = 16`. The drafter's noise input has length `N`
and produces `N` logits, but draft tokens come from positions `1..N-1`
(i.e., `D = N-1 = 15` candidates). Position 0 of the drafter sees
`carry_tok`'s embedding and is essentially a "seed" — its logits are
discarded.

**Per outer step:**

```
# A. Drafter forward
#    Input: noise_input = [carry_tok, MASK, MASK, ..., MASK]   (length = N)
#    Conditioning: target_ctx (covers positions [0..=processed_pos])
draft_logits  = drafter_forward(noise_input, target_ctx, processed_pos)  # [N, V]
# Read draft tokens from positions 1..N (D = N-1 candidates).
draft_tokens  = argmax(draft_logits[1..N], dim=-1)                       # [D]

# B. Target packed verify (one fused forward, N tokens)
#    Input: verify_input = [carry_tok, draft_tokens[0..D-1]]   (length = N)
#    Positions: [processed_pos+1 .. processed_pos+N]
#    Side effect: target session advances by N; conv + GDN state checkpoints
#                 written per token; multi-layer target hidden captured per token.
verify_argmax, hidden_capture, conv_ckpt, gdn_ckpt = target_packed_forward(
    tokens=verify_input,
    start_pos=processed_pos + 1,
    target_layer_ids,
    session=target_session,
)
# verify_argmax[i] = greedy target prediction at position processed_pos+1+i.

# C. Greedy accept-prefix
n_accepted = 0   # count of accepted DRAFT tokens (range 0..D)
for j in 0..D:
    # Position of draft_tokens[j] in the verify batch is index j+1.
    # Target's prediction at THAT position is verify_argmax[j].
    if draft_tokens[j] == verify_argmax[j]:
        n_accepted += 1
    else:
        break

# Bonus token: target's argmax at the first un-accepted position in the batch.
# That's verify_argmax[n_accepted].
bonus_tok = verify_argmax[n_accepted]
emitted_this_step = n_accepted + 1   # n_accepted drafts + bonus

# D. Commit accepted prefix + bonus, rollback rest
#    Total emitted: carry_tok was emitted in PREVIOUS step (or bootstrap);
#    this step emits draft_tokens[0..n_accepted-1] + bonus_tok.
#    Wait — carry_tok IS in verify_input as position 0; it gets processed
#    by target. So emit carry_tok now (it was selected last step but not
#    yet emitted to caller per the H4 carry contract).
emit(carry_tok)                                  # was selected last step
emit_all(draft_tokens[0..n_accepted])            # 0..n_accepted-1 (n_accepted tokens)
# bonus_tok will be emitted NEXT step as the new carry_tok.
# Total tokens flushed THIS step = 1 + n_accepted (range 1..N).
# Bonus is staged but NOT emitted, matching H4 carry semantics.

# Append target hiddens for processed positions to target_ctx:
#   verify processed positions [processed_pos+1 .. processed_pos+1+n_accepted]
#   (carry + n_accepted accepted drafts; bonus is staged, not committed).
target_ctx.append(hidden_capture[0..=n_accepted])

# Restore conv + GDN state to the state AFTER (n_accepted+1) tokens of verify.
# That's checkpoint index n_accepted in the [0..N) checkpoint array.
restore_state(target_session, conv_ckpt[n_accepted], gdn_ckpt[n_accepted])

# KV cache: rolled back via n_pos reset; physical bytes at rejected slots
# stay until next verify overwrites. n_pos := processed_pos + 1 + n_accepted + 1
# (i.e. covers carry + accepted_drafts; bonus position not yet committed).
target_session.kv_n_pos = processed_pos + 1 + n_accepted + 1

# Update cursors.
processed_pos = processed_pos + 1 + n_accepted        # advanced by carry + drafts
carry_tok     = bonus_tok                             # for next step
```

**Invariants verified per branch:**
- Tokens emitted this step: `1 + n_accepted` (range 1..N).
- `processed_pos` advanced by `1 + n_accepted`.
- `target_ctx` extended by `1 + n_accepted` columns.
- `carry_tok` carries the bonus (never double-emitted; matches H4 contract).
- On full reject (n_accepted=0): emit carry, advance by 1, like normal decode.
- On full accept (n_accepted=D=N-1): emit `N` tokens this step, max throughput.

### 1.4 Speedup math (rev)

Tokens emitted per outer step = `1 + n_accepted`. Let `α_draft =
mean(n_accepted) / D` (acceptance rate over drafts; in [0, 1]).

```
mean_emitted_per_step = 1 + α_draft · D = 1 + α_draft · (N-1)
wall_per_step         = T_drafter + T_verify_N + T_rollback
speedup_vs_no_spec    = (1 + α_draft · (N-1)) · T_base / wall_per_step
```

For N=16, α_draft=0.55, T_drafter=0.5·T_base, T_verify_N=4·T_base,
T_rollback=0.05·T_base:
```
mean_emitted   = 1 + 0.55 · 15 = 9.25 tokens/step
wall_per_step  = 0.5 + 4.0 + 0.05 = 4.55 · T_base
speedup        = 9.25 / 4.55 ≈ 2.03×
```

That matches spiritbuun's reported ~2× M-series tweet. The ratio is
fragile to:
- `T_verify_N`: how well our packed kernels amortize 16 query tokens
  through one weight pass (FFN/attn pack well; GDN is sequential).
- `α_draft`: prompt domain (code/JSON > prose > thinking — z-lab claims
  ~93% on quicksort, ~38% on prose) and quant choice.
- `T_rollback`: per-token GDN + conv checkpoints inline in the packed
  kernel (NOT pre-verify snapshot, which would force replay).

**Worst-case scenario (don't be surprised at H5.5):**
- α_draft on prose lands at 0.30 (not 0.55)
- T_drafter is 0.7·T_base (drafter still pays full lm_head over N rows + Q8_0 dequant cost)
- T_verify_N is 6·T_base (GDN sequential + checkpoint write BW + command overhead don't amortize cleanly)
- T_rollback is 0.10·T_base (2.3 GiB SSM copy + cache perturbation)

```
mean_emitted   = 1 + 0.30 · 15 = 5.5 tokens/step
wall_per_step  = 0.7 + 6.0 + 0.10 = 6.8 · T_base
speedup        = 5.5 / 6.8 ≈ 0.81×
```

That's a slowdown. Concretely: if T_verify_N actually lands at 6× and
α at 0.30, DFlash hurts. Not a corner case — that's a normal failure
mode if packed verify underdelivers OR α is bad. The H5.2.5 lazy
DFlash gate exists specifically to detect the α-bad arm of this risk
before we pay for packed verify.

### 1.5 The packed verify forward

The target processes `verify_input = [carry_tok, draft_tokens[0..D-1]]`
of length `N`. Packed multi-token decode — N queries advance through
every layer simultaneously, sharing weight loads.

What changes vs single-token decode:

- **Embed**: `get_rows` for N tokens → `[N, H_target]`.
- **Per-block residual stream**: shape `[N, H_target]` throughout.
- **GDN layers**: sequential per token (the SSM update can't pack along
  time). Kernel iterates internally over N tokens, advancing state
  per step. **Per-token state checkpoints** (both SSM and conv state)
  written inline.
- **Full-attn layers**: Q/K/V/O projections become mat-mat
  (`[H, n_q·head_dim] · [N, H] = [N, n_q·head_dim]`). KV cache appends N
  positions contiguously starting at `start_position`. Attention computes
  scores against `start_position + i + 1` keys for query at relative
  position i (causal mask).
- **FFN**: SwiGLU mat-mat across all N positions.
- **Output norm + lm_head**: per-position; produces `[N, V]` logits,
  then **GPU argmax** writes `verify_argmax[N]` (i32) — avoids the
  `N · V · 4 B` readback (16 · 248320 · 4 = 15.9 MB) per outer step.
  Spiritbuun does this; we should too.
- **Multi-layer hidden capture**: at `K = target_layer_ids.len()` layer
  indices, copy the layer's pre-output residual into a side buffer
  for next iter's drafter conditioning. Total side-buffer per verify =
  `K · H_target · N · 4 B` (e.g. `5 · 5120 · 16 · 4` = 1.6 MB).

`packed_forward` returns `(verify_argmax: [N] i32, hidden_capture:
MetalTensor [K, N, H_target], conv_ckpt + gdn_ckpt: MetalTensors)`.
**Does NOT return raw logits** (production path). A debug variant can
return logits for the bit-exactness test in H5.3.

### 1.6 GDN + conv state on rollback

**Inline per-token checkpoints in the packed-verify GDN kernel.** Two
state pieces both need checkpointing:

- **SSM state** per layer per token: `n_v · head_dim · head_dim · 4 B`.
  For 27B (`n_v=48, head_dim=128`): `48·128²·4 ≈ 3 MiB` per layer per token.
  All 48 GDN layers × 16 tokens × 3 MiB ≈ **2.3 GiB** per outer step.
  Fits on M4 Max's 128 GB unified easily; allocate once per session.
- **Conv state** per layer per token: `(conv_kernel-1) · conv_dim · 4 B`.
  For 27B: `3 · ((2·16 + 48)·128) · 4 = 122,880 B ≈ 120 KiB` per layer per
  token. All 48 layers × 16 tokens × 120 KiB ≈ **90 MiB** per outer step.
  Trivial.

Checkpoint policy: write `state_after_token[k]` AFTER the recurrence
step for token k completes. On reject at draft index `j`, accepted
prefix length is `n_accepted`, so the desired final state is "after
processing token `n_accepted+1`" in the verify batch (carry + n_accepted
drafts). Restore = `device_copy(state_session ← state_after_token[n_accepted])`
for both SSM and conv states.

Cost of restore: ~2.3 GiB / 500 GB/s = **4.6 ms** for SSM + 0.18 ms for
conv. Per-outer-step worst case: ~5 ms total. Speedup math used 0.05·T_base
(~2 ms) — actual will be ~5 ms; marginal speedup hit but well worth it.

Alternative if memory is tight: snapshot before verify, replay accepted
prefix. Replay cost = `n_accepted × T_base_GDN_only` (~10-15 ms per
token replayed). For `n_accepted = 9`, that's ~120 ms — worse than
checkpointing.

### 1.7 KV cache rollback

Trivial: KV append in packed verify writes N contiguous slots starting
at `start_position`. On reject, set `kv_n_pos = start_position +
n_accepted + 1`. The rejected K/V data physically remains in the cache
but is unreachable; the next verify writes over it. No explicit zero
or free.

### 1.8 Drafter KV — transient

The drafter's per-layer KV is **transient within an outer step**: each
forward processes a fresh `[carry_tok, MASK × (N-1)]` block under
self-attention with `target_ctx` providing cross-context. The drafter
doesn't accumulate KV across outer steps because the noise input is
always fresh. So no rollback needed for drafter KV.

### 1.9 SWA: sliding-window mask construction

For SWA layers (layers 0-3 in the 3.6 drafter), per-layer attention
mask combines:

- **k_pos < pos_ctx_first**: padded/inactive context slot — DENY (-INF).
  (Single-slot v1: no padding; this is for future multi-slot.)
- **k_pos in `target_ctx` (real slots)**: allow if
  `q_pos - k_pos ≤ window`, deny otherwise.
- **k in noise positions [0..N)**: allow if `k ≤ q` (block-causal).
  Strictly future positions DENY.

For full-attn layers (layer 4): only the block-causal rule applies; no
sliding-window restriction over context.

Mask is computed once per outer step on host and uploaded. Cost:
`N · (n_ctx + N) · 4 B` per mask × 2 (full + SWA). At n_ctx=4096, N=16:
~264 KB per mask, ~530 KB total. Trivial.

`pos_ctx[k]` for context slots is the absolute target position (e.g.
[0, 1, 2, ..., processed_pos]). Used for both RoPE (drafter K side)
and the SWA inequality.

If we treat `target_ctx` as bounded ring (cap at `GGML_DFLASH_MAX_CTX`,
default 4096 per spiritbuun), `pos_ctx` carries the actual absolute
positions of the most-recent slots, and slots beyond the cap are masked
away. v1 keeps unbounded; H5.6 adds the ring if memory grows.

**SWA must be implemented BEFORE the end-to-end equivalence test (H5.5).**
Without correct SWA mask, the drafter sees full bidirectional attention
on context that was trained to be windowed; the distribution corrupts
and α tanks. (Generation still works — verifier corrects every token
greedy — but α makes the bench meaningless.)

## 2. API surface

### 2.1 Loader (extends existing pattern)

```rust
pub struct DFlashHead<'a> {
    pub block_size: u32,
    pub mask_token_id: i32,
    pub target_layer_ids: Vec<u32>,    // K layer indices in the target
    pub n_target_features: u32,         // K · H_target

    pub fc: &'a TensorDesc,             // [n_target_features, H_drafter]
    pub hidden_norm: &'a TensorDesc,    // [H_drafter]
    pub output_norm: &'a TensorDesc,    // [H_drafter] — drafter's own final norm

    pub layers: Vec<DFlashLayer<'a>>,
}

pub struct DFlashLayer<'a> {
    pub attn_norm: &'a TensorDesc,
    pub q: &'a TensorDesc,              // NOT gated; shape [H, n_q · head_dim]
    pub k: &'a TensorDesc,
    pub v: &'a TensorDesc,
    pub o: &'a TensorDesc,
    pub q_norm: &'a TensorDesc,
    pub k_norm: &'a TensorDesc,
    pub post_attention_norm: &'a TensorDesc,
    pub ffn_gate: &'a TensorDesc,
    pub ffn_up: &'a TensorDesc,
    pub ffn_down: &'a TensorDesc,
    pub is_swa: bool,                   // from sliding_window_pattern[i]
}

pub struct DFlashConfig {
    pub n_layer: u32,                    // 5
    pub hidden_size: u32,                // H_drafter == H_target
    pub intermediate_size: u32,          // F_drafter
    pub n_q_heads: u32,                  // 32
    pub n_kv_heads: u32,                 // 8
    pub head_dim: u32,                   // 128
    pub rope_theta: f32,                 // 10M
    pub swa_window: u32,                 // 2048 (0 if no SWA layers)
    pub partial_rotary_factor: f32,      // 1.0 for drafter (full RoPE)
}

/// Loads a separately-distributed DFlash drafter GGUF and binds it to a
/// pre-existing target Model. **Validates compatibility** before binding:
///   * arch string == "dflash-draft"
///   * `target_layer_ids` are all valid indices into `target_model.blocks`
///   * `H_drafter == H_target` (required for shared tok_embd / lm_head)
///   * `vocab` matches between target and drafter (drafter has none of
///     its own; relies on target's via `Model::token_embd` / `lm_head`)
///   * `dflash_fc.shape[0] == target_layer_ids.len() · H_target`
pub fn open_dflash_drafter<'a>(
    drafter_gguf: &'a GgufFile,
    target_model: &Model<'_>,
) -> Result<DFlashHead<'a>, LoadError>;
```

### 2.2 Metal driver

```rust
pub struct MetalDFlashHead {
    pub config: DFlashConfig,
    pub target_layer_ids: Vec<u32>,
    pub mask_token_id: i32,
    pub block_size: u32,

    pub fc: MetalTensor,
    pub hidden_norm: MetalTensor,
    pub output_norm: MetalTensor,
    pub layers: Vec<MetalDFlashLayer>,
}

pub struct MetalDFlashLayer {
    pub attn_norm: MetalTensor,
    pub q: MetalTensor,                  // NOT gated
    pub k: MetalTensor,
    pub v: MetalTensor,
    pub o: MetalTensor,
    pub q_norm: MetalTensor,
    pub k_norm: MetalTensor,
    pub post_attention_norm: MetalTensor,
    pub ffn_gate: MetalTensor,
    pub ffn_up: MetalTensor,
    pub ffn_down: MetalTensor,
    pub is_swa: bool,
}

pub struct MetalDFlashSession {
    /// Persistent cross-context: [K · H_target, target_ctx_capacity].
    /// Grown by `1 + n_accepted` per outer step.
    pub target_ctx: MetalTensor,
    pub target_ctx_n: usize,             // valid columns

    /// Per-layer drafter KV (transient — recomputed per call).
    /// Shape per layer: [N, n_kv · head_dim].
    pub drafter_kv_k: Vec<MetalTensor>,
    pub drafter_kv_v: Vec<MetalTensor>,

    /// Drafter scratch — ALL shaped [N, ...] for packed forward.
    pub x: MetalTensor,                  // [N, H_drafter]
    pub h: MetalTensor,                  // [N, H_drafter]
    pub q_buf: MetalTensor,              // [N, n_q · head_dim]
    pub k_buf: MetalTensor,              // [N, n_kv · head_dim]
    pub v_buf: MetalTensor,
    pub attn_o: MetalTensor,
    pub ffn_inner: MetalTensor,          // [N, F_drafter]

    /// Per-step block input (carry_tok + (N-1) MASK).
    pub noise_ids: MetalTensor,          // [N] i32 (in F32 buffer, like ids_buf)

    /// Drafter logits (after lm_head). [N, V_target].
    pub draft_logits: MetalTensor,

    /// Per-layer SWA + full attention masks. Computed per outer step.
    /// Shape: [N, n_ctx + N] each.
    pub kq_mask_full: MetalTensor,
    pub kq_mask_swa: MetalTensor,
}

pub struct MetalDFlashVerifyScratch {
    /// Output of `packed_forward`. Owned by caller (DFlashDecoder).
    pub verify_argmax: MetalTensor,      // [N] i32
    pub hidden_capture: MetalTensor,     // [K, N, H_target]
    pub conv_ckpt: MetalTensor,          // [n_gdn, N, conv_state_elems]
    pub gdn_ckpt: MetalTensor,           // [n_gdn, N, ssm_state_elems]
}

pub struct DFlashDecoder<'a> {
    pub base: &'a MetalForward<'a>,
    pub head: &'a MetalDFlashHead,
    pub session: MetalDFlashSession,
    pub verify_scratch: MetalDFlashVerifyScratch,
}

impl<'a> DFlashDecoder<'a> {
    /// Run drafter forward: produce `[N]` draft tokens given the carry
    /// and the persistent target_ctx. Drafter KV is fully recomputed.
    pub fn draft_block(
        &mut self,
        carry_tok: i32,
        processed_pos: u32,
    ) -> Result<Vec<i32>, DFlashError>;       // length D = N-1

    /// Top-level greedy decode loop (§1.3).
    pub fn decode(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_id: i32,
        base_session: &mut MetalSession,
    ) -> Result<DecodeOutput, DFlashError>;
}
```

### 2.3 MetalForward extensions

```rust
impl<'a> MetalForward<'a> {
    /// Packed multi-token decode. Encodes ALL kernels for N tokens
    /// in one command buffer. Per-token GDN + conv state checkpoints
    /// are written inline. Multi-layer target hidden states captured
    /// at `target_layer_ids` indices.
    ///
    /// Returns argmax tokens (GPU-computed, no full-vocab readback).
    /// `tokens` length = `N`. `start_position` is the first slot.
    /// Caller pre-allocates the scratch buffers; this method only writes
    /// into them.
    pub fn packed_forward(
        &self,
        tokens: &[i32],
        start_position: u32,
        target_layer_ids: &[u32],
        scratch: &mut MetalDFlashVerifyScratch,
        session: &mut MetalSession,
    ) -> Result<Vec<i32>, MfError>;       // [N] argmax tokens

    /// Same as packed_forward but ALSO returns full logits for debug /
    /// bit-exactness validation (H5.3). Adds an N × V CPU readback.
    pub fn packed_forward_with_logits(
        &self,
        tokens: &[i32],
        start_position: u32,
        target_layer_ids: &[u32],
        scratch: &mut MetalDFlashVerifyScratch,
        session: &mut MetalSession,
    ) -> Result<(Vec<i32>, Vec<Vec<f32>>), MfError>;

    /// Restore SSM + conv state from the per-token checkpoint at
    /// `n_accepted_in_batch`. Resets KV `n_pos`. Used by H5.4 rollback.
    pub fn restore_after_partial_accept(
        &self,
        scratch: &MetalDFlashVerifyScratch,
        n_accepted_in_batch: usize,        // 0..N (count of tokens to keep
                                            // including the carry at index 0)
        start_position: u32,
        session: &mut MetalSession,
    ) -> Result<(), MfError>;
}
```

## 3. Phasing

### H5.0 — Loader binding

1. Probe drafter GGUF for `dflash_fc.weight`. Read `dflash-draft.*`
   metadata. Validate every dim against the target model:
   - arch string `"dflash-draft"`
   - `H_drafter == H_target`
   - `vocab` (drafter has no embedding; just reuses target's)
   - `target_layer_ids[i] < target.n_layer`
   - `dflash_fc.shape[0] == target_layer_ids.len() · H_target`
2. Test: `loads_dflash_drafter` on `~/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf`.
   Assert exact dims (block_size=16, n_layer=5, target_layer_ids=[1,16,31,46,61], etc.).

### H5.1 — Drafter Metal forward + SWA

Build the drafter forward right the first time, including SWA, since
SWA is needed for correct output distribution per spiritbuun.

1. Implement `DFlashDecoder::draft_block` — full Metal forward on
   the drafter:
   - Embed `noise_ids` via target's `tok_embd`.
   - Per-layer (5 layers): pre-attn norm → Q/K/V/O with cross-context
     concat → full or SWA mask depending on `is_swa` → FFN.
   - Final `output_norm` → target's `lm_head` → argmax → `[N]` tokens.
2. SWA mask construction (host-side, uploaded per outer step).

### H5.1.5 — Metal-vs-CPU drafter cosine gate (mandatory)

Finiteness alone is too weak. Add a bit-exactness test before any
infrastructure that depends on the drafter's distribution being
correct:

1. Pick a deterministic 4-token prompt (`"The quick brown fox"`).
2. Run target prefill on Metal AND CPU; capture multi-layer hiddens
   at `target_layer_ids` from each.
3. Run `DFlashDecoder::draft_block` (Metal) AND `Forward::dflash_draft`
   (CPU) with the same `(noise_ids, target_ctx_stacked)` inputs.
4. Pass criterion: cosine ≥ 0.9999 between Metal and CPU draft logits
   at every noise position. Argmax must match.

This is the same pattern that caught the cross-GGUF bug at H5.1
(round-2 codex review).

### H5.2 — Multi-layer target hidden capture (single-token first)

1. Add `MetalForward::single_token_with_multi_hidden`: like
   `single_token_with_hidden` but writes hiddens at K layer indices
   into a `[K, H]` buffer.
2. Use this to bootstrap `target_ctx` from prompt prefill.
3. Test: hidden values at `target_layer_ids` are non-zero and
   different across layers (sanity).
4. **Layout sanity test (cheap, catches silent bugs):** run drafter
   with the captured `target_ctx`, then run drafter with `target_ctx`
   shuffled along the K-layer axis (or with positions reversed).
   Correctly captured `target_ctx` must produce materially better α
   downstream — a no-op test is OK if the layout is wrong because
   shuffled noise on shuffled input still yields valid-looking
   logits. So we test *acceptance* sensitivity, not just shape.

### H5.2.5 — Lazy DFlash acceptance gate (NEW; mandatory before H5.3)

The most expensive thing on the road to H5 is packed verify
(H5.3). Before paying for it, prove DFlash actually drafts a useful
distribution under our engine's quants, hidden capture, SWA mask,
RoPE, and shared-lm_head path. **Cost-model α was the only thing the
review couldn't predict; this gate measures it cheaply.**

1. Implement `DFlashDecoder::decode_lazy` — uses the H4-style
   single-token verify path (already shipped) instead of packed
   verify:
   - Outer step: `draft_block(carry, processed_pos)` → `[D]` tokens
   - For each draft: run `single_token_with_multi_hidden` on the
     target → check `argmax == draft_tok`; on first mismatch, emit
     accepted prefix + bonus = `argmax(target)`, restart.
   - GDN/conv state is consistent at every accepted position because
     we processed each token sequentially (lazy verify; no rollback
     needed).
2. Bench `--dflash-lazy` reports α for code/prose/mixed workloads.
3. Cheap experiments to run while we're here:
   - **Effective-N**: use only first `m ∈ {4, 8, 12, 15}` draft rows;
     report α as a function of m. Reveals where α decays in the block.
   - **Top-k rank instrumentation**: for each draft position, record
     the rank of the target's argmax in the drafter's top-k logits.
     Tells us whether α=0.4 is "drafter is near-correct" (top-4 hit
     rate ~0.7 → tree verify viable) or "drafter is way off"
     (top-4 hit rate ~0.4 → no easy uplift from tree verify).
   - **Shuffled-target_ctx baseline** (continued from H5.2): if
     shuffled target_ctx achieves α within 30% of correct target_ctx,
     hidden capture is broken or the model isn't actually conditioning
     on it.

**GO/NO-GO gate:**
- α_draft ≥ 0.50 on code, ≥ 0.30 on prose: PROCEED to H5.3 (packed
  verify earns its complexity).
- α_draft < 0.30 on prose: STOP. Debug drafter forward, SWA mask,
  hidden capture, quant, recipe. Do NOT build packed verify on a
  broken drafter.

This phase produces no speedup (lazy verify is `(1+α)/(1+α+ε_drafter)`
≈ 0.5-0.8× depending on drafter cost) — that's expected. The
deliverable is a number, not a perf win.

### H5.3 — Packed verify forward (the perf-critical chunk)

1. Extend GDN kernel: iterate over N tokens internally with per-token
   SSM and conv checkpoint writes.
2. Extend full-attn kernel: mat-mat Q/K/V/O across N rows. KV append
   writes N contiguous slots. Attention scores against
   `start_position + i + 1` keys for query at relative position i.
3. Extend FFN kernel for N rows.
4. Final norm + lm_head per position; **GPU argmax** writes
   `verify_argmax[N]`.
5. Multi-layer hidden capture at K target layers.
6. Implement `MetalForward::packed_forward` and the `_with_logits`
   debug variant.
7. **Bit-exactness test (H5.3 cosine gate):** for fixed seed,
   `packed_forward(tokens, start, ...)` produces logits cosine ≥ 0.9999
   with the result of N successive `single_token(tokens[i], start+i, ...)`
   calls. This is the gate that says packed semantics match.

### H5.4 — Rollback primitive

1. Implement `restore_after_partial_accept`. Reads checkpoint at
   `n_accepted_in_batch`, writes into session SSM + conv state. Resets
   `kv_n_pos`.
2. Test: pick arbitrary `n_accepted ∈ [0, N-1]`. Run `packed_forward`,
   then `restore_after_partial_accept(n_accepted)`, then a single
   `single_token` at the next position. Compare the resulting state
   to one produced by N+1 single_token calls through carry +
   n_accepted drafts + bonus. Bit-exact match required.

### H5.5 — End-to-end DFlash decode + bench

1. Implement `DFlashDecoder::decode` per §1.3.
2. **Greedy equivalence test** (the H5 correctness gate): generate
   16 / 64 tokens with DFlash=on vs DFlash=off; sequences must be
   IDENTICAL.
3. **Cursor / EOS / max edge cases test matrix:**

   | Test                                              | Setup                                            | Pass criterion                                                              |
   |---------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------|
   | Full reject (all drafts wrong)                    | Drafter conditioning broken                      | Emit 1 token (carry); processed_pos += 1                                    |
   | Full accept (all D drafts right)                  | Easy code prompt                                 | Emit N tokens; processed_pos += N                                           |
   | Partial accept (n_accepted = D/2)                 | Mid-confidence prose                             | Emit n_accepted+1 tokens; processed_pos += n_accepted+1                     |
   | EOS as carry_tok                                  | Bootstrap or post-step                           | Emit EOS, stop. Don't run drafter or verify                                 |
   | EOS in accepted draft (j < n_accepted)            | Drafter outputs EOS at position j; target agrees | Emit prefix through EOS; stop                                               |
   | EOS as bonus                                      | verify_argmax[n_accepted] == EOS                 | Emit accepted prefix + EOS; stop. Carry is EOS but already emitted          |
   | EOS as drafted token at index ≥ n_accepted        | Drafter outputs EOS; target disagrees           | Reject at first disagreement; EOS not emitted (target's argmax wins)        |
   | max_new_tokens hit mid-block                       | limit hits within accepted prefix                | Emit until limit; abort rest of step. Session not resumable per contract.   |
   | All reject + max hit                              | Limit on next iter's carry                       | Emit prior carries, then EOS-or-limit-stop                                  |

4. **Bench:** extend `qwen-bench dflash` analogous to `mtp`. Report:
   - `α_draft = mean_accepted_drafts / D`
   - `mean_emitted_per_step = 1 + α_draft · D`
   - decode-only and total t/s
   - speedup vs no-spec (total wall vs total wall)
   - per-call counts (drafter, verify_N, restore)
   - **bytes_readback_per_outer_step**: anti-regression counter. If
     the bench reports `≥ N · V · 4` bytes per step, a debug logits
     readback path is enabled in production and the throughput number
     is invalid.
   - **Effective-N CLI flag** `--effective-n m`: ignore drafter
     positions `m..N` even though the drafter computed all of them.
     Reduces verify and rollback cost; reveals whether N=16 is
     actually the M4 Max sweet spot.
5. **Per-step state-hash diagnostic** (paranoid mode, off by default
   for prod runs): in addition to comparing emitted tokens between
   DFlash=on and DFlash=off, hash + compare per outer step:
   - `processed_pos`
   - per-layer `kv_n_pos`
   - GDN SSM state checksum
   - GDN conv state checksum
   - `target_ctx_n` and last-column hash
   Catches latent state corruption that would otherwise silently
   surface only as α/quality drift later.
6. **Workloads:** code (`humaneval_short`), prose (`pile_short`),
   mixed (`mmlu_easy`).

### H5.6 — Cross-impl validation against spiritbuun (no longer optional)

Promoted from optional. Cross-impl validation must run alongside
H5.2.5 (acceptance gate) — the cheapest place to discover that
"layer 31" means different things in different impls, or that our
SWA mask is off by one but still produces finite logits.

Phased:
- **Cheap trace-equivalence (alongside H5.2.5):** for one fixed
  prompt and seed, compare against MLX reference at
  `~/code/dflash/dflash/model_mlx.py`:
    * captured target hidden slices at `target_layer_ids`
    * `dflash_fc + hidden_norm` output
    * drafter logits at every noise position
    * accepted-prefix length under lazy verify
- **End-to-end token equivalence (after H5.5):** build spiritbuun's
  fork on M4 Max, run the same `(target_gguf, drafter_gguf, prompt,
  seed)`, compare emitted token sequences. They must be identical.
  Schedule-dependent.

### H5.7 — Probabilistic rejection sampling

Optional. Only after greedy works and α is in expected range. One
Metal kernel; one-hot draft fast path
(`HAS_DRAFT_LOGITS=False`).

## 4. Validation plan

### Correctness gates

- **H5.0**: drafter GGUF loader binds without error.
- **H5.1**: drafter forward produces finite logits.
- **H5.3**: packed_forward cosine ≥ 0.9999 vs N successive single_token
  on the same N tokens. Per-layer hidden capture matches sampled hiddens.
- **H5.4**: state after restore_after_partial_accept bit-exact to
  state after the equivalent number of single_token calls.
- **H5.5**: DFlash=on and DFlash=off produce IDENTICAL token sequences
  under greedy. Test matrix in §3 H5.5(3) passes.
- **H5.6** (optional): cross-impl matches spiritbuun.

### Throughput targets (apples-to-apples wall vs wall)

- **27B-Q4_K_M target + Q8_0 drafter, 64 tokens, code prompt**:
  ≥ 1.6× speedup over DFlash=off baseline.
- **Prose prompt**: ≥ 1.3×.
- **α_draft**: ≥ 0.50 on code, ≥ 0.30 on prose (z-lab claims ~93%
  / ~38% respectively; we account for quant noise).

If α_draft is suspiciously low (<0.20 on prose), the bug is in the
SWA mask, the cross-context hidden projection, or the multi-layer
hidden capture. Greedy equivalence still passes (verifier corrects),
but α tells us the drafter is broken.

## 5. Files to be touched

- `crates/qwen-llm/src/loader.rs` — `DFlashHead`, `DFlashLayer`,
  `open_dflash_drafter`.
- `crates/qwen-llm/src/forward.rs` — CPU drafter forward (oracle for
  H5.1/H5.3; small enough to live alongside `mtp_step`).
- `crates/qwen-llm/src/metal_dflash.rs` — new: `MetalDFlashHead`,
  `MetalDFlashSession`, `MetalDFlashVerifyScratch`, `DFlashDecoder`.
- `crates/qwen-llm/src/metal_forward.rs` — `packed_forward`,
  `packed_forward_with_logits`, `restore_after_partial_accept`,
  `single_token_with_multi_hidden`.
- `crates/qwen-llm/src/metal.rs` — packed-N kernels (extend mat-vec
  to mat-mat for F32/Q4_K/Q5_K/Q6_K/Q8_0; GDN packed; flash-attn
  v4 verified for N=16 queries).
- `kernels/*.metal` — new packed kernel templates. GDN gets per-token
  checkpoint write.
- `crates/qwen-cli/src/bench.rs` — `dflash` subcommand.
- `crates/qwen-llm/tests/dflash_correctness.rs` — equivalence + cosine
  + edge case matrix.

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Packed N=16 kernels slower than 16× single-token (no amortization win) | H5.3 measures each phase. Keep single-token fallback as the floor. |
| GDN checkpoint memory (2.3 GB SSM + 90 MB conv per outer step) | M4 Max 128 GB unified handles it. Cap context if blows up. |
| Conv state forgotten in checkpoint (rev-1 omission caught in review) | §1.6 explicitly checkpoints both SSM and conv per token. |
| SWA mask geometry wrong → α corrupted | H5.5 step (3) test matrix; reproduce α against spiritbuun ranges. |
| Drafter quant Q4_K_M acceptance drop on 3.6 (~15 pts) | Use Q8_0 (1.85 GB) per spiritbuun's published numbers. Bench Q4_K_M as a separate row. |
| `--enable_thinking` template enables thinking-mode by default | Document `--chat-template-kwargs '{"enable_thinking": false}'`; bench only without thinking. |
| Drafter `target_layer_ids` references invalid target layers | H5.0 validation rejects at load. |
| H_drafter != H_target on a future drafter | H5.0 validation rejects at load. |
| `packed_forward` spilling N×V logits to CPU | Default API returns argmax (GPU); `_with_logits` only for debug. |
| Multi-layer hidden capture cost (1.6 MB/step) | Trivial; inlined in packed_forward command buffer. |

## 7. Reference sources

- **`/tmp/dflash-forks/REPORT.md`** — 4-fork comparative analysis
  (Z-Lab v1 configurable vs Luce 5-layer fixed recipes).
- **`/tmp/dflash-forks/raw/spiritbuun089_dflash_draft.cpp`** (487 LOC) —
  primary reference. Single-slot path (lines 40-156) mirrors what we
  implement. SWA mask construction (lines 134-156).
- **`/tmp/dflash-forks/raw/ruixiang_dflash.cpp`** (160 LOC) — minimal
  z-lab v1 reference. No SWA.
- **`~/code/dflash/dflash/model_mlx.py`** (471 LOC) — z-lab official
  MLX impl. Cleanest read of the algorithm.
- **spiritbuun's HF README** — Q8_0 mandatory for 3.6 SWA layers,
  thinking-mode footgun, exact CLI invocation.
- **`~/models/spiritbuun-dflash/dflash-draft-3.6-q8_0.gguf`** — actual
  drafter; all dims in §1.1/§1.2 confirmed against this file.
- **z-lab paper** arXiv:2602.06036.

## 8. Position in the broader plan

H4 shipped the spec-decode infrastructure validated against a known
flat-0.91× baseline (MTP). H5 reuses the infrastructure and adds:

| Component | H4 (MTP) | H5 (DFlash) |
|---|---|---|
| Drafter | In-target MTP head | Separate ~1.7 B drafter GGUF |
| Block size | 1 | 16 (D=15 candidates) |
| Drafter KV | Persistent across drafts | Transient per outer step |
| Target hidden expose | Single layer (pre-output_norm) | K layers (sampled) |
| Verify forward | Sequential single-token | Packed N=16 |
| Rollback | None (lazy verify) | Per-token SSM + conv checkpoints + KV n_pos reset |
| Acceptance | Greedy match | Greedy match (probabilistic later) |
| SWA support | N/A | Required (drafter has SWA layers) |
| Predicted speedup | 0.91× (cost model) | 1.5-2.5× (depends on packed kernel quality + α) |

Shared infrastructure carried forward from H4: `SpeculativeDecoder`-style
wrapper pattern, drafter loader boilerplate, prompt-prefill streaming,
EOS / max_new_tokens edge cases, GPU-resident hidden carry, `processed_pos
+ carry_tok` cursor model.

## 9. Follow-on opportunities (after H5.5 lands)

Prioritized for "what to do if H5.5 hits α ≥ 0.5 and speedup ≥ 1.5×."
Order is bang-for-buck, easiest first.

1. **GPU-resident sampling kernel (top-k, top-p, argmax).** If any
   vocab-sized logits readback remains in the production path
   (per-iteration `N · V · 4 B`), kill it. spiritbuun's fork does this
   for a reason. The lm_head compute is unavoidable; the 16 MB CPU
   transfer + sync per outer step is not. Bench's
   `bytes_readback_per_outer_step` counter (per H5.5) flags it
   automatically.

2. **Indirect Command Buffers / MTL4** to amortize CPU encode cost.
   At N=9 tokens emitted per step, ~3-5 ms encode overhead per outer
   step. The architecture already has stable buffers and a fixed
   pipeline sequence — likely a clean win.

3. **Effective-N tuning.** Cheap experiment via `--effective-n m`
   (already in the bench per H5.5). If N=8 captures 85% of the
   emitted tokens at half the verify+rollback pain, latency wins
   even when throughput doesn't. M4 Max's BW profile may favor
   smaller N than the trained default.

4. **Profile-driven packed kernel work.** Optimize what's actually
   slow, not what we expected to be slow. Phases to break down:
   GDN recurrence, lm_head, FFN mat-mat, attention, checkpoint
   writes, command-buffer overhead. The phase profiler from
   `single_token_phase_profiled` extends to the packed path.

5. **PLD (prompt-lookup decoding) as a baseline.** Cheap experiment
   that reuses H5's packed verify primitive but with a trivial
   "next-N tokens copied from the prompt where they appeared earlier"
   drafter. If PLD + packed verify achieves 1.3× and DFlash + packed
   verify achieves 1.4×, the engineering investment beyond PLD is
   small. If the ratio is more like 1.3× vs 2.0×, the learned drafter
   is genuinely earning its complexity. Either answer is informative.

6. **Tree verify (DDTree) — only if instrumentation says so.** The
   H5.2.5 top-k rank diagnostic measures: rank-1 hit rate (= α_draft)
   and rank-4 hit rate. If rank-1 is mediocre but rank-4 is rich
   (e.g. α=0.4 but rank-4=0.8), DDTree can recover more emitted
   tokens per verify. If rank-4 is also poor, tree verify just burns
   compute. DDTree is a second speculative engine, not a driver
   change — defer until measurements demand it.

Lower priority / different product:

- **KV-Q8 on the 16 attn layers** (PLAN.md v2 item). Important for
  long context + memory pressure; not ahead of DFlash-specific
  bottlenecks unless attention KV bandwidth shows up dominant.
- **Continuous batching (multi-slot drafter)** — different product
  (multi-user serving). Don't let it distract H5 unless we decide
  the product is multi-user.
- **Approximate / pollution-tolerant mode** — skip GDN rollback,
  accept drift. Quality-vs-throughput tradeoff. Would surface as an
  opt-in `--dflash-approximate` flag with perplexity-drift telemetry.
- **Probabilistic rejection sampling** with retained drafter
  probabilities — needs sampling temp > 0 path; H5.7 stub.

If H5.5 does NOT hit 1.5×, the next action depends on which arm of
the failure:
- Low α: drafter correctness, SWA mask, hidden capture, quant choice
  (Q4_K vs Q8_0), or domain mismatch (try `enable_thinking=false` or
  different workloads).
- Slow verify: packed kernel investment, checkpoint strategy,
  command overhead.
- High α + low speedup: implementation problem, profile and fix.
- Low α + perfect kernels: this is a model/speculation problem;
  DFlash with current weights doesn't work for this stack.

## Document history

- **rev 3 (current).** Open-ended codex review after H5.0 + H5.1
  shipped. Insertions:
  - **H5.1.5 Metal-vs-CPU drafter cosine gate** — finiteness alone
    proved too weak in the cross-GGUF dequant bug (caught by codex
    in H5.1).
  - **H5.2 layout sanity test** — shuffled `target_ctx` should
    materially degrade α; if it doesn't, hidden capture is broken
    or the model isn't actually conditioning on `target_ctx`.
  - **H5.2.5 Lazy DFlash acceptance gate (NEW, mandatory before
    H5.3)** — measures α end-to-end using cheap H4-style
    single-token verify before paying for packed verify. GO/NO-GO:
    α ≥ 0.50 code, ≥ 0.30 prose. Includes effective-N sweep and
    top-k rank instrumentation as cheap experiments.
  - **§1.4 worst-case scenario** explicitly documented (0.81×
    speedup if α=0.30 and T_verify_N=6×). Not a corner case.
  - **§4 / H5.5 bench additions**: bytes-readback counter (production
    no-readback guard), `--effective-n m` flag, paranoid
    per-step state-hash diagnostic.
  - **H5.6 cross-impl validation** promoted from optional to
    mandatory; cheap trace-equivalence runs alongside H5.2.5
    against MLX, full token-equivalence against spiritbuun after H5.5.
  - **§9 reframed** as prioritized follow-on opportunities with a
    "what to do if speedup is bad" decision tree.
- **rev 2.** Reviewed and rewritten after codex round 1
  surfaced 3 BLOCKERs and 8 MAJORs:
  - Renamed cursor state to `processed_pos` + `carry_tok` (matches H4
    contract). Algorithm now flows through carry properly: draft input
    is `[carry, MASK × (N-1)]`, verify input is `[carry, draft[0..D-1]]`.
  - Defined `N = block_size`, `D = N-1`. Draft tokens read from
    drafter positions `1..N-1`. Bonus is `verify_argmax[n_accepted]`.
    Tokens emitted per step = `1 + n_accepted` ∈ `[1, N]`.
  - All dims pulled from real GGUF (§1.1, §1.2). H_drafter = 5120
    (= H_target, hard requirement for shared embed/lm_head).
  - SWA support moved into H5.1 (was H5.6).
  - Conv state added to checkpoint design (was missing).
  - `packed_forward` returns argmax tokens (GPU-computed); separate
    `_with_logits` debug variant for the H5.3 cosine test.
  - H5.5 test matrix expanded to cover full-reject, full-accept,
    partial accept, EOS / max_new_tokens edge cases.
  - α_draft definition pinned to mean_accepted_drafts / D.
- **rev 1.** Initial draft — placeholder dims, missing carry-token
  state, SWA deferred too late.
