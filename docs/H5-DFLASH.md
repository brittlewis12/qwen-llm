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

### H5.1.5 — Plumbing cosine gate (Metal-vs-CPU on the hybrid impl)

A plumbing test, NOT a DFlash correctness gate. The hybrid v1 of
`DFlashDecoder::draft_block` runs Metal-resident kernels for
projections + RMSNorms + RoPE and falls back to CPU for the
asymmetric SWA-masked attention + SwiGLU FFN. The cosine vs CPU
oracle therefore tests that:

* drafter weight tensors are scoped to the drafter GGUF (catches the
  cross-GGUF dequant bug class)
* `target_ctx_stacked` column layout, `pos_ctx` ordering, row
  slicing, and shared-target-lm_head wiring are coherent
* per-call command-buffer encode + readback rhythm doesn't introduce
  ordering hazards

It does NOT validate the drafter's actual *distribution* — most of
the per-layer compute that determines logits (attention with SWA
mask + FFN silu_mul) runs on CPU on both paths. The H5.5 greedy
equivalence test + the H5.2.5 measured α are the real correctness +
usefulness gates. **If H5.1.5 cosine fails, suspect plumbing
(buffers, scoping, layout). If H5.2.5 α is bad, suspect algorithm
(SWA mask, hidden capture, recipe).**

1. Pick a deterministic 4-token prompt (`"The quick brown fox"`).
2. Run target prefill on Metal AND CPU; capture multi-layer hiddens
   at `target_layer_ids` from each.
3. Run `DFlashDecoder::draft_block` (Metal hybrid) AND
   `Forward::dflash_draft` (CPU oracle) with the same
   `(noise_ids, target_ctx_stacked)` inputs.
4. Pass criterion: cosine ≥ 0.9999 between Metal hybrid and CPU draft
   logits at every noise position. Argmax must match.

A separate cheap CPU-vs-MLX trace equivalence check (H5.6) tests the
algorithm independently of the Metal dispatch.

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

### H5.3 — Packed verify forward

Codex partner session (post-H5.2.5-GREEN) re-sequenced this phase.
**Original instinct:** front-load Bucket A (mechanical N-grid extends) →
GPU argmax → naive C/D → Bucket B (per-token GDN/conv checkpoints).
**Codex pushback:** "risks getting a pretty packed API that cannot roll
back. The earliest high-value signal is: can this API advance N tokens,
checkpoint every intermediate state, restore any accepted prefix, and
match N single-token decode?"

Same playbook as the H5.2.5 pivot — split the phase. **H5.3a** ships a
state-correct packed verifier scaffold (production-shaped API,
correctness-verified, allowed to be slow). **H5.3b** is the perf pass
(tiled Q4_K mat-mat first because naive 16× weight re-reads is
structurally fatal; internalized GDN/conv recurrence second; packed
attention only if profiling forces it).

#### H5.3a — Correctness scaffold (no speedup expected)

The deliverable is the production API surface + bit-exactness, NOT
throughput. H5.3a packed_forward is naive: N successive single-token
encodes inside one command buffer, with checkpoint copies between
tokens. Allowed to be slower than 16× single_token wall — that's fine
because it lands the API + restore primitive cleanly.

1. **`MetalDFlashVerifyScratch` struct** (NEW; codex Q5 expansion).
   Owns ALL N-shaped state for packed forward — outputs (verify_argmax,
   hidden_capture, conv_ckpt, gdn_ckpt) AND target-side activations
   (packed_ids_buf, x_pack `[N,H]`, h_pack, q/k/v/o pack buffers, ffn
   pack buffers, debug logits). Threaded `&mut` through `packed_forward`.
   `MetalSession` stays unchanged; single_token keeps its scratch.
   Lifetime: per-`DFlashDecoder`, allocated once.
2. **Per-token checkpoint machinery via Metal blit copies** (codex Q4
   v1: design ii — keep `kernel_gdn_step_f32` as-is, append a blit copy
   into `gdn_ckpt[layer, n, ...]` after each step). Same for conv state
   into `conv_ckpt[layer, n, ...]`. Blits are bulk-memory ops on GPU DMA
   engines — faster than a compute "copy kernel" and avoid kernel
   dispatch overhead. The 2.3 GiB SSM checkpoint is contiguous bulk
   memory; treat it like bulk memory.
3. **`packed_ids_buf: MetalTensor [N]`** (codex Q7 mitigation — the
   bug we'd otherwise ship). NEVER mutate `MetalSession::ids_buf` from
   the host inside the packed encode loop; `get_rows` would race and
   every call would read the last-written CPU value. Upload once at
   the top of `packed_forward`; each block reads by offset
   `packed_ids_buf[n..n+1]`.
4. **`MetalForward::packed_forward(tokens, start_pos, target_layer_ids,
   scratch, session) → Vec<i32>`** — naive encoding: for `n in 0..N`
   { write tokens[n] at packed_ids_buf[n]; encode N single-token paths;
   blit GDN+conv state into checkpoints; capture hidden at K target
   layers into hidden_capture[K, n, :] }. Final lm_head + GPU argmax →
   verify_argmax[N].
5. **`packed_forward_with_logits` debug variant** — adds `[N, V]`
   readback for the bit-exactness test. Production `packed_forward`
   does NOT spill `[N, V]` (anti-regression assertion in tests).
6. **GPU argmax kernel** (Bucket E, mandatory in H5.3a, not deferred).
   Standard reduction-tree two-stage `[N, V] → [N] i32`. Avoids the
   15.9 MB readback per outer step.
7. **`restore_after_partial_accept`** — H5.4 was originally a separate
   phase, but lands in H5.3a because the checkpoint contract isn't
   testable without it. Blit `conv_ckpt[*, k]` and `gdn_ckpt[*, k]`
   back into session GDN+conv state; reset `kv_n_pos = start_pos +
   k + 1`. Same blit primitive as the forward writes, reversed.

**H5.3a gates** (per codex Q6 — 7 cheap tests, ALL gating). Status
as of v0.60:

- **G1 — packed-vs-single logits cosine.** ✅ DONE (v0.60).
  `packed_verify_with_logits(tokens, start)` cosine vs N successive
  `single_token(tokens[i], start+i)` per row. F32 path: **cos =
  1.0000000000 EXACTLY**, max|Δ| = 0, 100% bit-exact across 993,280
  vocab values. The H5.3 plan threshold of 0.9999 was designed for
  quantized paths; F32 trivially crushes it.
- **G2 — final state equivalence.** ✅ DONE (v0.57+v0.58, BITWISE).
  Post-packed `gdn_state`, `gdn_conv`, `kv_n_pos` bit-exact (no slack
  tolerance) vs N successive single_token's terminal state on all
  24 GDN layers. Codex review tightened from `< 1e-5` to BITWISE EQ;
  proves F32 dispatch order is genuinely deterministic.
- **G3 — checkpoint replay equivalence.** ✅ DONE (v0.59). For each
  n_keep ∈ {1, N/2, N}: run packed_verify, restore, marker
  single_token; bit-exact vs n_keep sequential single_tokens +
  marker. Proves checkpoint CONTENTS at each n ∈ [0, N) are exactly
  the right bytes to restore to.
- **G4 — hidden capture layout.** ✅ DONE (v0.60). For each (k, n):
  `scratch.hidden_capture[k, n, :]` bitwise-equal to
  `single_token_with_multi_hidden`'s per-token capture at row k.
  Test ALSO asserts captured layers are L2-distinct so a K↔N
  transpose bug couldn't pass silently.
- **G5 — GPU argmax vs CPU argmax.** ✅ DONE (v0.57). `verify_argmax[n]
  == cpu_lowest_idx_argmax(reference logits[n])` across all n.
  Lowest-index tie semantics tested explicitly across 8 cases
  including all-zero, all -inf, all-NaN, vocab-sized 248320×16
  random fuzz.
- **G6 — restore boundary cases.** ✅ DONE (v0.59 — folded into
  G3 test which exercises n_keep ∈ {1, N/2, N} simultaneously).
- **G7 — nonzero start_pos / attention partition exercise.**
  ⏳ DEFERRED to H5.5 integration. The lib-loop test
  `dflash_packed_verify_with_primed_session` (v0.58) covers the
  general start_pos > 0 case on 0.8B (which has no attn layers, so
  no kv_n_pos to exercise). The full G7 (start_pos > rows_per_partition
  on attn-v4 multi-partition mode) requires ~7K context on 27B-Q4_K_M
  → ~10 min CPU prefill; better to ship alongside the H5.5 end-to-end
  greedy-equivalence integration test.

**v0.60 H5.3a gate summary**: 6 of 7 lib-loop gates green. Six gates
all bit-exact / cos=1.0 for the F32 packed_verify path. The only
remaining gate (G7) is fundamentally a 27B integration test that
will land alongside H5.5.

**Production API anti-regression assertion** (in tests):
`bytes_readback_per_outer_step < N · V · 4` always for the production
path. Caught here, also surfaced by the H5.5 bench counter.

#### H5.3b — Performance pass (where the speedup actually lands)

Re-sequenced after a codex partner session at the start of v0.62.
Three sub-phases, each shippable independently with its own gates.

##### H5.3b.0–3 — Q4_K mat-mat kernel + bench (NO plumbing yet)

Lift llama.cpp's `kernel_mul_mm_q4_K_f32` (the non-MPS-tensor classic
path, lines 9440–9648) verbatim into `kernels/mat_mat_q4_k.metal`,
then bench in isolation.

* **Tile shape** = the lifted classic: NR0=64 (M), NR1=32 (N), NK=32
  (K-step), 4 simdgroups per TG = 128 threads. Codex Q4 (i): for
  N_QUERY=16 the column tile half-fills (uses the kernel's existing
  partial-output-tile path via threadgroup-mem `temp_str` shuffle);
  accept this for v1, profile-driven retune later if measurements
  show the half-fill is fatal.
* **Output stride** = column-major `dst[row + col * ne0 + im*ne1*ne0]`
  per the lifted kernel. **Codex Q7 failure-mode prediction**: this
  is THE pitfall — our scratch is row-major `[N, H]`. The host
  wrapper must explicitly transpose via stride args OR we wrap the
  kernel to write row-major. Tests must read the result the way
  downstream consumers will, NOT the way the host wrote it (else
  cosine passes via shared bug).
* **Per-kernel correctness gates** (codex Q3 correction — lifted
  mat-mat is NOT bit-exact with our scalar mat-vec because llama
  templates stage activations through `half`/simdgroup_matrix
  before float accumulation; expect cos ≥ 0.999, NOT cos = 1.0):
    1. `mat_mat_q4_k vs CPU mat-mat oracle` — cos ≥ 0.9999
    2. `mat_mat_q4_k vs N existing mat-vec outputs` — cos ≥ 0.999
       per row (relaxed from 0.9999 because of half-staging
       precision differences)
    3. **Layout gate** — pick a deterministic per-row signature
       (e.g. checksum of row 0, row N/2, row N-1) and assert the
       output buffer is row-major `[N, H]` as expected by
       downstream FFN/residual paths. This catches the column-major
       vs row-major bug class codex flagged.
* **Bench in isolation** before plumbing (codex Q5): chained64
  GiB/s + wall-time-vs-16×-mat-vec speedup. The naive H5.3a path
  is the floor; the lifted kernel is the ceiling for now. Report
  both.

H5.3a packed_verify is unchanged in this sub-phase. Lib loop stays
green (0.8B is F32, no Q4_K paths affected).

##### H5.3b.4–5 — Layer-major encode_block_packed + plumbing

Codex Q1: the FFN/projection mat-mat win only fires if
packed_verify is **layer-major** (loop over layers, each layer
processes all N tokens), NOT token-major (current naive). The
naive H5.3a path is token-major because it reuses single-token
scratch — packed_verify ran N successive single_token in one
cmd buffer.

* New `encode_block_packed(layer, N, ...)` that:
  * batched RMSNorm across all N tokens
  * batched mat-mat for projections (using H5.3b.0 Q4_K kernel)
  * **per-token inner loop for GDN + attn** (sequential by nature;
    these can't pack along time)
  * batched FFN gate/up via Q4_K mat-mat, batched silu_mul,
    batched FFN down via Q4_K mat-mat
  * residual #1 + #2 batched
* `packed_verify` switches to layer-major: outer loop over layers,
  inner per-token-loop only for GDN/attn pieces.
* Per-layer end: blit GDN+conv state checkpoints (same Codex-Q2
  design Y as H5.3a), capture hidden if target_layer.
* Per-token end (after final layer): lm_head mat-mat across all N
  rows (still mat-vec for now — see H5.3b.6), GPU argmax per row.

**H5.3b.4–5 gate**: ALL H5.3a gates G1–G6 still green. Especially
G3 (checkpoint replay) and G4 (hidden capture layout) — those test
that the layer-major rewrite didn't break the H5.3a contracts that
the kernel-level cosine alone can't see (codex Q7 second-most-likely
failure mode).

**H5.3b.4–5 instrumentation requirements** (codex bench-review,
v0.64): per the H5.3b.0 isolated bench results, the lifted
mat-mat kernel hits only 12-20% peak BW even at chained64.
That's real in-kernel underfill (chained64 already amortizes
launch/wait), NOT just dispatch overhead. So:

* **Track command-buffer structure explicitly**: encoders,
  dispatches, waits separately. If layer-major still waits per
  projection/layer, the scheduling win evaporates.
* **Record GPU time, not just wall**, so encoder overhead
  doesn't hide kernel cost. Wall speedup can improve from
  batching while the kernel itself stays bad — fatal for the
  H5.5 cost model.
* **Watch the skinny projections** (attn_gate at 1.68× isolated
  is the canary). Many ops in layer-major may resemble this
  partial-tile regime; end-to-end could be dragged down even
  if FFN/embed look healthy.

**H5.3b.4–5 design decisions** (codex layer-major partner session,
v0.65; baked in here so they aren't re-debated mid-implementation):

* **GDN/conv per-N checkpoints: Option A** — inline compute→blit
  transitions inside the per-N inner loop. ~1536 transitions per
  outer step total (48 GDN layers × 16 N × 2). On M-series each
  transition is ~µs; total overhead is ~1-3 ms / outer step,
  well under the 184+ ms FFN savings. Post-v1 optimization (NOT
  blocking H5.3b.4-5): a compute "save checkpoint" kernel that
  copies `gdn_state[k]` / `gdn_conv[k]` → `ckpt_slot(k, n)` in
  the same compute encoder, eliminating all transitions with no
  temp storage. Codex idea; defer.

* **Attn `o_proj`: batched mat-mat across N**. Per-token decode
  into `attn_o_pack [N, q_dim]`, then one Q4_K mat-mat. Without
  this, `o_w` (17 MB Q4_K) is read 16× per attn layer × 16 attn
  layers = ~4.4 GB redundant traffic per outer step — same loss
  pattern as un-batched FFN.

* **K/V projection fusion: deferred**. Separate Q4_K mat-mats
  for K and V. Q is special (gated, output `2 * q_dim`). Fusion
  is a new kernel surface; not blocking. If profile says skinny
  projections dominate post-ship, add fused K+V mat-mat as
  follow-on (mirrors how `encode_ffn_swiglu_q4_K_f32` fuses
  ffn_gate + ffn_up).

* **Token-major path stays as oracle**. Layer-major is new free
  fn `encode_packed_verify_layer_major_inner`; production
  `DFlashDecoder::packed_verify` defaults to layer-major behind
  a feature toggle, fallback to token-major. Tests run BOTH on
  identical inputs and compare bit-exact wherever possible.
  Token-major remains the algorithmic ground truth.

* **Dtype dispatch inside `encode_block_packed`**. Single call
  site decides Q4_K mat-mat vs F32 per-token mat-vec. Scattering
  dtype branches across layer/mixer/tail code makes rollback
  bugs hard to localize.

* **Parallel scratch struct `MetalDFlashLayerMajorScratch`**.
  Holds the new N-wide activation buffers (`x_pack`, `h_pack`,
  `mixer_out_pack`, `attn_q_full_pack`, K/V/O packs, FFN packs).
  Total ~5.6 MB at 27B N=16. Sits alongside
  `MetalDFlashVerifyScratch` (which keeps owning checkpoints,
  hidden_capture, packed_ids_buf, verify_argmax). Token-major
  callers don't allocate this.

**H5.3b.4-5 codex-flagged failure mode (mitigation built into
test plan, not just hoped for)**:

The mat-mat output is col-major `[n_out, n_query]`-flat;
layer-major scratch is naturally `[N, dim]` row-major. EVERY
consumer of mat-mat output has to either transpose-on-read or
be col-major-aware. The H5.3a gates on argmax and final state
may pass even if intermediate layouts are silently transposed
(argmax + final state can mask shape-only bugs on lucky
logits).

Mandatory NEW intermediate-layer correctness tests:
  * `attn_q_full_pack` row-cosine vs single-token equivalent
  * `attn_k_now_pack`, `attn_v_now_pack` ditto
  * `attn_o_pack` (post per-token decode, pre o_proj)
  * `ffn_gate_pack`, `ffn_up_pack`, `ffn_inner_pack`,
    `ffn_out_pack` ditto
Per-row cosine ≥ 0.999 vs N-times-single-token reference. Done
ONCE on a representative GDN+attn+FFN layer set; no need to
test every layer if the kernels are bit-exact.

**H5.3b.4–5 → tile retune tripwire** (codex bench-review,
v0.64; pre-authorized — no need to re-debate):

If after layer-major plumbing EITHER condition fires:
  * packed_verify is **< 2.5× over naive** on FFN-heavy paths, OR
  * Q4_K mat-mat steady-state is **< 150 GiB/s GPU-time**

then the tile retune is mandatory. Predeclared retune target:
NR1 = 16 (not 32; matches our N_QUERY=16 with no half-fill),
keep NR0 = 64 if register pressure allows, REMOVE the partial-N
hot path, reduce simdgroup layout to match full 16-column
occupancy. Becomes new sub-phase H5.3b.5.5 if triggered.

##### H5.3b.6 — Q6_K mat-mat (ffn_down + lm_head)

Same playbook as H5.3b.0: lift `kernel_mul_mm_q6_K_f32` from
llama, ship + bench + per-kernel gates, then plumb into
encode_block_packed (lm_head and ffn_down are the remaining
weight-traffic sinks per layer).

##### Deferred (post-H5.3b)

* Internalized packed GDN recurrence (single kernel, N steps
  inline, ckpt write per step). Saves 16× state-load BW +
  dispatch overhead vs the per-token blit-checkpoint approach
  shipped in H5.3a. Bit-exactness oracle: the H5.3a impl.
* Packed flash-attn-v4 — ONLY if profile data shows attention
  dominating. Codex H5.3 Q3: "real long-context cost is N·L,
  not N²/2. FFN/projection weight traffic is the larger
  structural waste first. Don't touch attn_v4 until profile
  data forces it — adding an N axis to m/l/o can pass
  short-context tests and fail at partition boundaries."

##### H5.3b.7 — Tensor API exploration (POST-H5.3b ship; not immediate)

llama.cpp ships TWO `kernel_mul_mm_q4_K_f32` implementations gated
on `GGML_METAL_HAS_TENSOR`:

* The **classic path** (lines 9440–9648, lifted in H5.3b.0): explicit
  `simdgroup_matrix<f16, 8x8>` + manual threadgroup-mem dequant tile
  + simdgroup loads. Portable across all M-series; supported on
  every macOS Metal version we'd ship to.
* The **tensor path** (lines 9315–9431, gated): Apple's
  `mpp::tensor_ops::matmul2d` cooperative tensor API + `tensor()`
  wrappers. Uses MPS-Graph-style cooperative-tensor primitives,
  which on M3+ map to dedicated matrix hardware. Materially faster
  on M3/M4/M5 in llama.cpp's own benchmarks; tightly coupled to
  the Apple-private tensor-ops header set; recompiles required when
  Apple ships changes.

**Timing — strictly AFTER H5.3b.6 ships and we have a stable
classic-path baseline.** Don't fork the bring-up over an unknown
perf delta; we need the classic path correct + benched first so we
have a reliable A/B comparator.

**Deliverables when we get to it:**

1. **Spike: tensor-path mat-mat for one shape** (e.g.,
   FFN gate at 5120×17408×16). Ship behind a `#ifdef
   QWEN_LLM_HAS_TENSOR` (mirror llama's gating). Compare
   against the H5.3b.0 classic baseline on identical inputs:
   wall, GiB/s, max|Δ| from the established correctness
   oracle.
2. **Real measurement on M4 Max** at our shapes (N=16
   query, not N=512 prefill). The tensor API was designed
   for prefill batches; our skinny-N case may not benefit
   as much as llama's pp512 numbers suggest.
3. **Maintenance evaluation** — the real question. Three
   axes:
   * **Apple version churn risk.** `mpp::tensor_ops` is
     header-versioned; Xcode SDK upgrades have broken it
     in the past (cite specific llama.cpp issues if found
     during the spike). Quantify how often we'd need to
     re-port vs. our classic path which has been stable
     since simdgroup_matrix shipped in 2018.
   * **Build-system complexity.** The `GGML_METAL_HAS_TENSOR`
     macro requires SDK probing at build time. Our build.rs
     would need a feature-detection step. Acceptable, but
     non-trivial.
   * **Backport surface.** If we adopt tensor-path for Q4_K,
     we'd want it for Q5_K/Q6_K/F32/F16 too for consistency
     — that's 5+ kernels to maintain in two flavors, OR a
     hard cutover with no fallback for older OS / non-tensor
     hardware.
4. **Decision matrix** for adoption:
   * **Adopt fully** (classic kernels deleted): only if
     speedup ≥ 30% AND Apple churn rate ≤ once per major
     macOS version AND our minimum-OS target is M3+.
   * **Adopt as opt-in** (both paths shipped, runtime
     pick): if speedup ≥ 15% AND we're willing to maintain
     both. This mirrors llama's posture.
   * **Skip** (classic kernels only): if speedup < 15% OR
     Apple churn rate is high OR we want one source of
     truth for Q-quant kernels.

The decision is entirely a measurements-driven call; not
suitable for design speculation in advance. Filed here so it
isn't lost; will surface as a separate phase after H5.3b.6
ships and we have reliable baseline numbers to compare
against.

**H5.3b gate**: every optimized kernel keeps cosine ≥ 0.999 against
the naive H5.3a baseline (relaxed from 0.9999 per codex Q3 — half-
staging in lifted kernels is a real-but-tiny precision diff).

### H5.4 — (folded into H5.3a)

`restore_after_partial_accept` originally a separate phase, now lands
inside H5.3a. The checkpoint contract isn't actually testable without
the restore primitive — H5.3a gate G3 (checkpoint replay equivalence)
requires it. Per codex partner session: "build the rollback contract
in the same phase that ships the checkpoints; the rest is unverifiable
without it."

The bit-exactness test that previously lived in H5.4 (test 2 above) is
H5.3a gate G3 + G6 combined.

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

- **rev 15 (current).** v0.75.0 skip-tail prefill SHIPPED. All-but-last
  prompt tokens now skip final RMSNorm + lm_head + readback (~5%
  TTFT win at every measured ctx). Established the prefill API shape
  for v0.75.1 (packed multi-token prefill) to swap in. Codex pressure-
  test caught and prevented a `session.ids_buf` reuse hazard that
  would have shipped as a silent prefill bug if we'd attempted async
  pipelining (Option 2). Bit-exact correctness gate (final logits +
  accumulated multi-hidden capture) added to lib loop (77 tests, was
  76).

  ### v0.75.0 measurements (production 27B Q4_K_M, 32-token gen)

  Apples-to-apples vs v0.74.4 baseline (DFlash prefill ms):

    | ctx | v0.74.4 | v0.75.0 | Δ ms  | Δ %    |
    |----:|--------:|--------:|------:|-------:|
    |   9 |     376 |     355 |   −21 | −5.6%  |
    | 181 |    7502 |    7116 |  −385 | −5.1%  |
    | 363 |   15278 |   14329 |  −949 | −6.2%  |

  Phase profiler at ctx=181 shows lm_head 2.07 ms / token (5.0% of
  phase_sum 41.27 ms). 180 skip-tails × 2 ms = 360 ms predicted; we
  measured 385 ms. Matches kernel-level expectation.

  Decode-only and total-wall ratios unchanged (the change only
  touches prefill). Greedy equivalence PASSES at every ctx.

  ### Codex pressure-test catches

  Pre-implementation review on /tmp/v0.75.0-codex-prompt.md:
  - **Q1 reshape**: dropped `Result<Option<Vec<f32>>>` signature in
    favor of new `_no_tail` methods returning `Result<()>`. Avoids
    breaking ~20 existing call sites; keeps prefill semantics out of
    the decode API.
  - **Q3 trap saved**: I was planning to skip the `waitUntilCompleted`
    after each no_tail call to chain prefill iterations. Codex
    flagged that `session.ids_buf` is reused across calls — Metal
    cmd-buffer ordering does NOT order CPU writes to shared buffers
    after commit. If get_rows for token i hadn't run yet when CPU
    overwrote ids_buf for token i+1, get_rows would read the wrong
    id. Silent prefill corruption. Decision: ship Option 1 (commit +
    wait), defer real async pipelining to v0.75.1 where packed
    prefill restructures the loop.

  Post-implementation review on /tmp/v0.75.0-codex-review-prompt.md:
  no blocking findings. Two non-blocking nits applied: removed
  `(no_tail=N tail=N)` instrumentation print (now obvious from code +
  test), `cargo fmt` clean.

- **rev 14.** v0.74 ctx caching SHIPPED + v0.74.2 drafter
  phase 3 → Q8_0 mat-mat SHIPPED + v0.74.3 hot-loop sync cleanup
  SHIPPED. **DFlash now beats no-spec at every measured ctx.**
  Cumulative since v0.72.4 baseline at default-ctx: **0.452× → 1.008×**
  (+123%). Reviewer's external code review caught the v0.74.2 lever
  (drafter phase 3 was still per-row mat-vec despite Q8_0 mat-mat
  having shipped 5 commits earlier).

  ### Cumulative trajectory (production 27B Q4_K_M, 32-token gen)

    | version    | default ctx (5) | ctx=181 | ctx=363 |
    |------------|----------------:|--------:|--------:|
    | v0.72.4    |          0.452× |       — |       — |
    | v0.73a.1   |          0.548× |       — |       — |
    | v0.73b.1   |          0.746× |       — |       — |
    | v0.73c.1   |          0.805× |  0.874× |  0.984× |
    | v0.74      |       (~0.81×)  |  1.136× |  1.321× |
    | v0.74.2    |          1.005× |  1.336× |  1.465× |
    | **v0.74.3**|       **1.008×**|**1.342×**|**1.482×**|

  Total wall (apples-to-apples) at v0.74.3:

    | ctx | decode-only | total wall |
    |----:|------------:|-----------:|
    |   5 |       1.008×|     1.005× |
    | 181 |       1.342×|     1.038× |
    | 363 |       1.482×|     1.046× |

  Total wall is dominated by prefill (which remains scalar-token
  replay; reviewer's #1 long-ctx lever). Decode-only beats no-spec
  on every short-mid ctx; that's the speculative-decode KPI.

  ### What v0.74 (collectively) did

  Shipped over 4 sub-commits (v0.74.0 + v0.74.1 + v0.74.2 + v0.74.3).
  Combined as the "v0.74" series in this doc because they share one
  thesis: the drafter was doing more redundant per-step work than the
  v0.73c profile data made obvious.

    * **v0.74.0**: phase 1 ctx_h cache. Watermark-based append-only.
      `for c in 0..ctx_len` → `for c in ctx_h_ready_n..ctx_len`.
      Eliminated re-projection of cached cross-context columns.
      ~50 LOC, no new kernels.

    * **v0.74.1**: phase 2 per-layer K_ctx/V_ctx cache. Same
      watermark pattern across all 5 drafter layers. Phase 3 attn
      reads from per-layer cache instead of shared k_ctx_buf /
      v_ctx_buf. ~70 LOC, no new kernels. Memory: 5 layers × ctx_capacity
      × kv_dim × 2 × 4 B (40 KB × ctx_capacity); fits trivially.

    * **v0.74.2**: drafter phase 3 → Q8_0 mat-mat (the reviewer's
      catch). 80 mat-vec dispatches/outer-step → 5 mat-mat. Drafter
      total GPU at default ctx: **1268 ms → 150 ms (-88%)**. Phase 3
      went from 49% of drafter GPU to 12%. Same playbook as v0.73a.1
      GDN / v0.73c.1 attn (eligibility predicate, batched mat-mat
      Step A/C, F32 fall-through preserved). LOW correctness risk;
      reuses validated kernels.

    * **v0.74.3**: hot-loop sync cleanup. Two cheap follow-ons
      flagged by the reviewer:
      - `append_target_ctx_columns_now` (batched commit): N waits
        per outer step → 1 wait. At α_chain≈5.2 typical for code,
        ~6 sync points collapsed to 1.
      - Skip `restore_after_partial_accept` when `n_keep == N`
        (full accept). At ctx=363 with α=16/16, both outer steps
        skip restore entirely (restore_calls: 2 → 0).
      Marginal speedup (~0.3-1.7% decode-only) because both target
      sync overhead, not GPU work. Greedy equivalence preserved.

  ### Reviewer's external code review credit

  External review pointed at three things we'd missed:
  1. **Drafter phase 3 still per-row mat-vec** despite Q8_0 mat-mat
     having shipped — biggest single short-ctx win available, and
     the reviewer was correct that the kernels were already there.
     "You already built the kernels, now use them." Shipped as v0.74.2
     for ~25% default-ctx speedup in 2 hours of work.
  2. **`append_target_ctx_column_now` in hot decode loop** creates a
     command buffer per call. Cheap fix; shipped as v0.74.3 part 1.
  3. **Restore is unconditional even on full accept**. Cheap fix;
     shipped as v0.74.3 part 2.

  The reviewer also noted docs drift (rev 13 still marked current
  while v0.74 had shipped). This rev 14 closes that gap.

  ### What's left from the reviewer's leverage map

  Still-applicable items, ranked:

    * **`prefill_tokens` API** (no `lm_head`, no readback per token):
      biggest TTFT lever in the repo. v0.75.0 SHIPPED the skip-tail
      half (no `lm_head`, no readback for all-but-last prefill
      tokens) for ~5% TTFT at ctx ≥ 181. The packed multi-token
      half — turning per-token mat-vec into per-chunk-of-P mat-mat
      — is v0.75.1 (~2 days, projected 3-5× TTFT at ctx ≥ 1K).
    * **Stream DFlash ctx-cache during prefill**: predicated on
      prefill_tokens; eliminates the cold first outer step.
      ~half day on top of prefill_tokens.
    * **Save-checkpoint compute kernel** (collapse compute/blit
      alternation in packed_verify GDN): 5-10% short-ctx; still on
      the table after v0.74.3.
    * **Packed-N target attn-v4**: long-ctx lever (47% of verify
      GPU at ctx=16K); modest at short ctx. Defer until prefill
      lands and we measure long-ctx end-to-end with cached prefill.
    * **MTP-N=3 fallback recipe**: zero-download alternate decode
      path at projected 1.35-1.85× per the codex MTP investigation.
      ~50 LOC reusing DFlash packed-verify infra.

  ### Stale comments cleaned in v0.74.x

    * "v0.72.4 will lift to mat-mat" comments at the per-row mat-vec
      call sites — replaced with v0.74.2 batched dispatch.
    * "Drafter F32 dequant resident" comments — drafter has been
      native Q8_0 since v0.73b.1.

  ### `target_ctx_n` is append-only

  Rev 13 sketched a "ctx_n decreases on restore" concern that was
  defensive future-proofing, not an actual code path. Verified: the
  bench appends only committed accepted-prefix hidden columns
  (n_accepted+1 per outer step) before calling restore; restore
  doesn't touch `target_ctx_n`. The watermarks `ctx_h_ready_n` /
  `kv_ctx_ready_n` therefore don't need clamp logic in practice
  (kept for future-proofing only).

- **rev 13.** Long-ctx bench discovery: DFlash decode
  collapses with ctx because the drafter re-projects the entire
  cross-context every outer step. **The drafter ctx caching item
  filed in rev 9 as v0.77+ is now the dominant lever. Promoting it
  to v0.74.**

  ### Long-ctx end-to-end bench (post-v0.73c.1, prompt = repeated quicksort code)

    | n_prompt | DFlash decode | no-spec decode | DFlash decode/no-spec | n_outer |
    |---------:|--------------:|---------------:|----------------------:|--------:|
    |        5 |      2600 ms  |       1334 ms  |              0.515×   |       5 |
    |      181 |      1546 ms  |       1352 ms  |              0.874×   |       3 |
    |      256 |      2362 ms  |       1364 ms  |              0.578×   |       4 |
    |      363 |      1434 ms  |       1412 ms  |              0.984×   |       2 |
    |      727 |      2215 ms  |       1399 ms  |              0.631×   |       2 |
    |     1455 |      3751 ms  |       1409 ms  |              0.376×   |       2 |
    |     2055 |      5045 ms  |       1483 ms  |              0.294×   |       2 |
    |     8223 |     16904 ms  |       1507 ms  |              0.089×   |       2 |

  No-spec decode wall stays ~1500 ms regardless of ctx (correctly:
  fixed work per generated token; KV-read scaling is the only
  ctx-sensitive piece). DFlash decode wall grows ~5×–11× with ctx,
  blowing past no-spec at ctx ≥ 727. **At ctx = 8K, DFlash is 11×
  SLOWER than no-spec — completely broken for code-context use.**

  ### Drafter --profile breakdown (per-call avg ms, n_outer 2 or 3)

    | n_prompt | phase1_ctx_fc_norm | phase2_proj_norm_rope | phase3 | TOTAL_DRAFTER |
    |---------:|-------------------:|----------------------:|-------:|--------------:|
    |        5 |              21 ms |                  6 ms |  39 ms |       1268 ms |
    |      181 |             105 ms |                 17 ms |  16 ms |        813 ms |
    |      363 |             213 ms |                 31 ms |  16 ms |        905 ms |
    |      727 |             427 ms |                 60 ms |  17 ms |       1637 ms |
    |     1455 |             830 ms |                115 ms |  21 ms |       3023 ms |

  **Phase 1 doubles exactly when ctx doubles** — confirmed linear-in-ctx.
  Same for phase 2 (the per-layer K/V ctx projection + RoPE).
  Phase 3 (attn + ffn) flat in ctx (independent of cross-context size
  by design — drafter SWA-masks the noise → ctx attention so attn
  is per-noise-token, ctx_len doesn't grow it).

  ### Root cause (`metal_dflash.rs:2700-2748` + `:2850-2866` + `:2942-2949`)

  `draft_block` re-runs every outer step:
    1. Phase 1: `for c in 0..ctx_len` → mat-vec dflash_fc per
       column (n_target_features=25600 → hidden=5120) + per-column
       RMSNorm. **Re-projects the entire target_ctx_stacked through
       dflash_fc + hidden_norm**, even though the first `ctx_n` rows
       are unchanged from the previous outer step.
    2. Phase 2: per drafter layer (5 layers): `for c in 0..ctx_len`
       K/V projections, then per-c RoPE. **Re-projects + re-RoPEs
       the entire ctx through each layer's K/V weights** every step.

  Per outer step:
    Phase 1 redundant work: ctx_len * (mat-vec + RMSNorm)
                          ≈ ctx_len * (mat-vec(25600→5120) + 5120 ops)
    Phase 2 redundant work: 5 layers * ctx_len * (2 mat-vec(5120→1024)
                          + 2 RoPE(1024 elem))

  At ctx=1455:
    Phase 1: 1455 * ~0.6 ms = ~830 ms (matches measurement)
    Phase 2: 5 * 1455 * ~0.016 ms = ~115 ms (matches)

  ### Fix: drafter ctx caching (was v0.77+, NOW v0.74)

  Persist `ctx_h` and per-layer post-norm post-RoPE `K_ctx_cache[L]`
  / `V_ctx_cache[L]`. On each outer step, only project + norm + RoPE
  the **newly appended positions** (1..N from the previous outer
  step's accepted prefix + bonus). Append to the caches.

  Implementation sketch (~150 LOC + new session fields):
    - Add `K_ctx_cache: Vec<MetalTensor>`, `V_ctx_cache: Vec<MetalTensor>`,
      `ctx_h_cache: MetalTensor` to `MetalDFlashSession`. Sized for
      `target_ctx_capacity * kv_dim` / `target_ctx_capacity * h`.
    - Track `ctx_h_ready_n: usize` (= number of cached positions
      that are post-fc-norm-rope through ALL layers).
    - At top of `draft_block`: compute `delta = ctx_n - ctx_h_ready_n`.
      Run phase 1 + phase 2 ONLY on positions `[ctx_h_ready_n,
      ctx_n)`, not on the entire `ctx_h`. Set
      `ctx_h_ready_n = ctx_n` after.
    - Phase 3 attn reads cached K_ctx/V_ctx of length ctx_len; no
      change to its core algorithm.
    - Restore semantics: on partial accept, ctx_n DECREASES (rolls
      back to processed_pos+1+n_accepted). Cache positions beyond
      that aren't valid; re-project on next outer step. Simplest
      correctness rule: `ctx_h_ready_n = min(ctx_h_ready_n, ctx_n)`
      after every restore.

  ### Projected impact

  At ctx=1455 (per-call savings):
    phase 1 redundant: ~830 ms per outer step → ~5 ms (only the new
                       1-N positions get projected)
    phase 2 redundant: ~115 ms per outer step → ~5 ms
    Total savings: ~935 ms per outer step
    × 2 outer steps in this bench = ~1870 ms total decode wall
    Decode wall: 3751 ms → ~1880 ms. Speedup vs no-spec: 1409/1880 = 0.749×.

  At ctx=8223 (extrapolated):
    phase 1: ~4700 ms → ~5 ms savings/step ⇒ ~9300 ms total saved
    phase 2: ~650 ms → ~10 ms ⇒ ~1280 ms total saved
    Decode wall: 16904 ms → ~6300 ms. Speedup vs no-spec: 1507/6300 = 0.239×.
    Still bad at 8K because verify-side per-token attn/GDN
    presumably ALSO scales with ctx (KV read BW), but the
    cliff is FAR less catastrophic.

  ### v0.74 plan (was Q8_0 drafter + N=16 mat-mat, now SHIPPED early as v0.73b)

  ~~Q8_0 drafter (v0.74) shipped early as v0.73b.~~

  v0.74 = drafter ctx caching. Sequence:
    * v0.74.0: add `ctx_h_cache` + `ctx_h_ready_n` field, refactor
      Phase 1 to only project the delta. Cosine gate vs current
      behavior at ctx=64 (where redundant re-project IS what
      happens).
    * v0.74.1: add `K_ctx_cache[L]` + `V_ctx_cache[L]`, refactor
      Phase 2 same way. cos gate.
    * v0.74.2: restore-after-partial-accept semantics. Test:
      a sequence that triggers partial accept must produce
      identical drafter output regardless of cache state.
    * v0.74.3: end-to-end bench at ctx ∈ {256, 1024, 4096, 8192}.
      Speedup must improve monotonically.

  ### Other notes

  - Long-ctx bench used a synthetic-repeated prompt (quicksort code
    repeated). Real code-context prompts should also exhibit this
    scaling because phase 1/2 are O(ctx) regardless of content.
  - Prefill is also LINEAR in ctx (~10 ms/token), which is the
    expected baseline. The dramatic prefill totals (369 s for 8K)
    are real but proportional. Not a separate bug.
  - greedy equivalence STILL PASSES at all measured prompt lengths.
    The re-projection isn't producing incorrect output — it's just
    wasteful redundant work.

- **rev 12.** v0.73 stretch consolidation. v0.73a.0
  through v0.73c.1 SHIPPED. v0.73c.2 (fused-SwiGLU FFN) attempted
  + bounded as a NEGATIVE RESULT. Codex MTP investigation
  (subagent) recorded.

  ### Sweep summary (commits cae89ce → 6c15237)

    | commit       | speedup | what                                           |
    |--------------|---------|------------------------------------------------|
    | v0.72.4 base | 0.452×  | rev-9 plan checkpoint                          |
    | v0.73a.0     | —       | Q5_K mat-mat lift + A-lite gate (3.59×)        |
    | v0.73a.1     | 0.548×  | GDN projection batching                        |
    | v0.73a.2     | —       | profile pivots v0.73b → Q8_0 drafter           |
    | v0.73b.0     | —       | Q8_0 mat-vec + mat-mat lift (A-lite 7.74×)     |
    | v0.73b.1     | 0.746×  | native Q8_0 drafter                            |
    | v0.73c.1     | 0.805×  | attn projection batching                       |
    | **v0.73c.2** | —       | **fused-SwiGLU FFN: NEGATIVE RESULT**          |

  Cumulative since v0.72.4 baseline: 0.452× → **0.805× = +78%**.
  Closing in on break-even (1.0×) on production 27B Q4_K_M code prompt.

  ### v0.73c.2 negative result — fused SwiGLU FFN

  Codex's #2 ranked v0.73c candidate after the post-v0.73c.1 profile
  showed FFN at 49.3% of verify GPU at ctx=1024. Wrote
  `kernel_ffn_fused_swiglu_q4_K_mm_n16_f32` (~280 LOC): one fused
  dispatch per FFN layer instead of (gate_mm + up_mm + silu_mul) ×
  3 dispatches. Bit-exact correctness vs unfused (cos = 1.000000,
  max|Δ| = 0).

  **A-lite go/no-go bench at production shape FAILED:**
    fused:    7.0 ms / 64 layers GPU
    unfused:  7.5 ms / 64 layers GPU
    ratio: 0.937 — only 1.07× speedup
  Codex threshold was ≤ 0.7 (≥ 30% speedup). Saved ~0.5 ms / call ×
  5 calls = ~2.5 ms total decode wall. Below noise floor.

  Codex's earlier estimate (5–15 ms/call honest, 10–25 ms/call
  optimistic) was **20–50× too high**. Why:
    1. Both paths run in ONE command buffer; Metal pipelines the
       dispatches well; codex's 3–6 ms "dispatch count savings"
       projection didn't account for pipelining.
    2. Intermediate I/O elimination (gate_pack + up_pack
       materialization) was correctly bounded at ~0.6 ms — that's
       essentially what we measured.
    3. The big BW cost (W_gate + W_up reads, ~94 MB / layer × 64 =
       ~6 GB / call) is unchanged — fusion can't avoid the weight
       reads.
    4. The unfused path is ALREADY scheduler-friendly inside one
       cmd buffer.

  **Action taken**: kernel labeled experimental in source comments,
  `#[ignore]` A-lite bench preserved as re-runnable "regime still
  capped?" probe (relaxed from panic to log). Kernel + correctness
  test stay in tree as institutional memory; not plumbed into the
  layer-major path.

  **Reusable lesson reinforced**: profile noise is real (2× variance
  run-to-run on the per-phase profiler), and codex-bounded
  estimates can still be 20–50× too high when the comparison's
  baseline is already well-pipelined. **A-lite isolated benches
  before plumbing infrastructure changes saved an estimated 1+ day**
  of plumbing → measuring → reverting that we'd otherwise burn.

  ### v0.73c.1 — attn projection batching (SHIPPED)

  Mirrors v0.73a.1 GDN restructure for the 16 attn layers. Production
  27B Q4_K_M attn projections are all Q4_K (Q gated, K, V, output);
  attn_q is `[5120, 12288]` (gated 2× q_dim), attn_k/v are
  `[5120, 1024]`, attn_output is `[6144, 5120]`. Eligibility
  predicate (`attn_mat_mat_eligible`) checks all 4 ∈ {Q4_K, Q5_K,
  Q6_K}; F32 oracle (0.8B) and any mixed-dtype case fall through
  to unchanged per-token `encode_attn`.

  Eligible path:
    Step A (1 enc/layer): mat_mat Q (gated) → split → mat_mat K,V
                          → batched Q-norm + K-norm
    Per-token loop (N=16): RoPE Q + RoPE K + KV-scatter F32→F16 +
                          attn-v4 (per-token: KV append + softmax
                          are inherently sequential)
    Step C (1 enc/layer): sigmoid(gate_pack) → mul on attn_o_pack
                          (flat N*q_dim) → mat_mat o_proj batched

  Net dispatch reduction: 16 per-token × 10 dispatches = 160 →
  74 (1 head + 64 inner + 1 tail). Reuses encode_split_q_gate_f32
  (per-head layout repeats across N rows — works as-is with
  n_heads = N × n_q_per_token).

  End-to-end: 0.746× → 0.805× (+8%); 27B layer-major-vs-token-major
  speedup 2.36× → 2.45× (essentially clears codex's 2.5× tripwire
  from H5.3b.6).

  ### Profile data (post-v0.73c.1, ctx sweep)

    | phase                    | ctx=1024 | ctx=4096 | ctx=16384 |
    |--------------------------|---------:|---------:|----------:|
    | ffn_plus_residual2       |    49.3% |    44.3% |     28.8% |
    | attn (Step A + per-tok + Step C) | 13.5% | 22.1% |   47.0% |
    | gdn aggregate            |    34.7% |    31.9% |     22.7% |

  Crossover at ctx 4K-16K: attn becomes dominant at long-ctx.
  Profile noise IS REAL (FFN measurements vary 2× run-to-run on
  the per-phase profiler that uses per-encoder commit/wait). Trust
  RELATIVE attribution; don't trust ABSOLUTE ms numbers from this
  profiler.

  ### Codex MTP investigation (informs roadmap, not v0.73c)

  Subagent investigated llama.cpp PR #22673 (MTP for Qwen 3.6 27B,
  draft as of May 6 2026) after a public claim of 2.5× speedup.

  Findings:
  - The 2.5× is the upper tail of a 1.35–1.85× distribution
    (M3 Ultra dense: 1.35×; RTX 5090 Q4_0: 1.66×; 3090 Q6_K: 1.85×).
    The 2.5× claim bundles MTP-N=3 × KV-Q4_0 × long-ctx × favorable
    prompt — three orthogonal wins stacked.
  - Our H4 was correct for what we shipped (N=1 lazy verify,
    mathematically capped at 0.91×). The PR ships an N=3 AR chain
    feeding `t_mtp_out` from step k-1 as the hidden for step k —
    a recurrence variant we never built.
  - **DFlash structurally beats MTP-N=3:** bigger drafter (1.7B
    params vs single-layer head), bigger N (16 vs 3), bigger
    amortization (17 query rows vs 4). DFlash ceiling 1.5–2.5×;
    MTP-N=3 ceiling 1.5–1.7×.
  - **Stay the H5 course.** After H5 lands break-even, MTP-N=3
    bolts onto our DFlash packed-verify infra in ~50 LOC for a
    zero-download fallback recipe. ~1–2 days work; reuses checkpoint
    machinery.

  ### v0.73c.3 / v0.74+ candidates (ranked post-v0.73c.2)

  1. **v0.73c.3 = packed-N attn-v4** (the original v0.75 pulled
     forward). Long-ctx lever: 47% of verify GPU at ctx=16K. Codex:
     "structurally simpler than fused-SwiGLU" — adds N axis to
     existing attn-v4 accumulators rather than a new fused kernel
     design. Short-ctx modest (~20-30 ms/call); long-ctx massive
     (~100-150 ms/call at 16K+). Multi-partition long-ctx tests
     mandatory (codex H5.3 era caution).
  2. **KV-Q4_0 / KV-Q8_0** — flagged in MTP investigation as a
     10–30% long-ctx independent win. Bigger scope (touches
     attn-v4 KV read path + new dtype across all 16 attn layers).
     Defer until after packed-N attn-v4 ships.
  3. **MTP-N=3 fallback recipe** — reuses DFlash packed-verify
     infra. Optional ship; helps users who can't/won't download
     the DFlash drafter GGUF. Defer until H5 hits ≥ 1.0×.
  4. **N-step GDN tail kernel (was rev-9 v0.73b)** — ceiling 57.6 ms
     / call at v0.73a.2 profile time; even smaller now post-v0.73a.1.
     Stays deferred indefinitely.

- **rev 11.** v0.73a.1 SHIPPED (0.452× → 0.548×, +21%
  speedup). Then v0.73a.2 surgical profile + drafter re-profile
  invalidated the rev-9/rev-10 locked v0.73b (N-step GDN tail
  kernel). Pivoting v0.73b to Q8_0 native drafter (was v0.74).

  ### v0.73a.1 results (production 27B Q4_K_M, code prompt, 32 tokens)

    | metric                            | v0.72.4 | v0.73a.1 | Δ        |
    |-----------------------------------|---------|----------|----------|
    | Speedup vs no-spec (total wall)   | 0.452×  | 0.548×   | +21%     |
    | Decode wall                       | 3192 ms | 2600 ms  | -592 ms  |
    | Layer-major packed_verify wall    | ~339 ms | ~287 ms  | -52 ms   |
    | Layer-major-vs-token-major        | 1.58×   | 2.36×    | (close to codex 2.5× tripwire) |
    | Greedy equivalence                | PASS    | PASS     | unchanged|
    | α_chain                           | 5.2     | 5.2      | unchanged|

  Bit-perfect cos=1.000000 across all 16 verify rows on the
  layer-major-vs-token-major 27B test. argmax 16/16. v0.73a.1
  landed in codex's predicted 0.50–0.55× range.

  ### v0.73a.2 surgical profile (NEW post-v0.73a.1, ctx=1024 N=16)

  Per-encoder commit/wait timing of one packed_verify_layer_major
  call. Reflects the ACTUAL post-v0.73a.1 phase mix (rev-9's profile
  was for the OLD per-token-mat-vec geometry):

    | phase                           | ms      | %     |
    |---------------------------------|---------|-------|
    | **ffn_plus_residual2**          | **128.37** | **41.0%** |
    | attn_per_token                  |  69.05  | 22.0% |
    | gdn_per_token_compute           |  49.26  | 15.7% |
    | gdn_step_a_proj_in_qkv_z        |  31.42  | 10.0% |
    | gdn_step_c_proj_out             |  19.48  |  6.2% |
    | gdn_per_token_blit              |   8.31  |  2.7% |
    | tail (norm+lm_head+argmax)      |   4.92  |  1.6% |
    | residual1 / norms / capture     |   ~2 ms |  ~1%  |
    | TOTAL_PHASE_GPU                 | 313.17  |       |
    | TOTAL_WALL (profiler artifact)  | 804.30  |       |
    | wall - phase_gpu (artifact)     | 491.13  | 61.1% |

  Critical takeaways:
  1. **FFN is now the largest GPU phase** (41%). Already mat-mat-
     batched (Q4_K gate/up + Q6_K down). Lift fused-SwiGLU pattern
     from single-token path to layer-major could save 10-25ms
     (codex haircut from my 30-50ms estimate; FFN is weight-traffic-
     bound, fusion saves intermediate buffer materialization but
     not weight reads).
  2. **GDN per-token compute + blit = 57.57 ms TOTAL.** This is
     the v0.73b ceiling. Codex's earlier 70 ms threshold for "ship
     N-step GDN tail kernel" is NOT MET. Even removing all of it
     would land at the upper end of codex's 0.58–0.66× estimate.
  3. **Wall - phase_gpu = 491 ms is a profiler artifact** (per-
     phase commit/wait). Production layer-major uses one cmd
     buffer; real overhead is small (production wall ≤ sum-of-GPU).
     ICB / encoder amortization (v0.76+) requires production-shaped
     A/B to measure honestly, not this profiler.
  4. **Attention at 22% is short-ctx**; long-ctx (16K, 64K) it
     scales linearly while GDN flat. Pull packed-N attn forward
     ONLY if long-ctx becomes the product target.

  ### Drafter re-profile (v0.72.3 profiler, post-v0.73a.1)

  Drafter unchanged since v0.72.1; ran the existing profiler again to
  get the post-v0.73a.1 wall-share:

    | metric                            | value      |
    |-----------------------------------|------------|
    | TOTAL_DRAFTER_GPU                 | 1268.54 ms |
    | TOTAL_DECODE_WALL                 | 2593 ms    |
    | **drafter share of wall**         | **48.9%**  |

  Drafter is now the **single largest phase** (was 40% pre-v0.73a.1;
  share grew because verify shrank). Q8_0 native drafter (was v0.74)
  is now clearly the highest-EV next move. Q8_0 mat-mat lift would
  shave estimated 100–150 ms per outer step × 5 = 500–750 ms total
  decode wall; cumulative speedup projection 0.548× → ~0.73–0.77×.

  ### v0.73b PIVOT — Q8_0 native drafter (was v0.74)

  Replaces the rev-9/rev-10 N-step GDN tail kernel (the GDN tail
  ceiling collapsed below the risk-justification threshold).

  * **v0.73b.0** — Lift Q8_0 mat-mat (kernel + host wrapper +
    NR1=16 fast path + isolated per-kernel cosine gate +
    `encode_mat_mat_dispatch` routing). Mirrors v0.63 (Q4_K),
    v0.67 (Q6_K), v0.73a.0 (Q5_K) lift playbook. Q8_0 has no
    high-bit path or super-block scale packing — structurally
    SIMPLER than Q5_K. Honest estimate: ~half day. Gate: per-row
    cos ≥ 0.999 vs N successive Q8_0 mat-vec on real Q8_0 weight,
    layout sanity check; isolated bench replay must beat 16 mat-vec
    by a large margin (A-lite go/no-go pattern).

  * **v0.73b.1** — Switch drafter loader from F32-dequant to
    native Q8_0 (5 projection types per drafter layer × 5 layers
    = 25 native Q8_0 projections). Wire Q8_0 into drafter's
    `encode_mat_vec_dispatch` and `encode_mat_mat_dispatch` paths.
    Greedy equivalence + H5.1.5 cosine gate must still pass.
    Drafter weight footprint 7.4 GB → 1.85 GB (also unblocks
    long-prompt memory headroom).

  * **v0.73b.2** — N=16 mat-mat fast path for drafter
    `draft_block` tail (lm_head batched lift already shipped in
    v0.72.0; Q8_0 changes the dispatch but not the orchestration).

  Estimated cumulative: 0.548× → 0.73–0.77×.

  ### v0.73b candidates explicitly NOT chosen (and why)

  * **N-step GDN tail kernel (was v0.73b through rev 10)**: 57.57 ms
    ceiling; codex flagged risk asymmetry. Defer indefinitely; may
    revisit at v0.76+ if encoder amortization surfaces it.
  * **Layer-major fused SwiGLU FFN (NEW idea)**: 10–25 ms ceiling
    (FFN is weight-traffic-bound; fusion saves materialization of
    `gate_pack` / `up_pack` intermediates, not weight reads). Real
    win but smaller than Q8_0; revisit as v0.73c after Q8_0 ships.
  * **Packed-N attention v4 pulled forward (was v0.75)**: 22% of
    verify GPU at ctx=1024, but linear-in-ctx. Defer to v0.75 unless
    long-ctx becomes the product target.

  ### Plan-doc honesty note

  This is the THIRD "wrong-phase" pivot in the v0.7x stretch:
  - v0.71 surfaced drafter-as-bottleneck (we'd been optimizing
    target verify)
  - v0.72.3 reframed verify > drafter (we'd been planning more
    drafter work)
  - v0.73a.2 reframed FFN > GDN tail and drafter > everything
    (we'd locked v0.73b on a stale phase attribution)

  Lesson reinforced: **profile before kernel work** when the
  surface area shifts substantially. v0.73a.2 cost 1-2 hours; saved
  an estimated 1+ day on a v0.73b-as-N-step-GDN-tail kernel that
  would have landed below its codex-projected speedup ceiling and
  carried higher correctness risk.

- **rev 10.** v0.73a dtype recon + scope split. Before
  touching control flow, dumped actual GGUF dtypes for production
  27B-Q4_K_M GDN blocks. Result invalidated rev-9's "all 5 GDN
  projections are Q4_K-batchable" assumption.

  ### Real GDN projection dtype layout (per-layer, 27B Q4_K_M)

    | GDN projection  | GGUF dtype | Shape           | Per-layer bytes |
    |-----------------|------------|-----------------|-----------------|
    | in_proj_qkv     | **Q6_K**   | [5120, 10240]   | ~42 MB          |
    | in_proj_z       | **Q4_K**   | [5120, 6144]    | ~22 MB          |
    | beta_proj       | **F32**    | [5120, 48]      | 0.94 MB         |
    | alpha_proj      | **F32**    | [5120, 48]      | 0.94 MB         |
    | out_proj        | **Q5_K**   | [6144, 5120]    | ~30 MB          |

  Critical implications for v0.73a as originally scoped:
    * `encode_mat_mat_dispatch` only supports Q4_K and Q6_K.
    * **Q5_K mat-mat is NOT lifted** in this codebase (only Q5_K
      mat-vec exists, in `encode_mat_vec_dispatch`).
    * out_proj (Q5_K, ~30 MB/layer × 16 tokens × 48 layers = ~23 GB
      redundant per outer step) is the second-largest projection.
      Without Q5_K mat-mat, v0.73a-as-scoped would batch only the
      front-end and leave out_proj per-token, capturing maybe half
      the codex impact estimate.
    * beta/alpha at F32 [5120, 48] are tiny; per-token mat-vec
      dispatch overhead probably exceeds mat-mat BW savings. Stay
      per-token.

  ### v0.73a SPLIT (codex consult: A-lite go/no-go)

  Codex recommended option A with two clean commits and a bench
  go/no-go between them:

  * **v0.73a.0** — Lift Q5_K mat-mat (kernel + host wrapper +
    NR1=16 fast path + isolated per-kernel cosine gate +
    `encode_mat_mat_dispatch` routing). Mirrors v0.63 (Q4_K) and
    v0.67 (Q6_K) playbook. Q5_K dequant has high-bit-path complexity
    not present in Q4_K/Q6_K; honest estimate **half day if it
    compiles cleanly, full day if dequant/layout bugs surface**.
    Gate: per-row cos ≥ 0.999 vs N successive Q5_K mat-vec on real
    `blk.*.ssm_out.weight` at N ∈ {1, 16, 32}; layout sanity check;
    isolated bench replay of `ssm_out` weight × 48 layers × N=16
    must beat 16 sequential mat-vecs by a large margin.
    **HARD GO/NO-GO**: if Q5_K mat-mat doesn't beat 16 mat-vecs
    materially on the isolated bench, stop and reassess; do NOT
    proceed to v0.73a.1 (because the GDN restructure would land
    smaller than estimated and we'd be deceiving ourselves about
    the leverage point).

  * **v0.73a.1** — GDN projection batching: in_proj_qkv (Q6_K) +
    in_proj_z (Q4_K) + out_proj (Q5_K) batched as mat-mat across
    N=16 in the layer-major path. beta/alpha stay per-token F32
    mat-vec (eligibility predicate `gdn_mat_mat_eligible(g)` returns
    true iff in_proj_qkv ∈ {Q4_K, Q5_K, Q6_K} AND in_proj_z ∈ same
    AND out_proj ∈ same; F32 falls through to per-token
    `encode_gdn`). Codex's restructure refinements baked in:
      - Step A (front-end batched projections) folded into the
        existing pre-mixer-norm encoder — no added encoder transitions.
      - Per-token loop uses `view_subrange` (zero-copy) instead of
        `copy_offset` for staging row N of pack buffers into the
        recurrence kernels.
      - Step C (back-end out_proj mat-mat) folded into the existing
        residual-#1 encoder.
    Gate: focused unit test (single-layer batched-GDN vs per-token
    encode_gdn cos ≥ 0.999); existing 27B layer-major-vs-token-major
    27B test must STILL pass cos ≥ 0.999 (loosened from the rev-9
    "expect cos = 1.0" — Q4_K/Q5_K/Q6_K mat-mat half-stages
    activations and the recurrence amplifies tiny projection
    deltas, per codex flag); greedy equivalence vs DFlash=off must
    pass over multiple prompts.

  ### v0.73a.0 / v0.73a.1 leftover scratch already landed (pre-recon)

  Before the dtype recon I had already added some scratch and
  kernel infrastructure assuming the all-Q4_K case. Kept where
  still useful, removed where not:

    * `MetalDFlashLayerMajorScratch::gdn_qkv_pack [N, conv_dim]`,
      `gdn_z_pack [N, v_dim]`, `gdn_normed_pack [N, v_dim]` — KEPT.
      Required by v0.73a.1 for the three batched projections.
    * `gdn_b_pack`, `gdn_beta_pack`, `gdn_a_pack`, `gdn_alpha_pack`
      — REMOVED. Beta/alpha stay per-token F32; no need for pack
      buffers.
    * `kernel_gdn_alpha_chain_batched_f32` + host wrapper +
      bit-exact unit test — KEPT as standalone infrastructure
      (small, isolated, useful if a future drafter quantizes
      alpha_proj). NOT used by v0.73a.1.
    * Q4_K partial-M test for `n_out=48` shape — REMOVED. Beta/alpha
      stay per-token F32; the partial-M Q4_K path is never exercised
      at this shape in production.

  ### Why split into two commits despite the "less commit noise"
  preference

  v0.73a.0 (kernel infrastructure) and v0.73a.1 (orchestration) are
  genuinely separate units of work with different correctness
  surfaces. v0.73a.0 has its own gate (the isolated bench), and the
  GO/NO-GO decision happens AT v0.73a.0's gate. If v0.73a.0 fails
  its bench, we stop and reassess BEFORE shipping orchestration
  work that depends on it. Combined commit would entangle two
  independent failure modes and would force a partial revert if
  v0.73a.1 surfaced a kernel bug.

- **rev 9.** v0.72.x drafter Metal-ization SHIPPED;
  v0.73 strategy pivoted (codex unbiased recon) from Q8_0 first to
  target packed-GDN first.

  ### v0.72.x summary

    * **v0.72.0** — drafter batched-tail port (lm_head Q6_K mat-mat
      + GPU argmax). Marginal end-to-end (tail wasn't dominant) but
      architectural posture correct.
    * **v0.72.1** — drafter Metal phase 3 (custom small-N fused
      SWA-masked attention + Metal Q/K/V/O proj + Metal SwiGLU FFN).
      End-to-end: 0.024× → 0.452× = **18.5× speedup in one commit**.
      Greedy equivalence vs DFlash=off PASSES on real 27B-Q4_K_M
      code prompt (32 tokens identical). H5.1.5 cosine vs CPU
      oracle PASSES bit-correct (cos = 1.000000 across all 16
      noise positions).
    * **v0.72.2** — codex code-review fixes for v0.72.1: real
      `q_idx >= n_rows` bound (was bogus `n_q_heads * 16`); reject
      head_dim > 256 host-side (kernel registers sized for ≤ 256);
      port CPU oracle's permissive full-attn semantics (full-attn
      ⇒ no causal restriction over ctx); SWA subtraction reordered
      to short-circuit on causal. NEW lib tests:
      `dflash_attn_matches_cpu_oracle_under_mask_regimes` (6 mask
      regimes including ctx_len=0, full-attn, SWA boundary,
      gapped pos_ctx) and `dflash_attn_rejects_head_dim_over_256`.
    * **v0.72.3** — drafter phase profiler (`qwen-bench dflash
      --profile`). Lightweight per-phase GPU timers via
      `MetalDFlashSession::maybe_record` + `take_phase_timings`.
      Aggregates same-name phases across all outer steps + layers.

  ### Profile data (v0.72.3, 27B-Q4_K_M, code prompt, ctx≈1-4K)

    | phase                              | sum (ms) | %   | n  | avg/iter |
    |------------------------------------|----------|-----|----|---------:|
    | phase3_attn_oproj_ffn_residuals    |   986.05 | 78% | 25 | 39.44 ms |
    | phase2_proj_norm_rope              |   148.98 | 12% | 25 |  5.96 ms |
    | phase1_ctx_fc_norm                 |   107.17 |  9% |  5 | 21.43 ms |
    | phase4_tail_norm_lmhead_argmax     |    23.85 |  2% |  5 |  4.77 ms |
    | phase2_embed                       |     0.03 |  0% |  5 | trivial  |
    | TOTAL_DRAFTER_GPU                  |  1266.07 |     |    |          |
    | TOTAL_DECODE_WALL                  |  3191.89 |     |    |          |

  Bench: 0.452× speedup (DFlash 9.4 t/s vs no-spec 21 t/s).
  Drafter is now **40% of wall**; packed_verify is the remaining 60%.

  ### v0.73 STRATEGIC PIVOT (codex unbiased recon, end of session)

  After v0.72.3 shipped the profile data, I delegated next-move
  selection to codex without leading framing. Codex returned a
  reframe I had missed:

  > "Treat packed verify as a real N-token target forward, NOT a
  > single-token forward loop with batched FFN islands."

  I had been thinking of layer-major packed_verify as "done" because
  H5.3b batched FFN/projections/lm_head. **It's not done — GDN and
  attn are STILL single-token-engine called N times** under the hood.
  `encode_gdn` is invoked 48 layers × 16 tokens = 768 times per
  outer step, each running the full single-token GDN mixer with
  per-call dispatch overhead AND per-call weight reads on the GDN
  projections (in_proj_qkv, in_proj_z, beta, alpha, out_proj).
  Same architectural debt for attn (16 attn × 16 tokens = 256
  per-token attn-v4 calls). Most of those projections are
  per-token-INDEPENDENT linear ops on h_pack rows that could batch
  as mat-mat — only the conv + recurrence + RMSNormGated must stay
  sequential per token.

  The profile labels "gdn_mixer_compute = 229ms / 50% of verify"
  as a single bucket. Codex's read: most of that 229ms is
  removable weight-reread + dispatch overhead from the per-token
  single-token-engine pattern, NOT the recurrence proper.

  ### v0.73+ locked sequence (codex stake)

  * **v0.73a** — Batch GDN projections via existing mat-mat
    dispatch. Replace per-token mat-vec calls (in_proj_qkv,
    in_proj_z, beta_proj, alpha_proj, out_proj) with one mat-mat
    per projection per layer using the H5.3b.5.5 NR1=16 fast-path
    Q4_K kernel. Preserve per-token recurrence (conv + gdn_step +
    rmsnorm_gated) loop unchanged. Estimated impact: verify ~50-100ms
    reduction at ctx=1-4K. LOW risk — reuses validated Q4_K mat-mat
    kernels; per-token recurrence semantics unchanged. Codex
    estimate: 0.45× → 0.50-0.55× speedup.
  * **v0.73b** — Internalized N-step GDN recurrence kernel: single
    kernel does (decay, sk, delta, update, output) for all N
    tokens internally with checkpoint write inline per step.
    Eliminates 768 sequential gdn_step dispatches + the 1536
    cross-encoder transitions for per-token blits (becomes one
    blit per layer for the final ckpt slot, or compute-side
    write inline). Estimated additional verify ~50-100ms.
    HIGHER risk — bit-exactness vs single-step recurrence is
    the gate; codex's "checkpoint indexing easy to get subtly
    wrong" warning applies. Codex estimate (cumulative with
    v0.73a): 0.55× → 0.58-0.66× speedup.
  * **v0.74** — Q8_0 native drafter + N=16 mat-mat fast path.
    Was previously v0.72.4 in the predeclared sequence; codex
    correctly demoted because verify is the larger phase.
    Drafter weight footprint 7.4 GB → 1.85 GB (also unblocks
    long-prompt memory headroom). Drafter ~250ms → ~100-150ms
    estimated. Cumulative speedup: ~0.7-0.8×.
  * **v0.75** — Packed-N target attention v4 (share K/V loads
    across the N=16 verify queries per attn layer). Long-ctx
    dominant lever (linear-in-ctx KV reads). Modest at ctx ≤ 4K
    (~30-50ms saved); major at ctx ≥ 16K (~100-200ms saved).
    Codex Q3 caution still applies: must exercise multi-partition
    long-ctx tests (codex's failure-mode call from H5.3 era).
    Cumulative speedup at code-prompt-class workloads: ~0.85-1.0×
    (crossover with no-spec). At 16K+: substantially better.
  * **v0.76+** — ICB / encoder-overhead amortization. Codex's
    risk-adjusted prediction: "as GPU work shrinks, host /
    encoder structure becomes visible. ICB looks premature now,
    may abruptly become the ceiling after v0.73 or v0.74."
    Watch the wall vs sum-of-phases gap; ICB fires when that
    gap grows beyond ~20%.

  ### Drafter ctx caching (codex's "next architectural debt")

  Phase 1 ctx_fc + hidden_norm at 21ms/outer step is small at
  current ctx_len = 5-37 but GROWS LINEARLY with prompt length.
  At ctx_len = 4K: ~1700ms/outer step from re-projection alone.
  Solution: persistent `ctx_h` cache (and per-layer K_ctx /
  V_ctx caches inside drafter attention) — append-only on each
  outer step. Defer to v0.77+ once long-prompt workloads are
  in scope; not the current bottleneck.

  ### Compounding realism (codex)

  Conservative cumulative trajectory:
  v0.72.x: 0.45×  (current)
  v0.73:  ~0.60-0.66×
  v0.74:  ~0.70-0.80×
  v0.75:  ~0.85-1.0× (code prompts; better at long ctx)
  v0.76+: ~1.0-1.5× depending on what becomes the next bottleneck

  Plan §1.4's 1.5-2× target is reachable but requires the FULL
  v0.73-v0.76 stack to land. Each individual chunk on its own
  is sub-2×.

  ### Things explicitly NOT in v0.73+

  * **Q8_0 mat-vec/mat-mat lift** (was v0.72.4): demoted to v0.74
    because verify > drafter on the current bench. Still mandatory
    for v0.74; just not the next move.
  * **Save-checkpoint compute kernel**: profile says blits are
    1-2% of verify. Defer until v0.73b absorbs the cross-encoder
    transitions.
  * **KV-Q8 cache compression**: long-context lever; comes after
    packed-N attn-v4 amortizes the algorithmic K/V reads.
  * **Apple tensor-API mat-mat (H5.3b.7)**: still post-ship.
  * **Acceptance-aware dynamic N policy**: algorithmic; ships
    when bench infrastructure is ready to A/B different N at
    different α regimes (post-v0.74).
  * **Compute-side concat/scatter elimination in drafter phase 3**:
    cleanup; deferred.

  ### Codex risk-adjusted prediction baked into roadmap

  > "The likely 'wrong phase' risk after v0.72.4 is host/encoder
  > structure and synchronization, not math kernels."

  We've now had two of these "we've been chasing the wrong phase"
  pivots in three sessions:
    * v0.71 surfaced drafter (we'd been optimizing target)
    * v0.72.3 reframed verify as bigger than drafter (we'd been
      planning more drafter work)

  v0.73+ predicted pivot: **once GPU work shrinks below ~1.5s/step,
  command-buffer + host-encode + sync overhead becomes the
  dominant cost.** ICB / encode-once amortization is the v0.76+
  catcher. Surface area to watch in v0.73-v0.75 benches:
  TOTAL_DECODE_WALL vs TOTAL_DRAFTER_GPU + TOTAL_VERIFY_GPU gap.
  If that gap stays > 30%, host orchestration is the cap.

- **rev 8.** H5.5 end-to-end DFlash decode SHIPPED
  (v0.71). Greedy equivalence with no-spec PASSES on 27B Q4_K_M
  code prompt. Acceptance excellent (alpha_pos1=1.000,
  alpha_chain=5.2 drafts/step, mean_emitted=6.2 tokens/step).

  But end-to-end speedup is 0.024x — DRAFTER consumes ~12.6 s per
  outer step. Surfaced via the bench as the actual bottleneck;
  ALL the H5.3b kernel work was attacking the wrong phase.

  ### Root cause

  draft_block has been the H5.1 "HYBRID DEBUG SCAFFOLD" the entire
  time: Metal projections + RoPE + norms but CPU attention + CPU
  SwiGLU FFN. CPU phase 3 of draft_block runs full attention with
  SWA mask + 3 mat-vecs through F=17408-intermediate-dim FFN per
  layer x 5 layers per outer step. Estimated 99% of wall.

  The H5.1 commit explicitly called this scaffold; the plan said
  "deferred to H5.3" but H5.3 ended up being the TARGET'S packed
  verify path. We never came back for the drafter.

  ### v0.72 — Drafter Metal-ization (codex partner session, v0.72-design)

  Codex flipped my proposed sequence on its head:
  "Q8_0 first optimizes a path you are about to delete, and
  native Q8_0 actively conflicts with the current CPU fallback."

  Locked sequence:

    1. **v0.72.0** — port batched-tail pattern from packed_verify
       into draft_block. Currently draft_block's tail still loops
       per row with mat-vec + CPU argmax. Replace with batched
       RMSNorm + mat_mat lm_head (Q6_K, shared with target) +
       batched argmax. Free win, isolates plumbing change.
    2. **v0.72.1** — Metal SwiGLU FFN for N-row activation on F32
       weights. Drafter weights stay F32-dequantized at load for
       now (matches CPU fallback contract). Reuses existing
       encode_mat_vec_f32 for individual mat-vecs; just batch N
       rows + remove CPU loop. Validate vs CPU oracle.
    3. **v0.72.2** — custom small-N fused SWA-masked attention
       kernel. Codex Q2: NOT multi-pass (would require new batched
       softmax + new GQA-aware F32 mat-mat for QK and V agg).
       Real fused kernel: ONE threadgroup per (q_idx, q_head),
       streaming softmax over ctx + noise, NO materialized scores.
       Same simdgroup-matrix tile pattern from H5.3b mat-mat plus
       the streaming-softmax pattern from attn_v4.
    4. **v0.72.3** — plumb attention + FFN + tail into draft_block;
       remove CPU phase 3 entirely. End-to-end bench.
    5. **v0.72.4** — ONLY after the above ships green: lift Q8_0
       mat-vec + mat-mat, switch drafter loader to native Q8_0,
       enable Q8_0 routing in mat_vec/mat_mat dispatch helpers.
       N=16 fast path lifted from H5.3b.5.5 pattern.
    6. **v0.72.5+** — ctx_h caching across outer steps + K_ctx /
       V_ctx caching. Asymptotic lever for long ctx; not the
       current bottleneck. Design attention buffers in v0.72.2
       so `[layer, ctx, kv]` slot-caching can be added later
       without reshape.

  ### Codex flags / mitigations baked in

    * **Mask correctness**: defensively require `k_pos <= q_pos`
      for ctx keys. Current saturating_sub would allow future
      ctx positions if an invariant slipped. Fused attention
      kernel asserts both `q_pos - k_pos <= swa_window` AND
      `k_pos <= q_pos`.
    * **CPU oracle for SWA boundary positions** mandatory in
      v0.72.2 tests. The fused attention kernel is exactly where
      silent acceptance regressions hide.
    * **Don't pull K_ctx/V_ctx caching forward** to v0.72.2 —
      bigger structural risk than asymptotic value at current ctx.
      But design `[layer, ctx, kv]` indexing in the attention
      kernel so v0.72.5 caching slots in cleanly.
    * **Skip dedicated dflash-profile subcommand**: diagnosis is
      already decisive (Phase 3 is ~99% of drafter wall).
      Lightweight timers in dflash bench only if cheap.

  ### Expected drafter speedup once v0.72.0-.4 ship

  Conservative: 12.6 s -> ~50 ms (250x), bringing end-to-end DFlash
  speedup from 0.024x to ~0.95x at current verify cost (~423ms),
  approaching break-even with no-spec at code-prompt alpha levels.

  Aggressive (with N-row mat-mat amortization in projections,
  ctx_h caching v0.72.5): ~10ms drafter, ~1.5-2x end-to-end DFlash
  speedup vs no-spec.

  At which point the H5.3b kernel work (packed GDN + packed
  flash-attn-v4) becomes the next dominant phase and v0.73+
  fires accordingly.

- **rev 7.** H5.3b SHIPPED (v0.62 → v0.69). Codex
  next-moves partner session staked the v0.70+ sequence.
  Headline H5.3b results:
    * Layer-major packed_verify bit-exact vs token-major oracle on
      F32 0.8B (993k logit values bit-identical)
    * 27B Q4_K_M end-to-end: cos = 1.000000 vs token-major across
      all 16 verify rows; argmax 16/16; max|Δ| 6.4e-3
    * Speedup: 1.58× over naive packed_verify (was 1.53× before
      NR1=16 retune; retune drove kernel BW +30-60% but Amdahl shows
      remaining wall time is in GDN/attn per-token loops + encoder
      transitions)
    * Codex tripwire (≥ 2.5×) NOT cleared end-to-end, but kernel-level
      objective achieved. Remaining gap is non-mat-mat phases that
      don't show up in isolated mat-mat benches.

  ### v0.70+ sequencing (codex stake: PROFILE BEFORE PLUMBING)

  Codex reframed: profile FIRST, even before H5.5 plumbing. Reasoning:
  "if profile is really 2-3 hr work, it should precede H5.5 because
  it can invalidate the whole ranked roadmap cheaply. If profiling
  starts ballooning into framework work, abort and ship H5.5 first."

  Plan:

    * **v0.70**: Per-phase profiler for layer-major packed_verify;
      run at ctx ∈ {1K, 4K, 16K, 64K}. Tightly scoped — INSTRUMENTATION
      ONLY, no kernel changes. Hard timebox: ≤ 3 hours; abort if
      growing into a framework rebuild. Establishes which phase
      dominates at which context length.
    * **v0.71**: H5.5 end-to-end DFlash decode loop +
      greedy-equivalence test + `qwen-bench dflash` subcommand.
      Plumbing we can't ship without (cursor state machine, EOS,
      max_new_tokens, draft+verify+rollback wrapper).
    * **v0.72**: First public bench table vs llama.cpp tg128 on
      representative prompts (code / prose / long-ctx). The actual
      headline number we've been working toward.
    * **v0.73**: ONE kernel bet, profile-driven. Likely packed
      flash-attn-v4 if long-ctx dominates the prompts we care about,
      packed GDN recurrence if short-mid ctx dominates.

  ### Bottleneck shift across context length (Britt + codex)

  Estimated phase contribution (no profile yet — that's v0.70):

    | ctx   | mat-mat | GDN  | attn (KV reads) | dominant            |
    |-------|---------|------|-----------------|---------------------|
    | 0–1K  | ~150ms  | ~150 | <5              | mat-mat / GDN tied  |
    | 4K    | ~150ms  | ~150 | ~10             | mat-mat / GDN tied  |
    | 16K   | ~150ms  | ~150 | ~35             | still mat-mat / GDN |
    | 64K   | ~150ms  | ~150 | ~135            | attn catches up     |
    | 256K  | ~150ms  | ~150 | ~540            | **attn 3:1 dominant** |

  KV BW math: K+V × n_kv_heads × head_dim × fp16 = 2 × 8 × 128 × 2
  = 4096 B/token/layer; at 10K ctx that's 40.96 MB per attn-v4 call,
  × 16 layer-major-N calls × 16 attn layers = 10.5 GB per outer step
  / ~500 GB/s eff BW = ~21 ms attention-only. Linear in ctx until
  partition / cache effects dominate at very long ctx.

  Codex's earlier "don't touch attn-v4 until profile demands it"
  presumed short ctx. At long ctx **profile already demands it**.

  ### What we previously missed (codex next-moves session)

  **Acceptance-aware dynamic `N` policy** — non-obvious algorithmic
  lever filed for v0.74-ish. Currently fixed N=16 every outer step;
  if α drops on prose, packed verify wastes rows after first
  mismatch. A policy `N(ctx, observed_acceptance)` could shrink N on
  prose-class behavior (cheaper verify, fewer wasted rows). At α=0.3
  the cost-model arithmetic favors going SMALLER on prose:
    * α=0.3 D=15: 5.5 emitted / ~9 wall = 0.6×
    * α=0.3 D=5:  2.5 emitted / ~3 wall = 0.83×
  Wins because verify cost shrinks faster than expected emission.
  Algorithmic change, no kernel work. Lands AFTER bench infrastructure
  exists to evaluate it on real workloads.

  ### Other deferred items per codex (unchanged status)

    * Save-checkpoint compute kernel (kills 1536 cross-encoder
      transitions): NOT NOW unless H5.5 measurement shows
      restore/checkpoint costs visible.
    * KV-Q8 cache compression: AFTER packed-attn (algorithmic change
      first; compression multiplies whatever's left).
    * Fused FFN gate+up mat-mat: lower priority; doesn't change
      asymptotics.
    * GPU sampling kernel (top-k/top-p): not until non-greedy is a
      product path.
    * KV append packing: cleanup, only if profile exposes it.
    * ICB / MTL4 (encode amortization): ~5-10% lever at short ctx;
      after profile.

- **rev 6.** H5.3b plan re-sequenced after codex partner
  session at start of v0.62. Three sub-phases: H5.3b.0–3 (Q4_K
  mat-mat kernel + bench, no plumbing), H5.3b.4–5 (layer-major
  encode_block_packed + plumbing), H5.3b.6 (Q6_K mat-mat for ffn_down
  + lm_head). Codex Q1: layer-major rewrite is mandatory (token-major
  + mat-mat doesn't actually share weights across tokens). Codex Q3
  correction: lifted mat-mat is NOT bit-exact with mat-vec (half-
  staging in templates); gate threshold relaxed from cos = 1.0 to
  cos ≥ 0.999. Codex Q4: lift the classic 64×32×32 simdgroup_matrix
  tile (NR0=64, NR1=32, NK=32) verbatim, accept the partial-output
  half-fill at N_QUERY=16 for v1. Codex Q7 failure-mode prediction:
  column-major dst stride pitfall (lifted kernel writes
  `dst[row + col*ne0]`; our scratch is row-major `[N, H]`); explicit
  layout gate added.

  **NEW H5.3b.7 (post-ship; not immediate)**: tensor API exploration
  spike. llama.cpp ships dual classic + tensor mat-mat paths gated on
  `GGML_METAL_HAS_TENSOR`. Tensor path uses `mpp::tensor_ops::matmul2d`
  cooperative-tensor primitives that map to M3+ matrix hardware,
  materially faster on M3/M4/M5 in llama's own benchmarks but Apple-
  private and version-coupled. Decision matrix is measurement-driven:
  adopt fully / opt-in dual-path / skip, depending on real speedup at
  our N=16 shapes and Apple version churn rate. Filed deliberately
  for after H5.3b.6 ships so we have a stable classic-path baseline
  to A/B against; not lost.

- **rev 5.** H5.3a SHIPPED. v0.55 → v0.60.
  - **6 of 7 H5.3a gates green on the lib loop, all F32 bit-exact /
    cos=1.0:**
    - G1 lite (argmax match) — v0.57
    - G1 full (cosine ≥ 0.9999 raw logits) — v0.60 (cos = 1.0)
    - G2 (final state, BITWISE) — v0.57+v0.58 (codex-tightened from 1e-5)
    - G3 (checkpoint replay equivalence) — v0.59
    - G4 (hidden capture layout) — v0.60
    - G5 (GPU argmax with tie semantics) — v0.55+v0.57
    - G6 (restore boundary cases) — v0.59 (folded into G3)
  - **G7 (nonzero start_pos / partition exercise)** deferred to H5.5
    integration phase — needs ~7K context on 27B-Q4_K_M, fundamentally
    not a fast-loop test.
  - **Codex H5.3a open-ended review (post-v0.57) caught the
    `kv_n_pos == start_position` precondition gap** — would have
    silently dropped previously-encoded session state on primed-session
    calls. Fixed in v0.58 with a fail-loud guard + primed-session test
    that proves packed_verify produces bit-identical state to continued
    single_token from a primed session.
  - **G2 tolerance tightened from `< 1e-5` to BITWISE EQUALITY**
    (codex: "slack masking signal"). F32 packed_verify is genuinely
    bit-deterministic; the slack was hiding usable signal.
  - **Slot-helper `debug_assert!`s promoted to runtime `assert!`s.**
    Failure mode if a slot is OOB is silent OOB blit on release builds.
  - **Restore primitive (originally H5.4) folded into H5.3a + shipped
    in v0.59.** API: `n_keep ∈ [1, N]` (not `n_accepted`; codex pinned
    the indexing). Algorithm: blit `gdn_ckpt_slot(k, n_keep-1)` →
    `session.gdn_state[k]` for every k; same for conv; host-update
    `kv_n_pos[i] := start_position + n_keep`.
  - Foundation pieces (v0.55–v0.56): BlitEncoder wrapper +
    GPU argmax kernel + `MetalDFlashVerifyScratch` /
    `MetalDFlashDebugScratch` two-struct split.
  - Packed_verify itself (v0.57): `DFlashDecoder::packed_verify` per
    codex Q1 placement (NOT method on `MetalForward`); inner free fn
    `encode_packed_verify_inner` lets tests run without a real DFlash
    drafter; `_with_logits` debug variant added in v0.60 with
    `Option<&MetalTensor>` plumbing in the inner impl (no code
    duplication; one inline scatter is the only divergence).

- **rev 4.** Open-ended codex partner session after H5.2.5
  GREEN — re-sequenced H5.3 before any code shipped.
  - **H5.3 split into H5.3a (correctness scaffold) + H5.3b (perf
    pass).** Same playbook as the rev-3 H5.2.5 pivot. Codex called the
    front-loaded "Bucket A → E → C-naive → D-naive → B" ordering as
    "risks getting a pretty packed API that cannot roll back."
  - **H5.3a is naive: N successive single-token encodes inside one
    command buffer, blit checkpoints between tokens.** Allowed to be
    slow. Deliverable is the production API + bit-exactness, not
    throughput. H5.3b is the perf pass (tiled Q4_K mat-mat first).
  - **H5.4 (rollback primitive) folded into H5.3a.** Checkpoint contract
    isn't testable without restore — gate G3 (checkpoint replay
    equivalence) requires it. The H5.4 bit-exactness test moves into
    G3 + G6.
  - **§H5.3a `MetalDFlashVerifyScratch` expansion**: not just outputs;
    also owns target-side N-shaped activations (packed_ids_buf, x_pack,
    h_pack, q/k/v/o pack, ffn pack, debug logits). Codex Q5 expansion.
  - **§H5.3a per-token checkpoint via Metal blit copies, NOT a compute
    "ckpt_write" kernel.** 2.3 GiB SSM checkpoint is bulk contiguous
    memory — blits use GPU DMA engines, faster than dispatch + faster
    than a compute copy. Codex Q4 v1 design (ii) refinement.
  - **§H5.3a `packed_ids_buf: [N]` is mandatory.** Catches the failure
    mode codex predicted: reusing single-token `ids_buf` inside the
    packed encode loop means every queued `get_rows` reads the
    last-written CPU value (CPU mutations don't sequence with the
    GPU encode). Would have shipped as a silent bug.
  - **§H5.3a 7-gate suite** (was 1 in rev 3): packed-vs-single cosine,
    final state equivalence, checkpoint replay equivalence, hidden
    capture layout, GPU argmax vs debug logits, restore boundary cases,
    nonzero start_pos / partition exercise. Anti-regression assertion
    that production `packed_forward` does not read back `[N, V]`.
  - **§H5.3b ordering = tiled Q4_K mat-mat → internalized GDN → packed
    attn-v4 (only if profile demands).** Codex Q3: "real long-context
    cost is N·L not N²/2; FFN/projection weight traffic is the larger
    structural waste first. Don't touch attn_v4 — adding an N axis to
    m/l/o state can pass short-ctx tests and fail at partition
    boundaries."

- **rev 3.** Open-ended codex review after H5.0 + H5.1
  shipped, then a follow-up codex partner session pressure-tested the
  v1 hybrid Metal drafter design before commit.
  - **H5.1.5 demoted** from "DFlash correctness gate" to "plumbing
    cosine gate" — most of the per-layer compute (attention + FFN
    silu_mul) is identical CPU code on both paths, so cosine doesn't
    actually validate the SWA-mask algorithm. Algorithmic check
    moves to a CPU-vs-MLX trace equivalence (part of H5.6).
  - **v1 hybrid drafter** explicitly labeled as a debug scaffold,
    not a perf path. Native Q8_0 storage and SWA Metal kernel both
    deferred to H5.3 — both are real wins, neither helps the H5.2.5
    α measurement.
  Insertions:
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
