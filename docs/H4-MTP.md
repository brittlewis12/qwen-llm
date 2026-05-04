# H4: Multi-Token Prediction (MTP) — speculative decode oracle

Implements Qwen3.6's native MTP head as a speculative-decode draft.
MTP itself is unlikely to deliver >1.05× tg multiplier on single-stream
Apple Silicon (cost analysis in §3 shows ~0.91×). The reason to ship
it anyway: it's the cheapest correctness oracle for the shared
speculative-decoding infrastructure that H5 (DFlash, the actual perf
target) builds on.

**What H4 ships:**
- `MtpHead` loader (binds tensors from MTP-aware GGUFs)
- CPU MTP forward (validation oracle vs vLLM ground truth)
- Metal MTP draft step (single-token, validated against CPU)
- `SpeculativeDecoder` wrapper with greedy lazy-sequential-verify
- Cursor accounting for accept/reject token bookkeeping
- Persistent MTP-KV with prompt-time prefill
- GPU-resident hidden carry across iterations
- Greedy generation equivalence test vs MTP=off

**What H5 ships (separate doc):**
- DFlash drafter loader (z-lab v1 recipe via spiritbuun GGUFs)
- Multi-layer hidden-state expose during base prefill
- Block-diffusion drafter Metal forward (block_size=16)
- Packed-N base verify (mat-mat, real perf-critical kernel work)
- Per-token GDN state checkpointing for hybrid rollback
- Greedy + probabilistic acceptance; bench vs spiritbuun's fork

**Non-goals (deferred):**
- N>1 MTP speculative tokens (Qwen3.6 ships `nextn_predict_layers=1`)
- Probabilistic rejection sampling (Leviathan/Chen) — H5 only
- Continuous batching — single-stream first
- ICB-cached MTP step — encode-from-scratch each step

`--mtp` fails closed if temperature ≠ 0 / sampling ≠ greedy until
H5 ships rejection sampling.

## 1. Architecture

The MTP head is one full-attention DecoderLayer with two extra
RMSNorms and one `Linear(2H, H)` in front. All sub-kernels already
exist in our Metal stack. The work is wiring + correctness, not new
kernels.

### 1.1 Tensor inventory (Qwen3.6-27B)

From `mtp_tensor_inventory()` (`crates/qwen-llm/src/metal_forward.rs:2222`)
and the converter remapping in
`brittlewis12/llama.cpp/convert_hf_to_gguf.py:4837-4844`:

| GGUF tensor (`blk.64.*`)        | Shape                       | dtype  | Role                  |
| ------------------------------- | --------------------------- | ------ | --------------------- |
| `nextn.eh_proj.weight`          | `[2*H, H]` = `[10240,5120]` | Q4_K   | 2H→H projection       |
| `nextn.enorm.weight`            | `[H]`                       | F32    | embed RMSNorm gain    |
| `nextn.hnorm.weight`            | `[H]`                       | F32    | hidden RMSNorm gain   |
| `nextn.shared_head_norm.weight` | `[H]`                       | F32    | post-MTP RMSNorm      |
| `attn_norm.weight`              | `[H]`                       | F32    | pre-attn RMSNorm      |
| `attn_q.weight`                 | `[H, 2*q_dim]`              | Q4_K   | Q + gate              |
| `attn_k.weight`                 | `[H, kv_dim]`               | Q6_K   | K                     |
| `attn_v.weight`                 | `[H, kv_dim]`               | Q6_K   | V                     |
| `attn_output.weight`            | `[q_dim, H]`                | Q4_K   | O                     |
| `attn_q_norm.weight`            | `[head_dim]`                | F32    | per-head RMSNorm      |
| `attn_k_norm.weight`            | `[head_dim]`                | F32    | per-head RMSNorm      |
| `attn_post_norm.weight`         | `[H]`                       | F32    | post-attn RMSNorm     |
| `ffn_gate.weight`               | `[H, F]`                    | Q4_K   | SwiGLU gate           |
| `ffn_up.weight`                 | `[H, F]`                    | Q4_K   | SwiGLU up             |
| `ffn_down.weight`               | `[F, H]`                    | Q6_K   | SwiGLU down           |

Tied tensors (TENSOR_NOT_REQUIRED, alias the main model):
- `nextn.embed_tokens` → `token_embd.weight`
- `nextn.shared_head_head` → `output.weight`

The MTP block is structurally a normal attn block + four adornments,
not a new graph.

### 1.2 The MTP input contract

At sequence slot `i`, the MTP head consumes:

```
mtp_input_at_i = (embed(t_{i+1}),  h_i,  position=i)
```

and predicts logits for `t_{i+2}`. The embedding is of the **next**
token (already known/sampled), the hidden is from the **previous**
slot (computed by the base model at position `i`), and RoPE rotates
by `position=i` (the slot of the hidden, not the slot of the predicted
token).

vLLM's proposer (`vllm/v1/spec_decode/llm_base_proposer.py:677-680`)
makes this explicit by shifting `target_token_ids` left by 1 and
stuffing the freshly-sampled `next_token_ids` into the last slot:

```python
self.input_ids[: num_tokens - 1] = target_token_ids[1:]  # shift
self.input_ids[token_indices_to_sample] = next_token_ids  # bonus slot
# self.hidden_states[:num_tokens] = target_hidden_states  # no shift
```

The MTP forward itself (`vllm/model_executor/models/qwen3_next_mtp.py:113-115`):

```python
inputs_embeds = self.pre_fc_norm_embedding(self.embed_tokens(input_ids))
hidden_states = self.pre_fc_norm_hidden(hidden_states)
hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))
```

Concat order is `[embed, hidden]` (vLLM canonical). The tensor name
`eh_proj` ("e then h") matches.

### 1.3 MTP forward algorithm

```
e_norm = RMSNorm(embed(next_tok), enorm)            # [H]
h_norm = RMSNorm(prev_hidden, hnorm)                # [H]
x      = eh_proj @ concat([e_norm, h_norm])         # [H], 2H→H matvec
x      = decoder_block(x, position, mtp_kv)         # standard full-attn block
                                                    # gated attn (q_norm, k_norm,
                                                    # partial RoPE, σ-gate, KV append,
                                                    # score, V) + SwiGLU FFN
x      = RMSNorm(x, shared_head_norm)
logits = lm_head @ x                                # shared with base lm_head
```

### 1.4 Per decode step (n_mtp=1, greedy)

State invariants carried across steps:

```
processed_pos:    u32      // base session has run through this position.
                          //   GDN state + KV cache reflect tokens at [0..=processed_pos].
hidden_at_proc:   Tensor   // base hidden state PRE-output_norm at processed_pos.
                          //   GPU-resident; lifetime = one step. Invalidated by next
                          //   base_forward call on the same session.
emit_tok:         i32      // next token to emit and process.
                          //   Selected by previous step's logits.
                          //   NOT yet emitted (loop emits at top).
                          //   NOT yet processed by base.
                          //   Position when processed = processed_pos + 1.

// Implicit invariant: mtp_processed_pos == processed_pos - 1 at the top of
// every iteration. Bootstrap establishes this; the loop preserves it by
// running the MTP bridge inline on accept (step E) before invalidating
// hidden_P with the second base_forward.
```

Bootstrap (after prompt prefill of `n` prompt tokens at positions `0..n-1`):

```
processed_pos     = n - 1
hidden_at_proc    = base_hidden_states[n-1]
emit_tok          = argmax(lm_head(output_norm(hidden_at_proc)))
// MTP-KV prefill (§1.5) leaves mtp_processed_pos = n - 2.
// First iteration's step B drafts slot n-1; no boundary bridge needed.
```

**Terminal-return contract (max_new_tokens / EOS):** when the loop hits
the limit and returns inside an iteration (after emitting `emit_tok` or
`D_tok`), the session state is left at "not resumable for further
generation." Specifically:
- After A's early return: base at `processed_pos`, MTP at
  `processed_pos - 1`, no work in flight.
- After D's early return on accept: base at `P_pos` (only stepped through
  P_tok), MTP at `P_pos - 1` (old `processed_pos`; step E did not run).
  Base and MTP are consistent only through `P_tok` — `D_tok` was emitted
  without being processed by either. Continuation would require running
  `mtp_draft_kv_only(D_tok, hidden_P, P_pos)` followed by
  `base_forward(D_tok, P_pos+1)` to bring state up to date — neither
  runs on terminal return.

This is fine for the current `decode()` API (caller takes the token
list and starts a fresh session if they want to continue). If
session continuation post-return becomes a requirement, change the
contract: do step E + step F unconditionally on accept, then check
the limit *after* the bonus is computed.

Per-step loop:

```
# A. Emit the carried token, check stop conditions.
emit(emit_tok)
if emit_tok == eos or emitted_count >= max_new_tokens:
    return

P_tok = emit_tok                                    # the just-emitted token
P_pos = processed_pos + 1                           # its sequence position

# B. MTP draft for slot processed_pos (predicts t at P_pos+1).
# Precondition: mtp_processed_pos == processed_pos - 1 (loop invariant).
D_tok = mtp_draft(
    next_tok    = P_tok,                            # t_{i+1}
    prev_hidden = hidden_at_proc,                   # h_i
    position    = processed_pos,                    # i
    mtp_session,
)
# After: mtp_processed_pos == processed_pos.

# C. Base forward on P_tok at P_pos.
# This call invalidates hidden_at_proc (session arena reuse).
hidden_P    = base_forward(P_tok, P_pos, base_session)   # advances base by 1
logits_P    = lm_head(output_norm(hidden_P))
target_next = argmax(logits_P)                           # what target wants at P_pos+1

# D. Lazy sequential verify.
if D_tok == target_next:
    # ACCEPT branch.
    emit(D_tok)
    if D_tok == eos or emitted_count >= max_new_tokens:
        return

    # E. Bridge MTP KV for the skipped middle slot P_pos, INLINE.
    # MTP currently at processed_pos = P_pos - 1. Base advanced through P_pos
    # in step C. Next iter wants mtp_processed_pos == P_pos.
    # We must run this bridge BEFORE the next base_forward (step F) because
    # that call would invalidate hidden_P which the bridge needs.
    _ = mtp_draft_kv_only(
        next_tok    = D_tok,                        # t_{i+1} for slot P_pos
        prev_hidden = hidden_P,                     # h_{P_pos} — still live
        position    = P_pos,                        # i = P_pos
        mtp_session,
    )
    # After bridge: mtp_processed_pos == P_pos.

    # F. Base forward on D_tok. Invalidates hidden_P, produces hidden_D.
    D_pos     = P_pos + 1
    hidden_D  = base_forward(D_tok, D_pos, base_session)
    logits_D  = lm_head(output_norm(hidden_D))
    next_emit = argmax(logits_D)

    # Update for next iter. mtp_processed_pos == P_pos == D_pos - 1, invariant holds.
    processed_pos    = D_pos
    hidden_at_proc   = hidden_D
    emit_tok         = next_emit

else:
    # REJECT branch.
    # Base sits at P_pos, MTP sits at processed_pos == P_pos - 1.
    # State is consistent — no rollback or bridge needed.
    processed_pos    = P_pos
    hidden_at_proc   = hidden_P
    emit_tok         = target_next
```

Invariants per branch:
- Accept: base advances by 2 (P_tok, D_tok); MTP advances by 2
  (step B + step E inline bridge); emitted 2 tokens. Loop invariant
  `mtp_processed_pos == processed_pos - 1` preserved.
- Reject: base advances by 1; MTP advances by 1; emitted 1 token.
  Loop invariant preserved.
- `emit_tok` is always the next-to-emit, never double-emitted.
- EOS short-circuits early. Skipped bridge/forward work is fine
  because we're not continuing.

The inline bridge (step E) is the key correctness point: `hidden_P` is
GPU-resident in the session arena and invalidated by the next
`base_forward` call. Running the bridge before step F's base call keeps
all hidden references valid throughout the iteration without copying.

### 1.5 Prompt-time MTP-KV prefill

Persistent MTP KV needs entries for prompt positions before the first
draft step. Otherwise the MTP attn block sees an empty KV history during
draft 1 and computes attention against itself only — different from
the trained distribution.

The MTP head at slot `i` consumes `(embed(t_{i+1}), h_i)`. For prompt
positions:
- `h_i` is available from the base prompt forward (positions `0..n-1`)
- `embed(t_{i+1})` is known from prompt token IDs (positions `1..n-1`)

For boundary slot `i = n-1` we have `h_{n-1}` but NOT `embed(t_n)` —
that's the first generated token, only known after the prompt logits
are sampled.

Prefill contract — streamed during prompt forward:

```
# Streamed: as base computes each h_i during prompt forward, immediately
# call MTP for slot i (using prompt[i+1] as the known next-token).
# This avoids needing to retain all h_i across the prompt forward, which
# would require copying out of the session arena (each base call invalidates
# the prior hidden).

for i in 0..n {                                    // i = 0, 1, ..., n-1
    h_i = base_step(prompt[i], position=i)         // sub-step of prompt forward
    if i + 1 < n {
        // Slots 0..n-2: pair (prompt[i+1], h_i, i). Discard logits.
        mtp_draft_kv_only(
            next_tok    = prompt[i+1],             // known from prompt
            prev_hidden = h_i,                     // still live; consumed before next base_step
            position    = i,
            mtp_session,
        )
    } else {
        // i == n-1: don't run MTP for this slot (would need first_generated_token,
        // not yet known). Retain h_{n-1} for bootstrap. The first decode
        // iteration's step B will draft slot n-1.
        h_{n-1} = h_i  // keep for bootstrap (callers must own this; see lifetime note below)
    }
}

# Postcondition: mtp_processed_pos == n - 2.
# h_{n-1} is the final live hidden, fed into bootstrap.
first_generated_token = argmax(lm_head(output_norm(h_{n-1})))
# First decode iteration's step B drafts slot n-1 using
# (first_generated_token, h_{n-1}, position=n-1), advancing mtp_processed_pos
# to n-1 and restoring the loop invariant before the iteration's step C.
```

Lifetime note: `h_{n-1}` must remain live across the prompt forward's
final `base_step` and into the first decode iteration's step B. That's
the natural lifetime since no further `base_step` calls happen in
prefill. The bootstrap argmax and first iter's `mtp_draft` both
consume `h_{n-1}` before any decode-time `base_forward`, so no copy
is needed.

If our prompt forward isn't structured to expose `h_i` per token, we
can either restructure it (cheaper) or copy each `h_i` into an owned
buffer for batch prefill. Owned-copy cost: `n × H × 4 B = 1024 × 5120
× 4 = 20 MiB` for a 1024-token prompt at H=5120 F32 — small. Streamed
prefill is still preferred (no copy + better cache behavior), and is
part of H4.3's deliverable.

Cost: `n-1` MTP forwards during prompt prefill, ~3 ms each on 27B
≈ ~3 s for a 1024-token prompt. Significant but not blocking for the
H4 oracle role. Mitigation paths if this becomes a workload-killer:

- **Packed prefill kernel.** Mat-mat through MTP weights for the whole
  prompt in one pass instead of per-token mat-vec.
- **Sliding-window MTP KV.** Cap history at K most recent positions.
  For K=64 prefill cost is bounded regardless of prompt length. Risk:
  attention to positions > K back may matter for some workloads.

Both are deferred until measured prefill cost justifies them.

### 1.6 Persistent MTP KV (not transient)

The MTP attn block is a real full-attention layer that consumes
positions through RoPE; its K/V at prior positions contribute real
attention scores during subsequent drafts. Resetting KV every draft
changes what the model computes. Persistent MTP KV (one ring per
sequence, one layer's worth — small) is the correct default.

Some references (notably Rapid-MLX `patches/qwen3_next_mtp.py:200-203`)
pass `mtp_cache=None` per draft. That's an approximation that may be
observationally close to persistent KV for `n_mtp=1` first-draft after
fresh state, but diverges as prompt length grows and as draft history
accumulates. We do not adopt it.

## 2. API surface

Loader (`crates/qwen-llm/src/loader.rs`):

```rust
/// Multi-Token-Prediction head. Present when GGUF contains the nextn tensors.
/// Block index is `arch.n_layer` (one past the last base layer; e.g. 64 for 27B).
pub struct MtpHead<'a> {
    pub block_idx: u32,
    pub eh_proj: &'a TensorDesc,
    pub enorm: &'a TensorDesc,
    pub hnorm: &'a TensorDesc,
    pub shared_head_norm: &'a TensorDesc,
    pub attn: AttnBlock<'a>,
    // embed_tokens / shared_head_head intentionally omitted — they alias the
    // main model's tensors. Caller resolves via Model.token_embd / Model.lm_head.
}

pub struct Model<'a> {
    // … existing fields …
    pub mtp: Option<MtpHead<'a>>,
}
```

Bind logic: probe for `blk.{n_layer}.nextn.eh_proj.weight`. If present,
bind `MtpHead`. **Don't trust `nextn_predict_layers` metadata** — the
upstream converter doesn't always write it; tensor presence is ground
truth.

Metal MTP module (new file `crates/qwen-llm/src/metal_mtp.rs`) — keep
MTP code OUT of `metal_forward.rs` so the base path stays
`Option`-free:

```rust
pub struct MetalMtpHead {
    pub eh_proj: MetalTensor,
    pub enorm: MetalTensor,
    pub hnorm: MetalTensor,
    pub shared_head_norm: MetalTensor,
    pub attn: MetalAttnBlock,
}

pub struct MetalMtpSession {
    /// KV ring for the single MTP attn layer. Persistent across drafts.
    pub kv_k: MetalTensor,
    pub kv_v: MetalTensor,
    pub kv_n_pos: usize,

    /// Pre-eh_proj scratch.
    pub e_normed: MetalTensor,
    pub h_normed: MetalTensor,
    pub eh_concat: MetalTensor,
    pub eh_out: MetalTensor,

    /// MTP attn scratch (separate from base session arena).
    pub attn_q_full: MetalTensor,
    pub attn_q: MetalTensor,
    pub attn_gate: MetalTensor,
    pub attn_q_normed: MetalTensor,
    pub attn_k_now: MetalTensor,
    pub attn_v_now: MetalTensor,
    pub attn_k_normed: MetalTensor,
    pub attn_scores: MetalTensor,
    pub attn_o: MetalTensor,
    pub mtp_logits: MetalTensor,
}

pub struct SpeculativeDecoder<'a> {
    pub base: &'a MetalForward<'a>,
    pub mtp_head: &'a MetalMtpHead,
    pub mtp_session: MetalMtpSession,
}

impl<'a> SpeculativeDecoder<'a> {
    /// Draft a single token. The MTP head at slot `position` consumes
    /// `(embed(next_tok), prev_hidden)` and predicts the token at
    /// `position + 2`. Caller must ensure
    /// `mtp_session.kv_n_pos == position`.
    /// Side effect: appends one entry to MTP KV at slot `position`.
    ///
    /// GPU command ordering: this call MUST complete all GPU reads of
    /// `prev_hidden` before returning (i.e., commit + waitUntilCompleted),
    /// OR enqueue them on the same MetalSession command queue used by
    /// subsequent `base_forward`/`single_token_with_hidden` calls so that
    /// the session arena's hidden buffer is not overwritten before the
    /// MTP read completes. v1: synchronous (commits + waits per call).
    pub fn draft(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
        ctx: &MetalContext,
    ) -> Result<i32, MfError>;

    /// Same as `draft` but discards logits. Used by the inline accept-branch
    /// bridge (§1.4 step E) and by prompt-time prefill (§1.5).
    /// Same GPU command ordering contract as `draft`.
    pub fn draft_kv_only(
        &mut self,
        next_tok: i32,
        prev_hidden: &MetalTensor,
        position: u32,
        ctx: &MetalContext,
    ) -> Result<(), MfError>;

    /// Top-level loop implementing §1.4.
    pub fn decode(
        &mut self,
        prompt_ids: &[i32],
        max_new_tokens: usize,
        eos_id: i32,
        base_session: &mut MetalSession,
    ) -> Result<DecodeOutput, MfError>;
}

pub struct DecodeOutput {
    pub tokens: Vec<i32>,
    pub stats: SpecStats,
}

pub struct SpecStats {
    pub steps: u32,
    pub accepted: u32,
    pub base_forward_calls: u32,
    pub mtp_calls: u32,        // includes bridge calls
    pub wall_ms: f64,
}
```

Base driver extension (`crates/qwen-llm/src/metal_forward.rs`) — one new
method, no new state:

```rust
impl<'a> MetalForward<'a> {
    /// Single-token decode that ALSO returns the GPU-resident pre-output-norm
    /// hidden state. Existing `single_token` becomes a thin wrapper.
    pub fn single_token_with_hidden<'s>(
        &self,
        token_id: i32,
        position: u32,
        session: &'s mut MetalSession,
    ) -> Result<SingleTokenOutput<'s>, MfError>;
}

pub struct SingleTokenOutput<'s> {
    pub logits: Vec<f32>,                   // CPU readback for sampling
    pub hidden: &'s MetalTensor,            // GPU-resident, no readback
}
```

The `'s` lifetime ensures the hidden borrow is tied to the session arena,
preventing accidental reuse across calls.

## 3. Cost model

Symbols:
- `T_base` = wall-time of one base single-token decode (~45 ms on 27B-Q4_K_M)
- `T_mtp`  = wall-time of one MTP draft step
- `α`      = acceptance rate (fraction of drafts that match target's argmax)
- `ε`      = `T_mtp / T_base`

### 3.1 `T_mtp` estimate

The MTP head is one transformer block + 4 RMSnorms + 1 mat-vec
(`eh_proj` 2H→H = 26 MB Q4_K). The base 27B has 64 such blocks plus
embedding lookup and lm_head.

By weight bytes touched per token (BW-bound regime):
- Base 27B-Q4_K_M: ~14.5 GB / token
- MTP block: `eh_proj 26 MB + attn ~40 MB + ffn ~110 MB ≈ 180 MB`
- BW ratio ≈ 180/14500 ≈ 1.2%

Fixed overheads inflate this:
- ~12-14 dispatches per MTP block at ~10 µs each = ~140 µs fixed cost
- MTP attention KV reads grow with context

Combined estimate: `ε ∈ [0.05, 0.15]`. Best case (BW-dominated, short
context) ≈ 0.05. Worst case (dispatch-dominated, long context with
growing MTP KV) ≈ 0.15. Empirical measurement is part of H4.3 bench.

### 3.2 Lazy sequential verify (Strategy A)

Per iteration (per §1.4):
- Always: 1 base forward + 1 MTP draft for current slot
- On accept (prob α): +1 base forward + 1 inline MTP bridge call

```
wall per iter = T_base + T_mtp + α·(T_base + T_mtp) = (1+α)·(T_base + T_mtp)
tokens per iter = 1 + α
speedup = (1+α)·T_base / [(1+α)·(T_base + T_mtp)] = 1 / (1 + ε)
```

Speedup is independent of α — both numerator and denominator scale by
(1+α), so they cancel.

| ε    | speedup |
|------|---------|
| 0.05 | 0.95×   |
| 0.10 | 0.91×   |
| 0.15 | 0.87×   |

**Lazy verify is a flat net loss.** Each accept costs not just an
extra base forward but also an extra MTP bridge call; they cancel
exactly against the extra emitted token. The amortization across
"tokens emitted" matches the amortization across "work performed."

This is correctness-only. We ship it as the H4 oracle for the shared
spec-decode infrastructure that DFlash (H5) builds on. The actual
perf wins come from H5's packed-N=16 amortization.

### 3.3 Why packed verify wins for DFlash but not for MTP

DFlash drafts 16 tokens in one parallel forward pass conditioned on
target hidden states. The verify pass processes 16 query tokens through
the base model in a single fused mat-mat forward — weight loads
amortize across 16 queries, not 2.

For MTP at N=1, packing buys only 2× amortization on FFN/attn weight
loads (and 0× on GDN, which is sequential per token). For DFlash at
N=16, the amortization is closer to 8-12× for FFN/attn. That's where
the measured ~2× tg multiplier comes from on M-series (per author tweet
about spiritbuun's fork; needs local verification).

MTP doesn't justify its own packed-verify implementation. The
infrastructure built for DFlash in H5 is recipe-agnostic and could be
revisited for MTP later if needed.

## 4. Phasing

### H4.0 — Loader binding

1. Add `MtpHead<'a>` to loader; bind on tensor presence (not on
   metadata).
2. Add `mtp_bind_27b()` test that verifies shapes match expected for
   our 27B-MTP-Q4_K_M GGUF and 0.8B-MTP-F32.

### H4.1 — CPU oracle

1. Extend `forward.rs` with `MtpHead` forward. Implement vLLM's exact
   concat order (`[embed, hidden]`).
2. Generate ground truth: run vLLM (or Rapid-MLX with `mtp_forward`)
   on Qwen3.6-0.8B-MTP for a fixed seed and dump
   `(prev_hidden, next_tok, position, draft_logits)` quadruples for
   ~10 steps. Here `next_tok = t_{i+1}` (the embedding input) and
   `prev_hidden = h_i` (the hidden input at slot `i`).
3. Feed our CPU MTP forward the same quadruples; compare logits.
4. Pass criterion: cosine ≥ 0.9999 vs the oracle.

If H4.1 fails: swap oracle to MLX DFlash on a small target (e.g.
Qwen3-4B-DFlash). The infra is what we're validating, not MTP
specifically — pivot directly to H5 DFlash bring-up.

### H4.2 — Metal MTP draft step

1. Implement `MetalMtpHead`, `MetalMtpSession`, `SpeculativeDecoder::draft`.
2. Reuse all existing kernels (RMSNorm, Q4_K mat-vec, RoPE, KV append,
   flash-attn v4, gated-attn output, SwiGLU FFN).
3. Test: draft a single token from a known prompt. Compare against CPU
   MTP forward (H4.1). Cosine ≥ 0.9999.

### H4.3 — Lazy sequential verify end-to-end

1. Implement `SpeculativeDecoder::decode` per §1.4.
2. Implement prompt-time MTP-KV prefill per §1.5.
3. Test: greedy generate 50 tokens with MTP=on vs MTP=off; sequences
   must be IDENTICAL.
4. Bench: report `(workload, α, t/s, speedup, mtp_calls_per_iter)`.
   Expected: ~0.91× per §3.2.
5. Move to H5 (DFlash) regardless of MTP speedup.

## 5. Validation plan

### Bit-exact correctness

- **CPU oracle (H4.1):** cosine ≥ 0.9999 draft logits vs vLLM ground
  truth on Qwen3.6-0.8B-MTP, 10 sequential steps.
- **Metal vs CPU (H4.2):** cosine ≥ 0.9999 draft logits Metal vs CPU
  on the same 0.8B model.
- **Greedy generation equivalence (H4.3):** with MTP on vs off,
  generated token sequences must be IDENTICAL up to max_new_tokens.

### Cursor, EOS, and bridge edge cases

| Test                                            | Setup                                                    | Pass criterion                                                                                          |
|-------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Accept then EOS-as-bonus                        | Accept, P_tok and D_tok both ≠ EOS, but argmax(logits_D)=EOS | Emit P_tok + D_tok + EOS in next iter; total = N+3                                                  |
| Reject with EOS as `target_next`                | Reject, argmax(logits_P) = EOS                           | Emit P_tok + EOS, stop                                                                                  |
| EOS as `emit_tok`                               | First step's emit_tok = EOS                              | Emit EOS, stop. Don't run base/MTP                                                                      |
| EOS as `D_tok`                                  | MTP drafts EOS, target's argmax ≠ EOS                    | Reject, emit P_tok, continue with target_next as next emit_tok                                          |
| max_new_tokens = 1, accept                      | Limit hits after P_tok                                   | Emit P_tok only; don't run D_tok forward, don't run inline bridge. Session not resumable (per §1.4 terminal-return contract) |
| max_new_tokens = 2, accept                      | Limit hits after D_tok                                   | Emit P_tok + D_tok; do NOT run inline bridge or step F (per terminal-return contract); don't fetch next_emit. Session not resumable |
| max_new_tokens = 3, accept-then-accept          | Two accepts in a row                                     | Emit exactly 3 tokens; first iter ran inline bridge + step F; second iter early-returns after emitting D_tok with no bridge/step F |
| max_new_tokens = N, all reject path             | All drafts rejected                                      | Emit N tokens, all from target_next chain; no bridge calls                                              |
| Accept-then-reject sequence                     | Accept iter 1, reject iter 2                             | Iter-1 inline bridge ran; iter-2 has no bridge; mtp_processed_pos correct after both                   |
| Two consecutive accepts                         | Accept iter 1, accept iter 2                             | Both iters ran inline bridge; mtp_processed_pos == processed_pos - 1 after each                         |
| Bootstrap → first iter accept                   | Prompt prefill leaves mtp at n-2; first iter accepts     | First iter's step B drafts slot n-1, step E bridges slot n; no double-append at slot n-1                |
| Bootstrap → first iter reject                   | First iter rejects                                       | First iter's step B drafts slot n-1; mtp_processed_pos == n-1 after iter (no bridge)                    |
| Prompt-time prefill streaming                   | n-token prompt                                           | After prefill, MTP KV has exactly n-1 entries (slots 0..n-2); base KV has n entries (slots 0..n-1)      |

Implement as a table-driven test in `crates/qwen-llm/tests/mtp_correctness.rs`.

### Throughput

- **H4.3:** lazy verify speedup vs MTP=off, measured on 27B-Q4_K_M.
  Report `(workload, α, t/s, speedup, mtp_calls_per_iter)`.
  Workloads: `mmlu_easy`, `humaneval_short`, `pile_short`.

### Greedy-only enforcement

`--mtp` + temperature ≠ 0 OR `--top-p < 1.0` OR `--top-k > 0` →
fail with explicit error: "MTP currently supports greedy decoding only;
remove sampling flags or wait for H5 rejection sampling." Do not
silently fall back.

## 6. Files to be touched

- `crates/qwen-llm/src/loader.rs` — add `MtpHead`, bind logic, test
- `crates/qwen-llm/src/forward.rs` — CPU MTP forward (H4.1 oracle)
- `crates/qwen-llm/src/metal_mtp.rs` — new: `MetalMtpHead`,
  `MetalMtpSession`, `SpeculativeDecoder`, draft + decode
- `crates/qwen-llm/src/metal_forward.rs` — `single_token_with_hidden`
  only (one new method, no Option fields)
- `crates/qwen-cli/src/main.rs` — `--mtp` flag with greedy enforcement
- `crates/qwen-llm/benches/qwen-bench.rs` — `--mtp` mode
- `crates/qwen-llm/tests/mtp_correctness.rs` — new

No new Metal kernels needed for H4. The packed-N path is built once in
H5 (DFlash) where N=16 amortization actually pays off.

## 7. Risks

| Risk                                                            | Mitigation                                                                                       |
|-----------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| MTP concat order wrong                                          | H4.1 oracle catches on first divergence                                                          |
| MTP RoPE position wrong                                         | Same                                                                                             |
| `eh_proj` loaded transposed                                     | Existing tensor desc parser handles GGUF orientation; covered by 0.8B oracle                     |
| Cursor accounting bug (especially inline-bridge ordering)       | §5 test matrix; particularly the `accept-then-reject`, `two consecutive accepts`, and `bootstrap → first iter accept` rows |
| `nextn_predict_layers` metadata absent                          | Probe by tensor presence (per H4.0)                                                              |
| MTP-KV prefill cost dominates short-prompt benchmarks           | Sliding window if measurement supports; not blocking for oracle role                             |
| α much lower than vLLM's claim due to quantization              | H4.3 bench reports per-workload α (informational only — MTP is correctness-only)                 |
| H4.1 cosine fails vs vLLM oracle                                | Pivot to MLX DFlash on Qwen3-4B-DFlash; the infra is what we're validating                      |
| H4.3 lazy verify shows the predicted ~0.91× slowdown            | Expected. Document and ship as infra-validation milestone                                        |

## 8. Position within the broader plan

H4 is one of two consecutive speculative-decode milestones:

- **H4 (this doc).** MTP head + lazy sequential verify. Correctness
  oracle; ~0.91× speedup expected.
- **H5 (separate doc, `docs/H5-DFLASH.md`).** DFlash drafter (z-lab v1
  recipe) + packed-N=16 verify + per-token GDN state checkpoints +
  greedy/probabilistic acceptance. Target: match or beat the
  ~2× tok/s figure that spiritbuun's fork author reported on M-series
  (verification is itself an H5.5 deliverable; public HF cards
  document only CUDA + RTX 3090 numbers).

The shared infrastructure between H4 and H5:
- Pre-output-norm hidden state expose from base forward
- `SpeculativeDecoder` wrapper owning drafter state + verify loop
- Drafter loader (from GGUF, with arch-specific tensor remapping)
- GPU-resident hidden carry across iterations
- Cursor accounting for accept/reject token bookkeeping
- EOS / max_new_tokens edge cases

H5 adds (not in H4):
- Multi-layer hidden taps from base
- Block-diffusion drafter graph (cross+self attn)
- Packed-N=16 base verify (mat-mat path through GDN+attn+FFN)
- Per-token GDN state checkpointing for hybrid rollback
- Probabilistic rejection sampling (Leviathan/Chen + one-hot variant)

## 9. Future directions (not in scope)

### Approximate MTP / DFlash mode

After H5 ships, an opt-in "approximate" mode could explore tolerating
GDN state pollution on reject (no rollback) for higher throughput.
Constraints:
- Bounded polluted steps (at most K consecutive rejects without resync)
- Periodic clean resync (every K steps run snapshot+restore)
- Quality metrics: perplexity drift vs greedy ground truth across
  ~1000 generated tokens, on multiple workloads
- Surface as `--{mtp,dflash}-approximate`, fail closed by default

This trades bit-exact correctness for throughput. Not in v1 scope.

### Probabilistic rejection sampling derivation

H4 does not retain MTP draft probabilities (greedy-only), and DFlash
references in production (vLLM, spiritbuun) use a one-hot draft fast
path that doesn't need them either. The "one-hot draft" variant of
Leviathan/Chen reduces the acceptance test to
`log p_target(draft) > log u`, which is what vLLM's
`probabilistic_rejection_sampler_utils.py` uses for
`HAS_DRAFT_LOGITS=False`. This is a fast-path simplification, NOT
the canonical Leviathan correction (which requires draft probs).

H5 ships:
1. Canonical formulation (academic correctness) for users who can
   provide draft probs via a custom drafter
2. One-hot fast path (production behavior, validated against spiritbuun)
3. Test confirming both produce identical accept decisions when `q` is
   one-hot

## Reference sources

- **vLLM** `vllm/model_executor/models/qwen3_next_mtp.py:113-115` (forward),
  `vllm/v1/spec_decode/llm_base_proposer.py:677-680` (input shifting):
  authoritative MTP forward and proposer reference. Only working public
  reference for MTP execution.
- **Rapid-MLX** `~/code/dflash/dflash/model_mlx.py` — z-lab official
  MLX DFlash impl (471 LOC). Reading-only algorithmic reference.
- **llama.cpp** `~/code/llama.cpp/src/llama-arch.cpp:447-452` — NEXTN
  tensor name templates. Loads MTP tensors but never executes them.
- **HF transformers** — explicitly drops `mtp.*` weights at load
  (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`). Not a reference
  for MTP execution.
- **DFlash llama.cpp forks** — full comparative report at
  `/tmp/dflash-forks/REPORT.md`. Four forks, two recipes:
  - z-lab v1 (configurable): `ruixiang63/llama.cpp@dflash` (PR #22105),
    `spiritbuun/buun-llama-cpp@experiment/SD-089-pflash` (Apple Silicon
    production fork; ships GGUFs; 532 lines added to ggml-metal.metal)
  - Luce 5-layer fixed: `Luce-Org/lucebox-hub/dflash` (CUDA-only),
    `Leechael/llama.cpp-dflash-ggml` (Luce port + DDTree + bit-equal
    test harness)
  H5 targets the z-lab v1 recipe via spiritbuun's GGUFs.
- **Spec-decode checkpointing** in upstream llama.cpp: PR #19493,
  PR #22227. Pattern for hybrid-target rollback that H5.4 mirrors.

## Document history

- **rev 6 (current).** Prefill loop syntax fix (covers slot n-1).
  Terminal-return contract documented. GPU command-ordering contract
  on `draft`/`draft_kv_only`. Test matrix updated to reflect
  not-resumable terminal returns.
- **rev 5.** Inline-bridge design (lifetime-safe under session-arena
  hidden invalidation). Bootstrap simplified. Streamed prefill.
- **rev 4.** Restructured as straightforward spec.
- **rev 3.** DFlash recon. Reframed MTP as oracle-only with DFlash
  (H5) as perf target.
- **rev 2.** Lazy sequential verify. Cursor tracking.
  SpeculativeDecoder wrapper. Added MTP-KV prefill.
- **rev 1.** Initial draft.
