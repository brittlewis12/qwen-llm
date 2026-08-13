# DeepSeek V4 Singleton Decode Dispatch Census and Fusion Design

Status: analysis packet, no GPU run and no code change. Source-derived census of
the one-encoder singleton decode path at a mid-window sparse position,
reconciled exactly against the measured 2,146-dispatch trace from the N2
verifier-floor packet. Prices the fusion family that item 5 of the DS4 queue
requires before front-end work resumes.

## Census

Path: `forward_token` -> `forward_token_inner_collapsed` -> `encode_token_layer`
x43 in one serial encoder. Per-layer dispatch counts at position ~2,384
(sparse CSA active, compressor frontiers mid-window, no publication boundary):

| Layer kind | Dispatches | Composition |
|---|---:|---|
| CSA (x21) | 55 | mHC pre 4, prepare 10, attn+indexer frontiers 6, indexer prep 5, scoring 1, selection 1, selected attention 1, inverse RoPE 1, output 9+1, mHC post 1, mHC pre (ffn) 4, router 3, routed experts 2, shared expert 4, combine 2, mHC post 1 |
| HCA (x20) | 45 | same minus indexer/scoring/selection; single ratio-128 frontier 3; cooperative dense attention 1 |
| SlidingWindow (x2) | 42 | no frontier; dense attention 1 |
| Layer-0 extras | 2 | embedding get-rows, hyper initial repeat |
| Tail | 5 | hc head 3, final norm 1, logits GEMV 1 |

`21x55 + 20x45 + 2x42 + 2 + 5 = 2,146` — exact match to the measured trace.
Boundary tokens add +11/CSA layer every 4th token (+231) and +4/HCA layer every
128th. The all-slot routed path is the 2-dispatch fused variant, so routed
experts are already at their dispatch floor: the 10.51 ms routed stage is
compute, not launch, and is out of scope for this family.

## Structural observations

- Roughly 800-900 of 2,146 dispatches are elementwise or utility kernels
  touching at most ~16K elements. The `hc_controls` Sinkhorn kernel is a
  1-thread, 1x1x1 dispatch launched twice per layer (86/token). The hash-route
  kernel is literally one thread. At the ~5 us effective serialized cost per
  dependent dispatch implied by stage arithmetic, the tiny-dispatch population
  accounts for ~4-6 ms of the 47.8 ms K160 token and most of the 8.6 ms gap to
  llama.cpp's 39.2 ms.
- Decode `encode_attention_output` is 1 inverse RoPE + 8 sequential per-group
  GEMVs (zero-copy views, no pack/unpack) + output-B = 10 dispatches/layer,
  430/token, on a stage measured at 7.89 ms whose weight traffic prices below
  1 ms. The packed-prefill 25-dispatch pack/unpack loop is a separate path.
- The prepare stage (7.68 ms measured) is 10 dispatches/layer of small
  norm->GEMV->norm->GEMV->RoPE->publish chains whose intermediate tensors have
  exactly one consumer each.

## Fusion family, priced

All contracts bitwise-preserving unless noted; per-token dispatch savings and
priced wall estimates at ~5 us effective cost plus stage-specific efficiency
recovery:

| # | Fusion | Dispatches saved | Est. wall | Contract |
|---|---|---:|---|---|
| F1 | Output: one z=8 grouped GEMV with inverse-RoPE load transform + output-B | 344 | 2.5-4.0 ms | bitwise (per-group dot order unchanged; RoPE moved from in-place store to load transform of the same values) |
| F2 | mHC pre: rms -> 16384->24 GEMV -> controls -> collapse in 1-2 kernels, x2/layer | 172-258 | 0.8-1.5 ms | bitwise if reduction orders preserved |
| F3 | Prepare: norm-into-GEMV staging, paired q/kv RoPE, RoPE+ring-publish fusion | 250-300 | 1.2-2.0 ms | bitwise for RoPE/publish pair; norm-epilogue variants numerical if reassociated, keep norm reduction intact for bitwise |
| F4 | Compressor frontier: dual-output GEMV + APE write epilogue, 3->1 per frontier | 124 | 0.6-1.0 ms | bitwise |
| F5 | Indexer prep: RoPE+Hadamard one pass; head-weight scale folded into GEMV epilogue | 40-60 | ~0.3 ms | bitwise |
| F6 | Shared expert: clone existing fused gate_up_swiglu, 4->2 | 86 | ~0.4 ms | bitwise (pattern already deployed for routed) |
| F7 | Combine: weighted-sum + add (+ optional hc_post) 3->1 | 43-86 | ~0.4 ms | bitwise |
| F8 | Router: rms -> gate GEMV -> route epilogue 3->1-2 | 43-86 | ~0.3 ms | bitwise |

Composed: 2,146 -> ~1,000-1,150 dispatches/token and a priced 6-10 ms of the
47.8 ms K160 token, consistent with the llama.cpp gap. This is also directly
the DSpark verifier denominator: every millisecond here moves the N=2 packet
floor toward the 58.34 ms external budget.

## Order of attack

F1 first: largest single saving, cleanest contract, and its stage is the most
over-measured relative to traffic. Then F2 (trivial kernels, pure launch
recovery), F6 (clone), F3, F4; F5/F7/F8 as batched cleanup. Each lands behind
its own env rollback with a bitwise differential at held positions
(2051/2052/3071 plus one boundary token for F4) before any timing bracket.

Gate per item 5's standing rule: structural deletions totaling >=2 ms/token
before local retunes; this family prices 3-5x that bar.

## Provenance

Census derived from `deepseek_v4_metal.rs` at `ef3fc76` (call-site line
citations retained in the working session); dispatch total reconciled against
the N2 verifier-floor trace (2,146 dispatches, one encoder) and the K160
five-way stage attribution (prepare 7.68 / output 7.89 / attention core 11.58 /
routed 10.51 / shared 2.98 ms).
