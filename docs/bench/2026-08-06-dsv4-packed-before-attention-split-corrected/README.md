# DeepSeek V4 Packed Before-Attention Split, Corrected

Status: separately frozen for one execution. This packet replaces no evidence
and inherits no timing sample from the held predecessor.

## Provenance

The predecessor at `../2026-08-06-dsv4-packed-before-attention-split/` emitted
`INCONCLUSIVE_HOLD` during its post-warm-pair warm-control identity assertion.
Its expected committed-token digest omitted one character and was therefore
only 63 characters. Both untimed warm arms completed, but it emitted no warm
timing values, timed arm, or performance direction.

The observed digest matched the canonical identity already retained by the
August 5 packed-attention packet:

```text
7c7d71da5df3b70e9d6f0555d949f7476acd5a68173331a228537c9e46dc728e
```

CX adjudicated the predecessor as immutable HOLD and authorized one separately
frozen corrected packet. This source changes only that canonical literal and
the preflight representation: every frozen 32-byte identity is now decoded by
a `const fn` requiring exactly 64 lowercase hexadecimal characters. Identity
comparisons use `[u8; 32]`, so malformed literals fail while compiling the
ordinary focused test binary, before Metal or model access.

## Unchanged Question

Split the accepted 477.570 ms lower bound around `BeforeAttentionBody` into:

1. `RawSetupAndHyperPre`: command ingress, raw-ring preservation, layer-zero
   embedding/repeat, and attention mHC pre.
2. `AttentionAndCompressorPrepare`: attention normalization, Q/KV preparation,
   and compressor batch projections.
3. `ChronologicalRows`: the complete 128-row RoPE, KV publication, and
   compressor-frontier loop.
4. `AfterChronological`: all remaining work through the pre-expert command end.

Only stages one and two above can authorize a static candidate-touched census.
The chronological stage remains closed by prior evidence; the final stage is a
validity sentinel. No result here can authorize a kernel, fusion, savings
claim, product change, or asset A/B.

## Frozen Fixture And Topology

- Base revision: `829de3d7bdb9f346ed84011ab5ee8441627390d1`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: current 97.05 GiB four-shard UD-IQ3_XXS GGUF
- Model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`
- Prefix: `[35, 201, 200, 34]` repeated to N=128
- Continuation: snapshot restore, then exact token ID 35
- Routing: Rust CPU routing plus qualified 25-layer grouped IQ2
- Ordinary topology: 86 encoders
- Sampled topology: 215 encoders
- Dispatches per arm: 47,990

Four serial sampled pre-expert encoders retain signed physical gap/overlap
accounting; the unchanged post-route command is the fifth encoder per layer.
Every physical stage requires `end > start`, and every layer must close:

```text
sum(stage duration) + sum(gap) - sum(overlap) = command duration
```

The accepted parent log remains
`../2026-08-06-dsv4-packed-pre-expert-residual-attribution/attention-split.log`
with SHA-256
`3fd96776530bd90903513aa790488cf314a81ce5a1ccfe96158b9d18a589278b`.

## Frozen Campaign And Gates

A fresh test process runs one untimed ordinary/sampled warm pair followed by
timed `A/B/A/B/A`, with fresh sessions sharing one residency. Neither
predecessor warm arm is inherited as evidence or conditioning for this
campaign. There is no filtering, retry, replacement, or excursion removal.
Wall time is diagnostic only.

All arms must preserve exact packed logits, normalized hidden output,
causal/prefix/compatibility identities, restored continuation, committed
tokens, dispatch count, and ordered full dispatch geometry. Evidence is valid
only with:

- ordinary and sampled GPU drift at most 5%;
- sampled-topology perturbation at most 10%;
- coverage error at most 2% per layer and 0.5% aggregate;
- transition ambiguity at most 5% each, 10% per layer, and 2.5% aggregate;
- stage repeat delta at most two percentage points;
- CSA/HCA stage repeat delta at most three points;
- parent aggregate reproduction within two points per sample and repeat delta
  at most two points; and
- parent CSA/HCA reproduction within three points per sample and repeat delta
  at most three points.

Normalize each sampled stage share against interpolated adjacent ordinary
controls. Charge the maximum of transition ambiguity, sampled perturbation, and
stage repeat delta. An actionable parent clears only with lower share at least
15%, lower time at least 158.3 ms, and mean CSA/HCA shares each at least 10%.
If both clear, select larger lower time, then larger lower share, then fixed
stage order. A valid no-qualifier result closes the lane. Any invalidity emits
`INCONCLUSIVE_HOLD` with no retry.

## Source Freeze

- Release diagnostics executable:
  `2a523c42079aab2c23a01631d955fbd9c894bf291785434784e6dcc0939e9e55`
- `deepseek_v4_metal.rs`:
  `e5f57e363beb2b1c7244dd4fbbf1cdcd56bdb57728a4500edbda40192fa56578`
- `deepseek_v4_metal/prefill.rs`:
  `308ff01bd47822bdb1990c7d0e17f7691f71fad984a051758ec62a55ff231c82`
- Frozen experiment diff:
  `a6090a656f58d26066aba6ca5d062600dc16e5bcb0f219f971eda991c331e725`
- Frozen boundary source:
  `5714dae28a0806a297cdb4fb969be7fa49719efb806c8fbccfa1837900d52c13`

The source diff, boundary slice, and checksums were frozen before execution.
CX session: `019fd685-acb3-7890-83c9-192cbea48c6e`.

## Frozen Command

```bash
set -o pipefail && QWEN_DSV4_PACKED_GROUPED_EXPERTS=auto \
  /usr/bin/time -l cargo test --release \
  -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::\
current_asset_packed_before_attention_split_attribution_packet \
  -- --ignored --exact --nocapture --test-threads=1 \
  2>&1 | tee \
  docs/bench/2026-08-06-dsv4-packed-before-attention-split-corrected/\
before-attention-split.log
```
