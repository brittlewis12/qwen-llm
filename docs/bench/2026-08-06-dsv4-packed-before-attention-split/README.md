# DeepSeek V4 Packed Before-Attention Split

Status: frozen for one current-asset execution. This packet only decides
whether one pre-attention parent merits a static candidate-touched census. It
cannot authorize a kernel, fusion, savings claim, product change, or asset A/B.

## Question

The accepted packed-attention packet left a conservative 477.570 ms lower
bound around `BeforeAttentionBody`. That containing interval includes several
unrelated families and is not itself removable cost. Split it once, without
adding sampled encoders, to ask whether either actionable parent retains the
158.3 ms current optimization floor:

1. `RawSetupAndHyperPre`: command ingress, raw-ring preservation, layer-zero
   embedding/repeat, and attention mHC pre.
2. `AttentionAndCompressorPrepare`: attention normalization, Q/KV preparation,
   and compressor batch projections.
3. `ChronologicalRows`: the complete 128-row RoPE, KV publication, and
   compressor-frontier loop.
4. `AfterChronological`: all remaining attention, output, hyper, FFN, and
   router work through the pre-expert command end.

The chronological interval remains closed by its prior packet. The fourth
stage is a coverage sentinel. Only stages one and two above can receive static
census authority.

## Frozen Instrument

The diagnostics-only stage recorder uses four serial compute encoders inside
each unchanged pre-expert command. The unchanged post-route command contributes
the fifth sampled encoder per layer. Therefore:

- ordinary topology is 86 encoders;
- sampled topology is 215 encoders;
- every arm retains exactly 47,990 dispatches; and
- adjacent physical timestamp envelopes retain signed gap/overlap accounting.

Every physical stage must have `end > start`. Each layer must satisfy:

```text
sum(stage duration) + sum(gap) - sum(overlap) = command duration
```

Stages zero through two must reproduce the accepted `BeforeAttentionBody`
aggregate and CSA/HCA cohort shares. The accepted parent log is
`../2026-08-06-dsv4-packed-pre-expert-residual-attribution/attention-split.log`
with SHA-256
`3fd96776530bd90903513aa790488cf314a81ce5a1ccfe96158b9d18a589278b`.

## Fixture And Campaign

- Base revision: `829de3d7bdb9f346ed84011ab5ee8441627390d1`
- Device: Apple M4 Max
- OS: macOS 15.6.1 (24G90)
- Rust/Cargo: 1.97.1
- Asset: current 97.05 GiB four-shard UD-IQ3_XXS GGUF
- Model content ID:
  `ae11d1ea13ccfd98509d248705a589384412cd67c502450158f84a8bd143b5e2`
- Prefix: `[35, 201, 200, 34]` repeated to N=128
- Continuation: restore the captured snapshot, then exact token ID 35
- Routing: Rust CPU routing plus the qualified 25-layer grouped-IQ2 policy

One untimed ordinary/sampled warm pair precedes timed `A/B/A/B/A`, where A is
ordinary and B is sampled. Every arm starts from a fresh session sharing one
residency. There is no filtering, retry, replacement, or excursion removal.
Wall time is diagnostic only.

## Frozen Gates

All warm and timed arms must preserve exact packed logits, final normalized
hidden output, causal/prefix/compatibility identities, restored continuation,
committed tokens, dispatch count, and ordered full dispatch geometry.

Evidence is valid only when all of the following hold:

- ordinary and sampled GPU drift at most 5%;
- sampled-topology perturbation at most 10% per sampled arm;
- timestamp coverage error at most 2% per layer and 0.5% aggregate;
- transition ambiguity at most 5% per transition, 10% per layer, and 2.5%
  aggregate;
- every stage repeat-share delta at most two percentage points;
- every CSA/HCA stage repeat-share delta at most three points;
- parent aggregate reproduction within two points per sample and repeat delta
  at most two points; and
- parent CSA/HCA reproduction within three points per sample and repeat delta
  at most three points.

For each stage, normalize each sampled share against its interpolated adjacent
ordinary controls. Use the conventional median of the two normalized values:

```text
common_uncertainty = max(transition_uncertainty,
                         abs(sampled_topology_perturbation))
stage_uncertainty = max(common_uncertainty,
                        abs(stage_share_0 - stage_share_1))
lower_share = mean(stage_shares) - stage_uncertainty
lower_ms = median(normalized_ms)
           - stage_uncertainty * ordinary_gpu_median_ms
```

An actionable parent clears only with lower share at least 15%, lower time at
least 158.3 ms, and mean CSA and HCA shares each at least 10%. If both clear,
select larger lower time, then larger lower share, then fixed stage order.
If neither clears validly, close this decomposition lane. Any identity,
stability, coverage, repeat, reproduction, or execution failure is
`INCONCLUSIVE_HOLD` with no automatic retry.

## Source Freeze

- Release diagnostics executable:
  `3d5ab605fe51fa349f95bee9599ff93bfa956fdde99088179eff71886ff7edbb`
- `deepseek_v4_metal.rs`:
  `0faf7762b88a853654c6bf71a7ad5ea076abb7c501e55b851b721f66558df43c`
- `deepseek_v4_metal/prefill.rs`:
  `308ff01bd47822bdb1990c7d0e17f7691f71fad984a051758ec62a55ff231c82`
- Frozen experiment diff:
  `0a1e56741a37a6c3a350d709b3508a56496f60cb4ec5c40f7669cf6737731919`
- Frozen boundary source:
  `5714dae28a0806a297cdb4fb969be7fa49719efb806c8fbccfa1837900d52c13`

`experiment-source-pre-run.diff`, `boundary-source.txt`, and
`checksums-pre-run.log` retain these artifacts before execution. CX gave the
static pre-execution GO in session
`019fd685-acb3-7890-83c9-192cbea48c6e`.

## Frozen Command

```bash
set -o pipefail && QWEN_DSV4_PACKED_GROUPED_EXPERTS=auto \
  /usr/bin/time -l cargo test --release \
  -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::\
current_asset_packed_before_attention_split_attribution_packet \
  -- --ignored --exact --nocapture --test-threads=1 \
  2>&1 | tee \
  docs/bench/2026-08-06-dsv4-packed-before-attention-split/\
before-attention-split.log
```
