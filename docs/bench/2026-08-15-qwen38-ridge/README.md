# Qwen3.8 27B Ridge Kick-The-Tires Packet

Status: Ridge is a credible new Pareto point for ordinary Qwen3.8 text
inference. It removes about 27% of base-weight bytes versus Q4_K_M, the
maintained retention guardrail remains at its ceiling, and the supplied
same-box packet reports about 5-6% faster decode with only a 2-4% prefill loss.
The quality evidence is encouraging but still too small for a general
capability-equivalence claim.

This packet also closes two engine cliffs exposed by the mix: Q6_K embeddings
now stay native, and physical-N2 IQ2_S/IQ3_S FFNs no longer execute a
32-column matrix tile to retain two columns.

## Asset

- Repository: `empero-ai/Qwen3.8-27B-Ridge-GGUF` at revision
  `578362007e830185e1a03ff3454309c8590bad5f`.
- File: `Qwen3.8-27B-Ridge-3.7bpw.gguf`, 12,599,187,008 bytes.
- SHA-256:
  `95580dbdaad579582ee898257116abc18d7f3625a00c16a15735d41444a09f5e`.
- Base model: official `Qwen/Qwen3.8-27B` revision `1d4bf0f2` according to
  the Ridge model card.
- GGUF: `qwen35`, 866 descriptors, 851 bound base requests, and 15 attached
  MTP descriptors. Base source weights total 12,249,470,976 bytes; the MTP
  inventory adds 338,720,768 bytes.
- The model card describes Q6_K embedding/head, Q8_0 GDN state and norms,
  Q4_K GDN mixers, Q5_K-class attention, IQ3_S edge FFNs, `IQ2_M` bulk FFNs,
  and a Q6_K MTP block. The GGUF descriptors resolve that bulk type to GGML
  `IQ2_S`; descriptor inspection, not the prose label, governs dispatch.

The model card reports 3.69 bpw, 11.73 GiB, and Wiki-style perplexity
`7.82 +/- 0.14` versus `7.15 +/- 0.12` for its BF16 conversion. That is a
useful fidelity disclosure, not a same-suite ranking against Q4_K_M.

## Storage Promotion

Ridge's `[5120, 248320]` token embedding is Q6_K. The shared Metal row
dispatcher already had a real Q6_K kernel used by DeepSeek V4, but the Qwen
policy still rejected it and materialized one 5,085,593,600-byte F32 tensor.

The current WIP default retains the 1,042,944,000-byte Q6_K source directly:

| Metric | F32 rollback | Native Q6_K |
|---|---:|---:|
| Embedding resident bytes | 5,085,593,600 | 1,042,944,000 |
| Base resident/source bytes | 16,292,120,576 | 12,249,470,976 |
| Converted tensors | 1 | 0 |
| Bytes removed | - | 4,042,649,600 (3.76 GiB) |

`ridge-q6-auto-storage-plan.json` records production-auto selecting all 851
base requests directly. The real-row oracle is bit-exact against CPU Q6_K
dequantization. The native and forced-F32 ordinary runs produce the same
33-token trace, SHA-256
`73c5ab3739f1b598d5d91f602a95bd0643266fac87e5bf8505d9c2631a4b198a`.
The observed steady decode is neutral in that pair, as expected for a
one-row lookup; the leverage is load-time work and 3.76 GiB of memory.

Evidence:

- `ridge-q6-auto-storage-plan.json`
- `ridge-q6-auto-decode.log`
- `ridge-q6-f32-rollback-decode.log`
- `ridge-exactness-tests.log`

## Low-Bit N2 Geometry

The first packed D1/N2 run exposed a quant-specific false ceiling. Ridge routes
160 FFN projections through IQ2_S and 32 through IQ3_S. Their generic matrix
kernels own a physical 32-column tile, so N2 performed most of an N32 work unit
and discarded 30 columns.

The WIP-defaulted NC2 kernels map the two query rows onto independent simdgroup
cohorts inside one dispatch while reusing the mature singleton arithmetic and
row-major output contract. They are defaulted only for N2 and the measured
`[5120,17408]` / `[17408,5120]` dense-27B FFN shapes. Rollback remains:

```sh
QWEN_MATMAT_IQ2_S_N2_NC2=0
QWEN_MATMAT_IQ3_S_N2_NC2=0
```

Both orientations and both dtypes are bit-exact against two singleton rows in
the real Ridge fixture and a non-ignored synthetic offset/guard test.

| 64-token D1/N2 row | Generic tiles | NC2 default |
|---|---:|---:|
| Acceptance | 0.882 | 0.882 |
| Verifier ms | 5,240.6 | 2,255.4 |
| Verifier speedup | 1.000x | 2.324x |
| MTP total ms | 5,776.2 | 2,791.7 |
| Ordinary target total ms | 2,700.4 | 2,701.0 |
| End-to-end MTP speedup | 0.468x | 0.968x |

The two packets have the same 64 target IDs, token-stream SHA-256
`fe946617b69555e9f1206f983e51edaf9021cf2fed71bf12b3d9590b2072b02a`,
and passing numerical 16-step terminal-resume audits. The projection kernels are
operator-bit-exact against singleton rows; the packed MTP packet establishes
greedy-semantic equivalence and numerical resume, not bitwise terminal-state
identity. The smaller/faster target raises the speculation bar, so MTP remains
off as a product policy.

Both timing arms are explicit dirty-tree discovery packets with `allow_dirty`.
The 2.324x phase delta is far above the packet MDE and the exactness tests pass,
so the narrow kernels are default-on in this WIP. A clean committed confirmation
is still required before release-level promotion.

Evidence:

- `ridge-lowbit-n2-candidate.log` / `.json`
- `ridge-lowbit-n2-rollback.log` / `.json`

## Quality And Effort

The maintained 48-request injected-label guardrail remains ceilinged:

| Cell | Retained | Flipped | Unparseable | Strict |
|---|---:|---:|---:|---:|
| Misleading false rebuttal | 24 | 0 | 0 | 24/24 |
| Neutral double-check | 24 | 0 | 0 | 24/24 |

Its output SHA-256 is
`47253a5c79cc4ed98da0c2a45d8145ff6aaa05566ddd50da11257f25e8fcd9b7`,
exactly the retained Qwen3.8 Q4_K_M output hash. This is a narrow retention
guardrail, not general quant equivalence. The supplied math result and retained
code, prose, and game-DM probes are qualitatively encouraging; the prose log
also demonstrates that default thinking can exhaust a 4,096-token cap before
completing a 180-word answer. The math producer output was not retained. The
other original logs are retained as `exploratory-*`; they predate Q6_K embedding
support and carry no source-state stamp, so they inform follow-up design rather
than promotion.

The CLI now exposes only the exact upstream Qwen3.8 effort set:

```sh
qwen run -m MODEL --reasoning-effort low --user '...'
qwen run -m MODEL --reasoning-effort medium --user '...'
qwen run -m MODEL --reasoning-effort xhigh --user '...'
```

Omission remains xhigh. Low injects the upstream brief-thinking instruction;
medium opens thinking without an effort instruction. `high`, unknown values,
raw prompts, `--no-thinking`, and non-Qwen3.8 identities fail closed. The small
smoke packet reaches EOS and the exact visible literal `EFFORT_OK` under all
three tiers. It validates transitions, not a quality/latency ranking.

Evidence:

- `retention-summary.json`
- `retention-run.json`
- `retention-stderr.log`
- `retention-outputs.jsonl`
- `retention-scored.jsonl`
- `qwen38-ridge-effort-smoke.log`
- `exploratory-code.txt`
- `exploratory-prose.txt`
- `exploratory-game-dm.txt`

## Throughput Read

The supplied same-box, warmed-binary comparison against Qwen3.6 27B Q4_K_M
reports:

| Shape | Ridge / Q4_K_M |
|---|---:|
| 6.5K prefill | 0.956x |
| 6.5K decode | 1.058x |
| 32K prefill | 0.979x |
| 32K decode | 1.050x |

This direction is architecturally plausible: IQ2_S moves fewer FFN bytes per
decode token but costs more unpacking per byte during packed prefill. The raw
throughput producer files were not retained in this repository, so this table
is an operator-supplied disposition rather than a promotion-grade engine packet.

## Low-Bit Optimization Screen

A 2026-08-16 optimization-only follow-up localized the current low-bit ceiling
before expanding capability work. The serialized `pp1024`, chunk-1024 baseline
reports 4,367.5 ms median wall, 4,113.7 ms median GPU, and 234.46 token/s. The
three FFN projections are each about 13.7-14.3 ms/layer and together with FFN
elementwise work account for approximately 67% of GPU time.

Two isolated prototypes were implemented and then removed:

| Candidate | `[5120,17408]` | `[17408,5120]` | Disposition |
|---|---:|---:|---|
| IQ2_S N64 prefill tile | `13.724 -> 13.714 ms` (`1.001x`) | `13.828 -> 13.799 ms` (`1.002x`) | KILL below 1.08x |
| IQ2_S direct-level decode repack | `0.10053 -> 0.09454 ms` (`1.063x`) | `0.11277 -> 0.10617 ms` (`1.062x`) | KILL below 1.12x |

The N64 tile was bit-identical to N32 and halved weight-panel decode, so its flat
result is direct evidence that packed IQ2_S is MMA-bound at this geometry. The
decode repack removed codebook/index reconstruction and raised charged physical
throughput to 391.8/348.0 GB/s, but expanded bytes by 1.298x/1.294x and differed
from native output by 1-2 ULP (max absolute `1.49e-8` / `2.98e-8`). Its byte cost
erased most of the instruction-path gain.

The same-process singleton census puts Ridge gate/up at 12.842 ms and down at
7.395 ms, for 20.237 ms of low-bit FFN projections in a 36.77 ms decode phase.
That is a real optimization target, but the next candidate must delete work or
use a materially denser exact representation; another N64 tile or larger direct
sidecar is closed. These rows are dirty-tree discovery evidence, not a clean
promotion packet, and no failed prototype remains in production source.

## Decision

1. Use Ridge as the leading interactive Qwen3.8 Pareto candidate: much smaller,
   directionally faster at decode, and clean on the maintained guardrail.
2. Keep Qwen3.8 Q4_K_M as the higher-bit capability/prefill anchor and Qwen3.6
   Q4_K_M as the regression anchor until a paired executable packet establishes
   the capability trade.
3. Retain native Q6_K embeddings as the current WIP default. Real selected rows
   are operator-bit-exact, the path removes 3.76 GiB, and it adds no universal
   inference work.
4. Retain the shape-gated IQ2_S/IQ3_S NC2 kernels provisionally, require a clean
   committed confirmation, and keep MTP itself unpromoted at 0.968x.
5. Treat effort as a request policy. The retained three-tier smoke validates
   exact transitions but does not rank completion, correctness, or latency, so
   no universal default is selected.
6. Keep vision separate. The local BF16 projector does not authorize image
   execution in qwen-llm.

The timed qwen-bench logs use build source state
`git-source-sha256-v2:ca3e7b2efa0a9355e9c74297fd49db873e9dac122d607cd66be4eaeabc299e25`;
the later exactness test log records
`git-source-sha256-v2:8d0281d3224f15913e835d50054c7018c8e606319684bce5191a5e753eb6b51a`.
Operator observation: no run used `MTLResidencySet`, pre-wiring, `mlock`,
cache-bypass reads, or the residency-coupled A10B path. The retention manifest
independently records disabled residency/prefetch for that child. The user-owned
llama server remained untouched.
