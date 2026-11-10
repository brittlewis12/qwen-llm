# Flash-Next native MTP: stage-1 artifact audit, 2026-11-09

Read-only research lane on `research/fn-mtp`, branched from main
`7613394d9`. Stage 1 only: no engine build, no GPU work, no M5 worktree.
The question was whether native Flash-Next MTP speculation pays for
single-agent decode on the M4, and for solo moments on the M5.

**Bottom line: the Flash-Next artifact both machines actually use has no MTP
head.** It declares 48 trunk blocks and no `nextn_predict_layers`, and its
1,224-tensor inventory contains zero `nextn`/`mtp` tensors. Published
Flash-Next MTP heads do exist, but as separate sidecar GGUFs (or grafted extra
shards), not inside the GSQ-RCO IQ3_S artifact. Changing artifacts is Britt's
call, so this lane stops here as the staged plan requires.

If a head-bearing artifact is adopted, the first arithmetic is **marginal on
the M5** (K=1/K=2 break-even acceptance about 0.68-0.76 against published
greedy acceptance 0.66-0.75, with agent sampling likely lower). **On the M4 the
answer is gated on a 2-3-row verifier the engine does not have**: with a cheap
multi-row verify it can pay; with serial verification it loses by construction.
The smallest proof is a stage-2 measurement probe on the M5, not a production
build, and no M4 build should start before that clears.

## Artifact audit

The artifact used by the Flash-Next standings is
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` (ISTA-DASLab, two shards). Both machines
hold the same bytes:

| Host | Path | Size |
| --- | --- | ---: |
| `zekrom` (M4 Max) | `/Volumes/wdblack/weights-archive/qwen3.8-flash-next-gsq-rco-iq3_s/IQ3_S/` | 78 GiB |
| `metagross` (M5 Ultra) | `~/models/ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/` | 78 GiB |

`scripts/research/fn-mtp/gguf_header.py` reads only metadata and the tensor
table. Full output: `header-audit.json` and `mtp-head-inventory.json` here.

| Shard | GGUF version | Tensors | KV keys | Tensor params |
| --- | ---: | ---: | ---: | ---: |
| `-00001-of-00002` | 3 | 1,223 | 67 | 125,743,653,760 |
| `-00002-of-00002` | 3 | 1 | 3 | 51,200,245,760 |
| total | | **1,224** | | **176,943,899,520** |

Relevant metadata on shard 0:

- `general.architecture = qwen4exp`
- `qwen4exp.block_count = 48`
- `qwen4exp.context_length = 262144`
- `qwen4exp.expert_count = 512`, `expert_used_count = 10`
- `qwen4exp.full_attention_interval = 4`
- `general.sampling.temp = 1.0`, `top_k = 20`, `top_p = 0.95`
- **no `qwen4exp.nextn_predict_layers`**
- **zero tensors matching `nextn`, `mtp`, `eh_proj`, `enorm`, `hnorm`, or
  `shared_head`**

The M5 copy at the path above produced an identical header: same version,
counts, metadata, parameter totals, and `nextn_predict_layers = null`. The
loader's MTP inventory exists (`crates/qwen-llm/src/qwen4exp_loader.rs:102`,
`:145-159`, `:506+`) but is dormant for this file; `ignored_mtp` is empty.

## What published artifacts carry the head

The popular main-model GGUFs do **not** embed MTP. The head ships beside them
or is grafted on as an extra shard:

| Source | File | Size | Notes |
| --- | --- | ---: | --- |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | `MTP/mtp-...-shared-Q8_0.gguf` | **2.79 GB** | recommended; borrows target embedding/output |
| same | `MTP/mtp-...-shared-Q4_K_M.gguf` | 1.91 GB | about 2 acceptance points lower |
| same | `MTP/mtp-...-shared-BF16.gguf` | 5.23 GB | larger and slower than Q8_0 |
| same | `MTP/mtp-...-Q8_0.gguf` / `Q4_K_M` / `BF16` | 4.14 / 2.79 / 7.77 GB | self-contained |
| `jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF` | `mtp-...-Q8_0.gguf` + `graft-mtp-shard.py` | 4.14 GB | adds a trailing shard, sets `nextn_predict_layers = 1`, `block_count = 49` |
| `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` | `unsloth-UD-Q4_K_XL-mtp-shards/` | +1.89 GB | pre-grafted five-shard Unsloth split |
| `agentionai`, `mxxm-t`, `drluoto` | standalone Q8_0 heads | 4.14 GB | community layouts, not all interchangeable |

The Unsloth head inventory is concrete: **32 tensors, 2,607,150,848
parameters**, `qwen4exp.block_count = 49`,
`qwen4exp.nextn_predict_layers = 1`, `nextn_shared_target_tensors = 1`. It is
one full QSA block (`attn_*`, `indexer.*`, 512-expert MoE with top-10,
hyper-connections) plus `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`,
`nextn.hc_head_*`. Its MoE has the same 512-expert / 10-active geometry as the
trunk, so a draft step reads only 10 of 512 routed expert banks.

The Unsloth `UD-Q3_K_XL` main split is not a candidate for an embedded head:
its first shard has **tensor_count = 0** (metadata only) and the head lives in
the separate `MTP/` directory. The same is true of the ISTA artifact: it has
1,223 data tensors in shard 0 and is not metadata-only.

**Loader compatibility.** `canonical_inventory` already accepts an appended
`blk.48.*` full-QSA MTP block with the `blk.48.nextn.eh_proj` signature and
excludes it into `ignored_mtp`
(`crates/qwen-llm/src/qwen4exp_loader.rs:506`, `:540-575`, and the
`..._appended_mtp` fixture). A **grafted full split** is therefore the
artifact shape the loader can already inventory. A sidecar that borrows the
target's embedding/output would need loader and runtime work before it looks
like the current model. Execution of the head is absent either way.

The cheapest loader-compatible route is a grafted split (Unsloth main + 4.14 GB
self-contained Q8_0 head) because its first shard is metadata-only. Grafting
onto the GSQ-RCO IQ3_S would require adapting the graft script to a split whose
first shard carries data tensors, or adding sidecar support. There is no
published GSQ-RCO-with-MTP artifact.

## Default sampling

Agents send no sampling fields, so the release preset is the effective
contract:

- `crates/qwen-llm/src/sampling.rs:84` `qwen38_flash_next`:
  **temperature 1.0, top_k 20, top_p 0.95, min_p 0**.
- `crates/qwen-cli/src/qwen/release_sampling.rs:63` selects it for
  `ReleaseIdentity::FlashNext`.
- The GGUF header independently declares the same
  `general.sampling.{temp,top_k,top_p}`.

This is the setting any acceptance measurement must use. Published Flash-Next
MTP acceptance numbers are mostly **greedy** (66.1% for Unsloth shared-Q8_0;
75.1% in one community greedy test; 44.6% for Q4_K_M on code/Pascal), so they
are an upper bound for agent traffic. The model card's non-thinking preset also
uses a presence penalty the sampler does not implement; the release
`generation_config.json` thinking preset is what this engine runs.

## Conditional economics (estimate, not measured on Flash-Next)

No Flash-Next head was executed in this lane. These are bounds built from the
measured engine numbers and the head's byte geometry. They are enough to judge
whether a stage-2 measurement is worth doing; they are not a substitute for it.

Measured inputs:

| Quantity | M5 Ultra | M4 Max | Source |
| --- | ---: | ---: | --- |
| Flash-Next 48K B=1 step | **13.619 ms** | not measured at 48K | PERF-LOG, "Flash-Next Widths..." |
| Flash-Next 48K B=2 / B=3 batched round | **20.919 / 25.667 ms** | no batched decode | same |
| Flash-Next B=1 8K / 32K decode | **86.72 / 86.45 tok/s** solo | **49.751 / 47.169 tok/s** = 20.10 / 21.20 ms | PERF-LOG, M4 GPU-loop entry |
| Streaming bandwidth | 1,175.47 GB/s | 506.96 GB/s | PERF-LOG, Nov 6 calibration |

`crates/qwen-llm/src/qwen4exp_runtime/decode_batch.rs:49` still refuses native
batch on anything but M5 with a Q6 head. On the M4 the only multi-row candidate
is the prefill/append path; no 2-3-row Flash-Next append is published (the
smallest measured append shape is 400 rows).

### Draft-step cost from bytes

For the shared-Q8_0 head with 10 of 512 experts active and the target's Q6_K
output projection borrowed, one draft step touches about:

| Part | Active bytes |
| --- | ---: |
| attention q/o/k/v | 54.6 MB |
| routed experts, 10/512 | 52.2 MB |
| shared expert | 5.2 MB |
| router (F32) | 5.3 MB |
| hyper-connections | 14.0 MB |
| indexer (BF16) | 3.3 MB |
| `nextn` projections | 20.9 MB |
| **head block subtotal** | **153.8 MB** |
| borrowed Q6_K output row | 521.5 MB |
| **total per draft step** | **675.3 MB** |

At the streaming calibrations that is 0.57 ms (M5) / 1.33 ms (M4) as a floor.
A small mixed head will not hit the streaming roof, so use 2-3x: **about
1.1-1.7 ms on the M5 and 2.7-4.0 ms on the M4** for the head alone. Add a CPU
draft draw (the GLM study measured 0.95 ms for a 154K row; Flash-Next's vocab
is 248K): **D = 2.2-3.2 ms M5, 3.7-5.5 ms M4** for head + draw.

### K=1 and K=2 arithmetic at B=1

`S` is the native forward, `V(m)` the m-row verify, `D` head + draft draw,
`a` one-step acceptance, and `a^2` the second-step acceptance. This follows the
GLM study's convention: target draws are excluded from both sides, the draft
draw is charged. Charging a target draw per verified row would shave a few
tenths of a point off the speculative side.

M5, `S = 13.619 ms`, `V(2) = 20.919`, `V(3) = 25.667` (the batched round is a
proxy for a causal verify; add ~0.5-1 ms of causal serialization):

| D | K=1 break-even `a` | K=2 break-even `a` |
| ---: | ---: | ---: |
| 2.0 ms | 0.683 | 0.695 |
| 2.5 ms | 0.720 | 0.725 |
| 3.0 ms | 0.756 | 0.755 |

For acceptance values we do not yet have for Flash-Next, this is what the
table says:

| a | K=1, D=2.5 | K=1, D=3.0 | K=2, D=2.5 | K=2, D=3.0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.930x | 0.911x | 0.870x | 0.843x |
| 0.66 | 0.965x | 0.945x | 0.931x | 0.901x |
| 0.75 | 1.018x | 0.996x | 1.027x | 0.995x |
| 0.80 | 1.047x | 1.025x | 1.084x | 1.049x |

So the M5 needs **at least ~0.72 acceptance just to break even**, and a
real 1.15x would need acceptance near 0.85-0.90 unless the verifier or head is
cheaper than the current shapes. The published greedy 0.66-0.75 bracket and
the lower acceptance expected at T=1 / top_k 20 make this marginal, not a clear
win. This is the main difference from GLM, whose measured 1.72 ms head and
18.60 ms serial step left room for 1.23-1.31x at a=0.80.

M4, `S = 21.2 ms` at 32K (48K is slightly worse):

| Multi-row verify assumption | K=1 break-even `a` | K=2 break-even `a` | At a=0.75 |
| --- | ---: | ---: | --- |
| Efficient: `V(2)=1.3S`, `V(3)=1.5S`, D=4.5 | 0.514 | 0.584 | K=1 **1.156x**, K=2 **1.202x** |
| Same, D=5.5 | 0.561 | 0.626 | K=1 1.121x, K=2 1.145x |
| Serial: `V(2)=2S`, `V(3)=3S`, D=4.5 | 1.212 | 1.135 | K=1 0.791x, K=2 0.675x |

The M4 verdict therefore depends entirely on whether 2-3 rows can be verified
for about 1.3-1.6x one token. If verification is K+1 serial forwards, the
method cannot win: it emits fewer than K+1 tokens while paying the K+1 forwards
plus the head. The 1.3S/1.5S row is an optimistic extrapolation, not a
measurement. This is the first thing a stage-2/3 lane must price.

### Head history and break-even length

The head needs its own state for the 48K prefix. A one-block QSA KV at
48K is roughly 100 MB, so memory is not the issue; the construction pass is.
The GLM one-block head history cost **3.88 s** for 47.7K rows. Flash-Next's
head is also one block, but it has a 512-expert MoE, so assume **1-3 s** until
measured. At the M5's marginal savings (about 0.2-0.9 ms per emitted token at
a=0.75-0.80), that is a break-even of roughly **2,000-12,000 emitted tokens**
in the central cases; the GLM study's comparable break-even was about 1,050
because its per-token saving was 3.7-4.7 ms. A separate head-prefill per
request would make short agent turns lose. The head history must be fused with
target prefill or reused across turns for this to be viable.

## Recommendation

1. **No build on the current artifact.** It cannot run MTP; changing artifacts
   is Britt's call. Do not add a speculative path to a model file that has no
   head.
2. **If Britt approves a head-bearing artifact**, the smallest proof is
   stage 2 on the M5, and it is a measurement probe, not an engine build:
   - Use a loader-compatible grafted split (Unsloth metadata-only first shard
     + 4.14 GB self-contained Q8_0 head) or adapt the graft to keep
     GSQ-RCO IQ3_S. Prefer keeping the tier-1 artifact if the graft can be
     made without changing its quantization.
   - Prefill one 48K frozen code prompt. Measure (a) head + draft draw per
     step, (b) one- and two-step acceptance at the real default
     T=1 / top_k 20 / top_p 0.95 and at greedy, (c) a causal 2- and 3-row
     verify through the native batched path, with the independent-batch number
     as a control, and (d) the head-history pass and its state.
   - Put the probe under `crates/qwen-llm/examples/` behind `perf-ablations`.
     Run K=1 first, then K=2; compute the K=1/K=2 arithmetic including the
     head-history break-even.
   - Success bar for any further build: **>=1.15x delivered inter-token at 48K
     on frozen code**, distribution-exact by fixed-seed replay against the
     non-speculative sampler, with the head-history pass amortized in a serve
     session. If the M5 probe is below ~1.10x, stop; do not build for the M4.
3. **Only if M5 clears**, price the M4 multi-row verifier. That proof must
   first show a 2-3-row verify cheaper than about 1.6S (or else the break-even
   acceptance exceeds 1.0). This is a verifier project, not a head project,
   and it is the real M4 gate.

Do not build K=2 or any multi-agent speculation first. On the M5, B=1 is the
only regime where speculation can even be considered; the batched B=3 round is
already 25.7 ms for three tokens, and a verify that blocks it loses.

## Limits

- Stage 1 only. No Flash-Next head was loaded or executed; no M4 or M5 build
  was made. The head cost, acceptance, and M4 verify rows are estimates.
- Published acceptance is text- and sampler-dependent and mostly greedy. The
  agent default T=1 / top_k 20 / top_p 0.95 will accept less; that is the
  number the stage-2 probe must measure.
- The B=2/B=3 verify numbers are independent-sequence batched decode, a
  conservative proxy for a causal multi-row verify. Causal serialization is
  not included.
- The M4 has no native batched decode. The prefill/append path may or may not
  make 2-3 rows cheap; no measurement exists below 400 rows.
- No GPU work was done, so no new thermal reading is claimed. Any stage-2 run
  must record thermal pressure and use the canonical Metal lease.

## Reproduction

```sh
python3 scripts/research/fn-mtp/gguf_header.py --json \
  /Volumes/wdblack/weights-archive/qwen3.8-flash-next-gsq-rco-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf

python3 scripts/research/fn-mtp/gguf_header.py --json --all-tensors \
  <path-to-mtp-head.gguf>
```

The M5 copy was checked with the same script over SSH; its header matched the
local one field for field. The `header-audit.json` file records the local
shards and the Unsloth metadata-only first shard; `mtp-head-inventory.json`
records the 32-tensor shared-Q8_0 head.

## References

- `crates/qwen-llm/src/qwen4exp_loader.rs` — MTP inventory and the
  `..._appended_mtp` fixture (`block_count` 49, `nextn_predict_layers = 1`).
- `crates/qwen-llm/src/sampling.rs:84` — Flash-Next release sampling.
- `crates/qwen-cli/src/qwen/release_sampling.rs:63` — release selection.
- `crates/qwen-llm/src/qwen4exp_runtime/decode_batch.rs:49` — M5-only native
  batch.
- `docs/PERF-LOG.md` — "Flash-Next Widths And Dispatch Dead Time Measured"
  (48K B=1/B=2/B=3), "Flash-Next M4: GPU Loop And Executor Carry Over"
  (M4 8K/32K), and the Nov 6 streaming calibration.
- `docs/bench/2026-11-09-spec-econ/README.md` — GLM NextN head cost,
  acceptance, and economics, the method reused here.
