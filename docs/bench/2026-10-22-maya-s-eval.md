# Maya-S artifact evaluation: +4.4% prefill at +0.080 held-text NLL

`peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF`, Maya-S, six shards,
339,477,953,988 bytes (339.5 GB). Same binary as the sink-tile ABBA
(`3e63384c` with `QWEN_V41_SINK_TILE=512`; the later `580b6aa0` default
changes no arithmetic). M5 Ultra 80-core, production lease. Raw:
`metagross:~/qwen-runs/2026-10-22-maya-s/`.

This is a quantization recipe for Project Maya expert offloading, not a
finetune or persona. `general.file_type` reports IQ1_S, but the routed
experts are not IQ1: layers 0-3 and 36-39 use IQ2_S gate/up and IQ3_S
down; layers 4-35 use IQ2_XXS gate/up and IQ3_XXS down. Attention,
shared experts, embeddings and output are Q8_0; norms are F32; the two
Engram tables (layers 1 and 14) are Q6_K and stay on the disk reader.
Architecture metadata matches the released V4.1 model: 40 layers, 384
experts, top-6, 1,048,576 context.

## Routing and load

At prefill the loader reports all 40 routed-expert layers on the
half-staged grouped route:

```
deepseek41: prefill rows=2048 routed experts: half_grouped=40 f32_grouped=0 native_rows=0 per_row=0
```

No layer falls back to native rows or per-row replay. The Engram tables
are validated and read by row, not mapped as resident weights; their
routing metadata (`deepseek41.engram.num_embeddings`, layers 1/14) matches
the tensor row counts.

## Throughput

`qwen-bench suite --pp 2048,4096 --runs 2 --seed 1`, A/B/B/A, means of
four samples:

| artifact | bytes | pp2048 | pp4096 |
|---|---:|---:|---:|
| Unsloth UD-IQ3_XXS | 394.7 GB | 755.6 | 723.8 |
| **Maya-S** | **339.5 GB (-14.0%)** | **788.5 (+4.4%)** | **755.3 (+4.3%)** |

Maya-S is 4.4% faster while its file is 14% smaller. The gap is smaller
than the byte reduction, but this probe does not isolate which stage
bounds the difference: file bytes include the disk-backed Engram tables,
and unchanged dense stages dilute any weight-streaming gain.

## Held-text quality

Six frozen documents, 512-token prefix, 64 teacher-forced targets each,
same tokenizer and target IDs as the IQ3 packet; only the packet's model
record (path/size/shards) was replaced for Maya-S. IQ3 witness and Maya-S
witness were scored with the same clean committed build.

| document | IQ3 NLL | Maya-S NLL | delta | IQ3 top-1 | Maya-S top-1 |
|---|---:|---:|---:|---:|---:|
| prose_1 | 2.604260 | 2.624313 | +0.020053 | 0.594 | 0.594 |
| prose_2 | 4.204816 | 4.349725 | +0.144909 | 0.188 | 0.172 |
| prose_3 | 2.260082 | 2.451706 | +0.191624 | 0.453 | 0.453 |
| code_4 | 1.252495 | 1.330226 | +0.077731 | 0.719 | 0.688 |
| code_5 | 1.570008 | 1.620570 | +0.050561 | 0.688 | 0.656 |
| code_6 | 0.986541 | 0.979620 | -0.006921 | 0.766 | 0.750 |
| mean | 2.146367 | **2.226027** | **+0.079659** | 0.5677 | 0.5521 |

Mean target rank moves 50.38 -> 53.44. The probe is descriptive document
means over 384 targets, not independent token trials.

## Interpretation

Maya-S is a viable memory/speed trade on disk: 55 GB smaller than
UD-IQ3_XXS (339.5 GB vs 394.7 GB) and 4.4% faster at both prefill shapes,
at +0.080 mean NLL (+3.7% relative) and -1.6 points top-1 on this probe.
Resident memory was not measured; the byte count is file size, not a
wiring or page-cache claim. Maya-S is not a quality-equivalent substitute
for IQ3. Maya-XS (IQ1_S routed experts) was not evaluated: the grouped
path rejects IQ1_S and the model card reports +8.7% NLL.

## Limits

- One M5 Ultra, one seed, one six-document probe; no independent quality
  corpus and no long-context or decode evaluation.
- The packet's documents/targets are frozen; substituting the model record
  is a path/size check bypass, not an artifact hash check. The helper
  compares the tokenizer summary metadata (vocab 129280, gpt2, pre) but
  not the full vocabulary/merges; target IDs are packet values and are
  never re-tokenized.
- Maya-S's Engram tables are Q6_K; the PLE read path was exercised by the
  quality run, but Engram-row throughput was not timed separately.
