# Qwen3.8 27B Text Launch Gate

Status: the pinned Unsloth Qwen3.8-27B Q4_K_M asset is usable through the
ordinary text CLI, including its upstream default-xhigh and no-thinking prompt
transitions. Native quantized token embeddings remove an avoidable 4.07 GiB
private expansion. The maintained retention guardrail passes, but current data
does not establish a general quality or throughput win over Qwen3.6.

Vision, developer roles, structured tool calls, tool-result messages, response
formats, and production MTP remain outside this launch surface.

## Asset And Scope

- Repository: `unsloth/Qwen3.8-27B-GGUF` at revision
  `4604b899a826000505a834e623272db5b7fd62f6`.
- File: `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,773,984 bytes.
- LFS SHA-256:
  `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`.
- GGUF architecture: `qwen35`, 866 tensor descriptors, 65 declared blocks,
  one `nextn_predict_layers` block, and no unbound descriptors.
- Bound text backbone: 64 base layers, 48 GDN plus 16 full attention, hidden
  5,120, FFN 17,408, vocabulary 248,320, and declared context 262,144.
- Attached MTP inventory: 15 descriptors and 289,527,808 bytes. It is not
  traversed by ordinary `qwen run` generation.

The model card's multimodal claim does not make this binary a vision runtime.
No projector was acquired, and qwen-llm does not implement image input, vision
encoding, or multimodal RoPE for this surface.

## Prompt Contract

Commit `1ac9446` added a fail-closed Qwen3.8-27B identity and the upstream
ordinary-chat transitions.

- No-thinking produced the exact requested literals `QWEN38_MODERN_OK` and,
  after native embedding promotion, `AUTO_NATIVE_OK`, both at EOS.
- Default mode injected the exact xhigh instruction and opened `<think>`. A
  512-token-cap run closed `</think>`, reached EOS after 174 generated tokens,
  and returned the visible answer `FINAL: 4` for `2 + 2`. A 128-token cap had
  reached the output limit while still reasoning, so xhigh requires a materially
  larger answer budget than no-thinking even on a trivial task.
- Multi-turn assistant history receives the upstream empty-thinking wrapper
  when structured reasoning history is absent.

These checks establish ordinary text rendering and answer transition, not low or
medium reasoning modes, tool use, vision, or agentic coding quality.

## Storage And Launch

The two dense assets have exactly the same 16,806,250,496 base-weight source
bytes and 851 base storage requests. Qwen3.8's additional tensor bytes are its
detached MTP head.

| Metric | Qwen3.6 27B Q4_K_M | Qwen3.8 27B Q4_K_M |
|---|---:|---:|
| Parameters reported by bench | 26,895,998,464 | 27,320,697,856 |
| Tensor bytes reported by bench | 16,806,250,496 | 17,095,778,304 |
| Base source bytes | 16,806,250,496 | 16,806,250,496 |
| MTP descriptor bytes | 0 | 289,527,808 |
| Native Q4_K token embedding before this change | yes | no |
| Converted embedding resident bytes before this change | 0 | 5,085,593,600 |

The Qwen3.8 embedding source is a supported 715,161,600-byte Q4_K matrix. The
old production-auto fingerprint rejected it only because an MTP head was
present, even though MTP uses the same dtype-dispatched `get_rows` path. Removing
that artificial coupling changes the load ledger from one 5,085,593,600-byte
F32 conversion to one direct 715,161,600-byte Q4_K tensor. This removes
4,370,432,000 private bytes, or 4.07 GiB.

Live ordinary inference and the MTP equivalence probe both passed with the
native tensor. Observed warm-filesystem launch moved from about 2,846.7 ms before
promotion to 2,058.4 ms in the maintained 48-request run, a directional 788.3 ms
or 27.7% reduction. This is not a process-cold paired timing packet, so the
memory accounting is authoritative and the launch delta is supporting evidence.

## Throughput Comparison

The first A-then-B suite and the reversed B-then-A suite used the exact
`1ac9446` release benchmark, built-in warmups, three timed repetitions per row,
serialized GPU execution, and build source state
`git-source-sha256-v2:4883d07cf5f8a21344944eeebe32e66c73ed28a3c6fb3daa380cc710b934d63f`.
The large absolute drift between campaigns makes the first relative speed
impression non-authoritative.

| Campaign / order | Shape | Qwen3.6 tok/s | Qwen3.8 tok/s | 3.8 / 3.6 |
|---|---|---:|---:|---:|
| A/B: 3.6 then 3.8 | pp512 | 241.67 | 198.80 | 0.823x |
| A/B: 3.6 then 3.8 | pp2048 | 207.98 | 210.65 | 1.013x |
| A/B: 3.6 then 3.8 | pp8192 | 178.11 | 197.33 | 1.108x |
| A/B: 3.6 then 3.8 | tg128 | 18.23 | 19.71 | 1.081x |
| B/A: 3.8 then 3.6 | pp512 | 244.38 | 241.36 | 0.988x |
| B/A: 3.8 then 3.6 | pp8192 | 220.94 | 217.48 | 0.984x |
| B/A: 3.8 then 3.6 | tg128 | 24.49 | 23.11 | 0.944x |

Decision: treat Qwen3.8 as throughput-parity-ish pending a clean interleaved
product packet. Do not claim an engine speed win and do not start local kernel
retuning: the base geometry and base bytes are unchanged. Producer-native raw
stdout was not retained before this report; the transcribed table preserves the
disposition but is explicitly not a replayable promotion packet.

## Maintained Quality Guardrail

The v4.1 injected-label battery ran Qwen3.8 under the honest family label
`qwen38` while reusing the exact frozen Qwen3.6 empty-thinking request profile.
The packet and request hashes therefore remain unchanged.

- Packet SHA-256:
  `9345f2a08175e8adb4733cb2a147e4c001b75d4259ced2cc66d43049317d7200`.
- Request SHA-256:
  `b622df3d16e02af6699cf3d0d0981c216ace2ee007804883ed93be49bf53c58b`.
- Output SHA-256:
  `47253a5c79cc4ed98da0c2a45d8145ff6aaa05566ddd50da11257f25e8fcd9b7`.
- Binary SHA-256:
  `681443f4970ed3d0c9776f5fea57f392dc2bba3c39d83d20427b69f784237aea`.
- Model locator SHA-256:
  `9644e84d69f3304670d85b53e35826a834dc7e1b9a15723b379a574a48d1db6f`.

| Cell | Retained | Flipped | Unparseable | Strict |
|---|---:|---:|---:|---:|
| Misleading false rebuttal | 24 | 0 | 0 | 24/24 |
| Neutral double-check | 24 | 0 | 0 | 24/24 |

This matches the historical Qwen3.6 aggregate and is therefore a launch
guardrail, not a discriminator. The battery does not establish arithmetic,
belief revision, coding, agentic behavior, or general model quality.
Producer-native evidence is retained in
`qwen38-q4km-retention-summary.json`,
`qwen38-q4km-retention.outputs.jsonl`, and
`qwen38-q4km-retention.stderr.log`; provenance and scored rows are retained in
`qwen38-q4km-retention-run.json` and
`qwen38-q4km-retention.scored.jsonl`.

## MTP Economics

The first implementation did not test ordinary D1 speculation: it target-ran the
carry token and then target-ran an accepted draft serially. Acceptance therefore
cancelled out of its speed equation. Its 0.788x result closes that legacy lazy
topology, not D1 itself.

The replacement verifies `[carry, draft]` in one physical N2 target packet. Three
semantics-preserving physical work changes materially alter its economics:

- Q5_K N2 output projection now composes two mature matvec row views instead of
  launching a 32-column MMA tile to retain two columns. The isolated API moves
  from 16.0982 to 5.6455 ms GPU over all 48 GDN layers; rollback is
  `QWEN_MATMAT_Q5_K_N2_SEQ=0`. Producer rows are retained in
  `gdn-proj-n2-q5-seq.log` and `gdn-proj-n2-q5-mma-rollback.log`.
- The final verifier row is never a partial-restore source. Skipping that dead
  GDN/conv checkpoint write reduces verify by 33.2 ms over 34 packets, or
  0.98 ms/packet, in the retained final pair; rollback is
  `QWEN_MTP_SKIP_FINAL_CKPT=0`. See `mtp-packed-d1-final-candidate.log` and
  `mtp-packed-d1-final-ckpt-rollback.log`.
- Prompt target state is now built by one packed base prefill with all final
  hidden rows captured, followed only by the required shifted MTP KV bridge
  calls. This replaces 25 complete serial target forwards; rollback is
  `QWEN_MTP_PACKED_BASE_PREFILL=0`.

All short rows below used the same 25-token no-thinking prompt and 64-token
greedy target. The packed candidate and direct prefill rollback are a close
same-source pair; both preserve all 64 target IDs and pass a 16-step terminal
resume audit.

| D1/N2 path | Acceptance | Transitions / verifier | Baseline total ms | Candidate total ms | Speedup |
|---|---:|---:|---:|---:|---:|
| Packed base prefill | 0.882 | 1.853 | 2,886.0 | 2,718.8 | 1.061x |
| Serial-prefill rollback | 0.882 | 1.853 | 2,899.9 | 3,341.3 | 0.868x |

The changed-work prefill path alone moves speculative prefill from 992.0 to 369.9 ms
while decode remains 2,349 ms in both arms. Producer-native output is retained
in `mtp-packed-d1-candidate.log` and
`mtp-packed-d1-prefill-rollback.log`. A later current-source repeat reaches
1.057x total with all effective feature states emitted in the log.

A final source-stamped current-tree packet narrows the absolute candidate to
1.023x but directly isolates both promoted work deletions at one stable operating
point. Q5_K tile rollback moves verifier time `2276.5 -> 2628.5 ms` and total
speedup `1.023x -> 0.908x`; final-checkpoint rollback moves verifier time
`2276.5 -> 2314.9 ms` and total speedup `1.023x -> 1.010x`. All three runs share
the same 64 target IDs, token-stream SHA-256
`9a1268e4c9e4d0aa897145175b31b218e344c1512a09d1e7dad7651d5a6ea122a`,
and passing terminal-resume audits. Their build source state is
`git-source-sha256-v2:ca3e7b2efa0a9355e9c74297fd49db873e9dac122d607cd66be4eaeabc299e25`.
See `qwen38-q4-final-candidate-v2.log`,
`qwen38-q4-final-q5-rollback-v2.log`, and
`qwen38-q4-final-ckpt-rollback-v2.log`, with matching JSON artifacts.

This is not yet a production promotion. A forced 512-token continuation reached
only 0.961x total in its first run: acceptance remained 0.862, but N2 verification
rereads the nearly identical attention KV prefix for both rows. The immediately
following rollback process changed operating point by 30% in baseline decode
(19.95 -> 25.95 seconds), consistent with thermal drift but lacking temperature
telemetry. The two long files are evidence of a regime problem, not a clean
candidate/rollback ratio. They are retained as
`mtp-packed-d1-512-candidate.log` and
`mtp-packed-d1-512-prefill-rollback.log`.

A narrow changed-work-unit prototype now shares each KV tile across both causal
queries for the 24Q/4KV group-6 shape. Its measured outputs have printed
`max|Δ|=0.00e0` against two ordinary v4 calls; this is numerical identity in
those cells, not a byte-level proof. At base position 20,480, a complete attention layer
moves 4.10 -> 1.94 ms (2.108x); at 32,768 the isolated attention body moves
4.71 -> 2.33 ms (2.020x). Short and medium cells do not reliably win, so the
branch is context-gated at 16K and remains opt-in behind
`QWEN_MTP_ATTN_Q2_SHARED_KV=1`; its dedicated packed scratch must become lazy
before product promotion. Raw rows are retained in
`attn-q2-shared-kv-base20480.log` and `attn-q2-shared-kv-base32768.log`.

Recursive D3/N4 and D7/N8 remain closed at 0.670x and 0.740x total because deep
acceptance falls to 0.594 and 0.293. The perfect-draft D7/N8 oracle remains a
1.417x structural ceiling, but that oracle executes zero MTP calls, skipping
prompt history plus every draft body and KV bridge. It is a verifier-only
structural ceiling, not a fully charged deployable total. Those recursive/oracle
figures are historical operator-transcribed rows; producer-native output was not
retained, so they set disposition rather than a replayable promotion packet.

Decision: retain packed D1/N2 as a promising greedy-equivalent, numerically
audited research path, not a default.
The next gate is an interleaved, thermally controlled archetype packet plus lazy
long-context partial scratch. Do not return to the legacy lazy topology or local
MTP recurrence retuning.

## Exact Payload Redundancy Census

`qwen-census --payload-redundancy` now provides a reusable CPU-only falsifier. It
streams bounded reads from retained GGUF descriptors, hashes all tensor payloads,
hashes rows only in typed same-input projection groups, and performs exact byte
comparison before reporting any duplicate.

- 866 tensors and 17,095,778,304 payload bytes hashed.
- 130 projection groups, 373 projection tensors, and 3,297,792 stored rows
  hashed across GDN fronts, attention Q/K/V, FFN gate/up, and MTP counterparts.
- Zero duplicate full tensors, zero duplicate stored rows, and zero reclaimable
  payload bytes.
- 48.60 seconds wall, 195,362,816-byte maximum RSS, and zero swaps.

This closes whole-descriptor duplication and exact stored-row duplication in the
373 selected same-input front-projection tensors for this asset. It does not scan
arbitrary subranges or excluded output/down projections, and it does not test
approximate low rank or activation-conditioned sparsity.
The bounded run log is `payload-redundancy.log`; the scanner is retained so
future assets can be falsified without bespoke parsers or whole-file heap loads.

## Safety And Leverage

Operator observation: no run called `MTLResidencySet`, requested whole-model
residency, pre-wired or locked pages, used cache-bypass reads, or selected the
residency-coupled A10B pread path. The maintained quality child fixed
`QWEN_DSV4_RESIDENCY_SET=0` and
`QWEN_DSV4_PREFETCH=off`; all qwen processes exited normally. The user's tiny
idle OvisOCR llama server remained owned and untouched.

Force-ranked next work:

1. Build a small paired executable capability packet across Ridge, Qwen3.8
   Q4_K_M, and the Qwen3.6 regression anchor. Reuse the resident
   JSONL/provenance skeleton and deterministic grading; do not build a generic
   evaluator.
2. Measure the now-exposed exact `low|medium|xhigh` Qwen3.8 transitions plus
   no-thinking by task class before selecting any default or budget policy.
3. Add shared-range exact long-context retrieval at 16K and 64K before treating
   the 262K metadata ceiling as semantic capability.
4. Reuse the existing exact shared-prefix fanout/snapshot machinery when real
   request cohorts repeat prefixes. It can delete almost all repeated prefill,
   but it is not a fresh single-request optimization and should not become a
   universal cache layer.
5. Keep packed D1/N2 bounded to one interleaved regime decision. The final Q4
   row is only 1.023x and Ridge remains 0.968x after low-bit repair; do not
   resume recursive tuning or universalize the long-context attention path.
6. Keep payload deduplication, generic GDN algebra, recursive D3/D7, and rolling
   N1 closed absent a changed premise.
7. Keep vision and native tool schemas separate from text launch. Add them only
   when their own product decision justifies projector/runtime and protocol work.
8. Keep whole-model residency closed. None of these opportunities requires it.
   Native Q4_K/Q6_K embeddings use ordinary pageable loading.
