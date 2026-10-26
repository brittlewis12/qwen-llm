# V4.1 Decode: Engram Row I/O, Small-Kernel Fixes, Split FFN Encoders (2026-10-26)

Branch `perf/v41-decode-stages`, M5 Ultra (metagross), GSQ-RCO IQ2_XS
(~144.6 GiB resident). Raw data: `~/qwen-runs/2026-10-26-decode-twk3/`
(`run*.sh` scripts, `r*-*.json` suites, `*.err` per-token hooks, `quality/`).
All arms lease-serialized; A/B arms are ABBA or in-binary env switches.

## Result (final ABBA, `run16`)

`suite --pp 512,2048 --tg 64 -d 0,2048 --runs 2`, A = main `ebb774b8`,
N = branch (`ccbd4827`; V4.1 decode arithmetic identical to the landed head).

| | main A1 / A2 | branch N1 / N2 |
|---|---:|---:|
| tg64 @ d0, rows cached | 54.0 / 53.9 | 61.9 / 61.9 |
| tg64 @ d2048, rows cached | 48.1 / 48.5 | 54.6 / 54.7 |
| pp512 | 717.3 / 716.7 | 720.2 / 722.3 |
| pp2048 | 856.4 / 855.2 | 861.7 / 859.8 |
| tg64 @ d0, fresh seeds (rows uncached) | 44.9 / 44.3 | 61.2 / 61.3 |
| first pp512 after pp64 warm-up, fresh seeds | 30.4 / 27.2 | 485.8 / 484.8 |

Cold arms are `--pp 64,512 --tg 64 --no-warmup` with a new seed per process
(A1 2101, N1 2202, N2 2303, A2 2404), so decode reps and the first pp512 read
uncached Engram rows; the second pp512 rep (rows then cached) is ~720 in
every arm. The synthetic tokens are the worst case (uniformly novel n-grams);
real text repeats common n-grams, so its uncached share is lower.

Quality (`quality/`): six-document witness, 384 targets, mean NLL 2.1421 and
top-1 212, equal to the 2026-10-25 GSQ baseline at four decimals (logits
differ by at most 5.0e-4; 8.7% bitwise identical). 22-task screen: code 9/10,
reasoning 1/8 (one truncated), JSON 0/4, identical to the baseline.

## 1. Engram rows were serial SSD reads in real use

The two Engram layers read 24 IQ4_NL rows each per token (144 B) from a
103 GiB table with one `pread` per row. The table cannot stay cached next to
the resident model (~2-8% resident in these runs), so novel text pays an SSD
round trip per row. Per-token timing (`QWEN_V41_DECODE_TIMING=1`, which now
prints process page-ins and disk bytes) on main-equivalent code:
`decode_engram_rows` 3.7-4.2 ms with 768 KiB read per token when uncached,
0.04 ms when cached; GPU time unchanged at ~17.2 ms. This is the
"first decode after load" effect recorded on 2026-10-25: cold decode was
never weight page faults, and warm repeats only looked warm because
seed-fixed runs reuse the rows the previous run cached.

Microbenchmark (`io/amp.c`, `io/rdadvise.c`): 48 uncached rows serial 3.35
ms; `F_RDADVISE` all then `pread` 0.24 ms; 12-13 KiB read per row either way
(no amplification); advising a cached row costs ~0.5 us.

Landed: decode advises both layers' rows, then reads; batched (prefill)
reads stay serial, probe the first 64 rows and queue read-ahead for every
remaining row once a block reads slower than 8 us/row.
`QWEN_V41_ENGRAM_IO=serial` restores the old path (A/B).

Negatives on the way (warm, rows cached): rayon global pool 0.5 ms/token
decode and -4% pp512 (36-thread wake-ups); a 4-thread pool -2.8% pp512; an
advisor thread advising every row -2.4% pp512.

## 2. Per-dispatch attribution and small-kernel fixes

`QWEN_V41_DECODE_DISPATCHES=1` runs every decode dispatch in its own sampled
encoder with replayed bindings (M5 has no dispatch-boundary counters) and
prints one JSON line per token with stage labels. Absolute times are inflated
(split timeline ~1.4x the unsplit command) and independent encoders overlap,
so read per-kernel ratios, then confirm in-pipeline. Removal pricing (an
experiment that skipped labeled stages; not landed) gave in-pipeline costs per
token at d0: attention projections+chains 2.69 ms, attention output 2.73,
shared expert 1.74, attention core 1.17, both HC pre sites 1.33, router 0.24,
Engram projection 0.22, HC post 0.13; routed experts ~5 ms by difference.

| change | evidence |
|---|---|
| Split-K BF16 GEMV (`kernel_mat_vec_bf16_f32_tg`, router/compressor) | router 44 -> 12 us per layer (one-simdgroup rows were latency-bound) |
| Wide decode Engram gate (1024 threads per stream for <= 16 rows) | 168 us -> negligible per call (two per token) |
| Vector HC pre (float4 loads issued together, controls on one simdgroup) | 20 -> 16 us per site |
| Vector-load native Q5_K (two 16-byte loads per 16 weights, same values) | head 522 -> 420 us (~1 TB/s); per-layer Q5_K -4 to -7% |
| Fused same-type native gate/up + SwiGLU (bitwise; V4.1 decode only) | tg64 +0.5-1.5% (run10 in-binary A/B) |

Warm tg64@d0 after these (run9/run10): 58.4-59.4 tok/s vs main 54.0.

Negatives (removed): multi-row native rows (2 / 4 channels per simdgroup
sharing activation fragments) -2% / -6% decode; unrolled load rounds (2 / 4
fragments in flight per lane) -24% / -41% (register pressure); 8-byte IQ2
grid loads neutral; parallelizing only the Sinkhorn loop neutral. The
isolated decode probe (`QWEN_EXPERT_DECODE_PROBE=1`, now with an IQ2_XXS
cohort) streams the routed stage at 514 GB/s (106 us for gate/up/down,
top-6), close to its in-pipeline cost: routed IQ2 needs a different kernel
design, not tuning of this one.

## 3. Split FFN encoders (default)

The decode FFN now encodes HC pre, shared expert, router plus routed experts,
and combine on four encoders. The shared expert and the routed chain only
read the HC pre output and write disjoint buffers, and Metal's resource
tracking overlaps them. In-binary ABBA (run15): tg64@d0 59.0/59.1 -> 61.8/61.8,
@d2048 52.7/52.7 -> 54.9/54.9, GPU 16.67 -> 15.82 ms/token; prefill neutral.
`QWEN_V41_FFN_ENCODERS=one` restores one encoder. The same split for
attention (query | KV + owner | selection/core) lost: 60.4/60.2 and
53.8/53.8 (two more encoders per layer cost more than the short chains
overlap); removed.

## Remaining budget

Decode is GPU-bound: ~15.8 ms GPU of ~16.2 ms per token at d0, ~490 GB/s
average over 7.28 GiB. Largest remaining items: routed IQ2 experts (~5 ms at
~440-510 GB/s; Q8_0 in the same kernel reaches ~1.1 TB/s, so low-bit decode
cost is per weight, not per byte), query projection q_b (K=1024 rows, ~440
GB/s), HC pre (~17 us per site for ~2 MB), attention core.
