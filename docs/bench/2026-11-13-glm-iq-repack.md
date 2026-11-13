# GLM Full-Chunk IQ Repack: Rejected

## Result

The M5 Ultra stage probe used the 49,152-token prefix and a 2,048-row chunk.
Three interleaved `routed_experts` spans measured **719.205 / 718.073 /
721.676 ms** with GGUF banks and **740.074 / 737.188 / 709.607 ms** with the
aligned repack. Median speedup was **0.976x**, below the required **1.25x**
isolated-stage gate. The candidate did not accelerate this full-chunk regime.

Supporting stage-profile wall medians were 2,203.73 ms and 3,447.67 ms. These
are not ordinary-wall A/B samples; the load average was high and variable, so
they are not used as the end-to-end result. Since the isolated stage failed,
no paired ordinary-wall, exactness/non-inferiority, or serve gate was run.

## Artifact formats and memory

The model’s routed-bank dtype census was:

- Gate/up: 82 IQ2_S banks and 2 IQ3_S banks.
- Down: 39 IQ3_S banks and 3 IQ4_XS banks.

The repack preserved the GGUF values, padded IQ2_S blocks from 82 to 84 bytes
and IQ3_S blocks from 110 to 116 bytes, and applied only to IQ2_S/IQ3_S banks.
IQ4_XS banks stayed on their incumbent kernels and still participated in the
measured stage. The first load attempt rejected block 11’s IQ3_S gate bank
under the original IQ2_S-only assumption; the second found IQ4_XS down banks.
Both stopped in repack eligibility checks before MTL expert-bank allocation
or stage timing. The third attempt supported the observed mix and emitted the
census above.

The three-session memory preflight **passed** with `AdmittedWithHostBudget`:

- Original retained weights plus repacked banks: **227,444,705,536 bytes**
  (211.8 GiB).
- Repacked banks: **109,886,570,496 bytes** (102.3 GiB).
- One 48K session buffer plan: **3,967,565,824 bytes**.
- Three 48K session buffers: **11,902,697,472 bytes** (11.1 GiB).

The M5 has 256 GiB physical memory and a 240 GiB recommended working set.
Thermal pressure was 0 through the 60-second settle and the stage samples.
Load average was 12.17/11.18/11.76 before loading and 14.16/12.71/12.28 after
measurement.

## Decision

The 1.25x stage gate failed, so paired wall, quality, and serve gates were
skipped. The aligned-bank storage and decoder code were removed. The next
lever is the 78 ms KDA recurrence span from the baseline breakdown: evaluate a
chunked TensorOps recurrence for full-chunk prefill.

No code from MLX, mlx-serve, MLX-LM, or oMLX was used or adapted.

## Evidence

- Successful stage and memory record:
  [`attempt-3-stage-gate/`](2026-11-13-glm-prefill/iq-repack/attempt-3-stage-gate/).
- Initial artifact-format refusals:
  [`attempt-1-mixed-dtype/`](2026-11-13-glm-prefill/iq-repack/attempt-1-mixed-dtype/),
  [`attempt-2-iq4xs-dtype/`](2026-11-13-glm-prefill/iq-repack/attempt-2-iq4xs-dtype/).
- M5 campaign output remains under
  `metagross:~/qwen-runs/2026-11-12-glm-prefill/iq-repack-stage/`.
