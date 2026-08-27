# Flash-Next Packed Router E8P32

Decision: **GO** for exact N=2,048 full chunks and **KILL** for N=18.
Promote only the full-chunk shape; keep interactive packed routing on the
generic F32 projection.

## Source And Protocol

- Measured source: `9aca035`, containing the default-off candidate from
  `b799a95` and the packed release-replay admission fix.
- Promotion source: `6df62fb`, which removes N=18 from strict eligibility and
  makes exact N=2,048 default-on with an explicit rollback.
- Device: Apple M4 Max with unified memory.
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q3_K_XL, revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`, on the external PCIe SSD.
- Protocol: separate-process A-B-A at each shape. A used the generic
  `kernel_mat_mat_f32_f32`; B used strict-order E8xP32. Every process ran a
  first, warm, and 128-sample profiled command.
- Acceptance: one packed command, observer GPU ratio `0.985-1.015`, observer
  wall ratio `0.98-1.02`, and raw timestamp coverage `0.995-1.005`.
- Correctness: synthetic N=18/N=2,048 projection and route metadata were
  bitwise equal. The pinned release A/B replay compared full-vocabulary
  endpoint and continuation logits, every raw persistent-state tensor before
  and after scalar handoff, QSA lengths, PLE history, and complete dispatch
  topology. On the measured source, the candidate replaced exactly 48 router
  dispatches at each shape. The promoted replay requires zero substitutions at
  N=18 and 48 at N=2,048.

Cold first-pass wall time is not a gate: it includes mmap first touch and the
candidate's first strict-pipeline use. Warm command GPU and the sampled router
leaf decide the experiment.

## A-B-A Results

| Tokens | Arm | Warm command GPU (ms) | Profiled command GPU (ms) | Router (ms/layer) | GPU ratio | Wall ratio |
|---:|:---|---:|---:|---:|---:|---:|
| 18 | A1 generic | 242.819375 | 244.740667 | 0.203500 | 1.007912 | 1.008175 |
| 18 | B strict | 244.416292 | 243.880125 | 0.199292 | 0.997806 | 0.999128 |
| 18 | A2 generic | 244.581708 | 244.419583 | 0.206666 | 0.999337 | 1.000153 |
| 2,048 | A1 generic | 3,764.607583 | 3,755.826625 | 15.903542 | 0.997667 | 0.997862 |
| 2,048 | B strict | 3,128.968083 | 3,118.744417 | 2.608167 | 0.996733 | 0.996726 |
| 2,048 | A2 generic | 3,764.298541 | 3,811.366625 | 15.878792 | 1.012504 | 1.012852 |

Raw timestamp coverage was `1.000000` in all six arms. N=18 produced `HELLO`
and reached producer EOS in every arm.

## Gate Evaluation

| Tokens | Metric | A/A mean | Candidate | Required | Decision |
|---:|:---|---:|---:|---:|:---|
| 18 | router ms/layer | 0.205083 | 0.199292 | <=0.146842 | KILL |
| 18 | warm command GPU ms | 243.700542 | 244.416292 | <=240.308 | KILL |
| 2,048 | router ms/layer | 15.891167 | 2.608167 | <=14.277188 | GO |
| 2,048 | warm command GPU ms | 3,764.453062 | 3,128.968083 | <=3,725.021 | GO |

At N=18, the leaf saves only `0.005791 ms/layer` (`2.82%`) and warm command
GPU regresses by `0.715750 ms` (`0.294%`). At N=2,048, the leaf saves
`13.283000 ms/layer` (`83.59%`) and warm command GPU saves `635.484979 ms`
(`16.88%`), raising warm command-GPU-equivalent throughput from `544.04` to
`654.53 tok/s`.

The 48-layer leaf prediction is `637.584 ms`; `635.485 ms` reaches the command,
or `99.67%` conversion. That agreement strongly corroborates the leaf
attribution; the row-for-row census establishes unchanged non-router topology.

## Promotion

- Enable strict E8xP32 by default only for Apple M4 Max, F32 router weights,
  `H=2560`, `E=512`, and exact N=2,048.
- Keep N=18 and every other packed shape on generic F32 routing.
- Roll back with `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0`.
- Do not reopen top-k/bucket fusion. The next packed-prefill gate is
  count-banded standard IQ3_XXS routed gate/up plus SwiGLU geometry.
