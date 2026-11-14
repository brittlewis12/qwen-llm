# GLM gate/up epilogue: audited edge is a route-range alias

**Decision: no fused candidate and no performance gate result.** The waste
audit's `moe_wide_probe → moe_up_clamped` and reverse edges are overlaps of
whole-buffer spans, not producer/consumer dependencies on the same routed
slots. The default-off prototype guard also excluded GLM's adaptive route
(`h=4096, f=2048, experts=288, top-8`); it produced zero fused-kernel calls
and was reverted. No NI path was used.

## What the kernels compute

For the root graph, causal nodes 289 and 290 are respectively
`kernel_moe_wide_probe_iq2_s_f16act` (grid 32×32×288) and
`kernel_moe_up_clamped_iq2_s_f16act_grouped_slots_tensor` (grid 1×32×288).
Both bind the same 128 MiB F32 `inner` buffer as a full-range read/write span.
The append graph similarly reports the full 64 MiB `inner` range. These spans
omit the per-expert `counts` range that determines which routed slots each
dispatch actually touches.

The GLM adaptive gate/up host path in
`crates/qwen-llm/src/metal/moe_grouped_generic/tensor.rs` dispatches four
count-partitioned passes:

| Pass | Count range | Kernel role | Epilogue |
|---|---:|---|---|
| Gate narrow | 0–32 | Tensor Down | `D = C` |
| Gate wide | 33+ | `moe_wide_probe` | `D = C` (`epi=0`) |
| Up narrow | 0–32 | Tensor UpClamped | `D = silu(min(D,L)) * clamp(C,-L,L)` |
| Up wide | 33+ | `moe_wide_probe` | `D = silu(min(D,L)) * clamp(C,-L,L)` (`epi=2`) |

Each expert count selects one of the disjoint narrow or wide slot ranges.
Thus the audit's wide-gate→narrow-up edge does not connect the same slots; the
reverse edge is also produced by the shared full-buffer span on later/reused
work. The GLM root capture has a 128 MiB `inner` span; the append capture has a
64 MiB span. `moe_up_clamped` computes the up projection as well as the
elementwise epilogue; it is not a standalone activation kernel. The actual
same-range producer/consumer pairs are narrow Down→UpClamped and wide
`moe_wide_probe` with `epi=0`→wide `moe_wide_probe` with `epi=2`.

The audit's 5.5 GB / 2.8 GB intermediate and 27.5 / 13.8 ms round-trip prices
are based on the aliased whole-buffer spans. They do not price a true transfer
between the named wide and narrow count partitions, so they cannot justify a
1.25× fusion gate for that pair.

## M5 attempt and disposition

The `iter` binary was built under the `glm-prefill` lease (SHA-256
`ba5922fae479aff11399fce6e0da0d82d0667bd30d780d883b460ad462350348`). The
driver settled at thermal 0 for 60 seconds, found no resident model process,
and ran the measurement at niceness 0. Loadavg at startup was 5.05/4.89/4.97;
system free memory was 95%. The model run took 02:07:44–02:08:55 UTC.

One root baseline dispatch profile completed: **324.90 ms** summed service
for 123 named wide-probe/UpClamped calls, with **2,005.91 ms** whole-cell wall
and a bitwise baseline warmup. The GLM adaptive route did not enter the
prototype guard (`adaptive == true`), so its measured arm had zero fused-kernel
calls; the harness stopped before a candidate bitwise comparison, paired
stage samples, append cell, or native-cell A/B. This is not a measured
candidate win or loss.

The code prototype was reverted. No source change remains from the attempt.
Raw capture, build/driver logs, and load snapshot are retained under
`metagross:~/qwen-runs/2026-11-14-glm-latent-fusion/` and the matching local
`~/qwen-runs/` directory. A future fusion must target one of the count-matched
pairs above and include the count mask in its correctness witness before
timing.
