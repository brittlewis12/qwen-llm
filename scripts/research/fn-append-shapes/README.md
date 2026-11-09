# Flash-Next shape-complete appends

Base `2e2f14e6`. Initial signal: ordinary fresh-process policy off/on, real
400/512/2000/2048 rows at 65536, then native common-prefix **ABBA + BAAB**.
No profiling/census or validation in retained timing; full endpoint readback.
The qualification test restores the same persistent snapshot outside timing.
Input generation is frozen by `prepare_inputs.py`; delete regenerated text.

```sh
python3 scripts/research/fn-append-shapes/prepare_inputs.py --out "$RUNS/inputs"
FLASH_PREFILL_MODEL="$MODEL" FLASH_PREFILL_OUT="$RUNS/native.jsonl" \
FLASH_APPEND_PLAN="$RUNS/inputs/timing-65536.json" \
  "$LIB_TEST" qwen4exp_runtime::tests::append_shapes::native_append_shape_packet \
  --ignored --exact --nocapture --test-threads=1
```

Build under the build lock, outside the Metal lease. Timing uses only the
canonical lease/wait and nice 10, cooled thermal zero, one-second stop-on-rise
monitoring. Native/quality/steady campaigns must run sequentially, one model.

Quality plans use four frozen Rust and four frozen manual-text continuations,
128 evenly spaced true next-token targets through each 400/2000-row append,
including the endpoint. Read complete vocabulary logits using the native head
over saved final hyper rows. Check the native endpoint against this readout.
Each case executes baseline/candidate/baseline-repeat/F32-GDN-control.
Repeat baseline must match bits; the arithmetic control retains F32 operands
while replacing chunked GDN by its released serial recurrence in an independently
executed prefix and append. The control matches the whole-policy scope: Q5 and
ragged HC changes can also affect prefix state. This scope is frozen before any
quality results are read.
No sampled targets use generated feedback. Contexts: 8192/65536; an additional
8693-row prefix exercises an additional irregular prompt tail. Every quality
prefix is executed independently per policy: long requests cross 2051 and can
end with an irregular packed chunk even when the total is 2048-aligned. The
native timing packet instead holds one baseline prefix snapshot fixed.

Prespecified decision: per depth/row cohort, hierarchical document/32-target-block
bootstrap, seed 20261109, 10000 resamples. Signed candidate-minus-baseline dNLL
one-sided 95% upper bound must fit the 95th percentile absolute per-document
arithmetic-control dNLL envelope; pooled mean KL <=0.01. Report each document,
16/64-block sensitivity, greedy agreement and deterministic repeat separately.
Operator F64 errors must be no worse for the operands actually consumed by
both routes: original F32 values for F32 operators, identical incumbent-staged
BF16/F16 values for HC/quantized matrices. Report original-F32 input error too,
so the inherited activation-conversion floor stays visible. Gate both RMS and
peak for newly admitted reductions; the Q5 N12288 tensor failed the peak gate
and keeps its incumbent. Existing aligned qualified kernels are controls.
Operator/tail/guard tests precede default promotion. Greedy text equality is
descriptive, not a gate. No automatic
promotion follows from the packet writer.

After promotion: production native ABBA and the committed steady-state harness
on both policy values, concurrent serving default, same seed/window. Compare
TTFT distributions with censored waits, stream gaps and rates; natural EOS and
unequal completed work are accepted. Full release lib + CLI tests and contracts.
Retain compact metrics, count/error summaries, commits and commands. Delete
full logits (in-memory only), corpus copies, build source/target and bundles.

Rejected probes are retained as patches, with their runtime paths and knobs
removed. Apply only in an isolated checkout with `git apply`:

- `rejected-panels.patch`: 16/32-slot panels; separate 5-KiB small tiles or
  fused gate/up staging. Four native formats match the original complete
  output bits. Whole appends did not improve: 400 was 314.97/314.75 ms and
  2000 was 1129.11/1135.78 ms without/with fusion. Keep the original N32 route.
- `rejected-hc-up2.patch`: aligned K160 partitions and M64 query tiles for
  the BF16 up projection. Rejected at the staged-F64 peak gate at M513;
  no whole-model timing or text qualification of this variant follows.
