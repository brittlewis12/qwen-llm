# IQ1 Native Capability Census Correction

This follow-up to `0e26313111e0477108173cb9edce9f1cc70c2dc8` is test-only
plus documentation. The only code change adds IQ1_S and IQ1_M to both
independent expected lists in
`crates/qwen-llm/src/metal/expert_native/tests/rows.rs`:
- The expected down-row capability list.
- The expected gate/up-row capability list.

Test repair commit: `1d9b2a14`.

The production adapters already support these types. The original expected
lists were stale; they are not replaced by imports of production capability
constants. The baseline arithmetic-body SHA assertion remains
`1bb4298433e54ba446cc5da2cc7d78d3b3431c7d54463763f9c7c55bef500b49`,
the NativeArgs size assertion remains 32 bytes, and the complete raw
0..=255 down matrix and 256-by-256 gate/up matrix remain unchanged.
There are no shader, host-adapter, codec, model-payload, or default-policy
changes. No main, remote, model, or PERF-document work is included.

## Parent M5 Evidence At The Previous HEAD

The parent supplied preserved logs for source commit `0e263131` and a frozen
M5 executable identity with SHA-256:
`c0820fbbe907c88f57646304f7e73f92cf6433566483829f4c2c2ee57be2a732`.

The logs show:
- Scoped IQ1 default: 33 passed, eight ignored.
- Unskipped IQ1 ignored: nine passed, zero failed. This is all eight M5
  tensor witnesses plus
  `metal_forward::role_tests::iq1_m_embedding_loader_and_gather_use_frozen_choice`.
- All tensor comparisons pass. In the copied stderr, the per-label maxima
  are `4.530e-6` for `IQ1 tensor vs canonical generic` and `1.526e-5` for
  `released mixed storage`. The reported `3.457e-6` is a passing IQ1_M
  comparison, not the maximum across these copied logs. These are numerical
  comparison results, not performance or full-model quality claims.
- The broader native module reports 25 passed, one failed, seven ignored.
  Its sole failure is the stale CPU capability expectation corrected here.

The parent's expected-eight harness stop after nine successful ignored
tests was a census mismatch, not a GPU failure. Original logs and identity
files under the private IQ1 root are retained:
`iq1-m5-default-v27.*`, `iq1-m5-tensor-v27.*`,
`iq1-m5-native-regressions-v27.*`, and
`iq1-experts-v27-core.identity.json`.

These are parent-provided actual M5 results at the previous HEAD, not a claim
that this follow-up was freshly executed on M5. The parent will fresh-build
and rerun the native module, scoped default, all nine ignored tests, and
three wide-activation tests before integration.

## New Local M4 Proof

New evidence, relative to the requested private IQ1 root:
`capability-census/freeze-v9/` and `capability-census/results.json`.
The new test binary and all three metallibs were frozen while the production
build lease remained held. GPU tests self-lease and run that frozen binary
directly with `MTL_DEBUG_LAYER=1`, `QWEN_METAL_LEASE_WAIT=1`, and one test
thread. No external double wrapper is used for GPU execution.

| Frozen M4 filter | Result |
| --- | --- |
| `metal::expert_native::tests::` | 26 passed, seven ignored |
| `metal::expert_native::tests::rows::native_rows_arithmetic_source_and_capability_contract --exact` | One passed |
| `iq1 --skip metal_forward::` | 33 passed, eight ignored |

The focused CPU filter exercises the unchanged arithmetic hash, ABI size,
and full capability matrix in addition to the repaired independent lists.
The changed test file passes rustfmt under the production lease.

New frozen M4 executable SHA-256:
`385fa064ff3333f11c6817d489bfc77ddfd2361a19271861a361eb4a57721c80`.
All three local metallib hashes are unchanged from `freeze-v8`:

| Artifact | SHA-256 |
| --- | --- |
| Product metallib | `b623201c7cd35f5cb7d6ec190ee6349d8074f66dbbd863cd63d97aa74887121c` |
| Tensor metallib | `64da3a665759978f44b21101ceb8c98a20806753c78c9953fb915c2397fc4580` |
| Research metallib | `00506cb04df3d957d932d029abd35ea1ca8280be2862e82402a33cfb5caf4189` |

The source manifest comparison with `freeze-v8` changes exactly one entry:
the native-row test source. Shader and host-adapter source hashes remain
unchanged, including the host tensor adapter; the parent M5 embedded source
hashes agree as well. Cross-toolchain M5 metallibs have their own identities
and are not asserted equal to these local M4 artifacts. `0e263131`,
`freeze-v8`, and the parent M5 evidence are preserved.

## Corrected Fresh M5 Rerun Census

Run each command against the newly frozen parent binary with
`MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1`, `--nocapture`, and
`--test-threads=1`. GPU requirements and ignore annotations are unchanged.

| Filter arguments | Expected M5 result |
| --- | --- |
| `metal::expert_native::tests::` | 26 passed, seven ignored |
| `iq1 --skip metal_forward::` | 33 passed, eight ignored |
| `iq1 --ignored` | Nine passed: eight tensor plus one legacy embedding |
| `iq1 --ignored --skip metal_forward::` | Eight passed: tensor-only alternative |
| `wide_activation_type --ignored` | Three passed |

The previously documented eight ignored witnesses applies only when the
legacy `metal_forward::` scope is excluded. Neither a local M4 unavailable
result nor a successful source-mapping test substitutes for M5 execution.
No full-model held-text quality or speed result is claimed by these checks.
