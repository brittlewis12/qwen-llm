# Flash Strict HC Projection Screen

Decision: **do not change production routing**. A strict four-output packed HC
injection projection improves a synthetic kernel but not ordinary native prefill.
Keep the bounded research kernel and test-only witness to avoid repeating the bet.

## Candidate And Correctness

The released Flash geometry projects F32 K=10,240 into four injection gates. The
generic F32 matrix kernel stages one weight row and executes 640 threadgroup
barriers per dispatch. The candidate shares each query lane's input across four
scalar accumulators without shared memory or barriers. It preserves ascending-K
traversal, multiplication operand order and safe math; it is not the earlier
short-packet GEMV experiment with a different reduction order.

The checked test encoder accepts only K=10,240, M=4 and N=1..2048. It validates
F32 counts, byte ranges, alignment, writable/disjoint output, device identity
and pipeline capacity before encoder mutation. Allocations use normal device
pricing, CPU staging allowance and dynamic reserve, under the production lease.
No new scratch, precision change, runtime option or production dispatch is added.

- Bitwise generic/candidate equality passes 36 cases: nine widths
  (1/2/7/31/32/33/128/512/2048), ordinary and cancellation-sensitive values,
  zero and guarded scalar-aligned nonzero offsets.
- Shape, overflow, bounds, aliasing, mutability and capacity refusals pass.
- Real-model prefill logits, final packed hyper row and active persistent state
  match at 128/512/2048. Four subsequent scalar-token logits and live state also
  match. Census confirms 96 candidate projections per packed command. Workspace
  pricing and geometry remain fixed. This is bounded equality, not universal
  quality evidence.

## Timing

M4 Max, 128 GiB, UD-Q3_K_XL, Metal API validation enabled, exclusive production
lease, no concurrent builds. All shaders use the normal optimized Metal build.
The native driver is a debug Rust test; GPU timestamps cover ordinary completed
prefill commands. **Its wall times are not release request latency evidence.**

Synthetic resident kernel: two excluded ABBA warmups, six alternating ABBA/BAAB
blocks, four launches per command. Compilation/allocation/readback are excluded;
both arms write the same output binding. Median GPU microseconds per launch:

| Rows | Generic | Strict | Time Reduction |
| --- | ---: | ---: | ---: |
| 128 | 662.203 | 631.823 | 4.6% |
| 512 | 718.812 | 685.750 | 4.6% |
| 2048 | 864.219 | 806.948 | 6.6% |

Native input is the existing frozen natural-SSH token fixture. Each width gets
two excluded census/correctness warmups followed by one A/B/B/A. Reset, hashing,
four-token continuation and checks are outside prefill timing. A second screen
tightens the witness to live state at both endpoints. Mean GPU milliseconds:

| Rows | First A / B | Candidate Time Change | Closure A / B | Candidate Time Change |
| --- | ---: | ---: | ---: | ---: |
| 128 | 498.52 / 506.36 | +1.57% | 502.52 / 506.44 | +0.78% |
| 512 | 1003.95 / 1011.08 | +0.71% | 1006.44 / 1008.05 | +0.16% |
| 2048 | 3721.90 / 3736.67 | +0.40% | 3704.45 / 3702.93 | -0.04% |

This is a rejection screen, not a precise regression estimate: only two measured
samples per arm per width in each screen, and the largest-width warmups show
drift. There is no consistent useful full-prefill gain. Do not multiply isolated
dispatch savings by 96 and call that a native speedup. No release gameplay A/B,
decode benefit, serving benefit or M5 conclusion is claimed.

An initial native attempt stopped before measured ABBA because the observer read
singleton scratch after packed execution. Prefill/continuation logits and state
already matched. The corrected observer reads the actual final packed row; the
closure compares only defined live QSA regions, not unused allocation tails.

[results.json](results.json) retains all successful native samples and kernel
medians. The candidate is test-only changes on `1c8b3735`, not a clean release
identity. Private raw logs and initial failed observation remain outside Git.

## Reproduction

Compile before taking the GPU lease:

```sh
cargo test -p qwen-llm --lib --no-run
```

Run the resulting test executable with `MTL_DEBUG_LAYER=1`,
`QWEN_METAL_LEASE_WAIT=1`, `QWEN_REQUIRE_METAL_TESTS=1` and `--test-threads=1`:

- `metal::moe::hc4_strict:: --include-ignored --nocapture` for contracts,
  bitwise comparisons and timings.
- `metal::moe::hc4_strict::hc4_strict_gpu_timing --exact --ignored --nocapture`
  for resident timings.
- `qwen4exp_runtime::strict_inject_probe::ordinary_prefill_exact_and_abba
  --exact --ignored --nocapture`, also setting `QWEN4EXP_Q3_K_XL_GGUF` to the
  first target shard and `QWEN_HC_STRICT_REPORT` to a new output file.

The witness acquires the production lease itself. Production binaries retain
their existing projection; the shader lives in the separate research library.
