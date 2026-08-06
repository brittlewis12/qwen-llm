# DeepSeek V4 Packed Grouped-IQ2 Phase Ceiling V2

Status:

- gate/up/SwiGLU: `GO - AUTHORIZE_CANDIDATE_DESIGN_ONLY`;
- down/scatter: `HOLD - INCONCLUSIVE`.

The gate authorization is exact to the 25-layer production-route IQ2 phase and
survives the later down instability because its complete packet is emitted
before down allocation. It is not a savings estimate, product-prefill claim,
correctness result, implementation authorization, or promotion evidence.

## Why V2 Exists

V1 stopped before retained samples because Metal assigned equal GPU start/end
timestamps to a truly empty command. V2 changes the empty-bracket
instrumentation and decision classification, and emits each fully validated
phase packet before allocating the next phase so independent claims survive a
later failure. Routes, banks, deployed kernels, phase operations, scratch,
validation, conditioning, retained sample order, and 5% stationarity gate
remain byte-for-byte equivalent to the reviewed V1 source.

Each empty command remains genuinely empty. V2 records host wall time from
immediately before command-buffer creation through immediately after completed
wait, while preserving raw Metal start/end timestamps. Equal GPU timestamps are
censored, never measured zero.

For empty command `i`:

```text
E_i = max(full_wall_i, positive_gpu_i_or_zero)
```

Each bracket retains exactly 25 raw records and sums all `E_i`. It also records
the outer 25-command wall interval as a scope diagnostic. Empty bounds are
one-sided uncertainty enclosures, not samples; they are never filtered,
averaged, or used for a latency claim.

For each phase:

```text
P = max(disjoint_p95, warm_p95)
R = 5 * max(disjoint_range, warm_range)
C = P + R
E = max(empty_pre_upper, empty_post_upper)
U = P + max(R, E)
```

Classification priority is:

1. any invalid command/validation or cell drift above 5%: `INCONCLUSIVE`;
2. `U < 158.3 ms`: exact phase-only `KILL`;
3. `C >= 158.3 ms`: candidate design authorized only; and
4. `C < 158.3 <= U`: `INCONCLUSIVE - bracket-limited`.

Thus an empty bracket may prevent a KILL but can never authorize candidate work.
Strict positive GPU timestamps remain mandatory for every nonempty command.

## Frozen Execution

The sole release-diagnostics execution retains V1's:

- exact route payload SHA-256
  `505cb93ff9c3e1557bbad8f27a773e08b0e8c3475fbb4d7096069667c1fbafdd`;
- H=4096, F=2048, E=256, K=6, N=128, and clamp 10;
- deployed grouped IQ2 gate/up/SwiGLU and IQ3 down/scatter encoders;
- 25 separate serial command buffers per retained phase sample;
- 25 disjoint logical banks plus one all-expert warm bank;
- selected-slice and complete-warm first touch with legal quant bytes;
- distinct production-sized F32 and slot buffers per layer;
- completed untimed gate production immediately before every timed down;
- one untimed `D/W/W/D` block and retained `(D/W/W/D) x 6`; and
- complete output, state, route, guard, input, weight, error, timestamp, and
  repeat-digest validation.

No model asset is loaded. The complete V2 diff receives static CX `GO` before
the sole run. Release diagnostics compilation, strict Clippy, formatting, and
patch whitespace pass before execution.

## Gate Result

Both gate cells are stable and complete:

| Metric | Disjoint | Warm |
|---|---:|---:|
| p95 GPU sum, 25 layers | 527.963833 ms | 526.355958 ms |
| Range | 1.817374 ms | 0.314167 ms |
| Drift | 0.344817% | 0.059705% |

Derived gate values:

| Quantity | Result |
|---|---:|
| `P` | 527.963833 ms |
| `R` | 9.086871 ms |
| `C` | 537.050704 ms |
| `E` | 0.492790 ms |
| `U` | 537.050704 ms |

The pre/post brackets censor 6/10 of 25 empty commands. Their conservative
totals are 0.492790/0.489959 ms, far below `R`, so they neither drive nor change
the decision.

Gate disjoint/warm output digests are
`bbf43456479d05e4097794dba99132df46588f25282f904917caf5ab68ad4cb2`
and
`850ccf1105481a836ef0425e005c96257d8878a84d6719c5189a2740f98f5211`.
The initialized weight digest is
`369a03db349ed82ba7dbce98115d54e09406d600ae5506ab734ba3647b7549c6`.

Decision: `GO - AUTHORIZE_CANDIDATE_DESIGN_ONLY`. `C` independently exceeds
the 158.3 ms economic floor by 3.39x. This supports design review for one exact-
route gate/up/SwiGLU-only grouped-IQ2 candidate across the 25 grouped layers.
It does not predict removable milliseconds or authorize coding without that
review.

## Down Result

Down disjoint samples pass stationarity, but the final warm sample is
75.617124 ms versus roughly 71.58-71.61 ms for the other warm samples.

| Metric | Disjoint | Warm |
|---|---:|---:|
| p95 GPU sum, 25 layers | 72.250041 ms | 75.617124 ms |
| Range | 0.667750 ms | 4.035915 ms |
| Drift | 0.928512% | 5.483643% |

Warm drift exceeds the frozen 5% limit. The emitted `C=U=95.796702 ms`, raw
samples, 14/16 censored pre/post records, and 0.477670/0.465166 ms bracket totals
are provenance diagnostics only. They support no down KILL, ceiling, candidate,
or product claim. The final sample remains included.

Decision: `HOLD - INCONCLUSIVE`. Do not filter, repair, or rerun the same
protocol. Down/scatter cannot be bundled into the gate design authorization.

## Containment

The complete gate packet is emitted after all gate validation and before down
allocation, so the later down failure cannot invalidate or broaden it. The test
then fails deliberately on the down stationarity assertion.

The run takes 98.21 seconds in-test and 98.32 seconds elapsed, reaches
9,317,285,888 bytes maximum RSS, and records zero swaps. Metal allocated bytes
return to the 475,136-byte baseline after both campaign lifetimes. These are
provenance and cleanup facts only.

Remove the live V2 profiler after archival. The retained source returns exactly
to the base revision. The one V2 execution authorization is consumed.

The first authorized gate-only design, exact multi-bin FFN-row packing, is
bit-exact but closes `HOLD - INCONCLUSIVE`: all GPU cells are stable and regress,
while two wall cells fail stationarity. Remove it without asset work or width
tuning. The active handoff is a materially different dispatch-neutral SIMD
weight-broadcast design; down remains excluded.

## Validation

- The exact executed source and binary receive static CX `GO`.
- Both retained route/schedule release diagnostics tests pass after removal.
- Strict retained release diagnostics Clippy passes with warnings denied.
- `cargo fmt --all -- --check` and `git diff --check` pass.
- The retained source diff is empty and all archived hashes recheck.

## Provenance

- Base revision: `113e84e54caebc19b25af5ac770f284f72dc0db7`.
- Device: Apple M4 Max.
- Executed source diff SHA-256:
  `80d9519ab635c03ce8c223842cfbb7cf58ec93765d3aac95bc31e1a3614ec751`.
- Raw log SHA-256:
  `9fd4ad3740f056a6f7d5dc98158a3b03ddb1cbd822ff9d0976b8db36d2149b8e`.
- Executed release test binary SHA-256:
  `a06fb42ba48b28c1fe65191b7bf6ca38710d6144d6babf7454efecb10c8317d9`.
- Final retained release test binary SHA-256:
  `7bf572a929c54c4300dd3c00458e79d1d5ccce47562310a6045ebf19b91a0d6e`.
- Retained source diff SHA-256, empty by construction:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Checksum transcript: `checksums.log`.
- CX review: `019fd588-96b0-7033-afd1-66d7e354e523`.

Executed command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::prefill::tests::\
profile_exact_route_grouped_iq2_production_phases_v2 \
  -- --ignored --exact --nocapture --test-threads=1
```
