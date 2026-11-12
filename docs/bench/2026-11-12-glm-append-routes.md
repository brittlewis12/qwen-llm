# 2026-11-12 — GLM append route sweep: IQ tensor below 1024 wins the expert cliffs, the traffic gate fails

Status: probe + decision. **No production change shipped.** The default-on
candidate was implemented and then reverted on the gate failure
(`46911dd35`, reverted at `aa83b92c0`). The route wins the isolated and wall
comparisons and passes the paired non-inferiority protocol; it does not reach
the required 20% traffic-weighted append-wall reduction on the archived
three-agent serve distributions.

GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra, `--profile iter`, `code.txt` corpus
(150,243 tokens), capacity 98,304, 2,048 prefill rows. Depths **49,152** and
**81,920**. Three warm repetitions per route with interleaved route order,
then one stage-profiled call and (at 32 and 512 rows) one dispatch census per
route. Thermal pressure was 0 or 1 during every timed repetition; one 48K row
reached 2 after its timed reps. GPU clocks remain unavailable unprivileged.

## 1. Routes and what the sweep forces

The sweep adds a `perf-ablations`-only route selector to
`glm5_next_metal/row_policy.rs`/`packed.rs` and a `routes` mode to
`examples/glm_small_append_price.rs`:

| arm | forced selection |
| --- | --- |
| `incumbent` | production gates: per-slot packet 1..=8, grouped buckets 9..=127, grouped tensor 128..=512, explicit tensor 1024..=4096 |
| `per_slot_native` | half-operand per-slot packet at any legal width (kernel bound 1..=128) |
| `grouped_generic` | generic clamped grouped gate/up and the `m128_n16_retile` small-counts down |
| `expanded_tensor` | grouped tensor gate/up (F32 activations) with the short/expanded down policy |
| `tensor_iq` | explicit tensor gate/up (Half activations) with IQ3_S tensor down |

The route selector is thread-local, restores to `None` after each campaign,
and is absent from product builds.

## 2. Wall and routed-expert stage

80K depth, warm median of three ordinary `prefill_packed` calls; the
routed-expert column is the stage-profiled `routed_experts` span. Both are
milliseconds.

| width | incumbent | packet | grouped | expanded | tensor IQ | incumbent routed | tensor-IQ routed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 119.9 | 120.0 | 188.1 | 138.0 | 136.3 | 37.4 | 53.5 |
| 16 | 277.0 | 189.2 | 277.5 | 199.0 | 197.0 | 162.3 | 82.4 |
| 32 | 315.1 | 267.1 | 314.8 | 234.4 | 231.7 | 200.2 | 116.5 |
| 64 | 431.8 | 475.7 | 432.5 | 316.4 | 312.2 | 275.0 | 156.6 |
| 128 | 334.8 | 788.3 | 487.0 | 334.3 | 327.5 | 202.7 | 195.8 |
| 161 | 394.5 | n/a | 564.1 | 394.0 | 386.1 | 226.6 | 218.5 |
| 512 | 720.9 | n/a | 1016.2 | 721.6 | 698.4 | 347.1 | 323.9 |
| 768 | 1695.3 | n/a | 1695.8 | 1270.8 | 1233.1 | 854.8 | 391.2 |
| 1023 | 2178.2 | n/a | 2181.4 | 1673.5 | 1618.6 | 1024.1 | 462.7 |
| 1024 | 1166.3 | n/a | 1732.9 | 1239.6 | 1166.4 | 450.7 | 450.9 |

Cheapest route per measured width:

- **48K:** packet at 8; tensor IQ at 16..=1024. Savings 27.8% at 16, 32.0% at
  24, 27.0% at 32, 28.1% at 64, 26.8% at 96, 25.7% at 112, 2.0-3.2% at
  128..512, 28.0% at 768, 26.3% at 1023, 0.1% at 1024.
- **80K:** incumbent at 8; packet at 16; tensor IQ at 24..=1023; incumbent at
  1024. Savings 31.7% at 16, 32.6% at 24, 26.5% at 32, 27.7% at 64, 26.2% at
  96, 25.7% at 112, 2.0-3.1% at 128..512, 27.3% at 768, 25.7% at 1023.

The 128..512 band is already on a tensor route in production, so the route
switch there is only the 2-3% between the grouped-tensor (F32-activation) and
explicit-tensor (Half-activation) compositions. The two large islands are
**9..=112** (incumbent grouped) and **768..=1023** (incumbent grouped again
because `iq_expanded` starts at 1024).

Fine grid (9..=24, same method): packet is cheapest at 9..=15 (31-34% below
the incumbent), packet and tensor tie at 16..=19, and tensor wins from 20.
The 80K packet/tensor crossover is 19-20; 48K is 16. The reverted policy kept
the existing packet boundary at 8 and used the tensor route from 9, which is
31-34% below the incumbent at 9..=15 but about 5-8% above the absolute cheapest
packet arm there.

Kernel witness at 32 rows: the incumbent runs
`moe_swiglu_clamped_iq2_s_f32_grouped_slots_generic` (111.5 ms) and
`moe_down_iq3_s_f32_grouped_slots_m128_n16_retile` (57.1 ms). The tensor arms
run `moe_up_clamped_iq2_s_f32_grouped_slots_tensor` / `moe_down_iq2_s_f32...
` (expanded) or `moe_up_clamped_iq2_s_f16act...` / `moe_down_iq2_s_f16act...`
(tensor IQ), totalling 110-117 ms of routed-expert work.

## 3. Traffic weighting: the gate fails

The sweep does not include a production 75-81K log. The archived M5
three-agent runs are 48K. Four real append distributions were used, all from
`2026-11-08-realistic-steady/2026-11-09-steady-concurrency-calls.csv`
(`glm-default`, 180 s) and `2026-11-11-glm-selective-layer/on-a`:

| distribution | n | median | 48K reduction | 80K reduction |
| --- | ---: | ---: | ---: | ---: |
| `new` (actual prefill width, glm-default) | 20 | 232 | 11.51% | 11.21% |
| `native_generated` (turn content, glm-default) | 20 | 146 | 12.37% | 12.18% |
| `new_content_budget` (glm-default) | 23 | 67 | 10.31% | 10.05% |
| `new` (2026-11-11 selective-layer on-a) | 13 | 342 | 13.23% | 12.82% |

Off-grid widths use the nearest measured width; widths above 1024 (and below
8) keep the incumbent and claim nothing. Count weighting is the most favorable
of count, token and wall weighting for this policy.

**Gate result: fail.** The best of these real distributions is 13.23%
(48K) / 12.82% (80K), against the required 20%. The arithmetic is the 128..512
band: the real median sits inside a band that is already tensor, and only the
small-width and 768..1023 islands have 26-32% headroom. The stated real median
of 161 rows is not present in the archived logs (their medians are 67-342 and
their depth is 48K); a production 75-81K distribution should be used to
recompute the gate before any re-land.

## 4. Numerics: paired non-inferiority passes

Protocol: three code prompts (`code.txt`, `metal-code-corpus.txt`,
`scripts/profile/dflash_k0s.py`) at 81,920 depth, teacher-forced teacher
targets, paired per-position signed dNLL of `tensor_iq` against the incumbent,
prompt-cluster bootstrap (20,000 resamples, central 90% CI, one-sided 95%
upper bound). Positive dNLL means the candidate gives the actual next token a
lower log-probability. Margin: +0.0100 nats/token.

| width | mean dNLL | 90% CI | 95% upper | mean KL | top-1 | verdict |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 16 | +0.00003 | [+0.00001, +0.00006] | +0.00006 | 3.2e-06 | 100.0% | PASS |
| 32 | -0.00017 | [-0.00034, -0.00000] | -0.00000 | 5.0e-05 | 100.0% | PASS |
| 64 | -0.00138 | [-0.00259, -0.00017] | -0.00017 | 3.5e-04 | 100.0% | PASS |
| 96 | -0.00435 | [-0.00807, -0.00064] | -0.00064 | 7.1e-04 | 99.7% | PASS |
| 161 | +0.00000 | [+0.00000, +0.00000] | +0.00000 | 0.0 | 100.0% | PASS |
| 512 | +0.00000 | [+0.00000, +0.00000] | +0.00000 | 0.0 | 100.0% | PASS |
| 768 | -0.00206 | [-0.00425, +0.00013] | +0.00013 | 1.5e-03 | 99.1% | PASS |
| 1023 | -0.00030 | [-0.00265, +0.00204] | +0.00204 | 1.4e-03 | 99.3% | PASS |

At 161 and 512 the candidate and incumbent are bitwise identical: both are
tensor compositions and the 2-3% wall difference is scheduling, not
arithmetic. At 16-96 and 768/1023 the candidate is at noise level or slightly
better. The prose control (`docs/PERF-LOG.md`) passes at every width with the
same bound. No `#[ignore]` test was added.

## 5. Decision and next steps

- **Do not ship default-on.** The traffic-weighted append-wall gate fails at
  10.05-13.23%; the candidate implementation stays in `46911dd35` (reverted at
  `aa83b92c0`) and the probe stays on `perf/glm-append-routes`.
- The route is a real 26-34% wall win on the 9..=112 and 768..=1023 islands,
  and the 128..512 switch is bitwise. Re-land only against a real 75-81K
  production append distribution, or after a step-2 kernel changes the
  128..512 economics.
- **Step 2 remains the floor.** At 128..512 the expert stage is 196-324 ms and
  the route is already tensor; a grouped kernel that reads each active expert
  once must beat ~1.25x there to be worth building. The Step 2 gate alone
  (1.25x on 196-324 ms) is worth ~5-8% wall in that band, not 20%.
- The serve validation (three-agent, interleaved pairs) was not run because
  the gate failed before it.

## Reproduction and raw records

Probe: `crates/qwen-llm/examples/glm_small_append_price.rs` (`routes`,
`route-quality`), route selector in `glm5_next_metal/{row_policy,packed}.rs`,
analysis `scripts/bench/glm_append_routes.py` and
`scripts/bench/glm_route_quality.py`.

Commands (M5, canonical lease, `nice -n 10`):

```sh
# build
cargo build --profile iter --features perf-ablations --example glm_small_append_price
# sweep
GLM53_GGUF=... target/iter/examples/glm_small_append_price code.txt routes 49152
GLM53_GGUF=... target/iter/examples/glm_small_append_price code.txt routes 81920
GLM_ROUTE_WIDTHS=9,10,...,24 ... routes 49152   # fine grid
# paired cohort (one prompt per invocation)
GLM_ROUTE_QUALITY_WIDTHS=16,32,64,96,161,512,768,1023 \
  GLM_APPEND_CAPACITY=98304 \
  target/iter/examples/glm_small_append_price <prompt> route-quality 81920
```

Raw: `metagross:~/qwen-runs/2026-11-12-glm-append-routes/`
(`routes-48k.jsonl`, `routes-80k.jsonl`, `routes-fine-{48k,80k}.jsonl`,
`quality-{code,metal,python,prose}-80k.jsonl`, `build*.log`). Branch
`perf/glm-append-routes`; not merged. M5 worktree and bundles removed after
the packet.
