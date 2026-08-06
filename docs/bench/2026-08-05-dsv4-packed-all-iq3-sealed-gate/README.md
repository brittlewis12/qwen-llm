# DeepSeek V4 Sealed All-IQ3 Promotion Gate

Status: `KILL` for production all-IQ3 grouped promotion under the current
asset, Apple M4 Max, and frozen observer contract. The exact grouped candidate
remains diagnostics-only. This packet authorizes no performance conclusion and
no same-condition retry.

## Question

The post-route attribution packet assigns about 355 ms of N=128 routed GPU work
to the 16 all-IQ3 layers. The retained four-dispatch grouped candidate is already
bit-exact and previously showed a directional stage saving, but its wall-only
campaign failed stationarity. A newly stable stage-specific observer authorized
one final balanced GPU-and-wall gate.

This packet asks whether that candidate clears the complete promotion contract
without deleting an observer-invalid sample or transferring an earlier timing
claim.

## Frozen Protocol

- A is the qualified grouped-IQ2 production policy.
- B adds the diagnostics-only grouped all-IQ3 candidate.
- Both policies use the same four-stage sampled post-route encoder topology.
- Capability checks for both grouped paths must pass before opening the GGUF.
- Run one untimed `ABBA BAAB` warm block.
- Time `(ABBA BAAB) x 2`, giving eight samples per arm and four per half.
- Warm arms must preserve exact output/state, topology, and 25/0 versus 25/16
  grouped invocation counts. They carry no observer-validity gate.
- Every timed sample must preserve exact output/state and invocation counts,
  repeat dispatch geometry within its arm, and pass the existing timestamp
  coverage and transition-ambiguity limits. No sample may be removed or
  replaced.
- Exactness and observer validity are adjudicated before timing.
- Conventional even half medians, half-to-half stationarity, matched-half
  savings, and nearest-rank overall p95 are computed for complete packed wall,
  total post-route command GPU, and the affected 16-layer routed GPU subtotal.
- Each arm and endpoint must remain within 5% half-to-half stationarity.
- Both halves must save at least 5% wall, 10% total post-route GPU, and 30%
  affected routed GPU.
- Candidate p95 must not exceed control p95 at any endpoint.
- Any failed condition ends the sole authorized attempt.

## Result

The complete warm block and all 16 timed executions ran. All eight warm arms
passed exactness, sampled topology, and grouped invocation checks.

After execution completed, adjudication began in timed schedule order. The first
timed sample is a control arm. It passed exactness, topology, invocation count,
and trace/census dispatch-count accounting, then failed the mandatory
observer-validity gate:

```text
sealed timed arm aggregate transition ambiguity
0.18274102410222184 exceeded 2.5%
```

The observed ambiguity is 18.27410241%, more than seven times the frozen 2.5%
limit. Because every timed sample had to be valid, the packet fails immediately.
The other 15 timed executions completed but were never adjudicated after the
first panic.

No sample is deleted, replaced, or reclassified. The gate is not relaxed, and
the condition is not rerun.

## Claim Limits

Authorized:

- all warm and timed executions completed;
- all warm arms passed exactness, topology, and invocation checks;
- the first timed control passed exactness and topology before observer
  adjudication;
- that control failed aggregate transition ambiguity at 18.27410241%; and
- the sealed packet is inadmissible for promotion.

Not authorized:

- any wall, post-route GPU, affected routed GPU, stationarity, saving, or p95
  claim;
- any all-IQ3 performance promotion or regression claim;
- a claim that the candidate caused the invalid observer sample;
- timed exactness, invocation, or dispatch-repeat claims for the other 15
  executions; or
- combining this run with earlier attribution to manufacture a result.

## Decision

Close production all-IQ3 grouped promotion for this device/asset/observer
contract. Do not retain the prior `HOLD` label: the one authorized gate is
complete and failed a mandatory validity condition.

Retain the exact candidate and the ignored failing harness under diagnostics as
executable negative evidence. Do not mark the test `should_panic`, move the
observer assertion, loosen the threshold, split routed sub-operations as a
rescue, or rerun this condition. Reopen only for material asset, device,
implementation, or measurement-capability drift that defines a genuinely new
contract.

The next force-ranked branch is the bounded selector product-promotion decision.
HCA tiling remains behind it.

## Evidence

- Base revision: `2506e7438b36cf1885bebd46b17f2236504fd9aa`.
- Device: Apple M4 Max; macOS 15.6.1 (24G90).
- Toolchain: Rust/Cargo 1.97.1.
- Asset: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`,
  104,207,848,032 bytes across the four census-pinned shards.
- Prefix: 128 exact IDs repeating `[35, 201, 200, 34]`.
- Continuation: canonical in-memory snapshot restore and exact token ID 35.
- `deepseek_v4_metal.rs` SHA-256:
  `b1a5fa084d592b20185cbfc52d2634082cd841c9bb97f00824dfba79c62d5777`.
- `deepseek_v4_metal/prefill.rs` SHA-256:
  `4f0ecea00de4c5dac7a7c8c273303957dd4f8f77b9499cce163b24850e354b95`.
- Release diagnostic executable SHA-256:
  `ac0a94b6abe801a412fba9ba2c074dad806a322d3db8ce43b36764c8e3e9ef6a`.
- Raw captured log SHA-256 before trailing-space normalization:
  `9201c69999c8405f5693f0c37329606f5c65747eb1522a4777adddd99b7b92ed`.
- Committed `integration.log` SHA-256:
  `616e82da84d8f617a0c3f776e56bfe009c03d3cf19470c6f2725d33a1b473dce`.
- CX review: `019fd4b0-a4d6-76f2-aee4-2172a72d8548`.

Command:

```bash
cargo test --release -p qwen-llm --features dsv4-diagnostics --lib \
  deepseek_v4_metal::tests::current_asset_packed_all_iq3_sealed_promotion_gate \
  -- --ignored --exact --nocapture
```

The process ran for 63.55 seconds in-test after release compilation. The test is
expected to fail under the frozen gate; that failure is the recorded decision,
not a validation omission.
