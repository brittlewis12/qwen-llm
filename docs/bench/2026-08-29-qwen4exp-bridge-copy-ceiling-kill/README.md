# Flash-Next Packed Bridge Copy Ceiling KILL

Decision: **KILL** packed bridge ownership work. Two accepted natural N=512
captures project only `1.69-1.70 ms` across all 92 post-PLE bridge copies,
17.5% of the preregistered `9.671346 ms` implementation gate.

## Screen

Each post-PLE block copies a `[2560, N]` F32 mixer output into a shared bridge
before attention combine, then repeats the copy for the MoE output before FFN
combine. At N=512, one copy has 5 MiB of payload and 10 MiB of aggregate
read/write traffic. Across 34 GDN and 12 QSA blocks, the 92 copies move 460 MiB
of payload, or 920 MiB counting reads and writes.

Existing six-stage profiles first supplied a zero-change upper bound: the four
copy-plus-combine spans projected `89.30-89.87 ms`, so they could not reject the
`9.671346 ms` gate without a split. The temporary profiling-only candidate then
split each representative layer's two copies from their combines. It changed
the selected block plan from six to eight serial sampled encoders, full-profile
samples from 128 to 136, and spans from 68 to 72. Ordinary packed execution and
all kernels remained unchanged.

Stage-boundary timing charges the new encoder boundary to each copy span and is
therefore an optimistic ownership screen, not a prediction of realizable
savings. The result also projects representative layer 5 and 7 measurements
across their mixer families rather than measuring every copy independently.

## Protocol

- Protocol and source commit: `cd6f4f068606cf649a280cf1628795d0e5c65bae`.
  Its preregistered roadmap blob is
  `a36269e2f45aa89a53ba8a9954176c5cf33c3bf1`.
- Temporary diff SHA-256:
  `86c86a0286240dff5824820cab437cb74259ce5f7c785ca185f0e27fd38be1ba`.
- Device: Apple M4 Max. Model: internal-SSD UD-Q3_K_XL from
  `unsloth/Qwen3.8-Flash-Next-GGUF` revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: tracked natural 512-token prompt, no special tokens, one generated
  token, exact 512-forward capacity, one packed command, and no scalar tail.
- Order: separate-process A1-B1-B2-A2 with five seconds between processes. A
  used the six-stage control; B used the eight-stage copy split.
- Projection per capture:
  `34*(L5 mixer copy + L5 MoE copy) + 12*(L7 mixer copy + L7 MoE copy)`.
  Positive split-composite inflation over interpolated controls was debited.

## Results

| Arm | Samples/spans | Warm GPU (ms) | Profiled GPU (ms) |
|:---|---:|---:|---:|
| A1 six-stage | 128 / 68 | 966.802125 | 968.041583 |
| B1 split | 136 / 72 | 967.313125 | 968.679125 |
| B2 split | 136 / 72 | 966.833375 | 968.458375 |
| A2 six-stage | 128 / 68 | 967.799125 | 969.051500 |

| Capture | Raw copy projection (ms) | Topology debit (ms) | Adjusted (ms) |
|:---|---:|---:|---:|
| B1 | 1.703394 | 0.009584 | 1.693810 |
| B2 | 1.697572 | 0.000000 | 1.697572 |

Every individual copy span was `0.018250-0.018625 ms`. The adjusted projections
miss the implementation gate by `7.977536/7.973774 ms` and represent only
0.175% of profiled command GPU. Six-stage control drift was 0.103% warm and
0.104% profiled. All observers passed, raw coverage was 1.0, and all four arms
produced the same generated-output SHA-256. These are timing and deterministic
replay checks, not an external semantic oracle.

The temporary split and both executable copies were removed. `cleanup.json`
records a clean source tree for the two touched files, absent temporary
executables, and the restored release/metallib hashes. Raw logs remain under
`target/profiles/qwen4exp-bridge-copy-ceiling-20260829/`.

Adversarial design and evidence review:
`01a04f60-7ce3-74f2-bc55-8378091e6d8f` (PASS).
