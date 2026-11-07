# Default-off performance knob cleanup

Measured code: `314461981eea1423ec25d529deba1f0310f39bad`. Apple M4 Max, 128 GiB, macOS 15.7.9, AC power.

25 retirement decisions are implemented in ten family commits: 23 environment names
are removed, plus the large-hidden Q4 fusion override and page-rounded-copy arm.
The existing small-dense fusion rollback and ordinary copy/pread controls remain.
Shared active kernels, retained-loader primitives and the serialized Qwen Q8
snapshot tag remain. The six FINISH items are untouched.

The retirement commits remove **6,969 Rust/Metal lines**, including tests, and
add 551 (net **6,418 fewer**). Including the three promotions, the code diff is
753 insertions / 7,088 deletions (net **6,335 fewer**).

[Retirement manifest](retirements.json) · [Measurements and checks](results.json)

| Automatic path | M4 confirmation |
| --- | --- |
| Untied dense 27B Q8 embedding | 5,085,593,600 → 1,350,860,800 tensor bytes; **3,734,732,800 B saved**. Thirteen gathered rows, prefill logits/state, 64 greedy logits/token steps and final KV/GDN state match bitwise. Ordinary CLI arms match 16 tokens and stdout. |
| A3B MTP block-40 Q4_K/Q4_K/Q5_K expert banks | 3,221,225,472 → 486,539,264 bank bytes; **2,734,686,208 B saved (84.9%)**. Three-position composed logits have cosine 1.000000, maximum difference 4e-6 and matching argmax. |
| A3B serial GPU greedy selection | Two complete ABBA campaigns, 128 tokens/process: mean generation **1395.336 → 1287.184 ms**, **1.08402×**, 7.75% less wall time. Every token hash and stdout matches. The JSONL rollback/positive-alias pair also matches. |

Each knob is an off-switch: `QWEN_NATIVE_QUANT_EMBED=0`,
`QWEN_MTP_MOE_NATIVE_BANKS=0`, `QWEN_GREEDY_GPU_ARGMAX=0`.
Positive values retain automatic qualification.

The greedy samples use one frozen release binary, fresh processes, both the build
lock and the inherited Metal lease, niceness 10, 30 seconds of stable normal
thermal pressure before each sample, and a compiler-overlap watcher. All accepted
samples have thermal pressure zero before/after and no sampled Rust compiler.
The per-campaign and chronological pair ratios are retained in `results.json`;
the pooled mean includes both campaigns, including their slower arms.

The embedding witness's post-load device counter also includes loader/driver
ownership. Both independent witness pairs and their observed totals are retained
in `results.json`; the tensor-byte reduction is identical in both.

## MTP decoder limitation

The D7/N8 normal decoder fails its terminal-resume state audit on a fresh build
of main `eb02cb0b7494710e092761bbe235d8aac13be887`.
Main F32 banks, this branch's F32 rollback and this branch's automatic native banks
produce the same 128 target tokens and **identical values for every reported
state-audit metric**. KV cosine is 0.9994419856 and continuation-logit cosine is
0.9988788770. This is an existing decoder limitation, separate from bank storage.
A legacy D1 F32 trial also differs from the product packed-prefill reference.
These rejected decoder probes supply no new end-to-end MTP throughput claim.
The bank promotion is supported by storage, composed-logit and relative-state checks.

## Verification

- `cargo fmt --all` and format checks.
- Full `cargo test --release -p qwen-llm --lib`: **2457 passed, 0 failed, 739 ignored**.
- `cargo test --release --workspace --no-run`: passed.
- `scripts/check-contracts.sh`: passed on the code commit; the result commit is sealed separately.
- Diagnostics-feature memory inventory and GPU-greedy policy tests passed.
- Twelve retention-evaluation tests and the changed text-capability environment test passed.
- Greedy smokes: Qwen 0.8B, dense 27B Q4/Q8, A3B, Flash-Next, Muse, K2, DS4 and GLM.
- DFlash greedy equivalence: eight tokens with online attention default and rollback.
- V4.1 target artifact is absent from the laptop and mounted model archive.

The broader text-capability suite has a pre-existing frozen-packet digest mismatch:
its actual packet is identical on main and this branch. Expected and actual hashes
are recorded in `results.json`; the packet and its frozen digest are unchanged.

The complete accepted receipts and setup/rejected probes remain locally under
`bench/results/knob-sweep/`. Native row/state packets are included here as
[native-off.json](native-off.json) and [native-on.json](native-on.json).
