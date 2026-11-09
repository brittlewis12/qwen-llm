# Agent append roofline decomposition

Research on main `2bad80c4`, ordinary Flash-Next GSQ-RCO IQ3_S and GLM
UD-IQ3_XXS appends, **400 / 2000 rows at 65536**. Exact 512/2048 controls
detect qualified-route cliffs. V4.1's Engram host/I/O floor requires a separate
probe and is outside this two-family screen.

The committed production library and shaders remain unchanged. Install the
explicit seams **only in an isolated scratch worktree**, then build outside
the Metal lease:

```sh
python3 scripts/research/append-roofline/prepare.py
lockf -k "$HOME/.cache/qwen-llm/locks/build.lock" nice -n 10 mise exec -- \
  cargo build --offline --release -p qwen-llm --example append_roofline
python3 scripts/research/append-roofline/run.py --family fn \
  --binary target/release/examples/append_roofline --model "$MODEL" \
  --corpus "$CORPUS" --out "$RUNS/fn"
```

`run.py` acquires only the canonical Metal lease/wait, cools 25 seconds, checks
three thermal-zero readings three seconds apart, monitors once a second and
stops on pressure/deadline. One model/process at a time. No build lock in timing.

The corpus is the existing 660043-byte Rust fixture, reproducible from the
recipe in `docs/bench/2026-11-09-steady-concurrency.md`. Each family tokenizes
it directly: first 65536 tokens are the real prefix; following tokens supply
each append. Every repeat restores the same native snapshot outside timing.
One excluded warm and three retained native calls/width include complete
endpoint logits/status handling; all repeat hashes must match. A native after
control checks that profiling did not corrupt state.

Instrumentation adds an ordinary GLM command GPU timer and a **separate** FN
route-count capture call: 48 tiny exact I32 copies preserve per-layer counts
from the shared routing scratch. GLM already retains per-layer route IDs;
read them after timing. Actual native bank bytes and touched expert counts
define the weight traffic floor, rather than active-parameters-per-token times
row count or the whole GGUF size. Embedding/PLE gathers are not full tables.
No captured future values, alternative model arithmetic or runtime policy is
used. The route-copy and dispatch-profile outputs must match native bits.

Per-dispatch profiling deliberately splits encoders and removes concurrency.
Its attention/non-GEMM shares are diagnostic allocations, not exclusive
measurements of ordinary GPU time. Normalize their completion-increment shares
to ordinary GPU time, retain the perturbation ratio, and leave an explicit
residual. Checked 1-GiB loads give current bandwidth; a dependent checked
empty-launch curve gives a command/control **model**, not recoverable hardware
time. The tensor peak reuses the measured 131.676 TFLOP/s F16/F32 calibration.
Do not add overlapping predictor savings.

The physical roof is `max(useful FLOPs / peak, useful bytes / bandwidth)`.
Keep that whole-call bound separate from the additive diagnostic ledger:
learned-matrix compute/weight floor, context attention, non-GEMM, synthetic
command/control, nonoverlapping host, then remaining kernel excess. Attention
arithmetic appears in the whole-call bound, not a second time in that ledger's
learned-matrix floor. Weight reuse/padding/quant unpack, extra scratch traffic
and occupancy belong to the excess. Any proposed runtime change still needs
the usual teacher-forced NLL/KL qualification; historical ragged IQ/attention
gate failures are not overturned by a timing profile.

Keep compact per-kernel aggregates, route histograms, ordinary timings,
thermals and commit + command. Delete regenerable per-dispatch bulk, binaries,
build checkout and bundles at campaign end.
