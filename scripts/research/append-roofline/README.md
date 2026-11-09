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
Separate uncovered transition gaps from interval cores; charge profile-minus-
native inflation to gaps first, then distribute the native kernel budget by
core shares. The maximum of residual gaps and checked dependent empty launches
is a command/serialization **model**, not exclusive hardware attribution.
Do not add those overlapping costs. This matters especially for GLM2000, where
instrumentation adds 21% GPU time, predominantly in uncovered transition gaps.
Keep the perturbation ratio and explicit kernel residual. The short 1-GiB
load control can still ramp its clock (retain all samples); hardware roofs use
the established warmed measured 1175.47 GB/s / 131.676 TFLOP/s calibrations.

The physical roof is `max(useful FLOPs / peak, useful bytes / bandwidth)`.
Keep that whole-call bound separate from the additive diagnostic ledger:
sum of per-projection learned compute/weight floors, context attention,
non-GEMM, command/serialization model, nonoverlapping host, then remaining
kernel excess. This stronger stage floor assumes the incumbent phase order.
Attention
arithmetic appears in the whole-call bound, not a second time in that ledger's
learned-matrix floor. Weight reuse/padding/quant unpack, extra scratch traffic
and occupancy belong to the excess. Any proposed runtime change still needs
the usual teacher-forced NLL/KL qualification; historical ragged IQ/attention
gate failures are not overturned by a timing profile. The later signed-IQ
qualification already widened >=1024 rows; sparse attention remains restricted.

Keep compact per-kernel aggregates, route histograms, ordinary timings,
thermals and commit + command. Delete regenerable per-dispatch bulk, binaries,
build checkout and bundles at campaign end.
