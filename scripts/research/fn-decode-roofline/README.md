# Flash-Next single-stream decode predictors

Research only, based on `09a3a521`. Production sources remain unchanged in the
committed tree. `capture.patch` instruments the existing encoder for this probe;
it is not a runtime optimization. The capture code extends the archived Nov 6
exact replay experiment. No batching-lane files or APIs are changed.

Evidence retention update: compact metrics, numerical checks, commands and
summaries remain under `~/qwen-runs/2026-11-09-fn-decode-roofline/`. Frozen
binaries, sealed sources, patches and regenerable capture bulk were removed;
`RETENTION.json` identifies source commit `8f2dedac`. This supersedes historical
evidence descriptions. Subsequent layer/cache results are in `layer/` and
`cache/`; the best causal executor and reusable GPU feedback remain integration
candidates despite missing the 120 tok/s design gate.

Apply the patch in an isolated worktree, copy `probe.rs` to
`crates/qwen-llm/examples/fn_decode_roofline.rs`, then build with the usual build
lock (outside the GPU lease):

```sh
git apply --unidiff-zero scripts/research/fn-decode-roofline/capture.patch
cp scripts/research/fn-decode-roofline/probe.rs crates/qwen-llm/examples/fn_decode_roofline.rs
lockf -k "$HOME/.cache/qwen-llm/locks/build.lock" nice -n 10 \
  cargo build --release -p qwen-llm --example fn_decode_roofline
python3 scripts/research/fn-decode-roofline/run.py \
  --model "$MODEL" --binary target/release/examples/fn_decode_roofline \
  --out "$HOME/qwen-runs/fn-decode-roofline"
python3 scripts/research/fn-decode-roofline/summarize.py \
  "$HOME/qwen-runs/fn-decode-roofline/probe.jsonl" \
  --output "$HOME/qwen-runs/fn-decode-roofline/summary.json"
```

The controller waits for three thermal-zero readings outside the lease and
monitors pressure once a second. Only the probe takes the canonical Metal
lease, with `QWEN_METAL_LEASE_WAIT=1`. Native measurements use the same salted
8192-token prefix and 64 warm decode steps as the Nov 6 census. Three retained
32-token repetitions restore one entry snapshot and copy all vocabulary logits.
All samples are retained. The after-probe native repetition detects drift and
state corruption.

The probe measures:

- **Native / full replay:** one ordinary serial encoder, all real dispatches.
  Full-vocabulary bits must match the native call.
- **Causal DAG replay:** real shared buffers and reflected used binding ranges;
  a concurrent encoder drains all pending work before every RAW, WAR or WAW
  conflict. This conservative schedule retains dispatch order and buffers.
  Full logits must match. It does not topologically reorder work or distinguish
  atomic status writes from ordinary writes.
- **Empty-launch curve:** checked one-threadgroup dependent increments, with
  48, 96, 192, 384, 768 and the real number of dispatches. This is a model of
  command/control cost, not an exclusive hardware attribution. It is warmed
  by a full graph before each sample; alternating count order limits drift.
- **Streaming bandwidth:** 1 GiB of checked nonzero reads, two initial probes
  excluded and seven retained. Logical traffic uses the prior census's
  unchanged 5.603241 GB/token; this is not a DRAM transaction counter.
- **Frozen serial / branch / global concurrency:** execute the real graph once
  one dispatch at a time. Save each dispatch's actual mutable inputs, preserve
  aliases within that dispatch, and give it private buffers. Immutable weights
  remain shared. Replays execute all original arithmetic at the recorded inputs.
  Every reflected output byte, including persistent state, must match the serial
  capture. Branch concurrency drains at each HC injection; global concurrency
  removes those boundaries as well. Both deliberately break model causality.
  They are optimistic occupancy predictors, never candidate decode forwards.

Replays flush a separate 1 GiB stream before each timed command. Snapshot
restores, private-buffer preparation, output checks and flushes are outside
timing. Real/frozen serial controls expose the effect of private buffers and
changed cache reuse. Order reverses on alternate repetitions.

The additive table separates logical byte time, a synthetic command/control
allocation, remaining kernel excess, and measured nonoverlapping host time.
Kernel excess includes unpacking, scratch/repeated traffic and occupancy; it
does not establish that all those milliseconds can be recovered. Frozen
concurrency and launch removal overlap: their projected savings must not be
added. Numerical equality here validates the predictor construction only.
Any future candidate still needs signed teacher-forced dNLL/KL with a repeat
noise floor and F64 operator error against the incumbent before rollout.

## First packet and decision

Raw data: `~/qwen-runs/2026-11-09-fn-decode-roofline/first/`. Native means
are **13.391 wall / 11.994 GPU / 0.863 encode ms/token**; the after check is
13.322 wall. Stream bandwidth is **1170.77 GB/s** (retained median). The
synthetic allocation is **4.786 byte + 1.866 command + 5.342 kernel excess +
1.397 host = 13.391 ms**. Empty-launch time has two alternating populations,
**1.548–2.152 ms** at 1316 dispatches; command cost is not a fixed constant.

Real serial / conservative DAG replay medians are **11.884 / 11.590 ms GPU**,
with **1153 barriers**, and exact full logits. Frozen serial / 97-branch /
global medians are **12.560 / 7.923 / 6.311 ms GPU**. All eight repetitions per
mode match **400,774,256 output bytes** each; private input/buffer storage is
668,165,476 bytes. The private serial control is slower than the real graph.
Consequently give both direct replacement and paired-delta projections:

| Predictor | Projected native tok/s | Meaning |
| --- | ---: | --- |
| Launch count 1316 to 48, fixed kernel work | 86 | Launch removal alone is insufficient |
| Conservative causal concurrent encoder | 76.4 | About 0.295 ms GPU saved; no reordering |
| Ideal independent work inside each HC branch | 107–114 | Noncausal bound; original host cost |
| Same branch bound, all 0.863 ms encode hidden | 118–127 | Optimistic design budget |
| All kernels independent across the model | 130–140 | Extreme noncausal bound; original host cost |

These are not confidence intervals or measured candidate decode rates. The
range reflects direct replacement versus correcting for the frozen serial
control's extra cost. Do not add launch savings to concurrency savings.

The design worth a next bounded experiment is a **multi-stage layer executor
that coarsens dependent native-quant work and fills the GPU across stage
boundaries, plus command preencoding**. A concurrent encoder alone leaves
1153 boundaries; replacing launches alone leaves the kernel excess. Keep the
large Q6 head stream. The branch-only ideal plus perfect encode hiding puts
120 tok/s at the edge of the evidence; this is not a forecast of a production
implementation. Use **about 120 tok/s, with an optimistic 118–127 band**, as
the design budget, and establish causal feasibility before a larger build.

Next increment, if selected: one real GDN layer and one QSA layer, with actual
incumbent inputs/weights/state and all dependencies preserved. A concrete
120 tok/s budget is **36 × 140 us GDN + 12 × 180 us QSA + 0.60 ms head/tail +
0.53 ms host = 8.33 ms/token**. These are acceptance targets, not measured
layer times. Failure to approach that weighted layer budget ends this design
before full-model integration. Persistent scheduling also needs a valid Metal
synchronization/forward-progress scheme; the frozen control proves no such
scheme. No runtime change or default is shipped by these predictors.
