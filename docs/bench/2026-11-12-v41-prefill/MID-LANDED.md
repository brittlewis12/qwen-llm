# V4.1 prefill: masked-index scoring at agent depths

Status: MID masked index promoted default on. This is a **MID-only excerpt** of
the shared `perf/v41-prefill` bench note; that branch retains the PV,
publication and Engram-ordering work and its own gates. Base `5ee5c8c4f`.
M5 Ultra, unchanged GSQ-RCO IQ2_XS four-shard GGUF plus disk IQ4_NL Engram,
384 experts/top-6. Both A/B arms use `iter`; serving must use release.
Runs are queued under task `v41-prefill`, holds at most 15 minutes.

Raw evidence: `metagross:$HOME/qwen-runs/2026-11-12-v41-prefill/`.
The corpus is the existing uncycled code witness, 164,054 tokens under this
artifact's tokenizer. Timings exclude model loading and state restoration.

## Lever: masked index scoring at agent depths

Prototype `95d622e9a`; `QWEN_V41_MASKED_REINDEX_MID`, now default on after the
gates below; `=0` restores the 128K eligibility threshold. Enable the existing
exact mask-skipping F32 MMA kernel for ordinary 512..2048-row appends from
49,152 to 131,071. The existing `QWEN_V41_MASKED_REINDEX` rollback still vetoes
it. Allowed scores retain their MMA and head-sum order; rejected scores remain
excluded by selection. No new memory, precision, candidate count or selection
policy, and no effect on cold roots or short appends.

The existing 128K guard leaves the requested 80K append on full-domain scoring.
Measure all score dispatches on the actual append, require >=1.25x in the
isolated dispatch census and bitwise logits/causal state/following decode,
then at least three native chronological pairs, then three-agent serving.

Initial niceness-19 M5 screen: **163.250 -> 108.310 ms, 1.507x** over all
index-score dispatches. Complete endpoint, causal state and following decode
are bitwise. Verified-niceness-10 repetition is **162.983 -> 107.521 ms,
1.51582x**, three retained pairs with the same complete bitwise witnesses.
Native/serve gates are recorded below.

### Native MID gate, completed 2026-11-13

Source `f69e9edeb`, effective niceness 10, same `iter` binary and restored
prefix in every arm. The predeclared warmup pair is excluded; retained order
is BA/AB/BA. At 1K@80K, baseline/candidate pairs in ms are
1053.505/1005.649, 1061.304/999.578 and 1054.623/999.916.
Means **1056.477 -> 1001.714 ms, 1.05467x**, 3/3 positive, pressure 0.
All eight arms' endpoint/state/following-teacher-step witnesses are bitwise.
Inactive 48K root means are 20048.214/20121.080 ms (-0.36%); claim no root gain.
The first baseline root takes 45043 ms and stays in raw evidence.
Compact evidence: `data/nice10-cell-masked-summary.json`.

### First MID served block, AB (reverse block pending)

Release SHA256 `08725bb811ecc9497c9c383be2d040ec7cbbf19bc180a34a55530740e76f18a6`,
niceness 10 recorded for both model children, cap 32, pressure 0/0. All nine
eligible equal-size appends improve; mean ratio **1.03881x**, median **1.03663x**.
All matched short appends: mean **1.03200x**, median **1.02924x**. Delivered
throughput **78.575 -> 80.517 tok/s**, gap p99 **0.15200/0.15126 s**.

Roots are inactive controls: native A **37.336/55.510/26.276 s**, B
**27.108/22.152/21.293 s**. Keep this large variation; claim no root speedup.
Visible client TTFTs are A 37.45/55.61/null, B 27.21/22.23/null seconds.
All three native root phases remain represented. No promotion before the
predeclared reversed block. `data/serve-masked-pair1w32-comparison.json`.

### MID reversed block and promotion, 2026-11-13

Same release SHA and cap 32. Reversed BA: all nine eligible matched appends
improve, mean **1.09442x**, median **1.07055x**. All matched short appends
mean **1.05123x**. Delivery **78.458 -> 80.283 tok/s**, gap p99
**0.15402/0.15110 s**. NI10 verified. Baseline A reaches pressure 2 for part
of the steady window; candidate B stays at 0. Pressure 2 is allowed and kept,
and this limits attribution of the larger reverse-block magnitude.

Across the two order directions, all 18 eligible matches improve. Visible
eligible client TTFT means are **1.5692 -> 1.5047 s (1.04283x)** in AB
and **1.6076 -> 1.4795 s (1.08655x)** in BA; keep every queued outlier. The
separate native 80K cell establishes **1.05467x** over three chronological
pairs, and the isolated stage **1.51582x**, all bitwise.
**Promote MID default on with `QWEN_V41_MASKED_REINDEX_MID=0` rollback.**
The existing `QWEN_V41_MASKED_REINDEX=0` veto remains.

The reverse roots are inactive controls, A **21.238/20.590/20.623 s**, B
**21.116/23.973/24.120 s**; visible TTFTs A 21.33/20.67/null, B
21.20/24.04/null. Keep all phases, claim no root speedup. All three sessions
fit and decode reaches width 3. `data/serve-masked-pair2w32-comparison.json`.

The shared branch note also carries the PV writeback, exact late-4K publication
and Engram-ordering records. Those candidates are not part of this landing.
