# N16 online attention: full-verifier phase PASS, product HOLD

The charged one-layer result survives real-model verification on
`Qwen3.8-27B-Q8_0.gguf`: warmed full-verifier wall falls **16.086%**, including
full-prefix V-transpose rebuilds on all16 attention layers. Partial8 restore
preserves a **16.865%** saving. With eight forced serial replay tokens included,
the saving is **5.696%**. These are measured packet costs, not layer extrapolation.

No production selector, scratch planner, drafter admission, template behavior,
long-context cutoff or default changes. The private workspace/hook exists only
under `cfg(test)`. The ordinary single-chunk-VT verifier rejection remains intact.

## Actual work and identity

Test implementation `8123e74d`; observer shape correction `e5e6abd9`. One loaded
model and one session, with a real32752-token prefix through existing packed1024
prefill. The next16 fixture tokens are teacher-forced, not drafter proposals.
Geometry: dense64,16 full-attention layers,24/4 heads,head256.

The Marcus fixture preserves all thinking/content using the documented raw ChatML
rendering, not a new production template implementation. This is a prefix/state
experiment, not template or conversation-quality qualification.

- Source: `docs/bench/tokenizer-messages/current-marcus-long.json`, SHA256
  `a41c38bacf3ff9ca32647e1485aacd2b8bcfddc3ab502fd4a45fa6cb05792238`.
- Rendered196541 bytes, SHA256
  `7f982cd896c7809430bdcdf489723e6eed6bfdc2a04882ee15bcf2a65c676352`.
- CPU tokenizer witness49561 tokens; prefix-ID SHA256
  `2f9c6aedee579283f7a74353e562d9222b15fa8fa1ba9eece73ca339b033398f`.
- Next16 IDs SHA256
  `3c9e65b983859fcf8c1ab00579a2742b4cc22e37e1d78e346168e90f86b0b7ce`.

All three runtime fixture assertions run before Metal initialization. Exact GDN
and convolution seed bytes plus the KV watermark are reset before every arm.
Unconsumed suffix bytes are overwritten by the next verify; all prefix K/V bytes
remain bitwise unchanged. Other activation/output/partial buffers are scratch
overwritten by the forward, not persistent continuation state.

## Frozen timing packet

Each mode warms A-B-B-A, then measures A-B-B-A once. Debug logits are allocated
only for the preceding correctness witness and dropped before timing. State
readbacks occur outside measured packets. Primary is synchronous
encode-through-completion verify+restore wall: full/partial8 must save >=10%
in aggregate and both pairs, with <=5% control spread. Forced replay8 has a <=3%
regression guard. No minimum selection or measured-loss rescue rerun.

| Mode | A1 ms | B1 ms | B2 ms | A2 ms | Saving | A spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full16 | 256.693750 | 216.350709 | 216.344708 | 258.945625 | 16.086% | 0.873% |
| Partial8 | 259.312958 | 216.009000 | 215.941792 | 260.263500 | 16.865% | 0.366% |
| Replay8 | 783.425000 | 739.498458 | 737.414375 | 782.690958 | 5.696% | 0.094% |

Full/partial means257.819688 ->216.347709 and259.788229 ->215.975396 ms.
Full pairs save15.716/16.452%; partial pairs16.699/17.030%. All gates pass.
Full-accept restore is the current no-op when final checkpoints are skipped;
partial restore costs approximately0.9 ms. Forced replay includes pre-block
restore plus eight serial forwards with hidden capture, approximately520 ms.
It excludes actual rejection decisions, drafter execution and ring publication.

The first A warmups are608.895/633.230/1158.576 ms for full/partial/replay;
their verify components are608.894/632.265/624.657 ms. They remain in the record
and are excluded only by the predeclared warmup protocol. No first-use or cold
claim follows, and no PSO/compression cause is established.

## State and allocation

All16 full-logit row argmaxes agree. All401 numerical comparisons pass cosine
>=0.999; the worst is0.999999850875 in a partial-restore V-cache tail. Maximum
absolute error across comparisons is0.03125 in an F16 V tail; no absolute-error
or bitwise packed-versus-packed gate was declared. Comparisons cover per-row
logits, full selected hidden capture, each GDN state/conv layer and each new KV
tail, including full/partial restore. Prefix KV is bitwise unchanged, and after
forced pre-block restore plus eight serial replays all persistent state payloads
agree bitwise. No sampled distribution, serial-oracle logits or generated greedy
continuation claim follows from teacher-forced A/B agreement.
At32K the existing fallback margin is0.75; teacher-forced rows6/10/12 are below
that margin in both arms. Only committed rows can incur the production guard,
so these gaps do not measure fallback incidence or accepted-chain quality.

Actual Metal allocation, reproduced in both attempts:

| Object | Bytes |
| --- | ---: |
| Model | 32321110016 |
| Session | 2332934144 |
| Ordinary verifier plus layer scratch | 2695266304 |
| Incremental one-layer VT/scores/ml | 93847552 |
| Combined model/session/scratch/online | 37443158016 |

The workspace remains allocated in both timing arms to match residency. The
baseline ledger is taken before allocating it; this is not a separate-process
peak-memory comparison or full-request admission result. Prefill allocation is
38053462016 B and its scratch is dropped before verification.

## Attempts, environment and review

Attempt01 fails before any verifier encoding: debug logits had the right byte
count but flat shape instead of the API's required `[16,248320]`. Its real-prefix
prefill162430.915 ms and allocation ledger remain recorded, with no A/B result.
Independent review approves the one-line observer repair under unchanged gates.
Attempt02 passes the full test in202.04 s; prefill162664.294 ms is setup, not
balanced timing authority. CPU fixture test and non-test release library check
also pass. Workspace-wide formatting check reports unrelated existing CLI/test
format differences; they are not changed by this work.

Both GPU attempts hold the ordinary production Metal lease externally around
the unit-test process, waiting for other owners without interruption. Across the
entire202-second second process, global compression/decompression grow109693/
28342, pageouts849, swapins4, swapouts0. These are not phase-local observations;
they neither explain the warmup outliers nor establish a quiet/cold host contract.
The frozen warmed-phase gates pass; cold and endpoint authority remain absent.

Raw artifacts: `target/profiles/full-verifier-online-n16/`, including original
`screen.log`, corrected `screen-02.log`, all build/execution/preflight/postflight
files, frozen fixture/CPU identity and `analysis.json`. The earlier primitive is
recorded in `docs/bench/2026-09-07-verifier-online-n16/RESULT.md`.

The independent checkpoint agrees on phase PASS and product HOLD. The production
default cutoff is16384, and the installed calibrated DFlash2 artifact uses physical
N8, not N16. Do not relabel this N16 result as a DFlash2 win or change block size
to manufacture reachability. An admitted active lane's block size, durable
acceptance and full charged economics must justify an endpoint experiment. The
measured mechanism is real; a user-visible improvement is not yet established.
