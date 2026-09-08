# Native Observer Controls

`qwen-lens read-full` supports an identity observer alongside fitted J/R
transports, and can retain every deployed vocabulary logit at a selected input
position. This uses the existing native model runtime, not the `llm` project's
inference or FFI stack. No dependency revisions are needed.

## Interface

```sh
qwen-lens read-full --model MODEL.gguf --logit-lens \
  --prompt 'The capital of France is' --layers 0,16,46,63 \
  --identity-cache CACHE --include-vector --full-output NEW_BUNDLE

qwen-lens read-full --model MODEL.gguf --full-lens FITTED_LENS \
  --token-ids 1,2,3 --layers 16,46,62 \
  --identity-cache CACHE --allow-unvalidated-transfer --full-output NEW_BUNDLE
```

- Choose exactly one of `--logit-lens` and `--full-lens`.
- Plain defaults to all actual model blocks; fitted defaults to the artifact's
  source layers. Layer IDs are zero-based and retain caller order.
- `--position` selects the token whose post-block state is read, predicting the
  next position. Only its causal prefix is forwarded. The supplied token list
  is retained, including any unused suffix. `--max-tokens` bounds supplied input
  without truncation; its default is 256 and can be raised explicitly.
- `--include-vector` reports the native source residual for plain, or the
  transported target-coordinate vector for J/R, before the output norm.
- `--full-output` creates an immutable directory and conflicts with `--output`.
  Compact top-k JSON is still printed to stdout. Without `--full-output`, the
  existing compact output contract remains.

Plain supports ordinary Qwen dense/MoE and Muse through their native passive
capture and deployed output tail. It requires no fitted artifact or transfer
acknowledgement. Published J/R artifact geometry and transfer checks remain:
having a native observer does not make a dense fitted transport valid on MoE.
Multi-stream residual architectures need a capture/head-coordinate adapter;
an arbitrary flattening is not a principled logit lens.

## What The Scores Mean

For plain, transport is identity, not an allocated or rounded F16 matrix.
Qwen applies the native output RMSNorm and dtype-dispatched LM head. Muse uses
its existing deployed tail, including its norm, logit scaling, and softcap.
Fitted readouts first apply their existing transport, then the deployed tail.

The binary is uncensored vocabulary coverage, not unscaled internal head
numerators: these are finite F32, pre-softmax deployed logits, including
architectural output transformations. Softmax-derived entropy, tail mass,
arbitrary token-set contrasts, and distribution divergences can be calculated
offline without treating missing top-k entries as zero. None of these makes an
early-layer lens distribution a calibrated belief or proves causal control.

Each bundle contains:

- `logits.f32le`: shape `[selected_layers, vocabulary]`, little-endian F32,
  ascending token IDs within each row and caller-selected layer order.
- `metadata.json`: dimensions, axes, byte count, payload BLAKE3, exact input,
  model content identity, observer/artifact identity, reader build identity,
  and the ordinary compact readout.

Rows are written into private staging and published without overwrite only
after completeness, finiteness, and metadata checks. Errors clean staging.
The payload is bounded to 16 GiB; metadata has a 16 MiB streaming limit and
retained-result preflight checks. Source captures and native scalar head work
also have allocation/admission checks. Build hashes do not comprehensively
bind runtime kernel settings, device, or driver: metadata explicitly disclaims
bitwise reproducibility from equal reader identities alone.

At Qwen's vocabulary size 248,320, one layer-position costs 993,280 bytes.
All 64 blocks at one position cost about 60.6 MiB; dense all-token retention
quickly becomes expensive. This is why full export is opt-in and scalar-first.
The packed tracer's optimized top-k path remains unchanged; it masks logits
during selection and must not expose that mutated buffer as full scores.

## Bounded Qualification (2026-09-07)

These checks qualify the measurement interface, not a new research hypothesis
or a new BF16/Q8 equivalence claim.

- The native Qwen3.6-27B Q8 final-block identity readout passes the runtime test
  against deployed inference logits on the same prompt.
- One frozen release binary compares scalar `read-full` against packed
  `trace-full` on three existing archived prefixes of 32, 128, and 512 tokens,
  both matched J and R, at layers 0, 16, 46, and 62: 24 paired cells.
- Top-1 agrees in 24/24 cells; all 24 top-8 sets agree. Maximum common-token
  logit difference is 0.006654; maximum top-1 margin change is 0.004874.
- At the J target-62 identity anchor, short/medium/long relative vector L2
  differences are 0.000624/0.000441/0.000698. Cosines are
  0.999999806/0.999999903/0.999999757; maximum coordinate differences are
  0.04011/0.01002/0.017296. Thus native serial/packed capture is close here,
  but not bit-exact. Other transported vectors mix forward and transport
  arithmetic differences. The shared J/R identity anchor is not independent
  corroboration by two observers.
- Nine full-score bundles, totaling 29 rows and 7,062,464 F32 values, pass
  shape/order/hash/finite checks and exact F32 agreement with compact scores.
  This includes Qwen plain layers 63,0, Muse plain layers 51,0 with top-40,
  and a fitted Muse J readout. Muse plain needs no transfer acknowledgement;
  the published fitted Muse artifact correctly still requires one.
- The 44 focused CLI CPU tests pass, covering parsing, native-vector semantics,
  existing fitted paths, streaming limits, incomplete output cleanup, ranking,
  and no-overwrite publication. The separate runtime CPU checks also pass.

Frozen executable SHA256:
`9e0b6bc9d5e027534c25334e156672ee4c02d13afaaec6a58b116a1600d3ae96`.
Private local artifacts, exact arguments, identities, logs, and analysis are in
`target/observer-controls/probes/live-verification/summary.json` in the active
observer-controls worktree. Archive text and model outputs are not committed.
Argument-rejection attempts are retained there; they are not successful runs.

This small panel does not qualify long contexts, all layers/positions, other
quantizations, MoE numerical equivalence, or Muse serial/packed equivalence.
It supplies no universal error tolerance, and near ties can reorder elsewhere.

## Research Use

The next high-leverage measurement is depth completion on existing prefixes,
not another wide completion campaign: cover early layers with R, compare plain
as an unfitted baseline, and keep J's distinct training objective visible.
R/J agreement is not a veto on early R-only structure. Use full scores at a
small number of sites to test whether top-k censoring changes apparent phase
structure; retain compact traces for wide coverage.

Do not infer basins from neighbors alone. A useful next analysis preserves
transition order and separates topic, communicative task, and induced metric.
The new interface helps test these alternatives without prematurely naming
clusters as model physiology or choosing intervention directions from labels.
