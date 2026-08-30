# Muse Source-50 Row-Slab Control KEEP

Decision: **KEEP** the source-50-only `fit-rows` path as the first production
consumer of the qualified full-attention VJP bank. It closes estimator layout,
prompt accumulation, resume, and artifact publication without launching a fit.

This packet records the pre-integration checkpoint. The preserving full-R merge
supersedes its source-50-only artifact schema with one all-source `[S,R,H]`
schema while retaining the same bank as the full-block motor.

## Contract

The first artifact is deliberately narrow:

- released Muse Q8_0 profile;
- R rule only;
- target block 51 to adjacent post-block source 50;
- B32 execution;
- contiguous shards of at most 256 target rows;
- output orientation `[source=1,target_row,source_coordinate]`;
- arithmetic mean over valid source positions, then used prompts.

Each prompt captures the production scalar trajectory with its F16 attention
KV path. The VJP is a local smooth-F32 block derivative at that captured
trajectory, not a coherent global F32 replay. Those coordinate, replay, and
production semantics are embedded in both the immutable config and fit
manifest.

## Execution

The library prepares block 51 once per prompt and allocates one current bank,
one next bank, and one workspace. Every row chunk clears the complete bank,
places one unit target coordinate at each `skip_first..T-1` position, executes
one caller-owned command, waits, and means the matching source positions.
Partial tails retain the fixed bank shape but emit only active rows.

The CLI accumulates prompt results in canonical row order and checkpoints after
every used or skipped record. Checkpoint publication writes immutable
generation sums, atomically replaces the cursor, then removes the previous
generation. Resume validates the exact corpus prefix, generation, counters,
shape, finite sums, and replay schedule before continuing.

## Identity

Muse row shards do not compute a model-content digest. Their local locator
hashes retained file stamps, artifact profile, tokenizer/chat metadata, and the
ordered tensor descriptor table. It hashes zero weight bytes and records
`content_authenticated=false` plus `locator_weight_bytes_hashed=0`. This is an
honest local resume/merge identity, not a portable content-authentication claim.

## Gates

- Model-free row-slab fitting matches scalar one-hot basis construction and
  source-position reduction under J and R, including B4 chunk reuse and a
  partial tail. Focused runtime: `0.08 s` after compilation.
- R-only/B32/256-row contract validation and replay-diagnostic merging pass.
- A filesystem-only checkpoint round trip preserves generation, cursor, sums,
  and diagnostics. Three focused CLI tests complete in `0.02 s`.
- Final adversarial verdict: KEEP, with no remaining source or artifact blocker.

No model asset, model-content hash, GPU corpus run, or broad suite was used for
this checkpoint. At 256 rows, the F32 accumulator is about 6.5 MiB. Eight
qualified B32 block commands project to about 0.41 GPU-seconds per prompt before
capture, preparation, reduction, and checkpoint overhead.

## Disposition

The next evidence is one bounded real-Q8 artifact: one prompt and 32 rows. Its
purpose is only to verify capture, command, checkpoint, and publication against
resident released weights. A multi-prompt or full-row fit remains prohibited.
After that smoke, inverse-RoPE sliding-block VJP integration is the next
mathematical seam.

Adversarial design and promotion review:
`01a05072-6951-7782-978a-2f274d90f474`.
