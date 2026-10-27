# Command-free prepared host ranges

Range advancement changes host execution only, not attention/FFN kernels,
publication arithmetic, selection,
control ordering, or GPU allocation/submission. No runtime switch is added.

`encode_rows_serial` validates the whole physical input, immutable weights,
serial encoder, active batch and first/last mode/capacity endpoints before
publication or advancement. Only prepared spans with neither Ordinary owner
publication nor per-row index work bypass the original row chain. Ordinary
owner bodies use precisely the span returned by the unchanged publisher;
heads/tails, unsupported publishers, pruning and per-row selection remain on
the original binding/restoration path. Batched selection still follows every
serial transition. The test-only old-host override does not change any GPU
publication, selection or arithmetic decision.

Checked cursor advancement is atomic on refusal and cannot cross a layer.
Ordinary final-layer advancement clears the batch and advances next_position.
CED advancement touches pending encoding only, including decoder `encoded`
at its end layer; it never finishes, publishes coverage, retires a group,
advances public completed frontiers, or marks the cursor ready.

## Exact selectors

All names have the prefix `deepseek_v41::forward::attention_gpu::`.
The acceptance inventory contains the complete names and ignore states.

CPU selectors (no Metal context):

```text
query_batch::range_tests::cursor_range_matches_repeated_rows_partial_exact_and_final
query_batch::range_tests::cursor_range_refusals_are_atomic
query_batch::range_tests::prepared_range_eligibility_preserves_publication_and_selection_fallbacks
query_batch::range_tests::ratio_head_body_tail_cursor_parity
query_batch::range_tests::old_host_override_is_nested_panic_safe_and_thread_local
ced::tests::pending_checked_ranges_match_rows_and_refuse_after_end_atomically
ced::tests::pending_ranges_encode_without_finishing_coverage_completion_or_group_readiness
tests::query_batch::range_execution::host_discovery_geometry_and_conservative_peak_fit_four_gib
tests::query_batch::range_execution::range_parity_schedule_keeps_ratio_two_and_ratio_one_candidate_ownership
```

Self-leased, ignored local discovery (not GPU execution or model timing):

```sh
QWEN_METAL_LEASE_WAIT=1 cargo test -p qwen-llm --lib deepseek_v41::forward::attention_gpu::tests::query_batch::range_execution::prepared_serial_host_encoding_discovery -- --exact --ignored --nocapture --test-threads=1
```

Use the repository opt3 **test** profile, not `--release`; debug assertions
must be on. Launch without MTL debug/shader validation, capture, injected
profilers, or device wrappers. This fixture needs a local Metal context but
neither M5 nor a model. Use an external <=15-second execution bound;
that runtime limit has not been measured or qualified.

The fixture prices one live N=4096 case plus immutable BF16 weights and the
workspace through production admission, conservatively counting half/weight
storage as F32 and adding 512 MiB CPU reserve, with a 4 GiB total ceiling.
N=128 and N=4096 share stable weights and one unsubmitted command/encoder.
Synthetic zero prepared payloads are never consumed by GPU work; the fixture
does not claim GPU preparation or completion. Setup is outside the clock;
the complete serial API, including all changed validation, is inside it.
Warm AB plus ABBA/BAAB/ABBA/BAAB yields 18 chronological JSON samples per case.
Both arms require empty dispatch/allocation censuses, unchanged original
bindings, and identical final cursor outcomes. There is no speed assertion.

Debug Metal parity selectors:

```text
deepseek_v41::forward::attention_gpu::tests::query_batch::range_execution::prepared_serial_range_old_host_bitexact_outputs_controls_pending_and_census
deepseek_v41::forward::attention_gpu::tests::ced::ced_owner_rows_reuse_preserves_published_cache_with_ordinary_batch_control
```

Run each with `QWEN_METAL_LEASE_WAIT=1 MTL_DEBUG_LAYER=1`, `--exact --ignored
--nocapture --test-threads=1`. The first uses BF16 head512/index128 for
all-selected, scored top-k, pruned fallback and odd head/body/tail cases,
with strict output/control/cache/pending/binding/cursor and ordered full
census equality. The second strengthens the existing CED poisoned-compressor
fixture with old-host/range bitwise and ordered-census comparisons and
encoded-but-not-finished/public-progress assertions. Neither is a claim of
Full-vs-dedicated-CED publisher parity.

## Composition and verification

Bounded history support in `6ee4173804b5ee14b99ea2bae79876bdcefa156b` includes
`active_history_floor` validation/counting in `encode_inner_with` and history
validation and bounded policy in `ced.rs`/`query_batch.rs`. These remain necessary
with range advancement: the range helper is policy-neutral and never infers a
history floor or approximation policy. Exact inventory entries are additive;
`docs/ENV.md` is generated from the composed source. Production provenance embeds
`attention_gpu.rs`, `query_batch.rs`, and `ced.rs`; all new modules are test-only.

Verification: six exact-source isolated cursor/override/pending CPU tests pass
using a temporary rustc harness (Result/require and unused control descriptor
stubs), not full-crate compilation. Rustfmt, diff whitespace checks, 14 offline
V4.1 source contracts, environment-knob and identity checks, inventory schema
and self-tests, witness self-tests and path-normalization checks pass. The
scalar encoder, row-binding wrapper and both publishers are byte-identical to
`9a7940c0`. Full-crate type-check/discovery, the other two new Rust CPU tests,
Metal fixtures, discovery timing and numerical qualification remain unverified.
No Cargo build, GPU test, model or remote execution was performed.
