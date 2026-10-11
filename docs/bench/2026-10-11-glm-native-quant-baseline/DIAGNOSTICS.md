# Build Provenance and Dispatch Attribution

Two concrete infrastructure defects surfaced during HC optimization. Neither
was a model-capacity problem or grounds for narrowing model admission.

## Dispatch Identity

The census recorded an ordered list of dispatches but labeled each using the
last pipeline lookup. Preflighting A and B, then binding A, incorrectly labeled
A's dispatch as B. It also attributed interleaved encoders using thread-global
last-encoder state.

Pipelines now retain their immutable names as shared strings; encoders track
the actual binding and their own capture identity. Recording uses that binding,
including for pipelines cached before capture and encoders surviving a capture
restart. Inactive tracing adds shared-reference bookkeeping, not string
allocation or Objective-C name queries. Pipeline cache keys keep the same string
contents. The previous native-expert test workaround is removed: tests now check
the ordered gate and up names correctly.

Focused real Metal tests cover preflight order, cache hits, failed lookups,
interleaved encoders, capture restarts, and inactive threads. Context, native
expert, IQ1/IQ2, generic grouped, and GLM regression suites passed. All workspace
targets checked successfully (existing unrelated CLI dead-code warning).

## Shared Target Provenance

The worktrees share a Cargo target directory. Portable relative build-script
watches allowed new Rust code to embed another worktree's older metallib. This
produced an observed `NoFunction` for the new HC kernel despite successful Rust
compilation; the cached source inventory omitted `mhc_pre_native.metal`.

Commit `2b5e92db` emits an absolute rerun path for the build script itself. Cargo
normalizes it relative to the current package root: switching worktrees changes
the dependency fingerprint and reruns Metal compilation. Runtime manifest lookup
and relative kernel watches remain unchanged. An `env!("CARGO_MANIFEST_DIR")`
substitution is insufficient because Cargo filters that variable from dep-info.

Verified in the actual shared target after applying the fix to both worktrees:
main check rebuilt (14.44 seconds), implementation-lane check rebuilt (13.20
seconds), and its unchanged repeat was fresh (0.05 seconds). Cached build output
changed from main's absolute path and shader inventory to the implementation
lane's absolute path and included `mhc_pre_native.metal`. Rebuilt HC GPU tests
passed. No cache purge or environment-limit change was used.

## Retrospective

Repeated diagnostics failures called for fixing attribution, not teaching more
tests to accept wrong labels. Build outputs also need source provenance when
multiple worktrees share caches. Keep these fixes separate from the numerical
optimization so each can be reviewed and reverted independently.
