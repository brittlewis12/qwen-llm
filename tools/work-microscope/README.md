# Work Microscope

An offline, source-linked experiment for discovering optimization opportunities in
qwen-llm. Follow a logical tensor through representations and accesses, compare
variants, inspect schematic storage use, and connect a pattern to cost evidence.
This is separate from the Lens workbench and never loads a model or launches GPU
work. The interface and source excerpts work without external assets or services.

## Run

From this directory:

```sh
bun install --frozen-lockfile
bun run dev
```

Open `http://127.0.0.1:4187`. `PORT` selects a different loopback port. `dev` builds
once, then serves; it is not a hot-reload server. After edits, stop it and run the
command again. Alternatively, use `bun run build` then `bun run preview`.

Builds need the repository's historical Git objects: `evidence.ts` reads pinned
commits, checks tracked regular files and line ranges, and bundles excerpts with
the full source files' SHA-256 hashes. Missing history is an error, not a fallback
to current files or a network fetch. Generated assets stay in ignored `dist/`.
The preview serves only that directory, handles GET/HEAD, and binds to loopback.
It neither reads arbitrary browser-requested source paths nor connects to the
resident inference server.

## First Investigations

- V4.1 query-half mirror: conversion once versus per key panel. Query reads still
  repeat; the per-query diagram is distinct from the 512-query probe and 8K CED
  throughput evidence.
- V4.1 scratch reuse: retain full-size scratch, reset incoming at every chunk,
  and drop the maximum set before allocating a smaller tail. Aggregate full-model
  bytes are deliberately unpriced.
- Flash-Next bridge copies: 460 MiB payload / 920 MiB logical read-plus-write,
  but only a reported 0.175% adjusted profiled-GPU cost projection. The two
  variants change observation boundaries, not tensor ownership or kernels.

These are curated historical source reconstructions, **not live traces**. They
are calibration cases: the instrument should expose the two reported wins and
help deprioritize the copy-ownership negative. It does not reproduce those runs
or turn reported projections into fresh measurements.

## Explore

1. Choose an investigation and a variant. Select any value or operation to see
   its immediate access neighborhood, repeat count, logical bytes, and sources.
2. Toggle Follow tensor to emphasize its identity across representations; Compare
   shows both variants while keeping the selected tensor as the common focus.
3. Scrub source order. The steps and storage bars are ordinal schematic use
   windows, never elapsed time, observed allocation lifetimes, or reuse proofs.
4. Open source buttons for pinned excerpts. Read the evidence ledger and limits,
   then write and download a hypothesis and discriminating experiment.

Selection is retained in the URL fragment for bundled cases. Hypothesis notes
stay only in the current tab's memory until downloaded. Switching cases preserves
those notes; importing a packet does not attach an earlier case's notes to a new
object merely because its ID matches. Neither notes nor packet selection are
uploaded or stored in browser persistence.

## Add An Investigation

Export a packet to use as a schema example. Edit the JSON and import it locally,
or create an `Investigation` in `cases.ts` for a maintained example. An imported
file can be a single investigation or an `EvidenceBundle` with investigations
and optional source snapshots (an empty `snapshots` array is valid).

`model.ts` defines the version-1 contract. `validate.ts` enforces a 2 MiB packet
limit, bounded arrays/text, finite safe-integer extents, unique identities,
connected acyclic source-order graphs, valid references, non-overlapping
schematic storage segments, restricted repo-relative paths, full Git revisions,
and matching snapshot metadata/line counts. Missing bytes use `null`, not zero;
unknown evidence can have no source citation. Unsupported versions and unknown
fields are refused rather than guessed. Imported strings render as text, never
HTML. No path or URL in a packet triggers a source fetch.

Validation establishes **structure, not truth**. An importer can claim a
measurement, citation, hash, or excerpt; the browser does not authenticate those
claims. Imported evidence is visibly labeled untrusted. To audit maintained
examples, use the Git-based build and review the source annotations.

Each edge's `logicalBytes` is a per-repetition, source-level quantity. Copy edges
price payload, conversion edges price result bytes, and read/write edges price
their stated accesses. `repeat` multiplies that quantity for the displayed
scope. These heterogeneous quantities must not be summed as unique traffic.
Views and reuse are not transfers; edge widths do not encode cost. The per-query
Q-half path also includes its separate F32 staging read.

## Evidence Boundaries

- Measured means **reported in the cited packet**, not newly measured here.
- Derived means arithmetic from stated source geometry, not hardware counters.
- Curated means a source-backed interpretation or layout, not an execution trace.
- Unknown is an explicit absence of evidence, not zero cost.

Logical bytes are not measured DRAM traffic. Allocated, logically live,
GPU-in-flight, and physically resident memory are different concepts. Sampled
stage intervals may overlap; isolated-dispatch profiling can remove concurrency.
This tool does not sum those intervals, infer exclusive stage shares, construct
a production critical path, or calculate hypothetical whole-model speedups.

The first version has no automatic dispatch/access capture. A binding trace alone
would not establish read/write semantics, indirect access ranges, or value
versions. Additional blocks need audited argument/access annotations; the UI does
not pretend to derive them by parsing arbitrary Rust or Metal.

## Verify

```sh
bun run typecheck
bun test
bun run build
bun run browser-check
```

The browser check uses the existing Chromium installation, an owned ephemeral
loopback preview, and an isolated profile. It disables GPU acceleration and tests
desktop/mobile navigation, graph/inspector linkage, scrubbing, source dialogs,
imports/exports, literal-text rendering, and absence of external page requests.
It does not install a browser or contact an inference service. Optional
`CAPTURE_SCREENSHOTS=1 bun run browser-check` attempts bounded screenshots;
artifacts stay in ignored `.browser-test/`.

Screenshot capture may be unavailable in headless Chromium; layout and input
checks are not screenshot-based visual review. Some CPU headless builds also
withhold mouse-move acknowledgements. The harness permits that transport warning
only after checking every requested range value, trusted input events, stable
element identity, and the matching descriptions, counters, and accessible values.
