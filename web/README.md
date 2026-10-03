# Qwen Lens Web

Preserved React/TypeScript workbench, recovered onto current serving semantics.
Bun owns development serving/HMR and production builds. No Vite, Next, SSR,
authentication flow, network assets or second production server. Current recovery
supports durable baseline generation and scoped plain readouts; fitted/intervention
and retained-capture producers remain pending. Compatible saved diagnostics remain
inspectable without inference.

## Run

From this directory:

```sh
bun install --frozen-lockfile
bun run build
```

From the repository root, start the existing resident server:

```sh
qwen serve -m MODEL --lens-data-dir JOB_DIRECTORY --web-root web/dist
```

`--web-root` requires `--lens-data-dir`, currently ordinary Qwen only. Native
generation is limited to metadata-qualified House Qwen3.6/3.8; unsupported
ordinary deployments retain readable history. Open the server's loopback URL.
Rust never invokes Bun. It loads immutable assets before model admission; the
manifest has a 64 MiB payload catalog limit, not a total allocation bound. API
paths cannot be shadowed; missing assets never receive an HTML fallback. Static
GET/HEAD requests use the bounded CPU control pool, not the execution reservation.

For development, `bun run dev` uses Bun HTML imports and HMR at
`127.0.0.1:3000` (`PORT` configurable). `/v1/*` proxies to the existing Rust server
at `QWEN_SERVE_ORIGIN=http://127.0.0.1:8737`; `QWEN_API_PREFIX` defaults to `/v1`.
This is a development hop, not an application backend. Path/query, statuses and
bodies are preserved. Both proxy and native API enforce their local browser-origin
boundary; connection failures remain visible. Authenticated remote access is not
part of this client.

`Bun.build` emits `dist/index.html`, hashed JS/CSS and `asset-manifest.json`.
The production bundle excludes tests, fixtures and development serving code.
No client URL router or subpath mount is needed.

## Workflow And Safeguards

Input, Execution and History have the same controls on phone and desktop. Messages,
typed assistant prefills and sampling are explicit. Capabilities determine supported
modes, layers, aliases and operations. An explicit baseline-only declaration is
validated; older capability responses retain the empty-list fallback. Unsupported
saved settings are preserved and blocked, never silently removed to make a run fit.

Versioned localStorage retains drafts, two selected job IDs, exact submitted body,
opaque key and attempt history. Web Locks serialize cross-tab creation. Persistence
must succeed before transmission; uncertain outcomes reuse identical bytes/key.
Only a matching-key `admission.state=not_accepted` response permits a corrected
request under a new key. Generic overload after a lost response remains uncertain.
Submission/cancellation is never automatically retried as a read. Read-only 503s
receive at most two short retries; discovery is sequenced to coexist with history
polling in the two-worker pool. Persistent failures remain in the error ledger.

Run-again reads the server's saved request, including another client's jobs, and
only prepares a draft. It does not submit. Reload, selection, history access and
viewer disconnect neither rerun nor cancel work. Cancellation is explicit and
does not promise immediate completion of an in-flight command.

New drafts bind discovery identities. Refresh, history copy and edits preserve
those assertions. They identify model metadata and registered asset fingerprints,
not authenticated weights or cross-version numerical equivalence. Explicit rebinding
requires confirmation and never changes saved retry bytes. Legacy history uses at
most one prepared-input record to recover complete known identities, otherwise
requires review. Async copies cannot replace newer edits; failed persistence cannot
silently change the in-memory target. Servers lacking preconditions remain readable
and recoverable but cannot accept new work from this client.

Result pages are validated, deduplicated by sequence and checked for mutation or
bad cursors. Empty live-end pages are valid. Exact sampled bytes determine continuous
output, including split UTF-8; individual fragments are marked, not invented text.
Prepared prompt bytes are checked separately from generated output. Generation,
observation and artifact publication failures remain distinct. Missing old prompt
text stays token-ID-only rather than being reconstructed from labels.

History previews show the last user message, with explicit truncation and distinct
empty/unavailable/not-supplied states. Escaped, wrapping, bidi-isolated text works on
phones. Refresh preserves older pages and known previews; conflicting immutable
previews are reported. No experiment/session hierarchy or full-text search is added.

## Saved Diagnostic Exploration

The recovered client retains original token/layer/candidate navigation for compatible
saved records. This is not a claim that the current server can produce every record.
Token windows cover 32 IDs/pieces; the optional numeric grid bounds presentation,
not requested backend coverage. Source phase/index, absolute position and predicted
position remain distinct. Missing measurements and unconsumed terminal samples
remain visible states instead of silently switching selection to a nearby result.

Saved readouts preserve native units, candidate universe, source/target layers and
provenance. Candidate pinning preserves exact scope and identity; unsupported new
diagnostic work remains blocked. Desktop uses adjacent source/measurement columns;
mobile retains the same actions in document order. Deployment/provenance/numeric
detail stays available in disclosures.

Retained arrays load only on explicit request. Hash, dimensions and source/head joins
are validated against saved metadata, not current discovery. Browser-only score,
rank, temperature-one full-vocabulary probability and entropy require no new GPU
work. Ties use token ID; only the selected score array stays cached. Whole-site
before/after metrics describe the complete ordered program, not per-operation
intermediates. Missing captures cannot be recovered from top-k history alone.

## Verification

```sh
bun run typecheck
bun test
bun run build
bun run browser-check.ts
bun run baseline-browser-check.ts
LENS_TEST_PLAIN_ONLY=1 bun run baseline-browser-check.ts
```

The historical browser check serves explicit fixture derivatives and exercises
phone editing, lost-ack recovery, identical retries, stale binding/rebinding,
scope pinning, operation order, retained arrays, late pages, history and persistence
failures. It uses existing Chromium via CDP, with GPU disabled; `BROWSER_BIN`
overrides `/Applications/Chromium.app/Contents/MacOS/Chromium`. Fixture capabilities
do not establish production execution support. No test observations enter the app.

The baseline browser check compiles and launches an isolated Rust test child with
the real control loop, static assets, store, native preparation and joined writer.
Only token forwards are synthetic CPU work. It checks mobile prefill/sampling,
actual submission/reload, recorded-byte output, history/copy without new jobs and
desktop parity under the real two-worker pool. It then signals only its owned
child, whose test harness verifies settlement. No model or GPU lease is used.
Plain mode additionally submits a decode-scoped readout and checks visible token
IDs/scores against the actual saved records. Its heads remain synthetic; it is not
a numerical Metal-head qualification.

Browser profiles/logs stay in ignored `.browser-test/`. Layout and interaction
assertions are not screenshot-based visual review. Optional screenshot capture
has historically stalled in this Chromium; it is not required by these checks.

For the separate bounded real-model lifecycle gate, run
`scripts/serve/lens_baseline_check.ts` from the repository root as described in
`docs/LENS-WEB.md`. Original richer live/oracle harnesses remain preserved in the
frozen worktree and return with their producing server slices. Their old evidence
does not qualify this branch. JS-unsafe u64 seeds, raw input, expanded operator
families, automatic discovery and richer plots are not added by this recovery.
