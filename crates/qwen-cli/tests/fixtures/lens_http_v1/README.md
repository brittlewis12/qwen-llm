# Lens HTTP v1 contract

These are representative wire fixtures, not claims about currently implemented
routes or the capabilities of a particular loaded model. Responses use JSON
except explicit retained-array downloads, which use binary F32LE payloads.
Capability values and limits are illustrative; clients must use returned values.

Baseline, plain/fitted readouts, scoped interventions and retained/pair capture are
wired on qualified ordinary Qwen deployments. The capability-gated browser, stored
history and static serving are recovered. Diagnostic schema descriptions also cover
compatible historical artifacts; illustrative fixtures do not qualify every model.

## Routes

- GET /v1/lens/capabilities: capabilities.json. Unsupported runtimes return 200
  with available=false and a stable unavailable_reason; never claim qualification.
- GET /v1/lens/assets: assets.json. No filesystem paths are exposed or accepted.
  `plain` is the native model readout alias; a fitted asset is optional.
- POST /v1/lens/jobs: request.json; 202 and Location header after durable acceptance.
  Body is status.json with state=queued, generation.state=pending, phase=null,
   zero counters, observations.state=pending (not_requested without readouts or
   residual pairs).
  These are the initial states; the returned status can already reflect progress
  or terminal cancellation/delivery failure that raced with the response.
  Identical idempotent submissions return 200 with the existing status. Different
  normalized content under the same key returns 409/error.json.
- GET /v1/lens/jobs?cursor=...&limit=...: {schema_version:1,jobs:[status],next_cursor:null|string}.
  Current responses additionally include `request_previews:{job_id:preview|null}`
  covering exactly the page's unique job IDs; clients accept older responses
  without that field. `request_preview.json` illustrates one preview.
- GET /v1/lens/jobs/{id}: status.json.
- GET /v1/lens/jobs/{id}/request: {schema_version:1,job_id:string,request:object}.
  Returns the retained authored request, without rendering or inference; it is
  available even when this server cannot execute that model or diagnostic plan.
- POST /v1/lens/jobs/{id}/cancel: empty body or {}; 200/status with cancellation
  requested. Repeated requests and terminal jobs are harmless; terminal state
  is not rewritten. A request is not an acknowledgment of GPU completion.
- GET /v1/lens/jobs/{id}/result?cursor=...&limit=...: result.json. Records are
  immutable and ordered by seq. Cursors are opaque. At the current live end,
  next_cursor is a resumable cursor and complete=false; only terminal publication
  uses next_cursor=null, complete=true. Empty pages are valid.
- GET /: prebuilt frontend; root-relative assets from independently configured
  `--web-root`. HEAD omits the body.
- GET /v1/lens/jobs/{id}/arrays/{record_offset}: verified committed F32LE array,
  `application/octet-stream`. Locator is the descriptor's record-log byte offset,
  not a filename or GPU query. Whole-array reads only; query parameters rejected.

No preview endpoint in edition 1. No HTTP file paths, model selection, transfer
overrides, raw prompts, raw tokens, or tool messages in the Lens input schema.
Ordinary /v1/responses remains its existing independent wire contract.

Current production wiring: `--lens-data-dir` enables stored history on every serving
family, with native generation for metadata-qualified ordinary House Qwen3.6/3.8.
Supported passive heads advertise plain readouts, retention and whole-site pairs.
Explicit registered fitted assets add fitted readouts and supported direction rows;
native operators remain `fixed_add`, `residual_l2_fraction`, `projection_ablate`.
Without a supported passive head, baseline generation and empty assets remain.
Other deployments remain native-execution-unavailable, not history-unavailable.
`capabilities_unavailable` is shared with browser tests: unavailable model metadata
is null, not zero dimensions or invented identities. Full diagnostic fixtures
remain illustrative. See `docs/LENS-WEB.md` for the actual qualification boundary;
old-worktree live evidence does not qualify this reconstruction.

Native Lens routes require a localhost/loopback Host. If Origin is present it
must match that local HTTP authority; headerless CLI clients remain supported.
The Bun dev proxy validates its own browser origin before forwarding to Rust.
These are local browser request checks, not an authentication or remote-access
feature.

## Request types

Unknown fields are rejected. schema_version is exactly 1. idempotency_key is a
nonempty bounded opaque string, retained for the job lifetime. input.kind is
messages. messages is a nonempty array of {role,content,reasoning?}; roles are
system/user/assistant with validated conversation grammar; reasoning is assistant
only. generation_mode uses LensMessageMode snake_case. assistant_prefill is absent
or {channel:reasoning|final,text:string}, subject to template validation. Its text
is prompt input, never new output. Unsupported channel/mode combinations fail.

generation and its sampling object are required. All five sampling fields are
required and use existing Sampler validation. max_new_tokens is positive.
diagnostics is optional (isolated diagnostic baseline) or contains directions,
operations, readouts arrays and optional residual_pairs. Each pair has a bounded
unique id and a numeric scope; it captures before/after the whole ordered site
program, not per-operation intermediates or a counterfactual baseline. Effective
operation IDs describe that local program. Committed pair records count toward
observations alongside readouts, while array descriptors do not. Pair scopes and
deduplicated before/after arrays are admitted against advertised row/raw-byte
budgets; after arrays can share with readout retention. Pair-only requests need
no readout head or fitted asset. All runs on this endpoint isolate ordinary caches.

`preconditions` is optional for API compatibility; new browser work requires it:
`{model_identity:string,asset_identities:{alias:string}}`. Unknown fields/null
are rejected. Identities contain 1-256 UTF-8 bytes; aliases contain 1-64 ASCII
letters/digits/underscores/hyphens, with at most 65 entries. The map covers exactly
the distinct aliases referenced by every direction/readout, including unused
directions and zero operations; baseline-only requests use an empty map.
Capabilities advertise `request_preconditions:true`; the assets envelope carries
`model_identity` so clients can reject discovery responses from different models.
Model/plain identity is advertised GGUF metadata, not weight-content authentication.
Fitted identities are registered manifest fingerprints, not quality claims.
Malformed assertions/coverage return `400`; valid mismatches return
`412 binding_mismatch` with a matching-key `not_accepted` decision, before queue
reservation or acceptance. Existing accepted-key lookup precedes current-profile
validation, even when full/unavailable; changed assertions under that key are
still a `409` conflict. Preconditions do not select a model or override binding.

Direction normalization: as_stored|unit_l2. Row kind: token_id, template_row_id,
or label, only when supported by the alias. The optional target_covector uses the
existing Lens enum. Operations use existing native Action serialization:
fixed_add, residual_l2_fraction, projection_ablate have direction/coefficient;
source_to_target and coordinate_swap have source/target/coefficient. The UI label
"Residual L2 add" maps to residual_l2_fraction, NOT residual_l2_add.
Multiple operations apply in array order at each layer.
The preserved diagnostic HTTP subset permits registered fitted token rows with
`deployed_logit_numerator` target covectors (also the default):
`M^T (gamma * LM-head-row)`, not normalized-logit gradients. Relative-L2 and
projection operators require `unit_l2`. Zero coefficients validate every field
and scope before omission from effective preparation/application. The shared
decoder rejects nonzero-to-zero F32 coefficient underflow and overflow.

Scope: layers required, prefill/decode optional. Selectors are {kind:all},
{kind:values,values:[sorted unique u32]}, {kind:range,start:u32,end:u32} with
INCLUSIVE end, or existing rendered_spans selectors for prefill only. Indices
are zero-based. Readouts specify id,lens,mode,scope,top_k. mode is selected or
full_vocabulary, gated by alias capabilities. Selected mode uses the registered
asset's candidate bank; it does not imply full-vocabulary normalization.
Readouts optionally accept `retain:"scores_and_residual"` (absent/null keeps
top-k-only). Capabilities advertise `readout_retention_modes`, model `hidden_size`,
`max_archive_bytes` (32 MiB) and `max_array_bytes` (4 MiB). Scoped admission
deduplicates source arrays by position/layer and score arrays by position/layer/
alias, checks exact bytes and reserves raw payload capacity before acceptance.
JSON result metadata is budgeted independently and may still fail publication.

## State and result types

Request previews contain `message_index` (zero-based), `message_count`, `text`
and boolean `truncated`. They select the last user message in recognized schema-1
messages input, then require string content; unsupported last-user content never
falls back to an earlier message. Missing/unsupported input or no user message
produces null. Empty content remains an empty string. Excerpts preserve authored
whitespace and contain at most 240 Unicode scalar values (960 UTF-8 text bytes);
truncated means more content exists. JSON escaping can expand those text bytes
to at most 1440 bytes. They are neither complete requests nor rendered prompts.
Derivation uses existing acceptance/recovery request reads, not history-poll IO;
previews are not persisted in status snapshots or included in request identity.
Clients retain known previews on status-only/old-server responses and report
conflicting supplied previews rather than silently rewriting them.

All status fields in status.json are required. Timestamps are Unix milliseconds;
revision is monotonically increasing per job. Overall state:
queued|running|finalizing|completed|cancelled|failed|interrupted.
Generation state: pending|running|completed|cancelled|failed|interrupted.
Generation phase: null|prefill|decode. Stop reason:
null|stop_token|token_limit|cancelled|execution_error|server_restart.
Observation state: pending|writing|complete|partial|failed|not_requested.
Nested errors are null or the error object from error.json. Generation completion
is independent of observation publication success. A generation completed with
failed observations has overall failed, not a rewritten generation outcome.
`result.error` independently reports output/artifact publication failure, including
baseline runs whose observations remain `not_requested`. `result.complete` means
publication has ended, not that all requested artifacts were successfully saved.
On restart unfinished jobs become interrupted, never automatically replayed.

Result record union is illustrated in result.json. prepared_input is first and
contains actual bound input, sampling, identities and execution policy.
New prepared-input records retain `prompt_text` and `prompt_bytes` together;
their UTF-8 bytes must agree exactly. `output_initial_state` is `pre_open` for
future reasoning bytes or `pre_closed` for final output. Assistant prefix bytes
are prompt-only. Older records may lack the text/bytes pair; clients must not
reconstruct it from token labels. The shared input helper provides an input
fragment; the HTTP executor supplies identities, execution and scopes. Baseline
model identity is tagged `runtime_gguf_metadata_not_content_hash`; it is not
fitted-lens content qualification. Baseline scopes and asset identities are empty.
`requested_preconditions` retains the authored assertions (null when absent).
`retention_admission` records source/score array count and raw-byte upper bounds,
saved `hidden_size`/`vocabulary_size`, F32LE dtype and capture stage. These counts
are reachable coverage bounds, not promises of execution through EOS/cancellation.
resolved_scopes is an array of {id,kind:operation|readout,scope}, using fully
resolved numeric scopes. rendering spans use the existing LensInputRendering
schema (byte_start/byte_end, nullable token boundaries and optional metadata).
sampled_token records include exact piece_bytes (u8 array); text is display-only.
Publication may wait until consumption is known so immutable consumed is truthful.
EOS and token-limit final samples are unconsumed; no readout is invented for them.
An attempted forward that fails emits `unresolved_sample` with token ID/bytes and
`reason=forward_failed_consumption_unknown`, not a fabricated consumption fact.
Operation order is zero-based within the authored operation list.
Plain readouts share one capture/head evaluation per consumed position/layer;
overlapping requests retain separate `readout_id` records and their own top k.
The first such record carries measured `cost.readout_ms`; subsequent records use
null with the same `shared_head_position` and `shared_head_layer`. Do not sum a
shared head more than once. At the final model layer when that forward produced
generation logits, `generation_logit_witness` compares against those exact logits
with absolute/relative tolerance 1e-4. It is null at other layers or no-tail
prefill positions. `within_tolerance=false` is diagnostic evidence, not hidden
or promoted to a passing qualification claim.
Fitted heads share capture by position/layer but score per position/layer/alias;
`cost.shared_head_lens` identifies that alias. Fitted rows also retain
`asset_identity`, `method` and `binding_status`, with a numeric `target_layer`.
They never receive generation-logit witnesses: a fitted target-layer score is
not the final generation distribution. Asset responses keep `transfer` as a
status string and add a `binding` object; prepared `asset_identities` retains the
selected fitted descriptors (manifest/payload identities and binding metadata).
These are local registered aliases, not HTTP asset paths or transfer overrides.
`prepared_input.effective_operation_ids` preserves effective authored order;
`intervention_admission` records application/projection/row bounds.
`direction_prepared` retains direction ID, alias/layer/token, target covector,
normalization, norm and a BLAKE3 digest of F32LE values. `operation_application`
uses `id`, `layer`, `phase`, `index`, absolute `position` and authored `order`;
it is published only after successful consumption, including operation-only jobs.
It does not claim rollback for a failed GPU command. Readout `applied_operation_ids`
lists effective operations at that same layer/event. Operation-only jobs keep
observations `not_requested` (that state counts readouts), while their result
stream still retains direction and application records.
Readout position is the consumed token position, predicts_position=position+1.
Capture is post_block_after_operations on original_forward. score_kind and
candidate_universe are backend semantic labels, not assumed probabilities.
Scores have nullable token_id, row_id, nullable label, finite score; target_layer
is nullable. Cost fields are measured milliseconds, nullable when unavailable,
never estimated zero or an instant-execution promise. Results are byte-bounded
pages, not a whole-run in-memory artifact.

Opt-in `retained_array` records contain `key`, `quantity` (`source_residual` or
`readout_logits`), source position/layer, consumed input token ID, phase/index,
capture stage, provenance and ordered applied-operation IDs. Score arrays also
retain lens, target layer, asset identity and source key. Their store-owned
`array` descriptor is `{dtype:"f32le",length,offset,byte_length,sha256,url}`;
length counts values, offset locates bytes inside the payload file, and URL
locates the descriptor record. Bytes are little-endian IEEE F32 with finite
values. Score indices are exact vocabulary token IDs. Source indices are model
residual coordinates, not vocabulary IDs. Keys are `source-{position}-{layer}`
and `logits-{position}-{layer}-{alias}`. Associated readouts carry
`retained:{source_key,logits_key}`; overlapping top-k requests share the arrays.

Payloads are synced before descriptors, then one snapshot publishes both
watermarks. Partial publication may retain a source without its subsequent head
or readout; it remains downloadable. Uncommitted payload/log tails are discarded
on recovery; missing/corrupt committed bytes fail closed. Binary reads verify
SHA-256 before returning success. Clients verify saved model dimensions and
source/head/readout joins, and never reinterpret arrays using current discovery.
Readout softmax derived from these scores is temperature one over the full
vocabulary, not the request's sampling distribution. Final unconsumed samples
still have no captures. No alternate-head computation occurs on these GETs.

## Errors and limits

Error envelope is error.json; param is nullable. HTTP statuses: 400 validation,
404 unknown route/job/alias, 409 idempotency conflict, 412 identity mismatch,
413 body too large,
429 queue capacity, 503 unavailable or storage admission failure, 500 internal
failure. Error type is invalid_request_error|server_busy|server_error;
code is a stable machine string and message is display text. Examples:
invalid_request, unsupported_capability, unknown_asset, idempotency_conflict,
queue_full, storage_unavailable, memory_admission_denied, observation_failed,
server_restart. Dispatch GPU admission failure appears in job status, not as a
retroactive rejection of a previously accepted HTTP request.
Likewise, post-acceptance matrix staging failures terminate the job with
`artifact_preparation_failed`, zero consumption and no execution phase. The
original accepted key continues to recover that terminal job, not new GPU work.
GPU direction/readout setup failures before the first token forward use
`diagnostic_preparation_failed`; a sequence memory refusal is labelled by kind
(`memory_admission_denied` under memory pressure; `memory_signal_unavailable`,
`memory_signal_invalid` or `memory_size_overflow` otherwise), and any other
preparation failure is `native_preparation_failed`. Artifact errors retain their independent result status.

A definitive submission rejection includes top-level
`admission:{schema_version:1,idempotency_key:string,state:"not_accepted"}` only
after the store verifies this key has no accepted job. Generic 4xx/503 responses,
including worker overload, are not proof of non-acceptance. Clients retain the
saved body/key across uncertain retries until they retrieve the job or receive
this matching keyed decision. Post-acceptance storage errors use 500; failed
queue delivery returns the accepted terminal job, including concurrent cancel.

Limits bound both array counts and aggregate work/bytes. CPU metadata, history,
committed results and static reads do not enter the model-owner queue. Subscriber
disconnect never cancels accepted Lens jobs; explicit cancellation does.
