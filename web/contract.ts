import { isObject } from "./draft";
import type { Sequenced } from "./records";

export type Selector = { kind: "all" } | { kind: "values"; values: number[] } | { kind: "range"; start: number; end: number };
export type Scope = { layers: Selector; prefill?: Selector; decode?: Selector };
export type Capabilities = {
  schema_version: 1; available: boolean; unavailable_reason: string | null;
  request_preconditions?: boolean;
  execution?: { baseline_only?: boolean };
  readout_retention_modes?: string[];
  residual_pair_capture?: boolean;
  model: { id: string; identity: string | null; template: string | null; layers: number | null; vocabulary_size: number | null; hidden_size?: number };
  generation_modes: string[]; assistant_prefill_channels: string[]; operations: string[]; readout_modes: string[];
  input_kinds: string[]; capture_stage: string;
  limits: Record<string, number> & { max_body_bytes: number; max_new_tokens: number; max_context_tokens: number; max_operations: number; max_directions: number; max_readouts: number; max_top_k: number; max_queued_jobs: number; max_result_page_records: number; max_result_page_bytes: number };
};
export type Asset = { alias: string; kind: string; identity: string; available: boolean; unavailable_reason: string | null; source_layers: number[]; target_layer: number | null; readout_modes: string[]; direction_rows: string[]; transfer: string };
export type JobState = "queued" | "running" | "finalizing" | "completed" | "cancelled" | "failed" | "interrupted";
export type Job = {
  deleted?: boolean;
  schema_version: 1; id: string; revision: number; created_at_ms: number; updated_at_ms: number; state: JobState; cancel_requested: boolean;
  generation: { state: "pending" | "running" | "completed" | "cancelled" | "failed" | "interrupted"; phase: "prefill" | "decode" | null; prompt_tokens: number; consumed_prompt_tokens: number; sampled_tokens: number; consumed_generated_tokens: number; stop_reason: string | null; error: unknown };
  observations: { state: "pending" | "writing" | "complete" | "partial" | "failed" | "not_requested"; committed_records: number; error: unknown };
  result: { available: boolean; complete: boolean; url: string; error: unknown };
  runtime?: { publication_error: { code: string; message: string }; execution_settled: boolean;
    generation: { stop_reason: string; counters: { prompt_tokens: number; consumed_prompt_tokens: number; sampled_tokens: number; consumed_generated_tokens: number }; error: unknown } | null };
};
export type Score = { token_id: number | null; row_id: number; label: string | null; score: number };
export type Readout = Sequenced & { kind: "readout"; readout_id: string; lens: string; phase: "prefill" | "decode"; index: number; position: number; input_token_id: number; predicts_position: number; source_layer: number; target_layer: number | null; capture_stage: string; applied_operation_ids: string[]; provenance: string; score_kind: string; candidate_universe: string; scores: Score[]; cost: { readout_ms: number | null } };
export type Sample = Sequenced & { kind: "sampled_token"; index: number; token_id: number; piece_bytes: number[]; text: string; consumed: boolean };
export type Prepared = Sequenced & { kind: "prepared_input"; token_ids: number[]; rendering: Record<string, unknown>; prompt_text?: string; prompt_bytes?: number[]; resolved_scopes: { id: string; kind: "operation" | "readout"; scope: Scope }[]; residual_pair_scopes?: { id: string; scope: Scope }[] };
export type ResultPage = { schema_version: 1; job_id: string; records: Sequenced[]; next_cursor: string | null; complete: boolean };
export type RequestPreview = { message_index: number; message_count: number; text: string; truncated: boolean };
export type StorageUsage = { retained_jobs: number; retry_identities: number; reserved_bytes: number; max_retained_jobs: number; max_retry_identities: number; max_store_bytes: number };
export type RecoveryReport = { read_only: true; unavailable_count: number; unavailable_jobs: string[] };
export type HistoryPage = { jobs: Job[]; next_cursor: string | null; request_previews?: Record<string, RequestPreview | null>; storage?: StorageUsage; recovery?: RecoveryReport };
export type RetainedArray = Sequenced & { kind: "retained_array"; key: string; quantity: "source_residual" | "source_residual_before" | "readout_logits"; position: number; source_layer: number; phase: "prefill" | "decode"; index: number; input_token_id: number; lens?: string; source_key?: string; applied_operation_ids: string[]; site_operation_ids?: string[]; capture_stage: "post_block_after_operations" | "post_block_before_operations"; provenance: "original_forward"; array: { dtype: "f32le"; length: number; offset: number; byte_length: number; sha256: string; url: string } };
export type PairMetrics = { norm_before: number; norm_after: number; delta_norm: number; relative_delta: number | null };
export type ResidualPair = Sequenced & { kind: "residual_pair"; id: string; before_key: string; after_key: string; position: number; source_layer: number; phase: "prefill" | "decode"; index: number; input_token_id: number; applied_operation_ids: string[]; metrics: PairMetrics };

export class ContractError extends Error {
  constructor(path: string, readonly payload: unknown) {
    super(`Lens v1 contract mismatch at ${path}. Received payload: ${JSON.stringify(payload)}`);
    this.name = "ContractError";
  }
}
function requireValue(ok: unknown, path: string, payload: unknown): asserts ok {
  if (!ok) throw new ContractError(path, payload);
}
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === "string");
const uint = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const u32 = (value: unknown): value is number => uint(value) && value <= 4_294_967_295;
const integers = (value: unknown): value is number[] => Array.isArray(value) && value.every(uint);
const nullableString = (value: unknown) => value === null || typeof value === "string";
const nullableNumber = (value: unknown) => value === null || (typeof value === "number" && Number.isFinite(value));
function object(value: unknown, path: string): asserts value is Record<string, unknown> { requireValue(isObject(value), path, value); }
function version(value: unknown): asserts value is Record<string, unknown> { object(value, "response"); requireValue(value.schema_version === 1, "schema_version", value); }

export function decodeCapabilities(value: unknown): Capabilities {
  version(value);
  object(value.model, "model"); object(value.limits, "limits");
  requireValue(typeof value.available === "boolean" && nullableString(value.unavailable_reason), "availability", value);
  requireValue(value.request_preconditions === undefined || typeof value.request_preconditions === "boolean", "request_preconditions", value);
  requireValue(value.readout_retention_modes === undefined || strings(value.readout_retention_modes), "readout_retention_modes", value);
  requireValue(value.residual_pair_capture === undefined || typeof value.residual_pair_capture === "boolean", "residual_pair_capture", value);
  if (value.execution !== undefined) {
    object(value.execution, "execution");
    requireValue(value.execution.baseline_only === undefined || typeof value.execution.baseline_only === "boolean", "execution.baseline_only", value);
  }
  requireValue(value.model.hidden_size === undefined || (uint(value.model.hidden_size) && value.model.hidden_size > 0), "model.hidden_size", value);
  requireValue(typeof value.model.id === "string" && nullableString(value.model.identity) && nullableString(value.model.template)
    && (uint(value.model.layers) || (!value.available && value.model.layers === null))
    && (uint(value.model.vocabulary_size) || (!value.available && value.model.vocabulary_size === null)), "model", value);
  if (value.available) requireValue(typeof value.model.identity === "string" && !!value.model.identity && typeof value.model.template === "string" && !!value.model.template, "qualified model identity", value);
  for (const key of ["input_kinds", "generation_modes", "assistant_prefill_channels", "operations", "readout_modes"]) requireValue(strings(value[key]), key, value);
  if (isObject(value.execution) && value.execution.baseline_only === true) {
    requireValue((value.operations as string[]).length === 0 && (value.readout_modes as string[]).length === 0 && value.residual_pair_capture !== true, "baseline-only diagnostic contradiction", value);
  }
  requireValue(typeof value.capture_stage === "string", "capture_stage", value);
  for (const key of ["max_body_bytes", "max_new_tokens", "max_context_tokens", "max_operations", "max_directions", "max_readouts", "max_top_k", "max_queued_jobs", "max_result_page_records", "max_result_page_bytes"]) requireValue(uint(value.limits[key]), `limits.${key}`, value);
  return value as unknown as Capabilities;
}
export function decodeAssets(value: unknown): Asset[] {
  version(value); requireValue(Array.isArray(value.assets), "assets", value);
  for (const asset of value.assets) {
    object(asset, "asset");
    requireValue(["alias", "kind", "identity", "transfer"].every(key => typeof asset[key] === "string")
      && typeof asset.available === "boolean" && nullableString(asset.unavailable_reason) && integers(asset.source_layers)
      && (asset.target_layer === null || uint(asset.target_layer)) && strings(asset.readout_modes) && strings(asset.direction_rows), "asset", value);
  }
  return value.assets as Asset[];
}
export function decodeJob(value: unknown): Job {
  version(value); object(value.generation, "generation"); object(value.observations, "observations"); object(value.result, "result");
  requireValue(typeof value.id === "string" && !!value.id && uint(value.revision) && uint(value.created_at_ms) && uint(value.updated_at_ms)
    && ["queued", "running", "finalizing", "completed", "cancelled", "failed", "interrupted"].includes(String(value.state)) && typeof value.cancel_requested === "boolean", "job", value);
  const generation = value.generation;
  requireValue(value.deleted === undefined || typeof value.deleted === "boolean", "deleted", value);
  requireValue(["pending", "running", "completed", "cancelled", "failed", "interrupted"].includes(String(generation.state))
    && (generation.phase === null || ["prefill", "decode"].includes(String(generation.phase)))
    && ["prompt_tokens", "consumed_prompt_tokens", "sampled_tokens", "consumed_generated_tokens"].every(key => uint(generation[key]))
    && nullableString(generation.stop_reason) && "error" in generation, "generation", value);
  requireValue(["pending", "writing", "complete", "partial", "failed", "not_requested"].includes(String(value.observations.state))
    && uint(value.observations.committed_records) && "error" in value.observations, "observations", value);
  requireValue(typeof value.result.available === "boolean" && typeof value.result.complete === "boolean" && "error" in value.result && typeof value.result.url === "string"
    && value.result.url.startsWith("/v1/lens/jobs/") && !/[\\\u0000-\u0020]/.test(value.result.url), "result", value);
  requireValue(value.deleted !== true || (["completed", "cancelled", "failed", "interrupted"].includes(String(value.state)) && value.result.complete && !value.result.available && value.runtime === undefined), "deleted job", value);
  if (value.runtime !== undefined) {
    object(value.runtime, "runtime"); object(value.runtime.publication_error, "runtime.publication_error");
    requireValue(typeof value.runtime.execution_settled === "boolean"
      && typeof value.runtime.publication_error.code === "string" && typeof value.runtime.publication_error.message === "string", "runtime", value);
    if (value.runtime.generation !== null) {
      object(value.runtime.generation, "runtime.generation"); object(value.runtime.generation.counters, "runtime.generation.counters");
      const outcome = value.runtime.generation;
      requireValue(["stop_token", "token_limit", "cancelled", "execution_error", "server_restart"].includes(String(outcome.stop_reason))
        && ["prompt_tokens", "consumed_prompt_tokens", "sampled_tokens", "consumed_generated_tokens"].every(key => uint((outcome.counters as Record<string, unknown>)[key]))
        && "error" in outcome, "runtime.generation", value);
    }
  }
  return value as unknown as Job;
}
export function decodeHistory(value: unknown): HistoryPage {
  version(value); requireValue(Array.isArray(value.jobs) && nullableString(value.next_cursor), "jobs page", value);
  const jobs = value.jobs.map(decodeJob);
  const ids = new Set(jobs.map(job => job.id));
  requireValue(ids.size === jobs.length, "duplicate history job", value);
  if (value.request_previews !== undefined) {
    object(value.request_previews, "request_previews");
    requireValue(Object.keys(value.request_previews).length === ids.size && Object.keys(value.request_previews).every(id => ids.has(id)), "request_previews coverage", value);
    for (const preview of Object.values(value.request_previews)) {
      if (preview === null) continue;
      object(preview, "request preview");
      requireValue(uint(preview.message_index) && uint(preview.message_count) && preview.message_index < preview.message_count
        && typeof preview.text === "string" && typeof preview.truncated === "boolean", "request preview", value);
      let scalars = 0;
      for (const char of preview.text as string) {
        const code = char.codePointAt(0)!;
        requireValue(++scalars <= 240 && !(code >= 0xd800 && code <= 0xdfff), "request preview text bound", value);
      }
      requireValue(!preview.truncated || scalars === 240, "request preview truncation", value);
    }
  }
  if (value.storage !== undefined) {
    object(value.storage, "storage");
    requireValue(["retained_jobs", "retry_identities", "reserved_bytes", "max_retained_jobs", "max_retry_identities", "max_store_bytes"].every(key => uint((value.storage as Record<string, unknown>)[key])), "storage usage", value);
  }
  if (value.recovery !== undefined) {
    object(value.recovery, "recovery");
    requireValue(value.recovery.read_only === true && uint(value.recovery.unavailable_count) && value.recovery.unavailable_count > 0
      && Array.isArray(value.recovery.unavailable_jobs) && value.recovery.unavailable_jobs.length <= 64
      && value.recovery.unavailable_jobs.length <= value.recovery.unavailable_count
      && value.recovery.unavailable_jobs.every(id => typeof id === "string" && id.length > 0 && id.length <= 1024), "recovery report", value);
  }
  return { jobs, next_cursor: value.next_cursor as string | null,
    ...(value.recovery === undefined ? {} : { recovery: value.recovery as RecoveryReport }),
    ...(value.storage === undefined ? {} : { storage: value.storage as StorageUsage }),
    ...(value.request_previews === undefined ? {} : { request_previews: value.request_previews as Record<string, RequestPreview | null> }) };
}
export function decodeSelector(value: unknown): Selector {
  object(value, "selector");
  requireValue(value.kind === "all" || (value.kind === "values" && integers(value.values) && value.values.length > 0 && value.values.every((n, i, a) => u32(n) && (i === 0 || n > a[i - 1]!)))
    || (value.kind === "range" && u32(value.start) && u32(value.end) && value.start <= value.end), "numeric selector", value);
  return value as Selector;
}
export function decodeScope(value: unknown): Scope {
  object(value, "scope"); decodeSelector(value.layers);
  if (value.prefill !== undefined && value.prefill !== null) decodeSelector(value.prefill);
  if (value.decode !== undefined && value.decode !== null) decodeSelector(value.decode);
  return value as Scope;
}
export function decodeResult(value: unknown): ResultPage {
  version(value);
  requireValue(typeof value.job_id === "string" && Array.isArray(value.records) && nullableString(value.next_cursor) && typeof value.complete === "boolean", "result page", value);
  requireValue(value.complete === (value.next_cursor === null), "result cursor/completion", value);
  for (const record of value.records) {
    object(record, "record"); requireValue(uint(record.seq) && typeof record.kind === "string", "record identity", value);
    if (record.kind === "retained_array") {
      object(record.array, "retained array descriptor");
      const array = record.array;
      requireValue(typeof record.key === "string" && !!record.key && ["source_residual", "source_residual_before", "readout_logits"].includes(String(record.quantity))
        && ["prefill", "decode"].includes(String(record.phase)) && ["position", "source_layer", "index", "input_token_id"].every(key => uint(record[key]))
        && record.capture_stage === (record.quantity === "source_residual_before" ? "post_block_before_operations" : "post_block_after_operations") && record.provenance === "original_forward" && strings(record.applied_operation_ids), "retained array identity", value);
      if (record.quantity === "source_residual_before") requireValue(strings(record.site_operation_ids) && (record.applied_operation_ids as string[]).length === 0, "before residual program", value);
      if (record.quantity === "readout_logits") requireValue(typeof record.lens === "string" && typeof record.source_key === "string" && record.score_kind === "logit" && record.candidate_universe === "full_vocabulary", "retained logits", value);
      requireValue(record.key === (record.quantity === "source_residual_before" ? `before-${record.position}-${record.source_layer}` : record.quantity === "source_residual" ? `source-${record.position}-${record.source_layer}` : `logits-${record.position}-${record.source_layer}-${record.lens}`)
        && (record.quantity !== "readout_logits" || record.source_key === `source-${record.position}-${record.source_layer}`), "retained site key", value);
      requireValue(array.dtype === "f32le" && uint(array.length) && array.length > 0 && uint(array.offset)
        && array.byte_length === array.length * 4 && Number(array.byte_length) <= 4 * 1024 * 1024
        && typeof array.sha256 === "string" && /^[a-f0-9]{64}$/.test(array.sha256)
        && typeof array.url === "string" && array.url.startsWith(`/v1/lens/jobs/${encodeURIComponent(value.job_id)}/arrays/`)
        && /\/arrays\/(0|[1-9][0-9]*)$/.test(array.url), "retained array bounds", value);
    } else if (record.kind === "residual_pair") {
      requireValue(typeof record.id === "string" && !!record.id && ["position", "source_layer", "index", "input_token_id"].every(key => uint(record[key]))
        && ["prefill", "decode"].includes(String(record.phase)) && record.provenance === "original_forward" && record.capture_stage === "whole_post_block_program"
        && strings(record.applied_operation_ids) && record.before_key === `before-${record.position}-${record.source_layer}` && record.after_key === `source-${record.position}-${record.source_layer}`, "residual pair site", value);
      object(record.metrics, "residual pair metrics");
      const m = record.metrics;
      requireValue(["norm_before", "norm_after", "delta_norm"].every(key => typeof m[key] === "number" && Number.isFinite(m[key]) && Number(m[key]) >= 0)
        && (m.norm_before === 0 ? m.relative_delta === null : typeof m.relative_delta === "number" && Number.isFinite(m.relative_delta) && m.relative_delta >= 0), "residual pair metrics", value);
    } else if (record.kind === "sampled_token") {
      requireValue(uint(record.index) && uint(record.token_id) && integers(record.piece_bytes) && record.piece_bytes.every(byte => byte <= 255)
        && typeof record.text === "string" && typeof record.consumed === "boolean", "sampled_token", value);
    } else if (record.kind === "readout") {
      requireValue(["readout_id", "lens", "capture_stage", "provenance", "score_kind", "candidate_universe"].every(key => typeof record[key] === "string")
        && ["prefill", "decode"].includes(String(record.phase)) && ["index", "position", "input_token_id", "predicts_position", "source_layer"].every(key => uint(record[key]))
        && (record.target_layer === null || uint(record.target_layer)) && strings(record.applied_operation_ids) && Array.isArray(record.scores), "readout", value);
      for (const score of record.scores) {
        object(score, "score"); requireValue((score.token_id === null || uint(score.token_id)) && uint(score.row_id) && nullableString(score.label)
          && typeof score.score === "number" && Number.isFinite(score.score), "score", value);
      }
      object(record.cost, "readout.cost"); requireValue(nullableNumber(record.cost.readout_ms), "readout.cost.readout_ms", value);
      if (record.retained != null) {
        object(record.retained, "readout.retained");
        requireValue(record.retained.source_key === `source-${record.position}-${record.source_layer}` && record.retained.logits_key === `logits-${record.position}-${record.source_layer}-${record.lens}`, "readout retained site", value);
      }
    } else if (record.kind === "prepared_input") {
      if (record.retention_admission !== undefined) {
        object(record.retention_admission, "retention admission");
        requireValue(["source_arrays_upper", "score_arrays_upper", "raw_bytes_upper", "hidden_size", "vocabulary_size"].every(key => uint((record.retention_admission as Record<string, unknown>)[key])), "retention admission counts", value);
        requireValue(["before_arrays_upper", "pair_rows_upper"].every(key => { const n = (record.retention_admission as Record<string, unknown>)[key]; return n === undefined || uint(n); }), "pair admission counts", value);
      }
      requireValue(integers(record.token_ids) && Array.isArray(record.resolved_scopes), "prepared_input", value);
      object(record.rendering, "rendering"); requireValue(typeof record.rendering.renderer === "string" && Array.isArray(record.rendering.spans), "rendering", value);
      if ("prompt_text" in record || "prompt_bytes" in record) {
        requireValue(typeof record.prompt_text === "string" && integers(record.prompt_bytes) && record.prompt_bytes.every(byte => byte <= 255), "prepared prompt bytes", value);
        const encoded = new TextEncoder().encode(record.prompt_text);
        requireValue(encoded.length === record.prompt_bytes.length && record.prompt_bytes.every((byte, index) => byte === encoded[index]), "prepared prompt text/bytes mismatch", value);
      }
      for (const scope of record.resolved_scopes) {
        object(scope, "resolved_scope"); requireValue(typeof scope.id === "string" && ["operation", "readout"].includes(String(scope.kind)), "resolved_scope", value); decodeScope(scope.scope);
      }
      if (record.residual_pair_scopes !== undefined) {
        requireValue(Array.isArray(record.residual_pair_scopes), "residual pair scopes", value);
        for (const pair of record.residual_pair_scopes) {
          object(pair, "residual pair scope"); requireValue(typeof pair.id === "string", "residual pair id", value); decodeScope(pair.scope);
        }
      }
    }
  }
  return value as unknown as ResultPage;
}
export const isReadout = (record: Sequenced): record is Readout => record.kind === "readout";
export const isSample = (record: Sequenced): record is Sample => record.kind === "sampled_token";
export const isPrepared = (record: Sequenced): record is Prepared => record.kind === "prepared_input";
export const isRetainedArray = (record: Sequenced): record is RetainedArray => record.kind === "retained_array";
export const isResidualPair = (record: Sequenced): record is ResidualPair => record.kind === "residual_pair";
export const isTerminal = (job: Job) => ["completed", "cancelled", "failed", "interrupted"].includes(job.state);
export const isBaselineOnly = (caps: Capabilities) => caps.available && (caps.execution?.baseline_only
  ?? (caps.operations.length === 0 && caps.readout_modes.length === 0 && caps.residual_pair_capture !== true));
export function selectorContains(selector: Selector | undefined | null, index: number) {
  if (!selector) return false;
  return selector.kind === "all" || (selector.kind === "values" ? selector.values.includes(index) : index >= selector.start && index <= selector.end);
}
