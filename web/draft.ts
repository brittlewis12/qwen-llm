import { decodeScope, selectorContains, type Asset, type Capabilities, type Readout, type Scope } from "./contract";
import { bindingErrors, boundPreconditions, parseBinding, parsePreconditions, type DraftBinding } from "./bindings";

export type Message = { role: "system" | "user" | "assistant"; content: string; reasoning?: string };
export type OperationRow = { key: string; enabled: boolean; document: unknown };
export type Draft = {
  version: 1;
  binding: DraftBinding;
  messages: Message[];
  generationMode: string;
  prefix: { enabled: boolean; channel: "reasoning" | "final"; text: string };
  generation: {
    max_new_tokens: number;
    sampling: { temperature: number; top_k: number; top_p: number; min_p: number; seed: number };
  };
  directions: unknown[];
  operations: OperationRow[];
  readouts: unknown[];
  residualPairs: unknown[];
};

export const DRAFT_KEY = "qwen-lens.draft.v1";
export const SELECTION_KEY = "qwen-lens.selection.v1";
export type Selection = { version: 1; view: "baseline" | "variant"; baseline: string; variant: string };
export const emptySelection: Selection = { version: 1, view: "baseline", baseline: "", variant: "" };

export function newDraft(): Draft {
  return {
    version: 1,
    binding: { state: "fresh" },
    messages: [{ role: "user", content: "" }],
    generationMode: "",
    prefix: { enabled: false, channel: "final", text: "" },
    generation: { max_new_tokens: 128, sampling: { temperature: 1, top_k: 20, top_p: .95, min_p: 0, seed: 17 } },
    directions: [], operations: [], readouts: [], residualPairs: [],
  };
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseDraft(raw: string): Draft {
  const value: unknown = JSON.parse(raw);
  if (!isObject(value) || value.version !== 1 || !Array.isArray(value.messages)
    || !value.messages.every(message => isObject(message) && ["system", "user", "assistant"].includes(String(message.role)) && typeof message.content === "string" && (message.reasoning === undefined || typeof message.reasoning === "string"))
    || typeof value.generationMode !== "string" || !isObject(value.prefix)
    || typeof value.prefix.enabled !== "boolean" || !["reasoning", "final"].includes(String(value.prefix.channel))
    || typeof value.prefix.text !== "string" || !isObject(value.generation)
    || !isObject(value.generation.sampling) || !Number.isFinite(value.generation.max_new_tokens)
    || !["temperature", "top_k", "top_p", "min_p", "seed"].every(key => Number.isFinite((value.generation as { sampling: Record<string, unknown> }).sampling[key]))
    || !Array.isArray(value.directions) || !Array.isArray(value.readouts) || !Array.isArray(value.operations)
    || (value.residualPairs !== undefined && !Array.isArray(value.residualPairs))
    || !value.operations.every(row => isObject(row) && typeof row.key === "string" && typeof row.enabled === "boolean" && "document" in row)) {
    throw new Error("Saved draft has an unsupported or invalid schema. Original storage was not overwritten.");
  }
  return { ...value, residualPairs: value.residualPairs ?? [], binding: parseBinding(value.binding) } as Draft;
}

export function parseSelection(raw: string): Selection {
  const value: unknown = JSON.parse(raw);
  if (!isObject(value) || value.version !== 1 || !["baseline", "variant"].includes(String(value.view))
    || typeof value.baseline !== "string" || typeof value.variant !== "string") {
    throw new Error("Saved job selection has an unsupported schema. Original storage was not overwritten.");
  }
  return value as Selection;
}

export function validateDraft(draft: Draft): string[] {
  const errors: string[] = [];
  if (!draft.messages.length || !draft.messages.some(message => message.role === "user")) errors.push("Add at least one user message.");
  draft.messages.forEach((message, index) => {
    if (!message.content.trim()) errors.push(`Message ${index + 1} is empty.`);
    if (message.role === "system" && index !== 0) errors.push("A system message must be first.");
  });
  if (!draft.generationMode) errors.push("Choose a server-supported generation mode.");
  const { max_new_tokens, sampling } = draft.generation;
  if (!Number.isSafeInteger(max_new_tokens) || max_new_tokens < 1) errors.push("Maximum new tokens must be a positive integer.");
  if (!Number.isSafeInteger(sampling.top_k) || sampling.top_k < 0) errors.push("Top k must be a nonnegative integer.");
  if (!Number.isSafeInteger(sampling.seed) || sampling.seed < 0) errors.push("Seed must be a nonnegative exact integer.");
  if (!Number.isFinite(sampling.temperature) || sampling.temperature < 0) errors.push("Temperature must be finite and nonnegative.");
  for (const key of ["top_p", "min_p"] as const) {
    if (!Number.isFinite(sampling[key]) || sampling[key] < 0 || sampling[key] > 1) errors.push(`${key} must be between 0 and 1.`);
  }
  return errors;
}

// Only the submission fields supplied by the backend owner are serialized here.
// Capability validation and authoritative response decoding remain integration gates.
export function submissionConfig(draft: Draft) {
  const errors = validateDraft(draft);
  if (errors.length) throw new Error(errors.join("\n"));
  return {
    schema_version: 1 as const,
    preconditions: boundPreconditions(draft),
    input: {
      kind: "messages" as const, messages: structuredClone(draft.messages),
      generation_mode: draft.generationMode,
      ...(draft.prefix.enabled ? { assistant_prefill: { channel: draft.prefix.channel, text: draft.prefix.text } } : {}),
    },
    generation: structuredClone(draft.generation),
    diagnostics: {
      directions: structuredClone(draft.directions),
      operations: structuredClone(draft.operations.filter(row => row.enabled).map(row => row.document)),
      readouts: structuredClone(draft.readouts),
      ...(draft.residualPairs.length ? { residual_pairs: structuredClone(draft.residualPairs) } : {}),
    },
  };
}

export function moveRow<T>(rows: readonly T[], index: number, delta: -1 | 1): T[] {
  const target = index + delta;
  if (index < 0 || index >= rows.length || target < 0 || target >= rows.length) return [...rows];
  const result = [...rows];
  [result[index], result[target]] = [result[target]!, result[index]!];
  return result;
}

export function layerValues(text: string): number[] {
  const values = new Set<number>();
  for (const part of text.split(",")) {
    const match = /^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$/.exec(part);
    if (!match) throw new Error("Use layer numbers or inclusive ranges, e.g. 3, 5-7.");
    const first = Number(match[1]);
    const last = Number(match[2] ?? match[1]);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last < first) throw new Error("Layer ranges must be increasing exact integers.");
    // Expansion beyond this guard requires capabilities instead of allocating unbounded memory.
    if (last - first > 100_000) throw new Error("Range too large to expand safely without model capabilities.");
    for (let value = first; value <= last; value++) values.add(value);
  }
  return [...values].sort((a, b) => a - b);
}

export function allPositionsScope(layers: number[]) {
  if (!layers.length || layers.some(layer => !Number.isSafeInteger(layer) || layer < 0)) throw new Error("Choose nonnegative exact layer indices.");
  return { layers: { kind: "values" as const, values: [...new Set(layers)].sort((a, b) => a - b) }, prefill: { kind: "all" as const }, decode: { kind: "all" as const } };
}

export function selectedSource(record: { phase: string; index: number; position: number; predicts_position: number }) {
  if ((record.phase !== "prefill" && record.phase !== "decode") || !Number.isSafeInteger(record.index) || record.index < 0) {
    throw new Error("Unknown source phase or invalid phase index; no scope created.");
  }
  // Never substitute the absolute position or the predicted-token position for the source index.
  return { phase: record.phase, index: record.index };
}

export function sourceScope(record: Pick<Readout, "phase" | "index" | "position" | "predicts_position" | "source_layer">): Scope {
  const source = selectedSource(record);
  return { layers: { kind: "values", values: [record.source_layer] }, [source.phase]: { kind: "values", values: [source.index] } };
}

export function validateCapabilities(draft: Draft, caps: Capabilities | null, assets: Asset[]): string[] {
  const errors = validateDraft(draft);
  if (!caps) return [...errors, "Capabilities have not been loaded."];
  if (!caps.available) return [...errors, `Lens unavailable: ${caps.unavailable_reason ?? "No reason supplied"}`];
  errors.push(...bindingErrors(draft, caps, assets));
  const modelLayers = caps.model.layers;
  const vocabularySize = caps.model.vocabulary_size;
  if (modelLayers === null || vocabularySize === null) return [...errors, "Model dimensions are unavailable."];
  if (!caps.input_kinds.includes("messages")) errors.push("This server does not support messages input.");
  if (!caps.generation_modes.includes(draft.generationMode)) errors.push("The selected generation mode is not supported by this server.");
  if (draft.prefix.enabled && !caps.assistant_prefill_channels.includes(draft.prefix.channel)) errors.push("The selected prefill channel is not supported.");
  if (draft.generation.max_new_tokens > caps.limits.max_new_tokens) errors.push(`New-token limit: ${caps.limits.max_new_tokens}.`);
  const active = draft.operations.filter(row => row.enabled);
  if (active.length > caps.limits.max_operations) errors.push(`Operation limit: ${caps.limits.max_operations}.`);
  if (draft.directions.length > caps.limits.max_directions) errors.push(`Direction limit: ${caps.limits.max_directions}.`);
  if (draft.readouts.length > caps.limits.max_readouts) errors.push(`Readout limit: ${caps.limits.max_readouts}.`);
  const directionIds = new Set<string>();
  const ids = new Set<string>();
  const aliasFor = (name: unknown) => assets.find(asset => asset.alias === name && asset.available);
  const checkId = (value: Record<string, unknown>, label: string) => {
    if (typeof value.id !== "string" || !value.id || ids.has(value.id)) errors.push(`${label}: IDs must be nonempty and unique.`);
    else ids.add(value.id);
  };
  const checkScope = (value: unknown, label: string, allowedLayers?: number[]) => {
    try {
      const scope = decodeScope(value);
      if (!scope.prefill && !scope.decode) errors.push(`${label}: choose prefill and/or decode indices.`);
      const layers = Array.from({ length: modelLayers }, (_, index) => index).filter(index => selectorContains(scope.layers, index));
      if (!layers.length) errors.push(`${label}: no model layers selected.`);
      if (scope.layers.kind === "values" && scope.layers.values.some(layer => layer >= modelLayers)) errors.push(`${label}: layer outside model bounds.`);
      if (scope.layers.kind === "range" && scope.layers.end >= modelLayers) errors.push(`${label}: layer range outside model bounds.`);
      if (allowedLayers && layers.some(layer => !allowedLayers.includes(layer))) errors.push(`${label}: selected layers are not supported by the registered alias.`);
    } catch (error) { errors.push(`${label}: unsupported scope retained, not silently translated. ${String(error)}`); }
  };
  for (const [index, value] of draft.directions.entries()) {
    if (!isObject(value)) { errors.push(`Direction ${index + 1} is opaque or invalid.`); continue; }
    checkId(value, `Direction ${index + 1}`);
    if (typeof value.id === "string") directionIds.add(value.id);
    const asset = aliasFor(value.lens);
    if (!asset) errors.push(`Direction ${index + 1}: unavailable or unknown lens alias.`);
    if (!isObject(value.row) || !asset?.direction_rows.includes(String(value.row.kind))) errors.push(`Direction ${index + 1}: unsupported row selector.`);
    if (isObject(value.row) && value.row.kind === "token_id" && (!Number.isSafeInteger(value.row.token_id) || Number(value.row.token_id) < 0 || Number(value.row.token_id) >= vocabularySize)) errors.push(`Direction ${index + 1}: token ID outside vocabulary.`);
    if (!["as_stored", "unit_l2"].includes(String(value.normalization))) errors.push(`Direction ${index + 1}: unknown normalization retained.`);
  }
  for (const [index, row] of active.entries()) {
    const value = row.document;
    const label = `Operation ${index + 1}`;
    if (!isObject(value) || !isObject(value.action)) { errors.push(`${label}: opaque operation retained; disable it or edit its JSON.`); continue; }
    checkId(value, label);
    const action = value.action;
    if (!caps.operations.includes(String(action.kind))) errors.push(`${label}: ${String(action.kind)} is not advertised by the server.`);
    if (!["fixed_add", "residual_l2_fraction", "projection_ablate", "source_to_target", "coordinate_swap"].includes(String(action.kind))) errors.push(`${label}: unknown operator retained verbatim; disable it until its native schema is supported by this editor.`);
    if (typeof action.coefficient !== "number" || !Number.isFinite(action.coefficient)) errors.push(`${label}: coefficient must be finite.`);
    const references = ["source_to_target", "coordinate_swap"].includes(String(action.kind)) ? [action.source, action.target] : [action.direction];
    for (const reference of references) {
      if (typeof reference !== "string" || !directionIds.has(reference)) errors.push(`${label}: missing direction ${String(reference)}.`);
      const direction = draft.directions.find(item => isObject(item) && item.id === reference);
      const asset = isObject(direction) ? aliasFor(direction.lens) : undefined;
      checkScope(value.scope, label, asset?.source_layers);
    }
  }
  for (const [index, value] of draft.readouts.entries()) {
    const label = `Readout ${index + 1}`;
    if (!isObject(value)) { errors.push(`${label}: opaque readout retained.`); continue; }
    checkId(value, label);
    const asset = aliasFor(value.lens);
    if (!asset) errors.push(`${label}: unavailable or unknown alias.`);
    if (!caps.readout_modes.includes(String(value.mode)) || !asset?.readout_modes.includes(String(value.mode))) errors.push(`${label}: unsupported mode.`);
    if (!Number.isSafeInteger(value.top_k) || Number(value.top_k) < 1 || Number(value.top_k) > caps.limits.max_top_k) errors.push(`${label}: top k must be 1-${caps.limits.max_top_k}.`);
    if (value.retain != null && (value.retain !== "scores_and_residual" || !caps.readout_retention_modes?.includes(value.retain))) errors.push(`${label}: retention mode is unavailable; saved request preserved.`);
    checkScope(value.scope, label, asset?.source_layers);
  }
  if (draft.residualPairs.length && caps.residual_pair_capture !== true) errors.push("This server does not support residual pair capture.");
  if (draft.residualPairs.length > (caps.limits.max_residual_pairs ?? 0)) errors.push("Residual pair request limit exceeded.");
  for (const [index, value] of draft.residualPairs.entries()) {
    const label = `Residual pair ${index + 1}`;
    if (!isObject(value)) { errors.push(`${label}: opaque measurement retained.`); continue; }
    checkId(value, label); checkScope(value.scope, label);
  }
  return errors;
}

export function draftFromBody(body: string): Draft {
  const request = JSON.parse(body);
  if (!isObject(request) || request.schema_version !== 1 || !isObject(request.input) || request.input.kind !== "messages" || !isObject(request.generation)) throw new Error("Stored request cannot be copied into the editor.");
  const diagnostics = isObject(request.diagnostics) ? request.diagnostics : {};
  const prefix = isObject(request.input.assistant_prefill) ? request.input.assistant_prefill : null;
  return parseDraft(JSON.stringify({ version: 1, binding: request.preconditions === undefined ? { state: "legacy" } : { state: "bound", preconditions: parsePreconditions(request.preconditions) }, messages: request.input.messages, generationMode: request.input.generation_mode,
    prefix: prefix ? { enabled: true, ...prefix } : { enabled: false, channel: "final", text: "" }, generation: request.generation,
    directions: diagnostics.directions ?? [], readouts: diagnostics.readouts ?? [],
    residualPairs: diagnostics.residual_pairs ?? [],
    operations: Array.isArray(diagnostics.operations) ? diagnostics.operations.map(document => ({ key: crypto.randomUUID(), enabled: true, document })) : [],
  }));
}

export function draftFromSavedRequest(value: unknown, jobId: string): Draft {
  if (!isObject(value) || value.schema_version !== 1 || value.job_id !== jobId || !isObject(value.request)) {
    throw new Error("Server request record is invalid or belongs to a different job.");
  }
  return draftFromBody(JSON.stringify(value.request));
}
