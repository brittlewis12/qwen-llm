import { decodeAssets, decodeCapabilities, type Asset, type Capabilities, type Prepared } from "./contract";
import { isObject, type Draft } from "./draft";

export type Preconditions = { model_identity: string; asset_identities: Record<string, string> };
export type DraftBinding = { state: "fresh" } | { state: "legacy" } | { state: "bound"; preconditions: Preconditions };
const identity = (value: unknown): value is string => typeof value === "string" && value.length > 0 && new TextEncoder().encode(value).length <= 256;

export function parsePreconditions(value: unknown): Preconditions {
  if (!isObject(value) || Object.keys(value).some(key => !["model_identity", "asset_identities"].includes(key))
    || !identity(value.model_identity) || !isObject(value.asset_identities) || Object.keys(value.asset_identities).length > 65
    || !Object.entries(value.asset_identities).every(([alias, id]) => /^[A-Za-z0-9_-]{1,64}$/.test(alias) && identity(id))) {
    throw new Error("Invalid identity preconditions; original data was not rewritten.");
  }
  return { model_identity: value.model_identity, asset_identities: Object.fromEntries(Object.entries(value.asset_identities)) as Record<string, string> };
}

export function parseBinding(value: unknown): DraftBinding {
  if (value === undefined) return { state: "legacy" };
  if (!isObject(value)) throw new Error("Invalid saved draft binding.");
  if ((value.state === "fresh" || value.state === "legacy") && Object.keys(value).length === 1) return { state: value.state };
  if (value.state === "bound" && Object.keys(value).every(key => ["state", "preconditions"].includes(key))) return { state: "bound", preconditions: parsePreconditions(value.preconditions) };
  throw new Error("Invalid saved draft binding.");
}

export function referencedAliases(draft: Draft): string[] {
  return [...new Set([...draft.directions, ...draft.readouts].flatMap(value => isObject(value) && typeof value.lens === "string" ? [value.lens] : []))].sort();
}

export function decodeDiscovery(capabilities: unknown, catalog: unknown) {
  const caps = decodeCapabilities(capabilities);
  const assets = decodeAssets(catalog);
  if (caps.available && caps.request_preconditions === true && (!isObject(catalog) || catalog.model_identity !== caps.model.identity)) {
    throw new Error("Capabilities and assets describe different model identities. Refresh discovery; no work was submitted.");
  }
  return { caps, assets };
}

// This preserves old assertions; refresh must never silently retarget a draft.
export function reconcileBinding(draft: Draft, caps: Capabilities | null, assets: Asset[]): Draft {
  if (draft.binding.state === "legacy") return draft;
  if (draft.binding.state === "fresh" && (!caps?.available || caps.request_preconditions !== true || !caps.model.identity)) return draft;
  const prior = draft.binding.state === "bound" ? draft.binding.preconditions : { model_identity: caps!.model.identity!, asset_identities: {} };
  const entries = referencedAliases(draft).flatMap(alias => {
    if (Object.hasOwn(prior.asset_identities, alias)) return [[alias, prior.asset_identities[alias]!] as const];
    const asset = caps?.model.identity === prior.model_identity ? assets.find(asset => asset.alias === alias && asset.available) : undefined;
    return asset ? [[alias, asset.identity] as const] : [];
  });
  return { ...draft, binding: { state: "bound", preconditions: { model_identity: prior.model_identity, asset_identities: Object.fromEntries(entries) } } };
}

export function rebindToCurrent(draft: Draft, caps: Capabilities | null, assets: Asset[]): Draft {
  if (!caps?.available || caps.request_preconditions !== true) throw new Error("Current server does not advertise identity preconditions.");
  const next = reconcileBinding({ ...draft, binding: { state: "fresh" } }, caps, assets);
  boundPreconditions(next);
  return next;
}

export function boundPreconditions(draft: Draft): Preconditions {
  if (draft.binding.state !== "bound") throw new Error("Review and bind this draft to a model and its assets before submitting.");
  const expected = draft.binding.preconditions;
  const entries = referencedAliases(draft).map(alias => {
    if (!Object.hasOwn(expected.asset_identities, alias)) throw new Error(`Draft has no identity assertion for alias ${alias}.`);
    return [alias, expected.asset_identities[alias]!] as const;
  });
  return parsePreconditions({ model_identity: expected.model_identity, asset_identities: Object.fromEntries(entries) });
}

export function bindingErrors(draft: Draft, caps: Capabilities, assets: Asset[]): string[] {
  if (caps.request_preconditions !== true) return ["This client requires server identity preconditions for new work. History and exact-key recovery remain available."];
  if (draft.binding.state === "legacy") return ["This draft has unresolved legacy identities. Review before using the current model and assets."];
  if (draft.binding.state === "fresh") return [];
  const errors: string[] = [];
  try {
    const expected = boundPreconditions(draft);
    if (expected.model_identity !== caps.model.identity) errors.push("Draft model metadata identity differs from the current server.");
    for (const [alias, id] of Object.entries(expected.asset_identities)) {
      if (!assets.some(asset => asset.available && asset.alias === alias && asset.identity === id)) errors.push(`Draft asset identity differs or is unavailable: ${alias}.`);
    }
  } catch (error) { errors.push(String(error)); }
  return errors;
}

export function mergePinnedBinding(draft: Draft, model: string, alias: string, asset: string): Draft {
  if (draft.binding.state === "legacy") throw new Error("Resolve this draft's legacy identities before pinning into it.");
  const prior = draft.binding.state === "bound" ? draft.binding.preconditions : { model_identity: model, asset_identities: {} };
  if (prior.model_identity !== model || (Object.hasOwn(prior.asset_identities, alias) && prior.asset_identities[alias] !== asset)) {
    throw new Error("Saved score conflicts with the draft's bound model or asset. Pinning did not change the draft.");
  }
  return { ...draft, binding: { state: "bound", preconditions: parsePreconditions({ model_identity: model, asset_identities: { ...prior.asset_identities, [alias]: asset } }) } };
}

export function recoverLegacyBinding(draft: Draft, prepared: Prepared | undefined): Draft {
  const model = savedModelIdentity(prepared);
  if (draft.binding.state !== "legacy" || !prepared || !model || !isObject(prepared.asset_identities)) return draft;
  const entries: [string, string][] = [];
  for (const alias of referencedAliases(draft)) {
    const saved = prepared.asset_identities[alias];
    if (alias === "plain" && identity(saved)) entries.push([alias, saved]);
    else if (isObject(saved) && saved.identity_kind === "blake3_compact_json_sorted_objects_array_order_preserved"
      && typeof saved.identity === "string" && /^[a-f0-9]{64}$/.test(saved.identity)) entries.push([alias, saved.identity]);
    else return draft;
  }
  return { ...draft, binding: { state: "bound", preconditions: parsePreconditions({ model_identity: model, asset_identities: Object.fromEntries(entries) }) } };
}

export function savedModelIdentity(prepared: Prepared | undefined): string | undefined {
  return prepared?.model_identity_kind === "runtime_gguf_metadata_not_content_hash" && identity(prepared.model_identity) ? prepared.model_identity : undefined;
}

export function requireUnchangedDraft(original: Draft, current: Draft) {
  if (current !== original) throw new Error("The draft changed while history was loading. The newer edits were preserved; copy the job again explicitly.");
}
