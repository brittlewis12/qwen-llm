import { expect, test } from "bun:test";
import { bindingErrors, boundPreconditions, decodeDiscovery, mergePinnedBinding, reconcileBinding, rebindToCurrent, recoverLegacyBinding, requireUnchangedDraft } from "./bindings";
import { decodeAssets, decodeCapabilities, type Prepared } from "./contract";
import { draftFromBody, newDraft, parseDraft, submissionConfig } from "./draft";
import { DurableSubmission, readIntent } from "./durable";
import { ApiHttpError } from "./api";

const root = `${import.meta.dir}/../crates/qwen-cli/tests/fixtures/lens_http_v1`;
const caps = decodeCapabilities(await Bun.file(`${root}/capabilities.json`).json());
const assets = decodeAssets(await Bun.file(`${root}/assets.json`).json());
const draft = () => ({ ...newDraft(), messages: [{ role: "user" as const, content: "Hello" }], generationMode: "no_thinking",
  readouts: [{ id: "r", lens: "plain" }] });
const bind = () => rebindToCurrent(draft(), caps, assets);

test("new work freezes advertised identities; discovery refresh preserves old assertions", () => {
  const bound = bind();
  expect(boundPreconditions(bound)).toEqual({ model_identity: caps.model.identity!, asset_identities: { plain: assets[0]!.identity } });
  const changed = assets.map(asset => ({ ...asset, identity: "replacement" }));
  expect(reconcileBinding(bound, caps, changed).binding).toEqual(bound.binding);
  expect(bindingErrors(bound, caps, changed).join(" ")).toContain("Draft asset identity differs");
  expect(boundPreconditions(rebindToCurrent(bound, caps, changed)).asset_identities.plain).toBe("replacement");
  expect(boundPreconditions(rebindToCurrent(bound, caps, changed)).model_identity).toBe(caps.model.identity!);
});

test("alias deletion and readdition never mixes deployments or silently changes retained identities", () => {
  const bound = bind();
  const added = { ...bound, readouts: [...bound.readouts, { lens: "fitted" }] };
  const changedModel = { ...caps, model: { ...caps.model, identity: "different-model" } };
  const unresolved = reconcileBinding(added, changedModel, assets);
  expect(() => boundPreconditions(unresolved)).toThrow("no identity assertion");
  const resolved = reconcileBinding(added, caps, assets);
  expect(Object.keys(boundPreconditions(resolved).asset_identities)).toEqual(["fitted", "plain"]);
  const deleted = reconcileBinding({ ...resolved, readouts: [] }, caps, assets);
  expect(boundPreconditions(deleted).asset_identities).toEqual({});
  const replacement = assets.map(asset => ({ ...asset, identity: "re-added" }));
  expect(boundPreconditions(reconcileBinding({ ...deleted, readouts: [{ lens: "plain" }] }, caps, replacement)).asset_identities.plain).toBe("re-added");
});

test("copied request preserves guards and legacy drafts require explicit provenance resolution", () => {
  const config = submissionConfig(bind());
  const copy = draftFromBody(JSON.stringify({ ...config, idempotency_key: "old" }));
  expect(submissionConfig(copy)).toEqual(config);
  const legacy = { ...config } as Record<string, unknown>; delete legacy.preconditions;
  const unbound = draftFromBody(JSON.stringify(legacy));
  expect(unbound.binding.state).toBe("legacy");
  expect(reconcileBinding(unbound, caps, assets).binding.state).toBe("legacy");
  expect(() => submissionConfig(unbound)).toThrow("Review and bind");
  const oldDraft = { ...bind() } as Record<string, unknown>; delete oldDraft.binding;
  expect(parseDraft(JSON.stringify(oldDraft)).binding.state).toBe("legacy");
});

test("legacy prepared records need known identity kinds and complete alias coverage", () => {
  const legacy = { ...draft(), readouts: [{ lens: "j" }], binding: { state: "legacy" as const } };
  const prepared = { kind: "prepared_input", model_identity: "saved-model", model_identity_kind: "runtime_gguf_metadata_not_content_hash",
    asset_identities: { j: { identity: "a".repeat(64), identity_kind: "blake3_compact_json_sorted_objects_array_order_preserved" } } } as unknown as Prepared;
  expect(boundPreconditions(recoverLegacyBinding(legacy, prepared)).model_identity).toBe("saved-model");
  for (const incomplete of [undefined, { ...prepared, asset_identities: {} }, { ...prepared, model_identity_kind: "unknown" },
    { ...prepared, asset_identities: { j: { identity: "new", identity_kind: "unknown" } } }]) {
    expect(recoverLegacyBinding(legacy, incomplete).binding.state).toBe("legacy");
  }
});

test("score pin merges are atomic and prototype-like aliases remain ordinary keys", () => {
  const bound = bind();
  expect(() => mergePinnedBinding(bound, "other-model", "plain", assets[0]!.identity)).toThrow("conflicts");
  expect(() => mergePinnedBinding(bound, caps.model.identity!, "plain", "other-fit")).toThrow("conflicts");
  const original = JSON.stringify(bound);
  const pinned = mergePinnedBinding(bound, caps.model.identity!, "__proto__", "fingerprint");
  expect(pinned.binding.state === "bound" && Object.hasOwn(pinned.binding.preconditions.asset_identities, "__proto__")).toBe(true);
  expect(JSON.stringify(bound)).toBe(original);
});

test("split discovery generations and asynchronous draft replacement fail closed", () => {
  expect(() => decodeDiscovery(caps, { schema_version: 1, model_identity: "other-model", assets })).toThrow("different model identities");
  expect(decodeDiscovery(caps, { schema_version: 1, model_identity: caps.model.identity, assets }).assets).toEqual(assets);
  const original = bind();
  expect(() => requireUnchangedDraft(original, { ...original })).toThrow("newer edits were preserved");
  expect(() => requireUnchangedDraft(original, original)).not.toThrow();
});

test("412 recovery preserves exact request bytes; only a matching no-acceptance envelope unlocks a new key", async () => {
  for (const rejectionKey of ["guarded", "wrong", null]) {
    const values = new Map<string, string>(); const bodies: string[] = [];
    const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value); } };
    const service = new DurableSubmission(storage, async body => {
      bodies.push(body);
      throw new ApiHttpError(new Response(null, { status: 412 }), JSON.stringify(rejectionKey === null ? {} : {
        admission: { schema_version: 1, idempotency_key: rejectionKey, state: "not_accepted" },
      }));
    });
    const saved = service.prepare(submissionConfig(bind()), "guarded");
    await expect(service.retry()).rejects.toThrow("412");
    await expect(service.retry()).rejects.toThrow("412");
    expect(bodies).toEqual([saved.body, saved.body]);
    expect(readIntent(storage)?.rejected).toBe(rejectionKey === "guarded");
  }
});
