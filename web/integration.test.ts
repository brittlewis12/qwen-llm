import { describe, expect, test } from "bun:test";
import { ApiHttpError, createLensApi } from "./api";
import { decodeAssets, decodeCapabilities, decodeHistory, decodeJob, decodeResult, isBaselineOnly, isReadout } from "./contract";
import { draftFromBody, draftFromSavedRequest, layerValues, moveRow, newDraft, parseDraft, sourceScope, submissionConfig, validateCapabilities } from "./draft";
import { DurableSubmission, INTENT_KEY, readIntent, type StoragePort } from "./durable";
import { RecordAccumulator, startPolling, type Sequenced } from "./records";
import { cellAvailability } from "./viewer";
import { forwardRequest, proxyConfig, proxyTarget } from "./proxy";

const fixtureRoot = `${import.meta.dir}/../crates/qwen-cli/tests/fixtures/lens_http_v1`;
const fixture = (name: string) => Bun.file(`${fixtureRoot}/${name}.json`).json();
const [capabilities, assets, status, result, request, errorEnvelope] = await Promise.all(["capabilities", "assets", "status", "result", "request", "error"].map(fixture));
class MemoryStorage implements StoragePort {
  values = new Map<string, string>();
  fail = false;
  getItem(key: string) { return this.values.get(key) ?? null; }
  setItem(key: string, value: string) { if (this.fail) throw new Error("Storage quota"); this.values.set(key, value); }
}

describe("authoritative Lens v1 fixtures", () => {
  test("explicit baseline-only declarations are decoded and contradictory capabilities refused", () => {
    const value = { ...capabilities, operations: [], readout_modes: [], residual_pair_capture: false,
      execution: { baseline_only: true } };
    expect(isBaselineOnly(decodeCapabilities(value))).toBe(true);
    expect(() => decodeCapabilities({ ...value, execution: { baseline_only: "true" } })).toThrow("execution.baseline_only");
    expect(() => decodeCapabilities({ ...value, readout_modes: ["full_vocabulary"] })).toThrow("contradiction");
    expect(() => decodeCapabilities({ ...value, residual_pair_capture: true })).toThrow("contradiction");
    expect(isBaselineOnly(decodeCapabilities({ ...value, execution: { baseline_only: false } }))).toBe(false);
  });
  test.each([502, 503, 504])("only reads retry transient HTTP %i and failures remain bounded", async (status) => {
    let calls = 0;
    const paths: string[] = [];
    const api = createLensApi(async path => { paths.push(path); return ++calls < 3 ? new Response("transient", { status }) : Response.json(capabilities); });
    expect(await api.capabilities()).toEqual(capabilities);
    expect(calls).toBe(3);
    expect(new Set(paths).size).toBe(1);
    calls = 0;
    const busy = createLensApi(async () => { calls++; return new Response("transient", { status }); });
    await expect(busy.assets()).rejects.toBeInstanceOf(ApiHttpError);
    expect(calls).toBe(3);
    calls = 0;
    await expect(busy.submit("exact saved body")).rejects.toBeInstanceOf(ApiHttpError);
    expect(calls).toBe(1);
    calls = 0;
    await expect(busy.cancel("job_example")).rejects.toBeInstanceOf(ApiHttpError);
    expect(calls).toBe(1);
  });
  test("read retries preserve cursors and do not retry malformed success or permanent failures", async () => {
    const paths: string[] = [];
    const api = createLensApi(async path => { paths.push(path); return paths.length === 1 ? new Response("proxy", { status: 502 }) : Response.json(result); });
    expect(await api.result("job_example", "cursor", 17)).toEqual(result);
    expect(paths).toEqual(["/v1/lens/jobs/job_example/result?cursor=cursor&limit=17", "/v1/lens/jobs/job_example/result?cursor=cursor&limit=17"]);
    for (const response of [new Response("invalid json"), new Response("forbidden", { status: 403 }), new Response("error", { status: 500 })]) {
      let calls = 0;
      const failing = createLensApi(async () => { calls++; return response; });
      await expect(failing.jobs()).rejects.toThrow();
      expect(calls).toBe(1);
    }
  });
  test("plain readout capabilities admit scoped observations without interventions", () => {
    const caps = decodeCapabilities({ ...capabilities, operations: [], readout_modes: ["full_vocabulary"],
      limits: { ...capabilities.limits, max_directions: 0, max_operations: 0 } });
    const plain = decodeAssets({ schema_version: 1, assets: [{ alias: "plain", kind: "plain_logit_lens", identity: caps.model.identity,
      available: true, unavailable_reason: null, source_layers: Array.from({ length: caps.model.layers! }, (_, layer) => layer),
      target_layer: null, readout_modes: ["full_vocabulary"], direction_rows: [], transfer: "identity" }] });
    const value = { ...request, preconditions: { model_identity: caps.model.identity, asset_identities: { plain: caps.model.identity } }, diagnostics: { directions: [], operations: [], readouts: [{ id: "plain", lens: "plain", mode: "full_vocabulary", top_k: 5,
      scope: { layers: { kind: "values", values: [0, 1] }, prefill: { kind: "all" }, decode: { kind: "values", values: [0] } } }] } };
    expect(validateCapabilities(draftFromBody(JSON.stringify(value)), caps, plain)).toEqual([]);
    expect(isBaselineOnly(caps)).toBe(false);
    expect(validateCapabilities(draftFromBody(JSON.stringify(request)), caps, plain).length).toBeGreaterThan(0);
  });
  test("baseline-only capabilities enable generation but never diagnostics", () => {
    const caps = decodeCapabilities({ ...capabilities, operations: [], readout_modes: [], residual_pair_capture: false,
      limits: { ...capabilities.limits, max_operations: 0, max_directions: 0, max_readouts: 0, max_top_k: 0 } });
    expect(isBaselineOnly(caps)).toBe(true);
    expect(isBaselineOnly(decodeCapabilities(capabilities))).toBe(false);
    const draft = draftFromBody(JSON.stringify(request));
    expect(validateCapabilities(draft, caps, []).length).toBeGreaterThan(0);
    const baseline = { ...request, diagnostics: { directions: [], operations: [], readouts: [] } };
    expect(validateCapabilities(draftFromBody(JSON.stringify(baseline)), caps, [])).toEqual([]);
  });
  test("unavailable diagnostic metadata is not a zero-layer model", async () => {
    const value = await fixture("capabilities_unavailable");
    const decoded = decodeCapabilities(value);
    expect(decoded.model.layers).toBeNull();
    expect(decoded.available).toBe(false);
    expect(validateCapabilities(draftFromBody(JSON.stringify(request)), decoded, []).join(" ")).toContain("diagnostic_executor_not_connected");
    expect(() => decodeCapabilities({ ...value, available: true })).toThrow("model");
  });
  test("run-again reads the server request without local submission history or GPU work", async () => {
    const paths: string[] = [];
    const api = createLensApi(async (path, init) => {
      paths.push(path);
      expect(init?.method).toBeUndefined();
      return Response.json({ schema_version: 1, job_id: "job_remote", request });
    });
    const saved = await api.request("job_remote");
    const draft = draftFromSavedRequest(saved, "job_remote");
    expect(submissionConfig(draft)).toEqual((({ idempotency_key, ...body }) => body)(request));
    expect(paths).toEqual(["/v1/lens/jobs/job_remote/request"]);
    expect(() => draftFromSavedRequest(saved, "wrong_job")).toThrow("different job");
  });
  test("decodes actual capabilities, assets, status and results", () => {
    expect(decodeCapabilities(capabilities).generation_modes).toEqual(capabilities.generation_modes);
    expect(decodeAssets(assets).map(asset => asset.alias)).toEqual(assets.assets.map((asset: { alias: string }) => asset.alias));
    expect(decodeJob(status).revision).toBe(status.revision);
    expect(decodeResult(result).records).toEqual(result.records);
    expect(decodeHistory({ schema_version: 1, jobs: [status], next_cursor: null }).jobs[0]).toEqual(status);
  });
  test("request round trip preserves native scope, prefix and sampling", () => {
    const draft = draftFromBody(JSON.stringify(request));
    expect(submissionConfig(draft)).toEqual((({ idempotency_key, ...body }) => body)(request));
    expect(validateCapabilities(draft, decodeCapabilities(capabilities), decodeAssets(assets))).toEqual([]);
  });
  test("baseline publication failure remains distinct from generation success", () => {
    const failed = decodeJob({ ...status, state: "failed",
      generation: { ...status.generation, state: "completed", phase: null, stop_reason: "token_limit" },
      observations: { state: "not_requested", committed_records: 0, error: null },
      result: { ...status.result, complete: true, error: errorEnvelope.error },
    });
    expect(failed.generation.state).toBe("completed");
    expect(failed.observations.state).toBe("not_requested");
    expect(failed.result.error).toEqual(errorEnvelope.error);
    const missing = structuredClone(status); delete missing.result.error;
    expect(() => decodeJob(missing)).toThrow("result");
  });
  test("prepared prompt is exact retained input, not reconstructed from token labels", () => {
    const value = structuredClone(result);
    const prepared = value.records.find((record: Sequenced) => record.kind === "prepared_input");
    prepared.prompt_text = "  <|im_start|>assistant\nAnswer:\n";
    prepared.prompt_bytes = [...new TextEncoder().encode(prepared.prompt_text)];
    expect(decodeResult(value).records[0]?.prompt_text).toBe(prepared.prompt_text);
    prepared.prompt_bytes[0] = 65;
    expect(() => decodeResult(value)).toThrow("text/bytes mismatch");
    delete prepared.prompt_bytes;
    expect(() => decodeResult(value)).toThrow("prepared prompt bytes");
    expect(decodeResult(result).records[0]?.prompt_text).toBeUndefined();
  });
  test("rejects schema drift, nonfinite scores and inconsistent cursor publication", () => {
    expect(() => decodeJob({ ...status, revision: "18" })).toThrow("job");
    expect(() => decodeCapabilities({ ...capabilities, schema_version: 2 })).toThrow("schema_version");
    expect(() => decodeResult({ ...result, complete: false, next_cursor: null })).toThrow("cursor/completion");
    const altered = structuredClone(result); altered.records.find((record: Sequenced) => record.kind === "readout").scores[0].score = Infinity;
    expect(() => decodeResult(altered)).toThrow("score");
  });
  test("native HTTP adapter uses exact paths, opaque cursors and errors", async () => {
    const paths: string[] = [];
    const api = createLensApi(async (path, init) => {
      paths.push(path);
      if (init?.method === "POST") return new Response(JSON.stringify(errorEnvelope), { status: 409 });
      return Response.json(path.includes("result") ? result : status);
    });
    await api.result("job_example", "a+/=&", 12);
    expect(paths[0]).toBe("/v1/lens/jobs/job_example/result?cursor=a%2B%2F%3D%26&limit=12");
    await expect(api.submit(JSON.stringify(request))).rejects.toBeInstanceOf(ApiHttpError);
    expect(paths[1]).toBe("/v1/lens/jobs");
    await expect(api.cancel("job_example")).rejects.toThrow("idempotency_conflict");
    expect(paths[2]).toBe("/v1/lens/jobs/job_example/cancel");
  });
  test("development proxy preserves bodies, errors and /v1 paths over localhost", async () => {
    let received: { path: string; body: string; length: string | null; origin: string | null } | undefined;
    const upstream = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
      received = { path: new URL(request.url).pathname, body: await request.text(), length: request.headers.get("content-length"), origin: request.headers.get("origin") };
      return new Response(JSON.stringify(errorEnvelope), { status: 409, headers: { "x-request-id": "proxy-fixture" } });
    } });
    try {
      const config = proxyConfig({ QWEN_SERVE_ORIGIN: upstream.url.origin });
      const body = JSON.stringify(request);
      const incoming = new Request("http://localhost:3000/v1/lens/jobs", { method: "POST", body, headers: { "Content-Type": "application/json", "Content-Length": String(new TextEncoder().encode(body).length), Origin: "http://localhost:3000" } });
      const target = proxyTarget(incoming.url, config)!;
      const response = await forwardRequest(incoming, target);
      expect(response.status).toBe(409);
      expect(response.headers.get("x-request-id")).toBe("proxy-fixture");
      expect(await response.json()).toEqual(errorEnvelope);
      expect(received?.path).toBe("/v1/lens/jobs");
      expect(received?.body).toBe(body);
      expect(received?.length).toBe(String(new TextEncoder().encode(body).length));
      expect(received?.origin).toBe(upstream.url.origin);
      for (const [url, origin] of [["http://rebind.example:3000/v1/lens/jobs", "http://rebind.example:3000"], ["http://localhost:3000/v1/lens/jobs", "http://evil.example"]]) {
        const rejected = await forwardRequest(new Request(url!, { method: "POST", body, headers: { Origin: origin! } }), target);
        expect(rejected.status).toBe(403);
      }
    } finally { await upstream.stop(true); }
  });
});

describe("durable submissions", () => {
  test("persists before fetch, survives reload, and retries identical bytes and key", async () => {
    const storage = new MemoryStorage(); const bodies: string[] = [];
    const api = async (body: string) => {
      expect(readIntent(storage)?.body).toBe(body);
      bodies.push(body);
      if (bodies.length === 1) throw new Error("Response connection lost after acceptance");
      return status;
    };
    const service = new DurableSubmission(storage, api);
    service.prepare(submissionConfig(draftFromBody(JSON.stringify(request))), "stable-key");
    await expect(service.retry()).rejects.toThrow("lost after acceptance");
    expect(() => service.prepare({ schema_version: 1 }, "different-key")).toThrow("Resolve");
    const recovered = new DurableSubmission(storage, api);
    expect(await recovered.retry()).toEqual(status);
    expect(bodies[0]).toBe(bodies[1]); expect(JSON.parse(bodies[0]!).idempotency_key).toBe("stable-key");
    expect(await recovered.retry()).toEqual(status); expect(bodies).toHaveLength(2);
    recovered.confirmJob("stable-key", status.id);
    await expect(recovered.retry()).rejects.toThrow("already has a job");
    recovered.prepare({ schema_version: 1 }, "new-run-key");
    expect(storage.getItem(`${INTENT_KEY}.archive.stable-key`)).not.toBeNull();
    expect(readIntent(storage)?.key).toBe("new-run-key");
  });
  test("quota failure prevents transmission and corrupt storage is not overwritten", async () => {
    const storage = new MemoryStorage(); let calls = 0;
    const service = new DurableSubmission(storage, async () => { calls++; return status; });
    storage.fail = true;
    expect(() => service.prepare({ schema_version: 1 })).toThrow("quota");
    expect(calls).toBe(0);
    storage.fail = false; storage.setItem(INTENT_KEY, '{"version":99}');
    expect(() => service.prepare({ schema_version: 1 })).toThrow("invalid");
    expect(storage.getItem(INTENT_KEY)).toBe('{"version":99}');
    expect(calls).toBe(0);
  });
  test("submission concurrency guard rejects simultaneous retries", async () => {
    const storage = new MemoryStorage(); let finish!: (value: unknown) => void;
    const service = new DurableSubmission(storage, () => new Promise(resolve => { finish = resolve; }));
    service.prepare({ schema_version: 1 }); const first = service.retry();
    await expect(service.retry()).rejects.toThrow("already in progress");
    finish(status); await first;
  });
  test("rejected admission can be corrected but conflict/500 cannot cause a fresh rerun", async () => {
    for (const httpStatus of [400, 409, 429, 500, 503]) {
      const storage = new MemoryStorage();
      const service = new DurableSubmission(storage, async () => { throw new ApiHttpError(new Response(null, { status: httpStatus }), JSON.stringify({ ...errorEnvelope, admission: { schema_version: 1, idempotency_key: "original", state: "not_accepted" } })); });
      service.prepare({ schema_version: 1 }, "original");
      await expect(service.retry()).rejects.toThrow("HTTP");
      expect(readIntent(storage)?.attempts.at(-1)?.detail).toContain("idempotency_conflict");
      if ([400, 429, 503].includes(httpStatus)) expect(service.prepare({ schema_version: 1 }, "corrected").key).toBe("corrected");
      else expect(() => service.prepare({ schema_version: 1 }, "unsafe")).toThrow("Resolve");
    }
  });
  test("accepted/lost response followed by generic overload retains the same uncertain intent", async () => {
    const storage = new MemoryStorage(); const bodies: string[] = [];
    const transport = async (body: string) => {
      bodies.push(body);
      if (bodies.length === 1) throw new Error("Accepted, response lost");
      if (bodies.length === 2) throw new ApiHttpError(new Response(null, { status: 503 }), JSON.stringify({ error: { code: "server_busy" } }));
      return status;
    };
    const first = new DurableSubmission(storage, transport);
    first.prepare({ schema_version: 1 }, "uncertain");
    await expect(first.retry()).rejects.toThrow("lost");
    const reloaded = new DurableSubmission(storage, transport);
    await expect(reloaded.retry()).rejects.toThrow("503");
    expect(readIntent(storage)?.rejected).toBe(false);
    expect(() => reloaded.prepare({ schema_version: 1 }, "duplicate")).toThrow("Resolve");
    expect(await reloaded.retry()).toEqual(status);
    expect(new Set(bodies).size).toBe(1);
  });
  test("status codes and another key's rejection do not prove non-acceptance", async () => {
    for (const payload of [errorEnvelope, { ...errorEnvelope, admission: { schema_version: 1, idempotency_key: "different-key", state: "not_accepted" } }]) {
      const storage = new MemoryStorage();
      const service = new DurableSubmission(storage, async () => { throw new ApiHttpError(new Response(null, { status: 429 }), JSON.stringify(payload)); });
      service.prepare({ schema_version: 1 }, "mine");
      await expect(service.retry()).rejects.toThrow("429");
      expect(readIntent(storage)?.rejected).toBe(false);
      expect(() => service.prepare({ schema_version: 1 })).toThrow("Resolve");
    }
  });
});

describe("record pagination and viewer independence", () => {
  test("deduplicates replay, preserves unknown records and stages pages atomically", () => {
    const records = new RecordAccumulator<Sequenced>();
    records.append(result.records.slice(0, 3), undefined, "cursor-1");
    records.append(result.records.slice(2), "cursor-1", undefined);
    expect(records.values()).toEqual(result.records);
    const original = records.values();
    expect(() => records.append([{ ...result.records[0], prompt_digest: "changed" }], undefined, "next")).toThrow("Immutable record");
    expect(records.values()).toEqual(original);
    records.append([{ seq: 99, kind: "future_event", opaque: { retained: true } }], undefined, "future");
    expect(records.values().at(-1)?.opaque).toEqual({ retained: true });
  });
  test("allows empty live-end polling, rejects stale/cyclic cursors and unsorted seq", () => {
    const records = new RecordAccumulator<Sequenced>();
    records.append([], undefined, "live"); records.append([], "live", "live");
    expect(records.cursor).toBe("live");
    expect(() => records.append([], "stale", "next")).toThrow("Stale");
    expect(() => records.append([{ seq: 2, kind: "x" }, { seq: 1, kind: "x" }], "live", "next")).toThrow("strictly increasing");
    expect(records.cursor).toBe("live"); expect(records.values()).toEqual([]);
    records.append([{ seq: 1, kind: "x" }], "live", "next");
    expect(() => records.append([], "next", "live")).toThrow("cursor did not advance");
  });
  test("disconnect stops viewer polling without cancelling in-flight work", async () => {
    let calls = 0; let finish!: () => void; let completed = false;
    const stop = startPolling(async () => { calls++; await new Promise<void>(resolve => { finish = resolve; }); completed = true; }, () => {}, 1);
    stop(); finish(); await Bun.sleep(10);
    expect(completed).toBe(true); expect(calls).toBe(1);
  });
  test("distinguishes unconsumed/missing capture from pending records", () => {
    expect(cellAvailability(result.records, "r1", "decode", 1, 12, false)).toContain("unconsumed");
    expect(cellAvailability(result.records, "r1", "decode", 2, 12, true)).toContain("Not captured");
    const scopeRecords: Sequenced[] = [{ seq: 0, kind: "prepared_input", token_ids: [], rendering: {}, resolved_scopes: [{ id: "r", kind: "readout", scope: { layers: { kind: "all" }, decode: { kind: "all" } } }] }];
    expect(cellAvailability(scopeRecords, "r", "decode", 3, 2, false)).toContain("Pending");
    expect(cellAvailability(scopeRecords, "r", "prefill", 3, 2, false)).toContain("outside scope");
  });
});

describe("scope precision and draft preservation", () => {
  test("selected scope uses phase index, not position or prediction", () => {
    const readout = decodeResult(result).records.find(isReadout)!;
    expect(sourceScope(readout)).toEqual({ layers: { kind: "values", values: [12] }, decode: { kind: "values", values: [0] } });
    expect(sourceScope({ ...readout, phase: "prefill", index: 7, position: 100, predicts_position: 101 })).toEqual({ layers: { kind: "values", values: [12] }, prefill: { kind: "values", values: [7] } });
    expect(layerValues("3, 5-7, 6")).toEqual([3, 5, 6, 7]);
    expect(() => layerValues("7-5")).toThrow();
  });
  test("ordered opaque operations survive persistence, reordering and disabling", () => {
    const draft = draftFromBody(JSON.stringify(request));
    const opaque = { id: "future", scope: { strange: true }, action: { kind: "future_operator", native_fields: [1, 2] } };
    draft.operations.push({ key: "opaque", enabled: true, document: opaque });
    const saved = parseDraft(JSON.stringify(draft));
    saved.operations = moveRow(saved.operations, 1, -1);
    expect(submissionConfig(saved).diagnostics.operations[0]).toEqual(opaque);
    saved.operations[0]!.enabled = false;
    expect(submissionConfig(saved).diagnostics.operations).toHaveLength(1);
    expect(saved.operations[0]!.document).toEqual(opaque);
  });
  test("capability limits block unsupported settings instead of silently clamping", () => {
    const draft = draftFromBody(JSON.stringify(request));
    draft.generationMode = "invented-mode";
    draft.generation.max_new_tokens = capabilities.limits.max_new_tokens + 1;
    expect(validateCapabilities(draft, decodeCapabilities(capabilities), decodeAssets(assets)).join("\n")).toContain("not supported");
    expect(draft.generation.max_new_tokens).toBe(capabilities.limits.max_new_tokens + 1);
    expect(() => parseDraft('{"version":2}')).toThrow("not overwritten");
    expect(newDraft().generationMode).toBe("");
  });
});
