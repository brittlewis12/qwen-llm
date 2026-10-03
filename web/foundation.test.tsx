import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { App } from "./App";
import { ApiHttpError, requestJson, type Transport } from "./api";
import { proxyConfig, proxyTarget } from "./proxy";
import { matchingDirectionAsset } from "./viewer";
import type { Asset, Prepared, Readout } from "./contract";
import { savedModelIdentity } from "./bindings";

function transport(response: Response): Transport {
  return async () => response;
}

describe("frontend shell", () => {
  test("historical fitted scores cannot pin against a replaced or missing asset identity", () => {
    const asset = { alias: "j", identity: "current", available: true } as Asset;
    const record = { lens: "j", target_layer: 62, asset_identity: "current" } as unknown as Readout;
    const prepared = { model_identity: "model", model_identity_kind: "runtime_gguf_metadata_not_content_hash" } as unknown as Prepared;
    const model = savedModelIdentity(prepared);
    expect(matchingDirectionAsset(record, [asset], model)).toBe(asset);
    expect(matchingDirectionAsset({ ...record, asset_identity: "older" }, [asset], model)).toBeUndefined();
    expect(matchingDirectionAsset({ ...record, asset_identity: undefined }, [asset], model)).toBeUndefined();
    expect(matchingDirectionAsset(record, [{ ...asset, available: false }], model)).toBeUndefined();
    for (const invalid of [{ ...prepared, model_identity_kind: undefined }, { ...prepared, model_identity_kind: "unknown" }, { ...prepared, model_identity: "x".repeat(257) }]) {
      expect(matchingDirectionAsset(record, [asset], savedModelIdentity(invalid))).toBeUndefined();
    }
    const plain = { lens: "plain", target_layer: null } as Readout;
    const head = { ...asset, alias: "plain", identity: "model" };
    expect(matchingDirectionAsset(plain, [head], model)).toBe(head);
    expect(matchingDirectionAsset(plain, [{ ...head, identity: "other" }], model)).toBeUndefined();
  });
  test("does not claim results or enable submission before discovery", () => {
    const html = renderToStaticMarkup(<App />);
    expect(html).toContain("Connecting to Lens");
    expect(html).toContain("Capabilities have not been loaded.");
    expect(html).toContain('disabled="">Run experiment');
    for (const name of ["Input", "Execution", "History"]) expect(html).toContain(`section-${name}`);
  });

  test("JSON remains unknown until contract validation", async () => {
    expect(await requestJson("/fixture", undefined, transport(Response.json({ fixture: true })))).toEqual({ fixture: true });
  });

  test("preserves complete HTTP error body, status and headers", async () => {
    const response = new Response("first error\nsecond error", { status: 422, headers: { "x-request-id": "fixture" } });
    try {
      await requestJson("/fixture", undefined, transport(response));
      throw new Error("Expected HTTP error");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiHttpError);
      const failure = error as ApiHttpError;
      expect(failure.body).toBe("first error\nsecond error");
      expect(failure.response.status).toBe(422);
      expect(failure.response.headers.get("x-request-id")).toBe("fixture");
    }
  });

  test("preserves malformed JSON and network failures", async () => {
    await expect(requestJson("/fixture", undefined, transport(new Response("not json")))).rejects.toThrow("not json");
    const failure = new Error("connection lost");
    await expect(requestJson("/fixture", undefined, async () => { throw failure; })).rejects.toBe(failure);
  });

  test("rejects cross-origin paths", async () => {
    for (const path of ["//example.com", "/\\example.com"] as const) {
      await expect(requestJson(path)).rejects.toThrow("same-origin");
    }
  });

  test("proxy is local, prefix-bounded and preserves paths and query", () => {
    const config = proxyConfig({ QWEN_SERVE_ORIGIN: "http://localhost:9090", QWEN_API_PREFIX: "/v1" });
    expect(proxyTarget("http://localhost:3000/v1/jobs?after=2", config)?.href).toBe("http://localhost:9090/v1/jobs?after=2");
    expect(proxyTarget("http://localhost:3000/v10", config)).toBeNull();
    expect(() => proxyConfig({ QWEN_SERVE_ORIGIN: "http://example.com" })).toThrow();
    expect(() => proxyConfig({ QWEN_API_PREFIX: "/" })).toThrow();
  });
});
