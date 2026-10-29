import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdtemp, mkdir, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { previewResponse } from "./preview";

let owned: string;
let root: string;
beforeAll(async () => {
  owned = await mkdtemp(join(tmpdir(), "work-microscope-preview-"));
  root = join(owned, "dist");
  await mkdir(root);
  await Bun.write(join(root, "index.html"), "<!doctype html><title>Owned fixture</title>");
  await Bun.write(join(root, "evidence.json"), '{"fixture":true}');
  await Bun.write(join(owned, "private.txt"), "outside preview");
  await symlink(join(owned, "private.txt"), join(root, "outside.txt"));
});
afterAll(async () => { if (owned) await rm(owned, { recursive: true }); });

test("serves only explicit assets, with no HTML fallback", async () => {
  const index = await previewResponse(new Request("http://127.0.0.1/"), root);
  expect(index.status).toBe(200);
  expect(await index.text()).toContain("Owned fixture");
  expect(index.headers.get("Content-Type")).toContain("text/html");
  const json = await previewResponse(new Request("http://127.0.0.1/evidence.json"), root);
  expect(await json.json()).toEqual({ fixture: true });
  expect((await previewResponse(new Request("http://127.0.0.1/missing"), root)).status).toBe(404);
});

test("HEAD has no body; mutation methods are refused", async () => {
  const head = await previewResponse(new Request("http://127.0.0.1/", { method: "HEAD" }), root);
  expect(head.status).toBe(200);
  expect(await head.text()).toBe("");
  expect(head.headers.get("Content-Length")).toBe(String(Bun.file(join(root, "index.html")).size));
  for (const method of ["POST", "PUT", "DELETE", "OPTIONS"]) {
    const response = await previewResponse(new Request("http://127.0.0.1/", { method }), root);
    expect(response.status).toBe(405);
    expect(response.headers.get("Allow")).toBe("GET, HEAD");
  }
});

test("encoded traversal and malformed escapes cannot read outside dist", async () => {
  for (const path of ["/..%2fprivate.txt", "/%2e%2e%2fprivate.txt", "/%252e%252e%252fprivate.txt", "/..%2f..%2fprivate.txt"]) {
    const response = await previewResponse(new Request(`http://127.0.0.1${path}`), root);
    expect(response.status).toBe(404);
    expect(await response.text()).not.toContain("outside preview");
  }
  expect((await previewResponse(new Request("http://127.0.0.1/%zz"), root)).status).toBe(400);
  expect((await previewResponse(new Request("http://127.0.0.1/%00"), root)).status).toBe(400);
  expect((await previewResponse(new Request("http://127.0.0.1/outside.txt"), root)).status).toBe(404);
});

test("security policy confines the client to its offline origin", async () => {
  const response = await previewResponse(new Request("http://127.0.0.1/"), root);
  expect(response.headers.get("Cache-Control")).toBe("no-store");
  expect(response.headers.get("X-Content-Type-Options")).toBe("nosniff");
  const policy = response.headers.get("Content-Security-Policy")!;
  expect(policy).toContain("script-src 'self'");
  expect(policy).toContain("connect-src 'self'");
  expect(policy).toContain("object-src 'none'");
  expect(policy).toContain("frame-ancestors 'none'");
});
