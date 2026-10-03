import { mkdir } from "node:fs/promises";
import { strict as assert } from "node:assert";
import { draftFromBody } from "./draft";
import { decodeResult, isReadout, isRetainedArray, isSample } from "./contract";
import { createHash } from "node:crypto";
import { residualMetrics } from "./retention";

// Fixtures by default; opt-in baseline mode contacts the live-check-owned server.
// Neither mode is imported by the application or production build.
const liveOrigin = Bun.env.LENS_BASELINE_ORIGIN;
if (liveOrigin) {
  const url = new URL(liveOrigin);
  if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || url.origin !== liveOrigin) throw new Error("Live browser check requires an explicit loopback origin.");
}
const fixtures = `${import.meta.dir}/../crates/qwen-cli/tests/fixtures/lens_http_v1`;
const load = (name: string) => Bun.file(`${fixtures}/${name}.json`).json();
const [caps, assets, statusFixture, resultFixture, requestFixture, errorFixture] = await Promise.all(["capabilities", "assets", "status", "result", "request", "error"].map(load));
const output = `${import.meta.dir}/.browser-test`;
await mkdir(output, { recursive: true });
const profile = `${output}/profile-${crypto.randomUUID()}`;
const manifest = await Bun.file(`${import.meta.dir}/dist/asset-manifest.json`).json();
const assetPaths = new Set<string>(manifest.files.map((file: { path: string }) => `/${file.path}`));
type TestJob = { id: string; key: string; body: string; ordinal: number; cancelled: boolean; emptyPages: number };
const jobs = new Map<string, TestJob>();
const posts: string[] = [];
let cancelCount = 0;
let requestReads = 0;
let firstFailure = true;
let overloadRecovery = true;
let assetRevision = 0;
let preparedIdentityKind: string | undefined = "unknown";
let heldRequest: Promise<void> | null = null;
let historyPageSize = 64;
let historyPreviews = true;
let corruptPreview = false;
let arrayReads = 0;
let navigationStage = 0;
const archiveScores = new Float32Array(caps.model.vocabulary_size).fill(-4);
archiveScores[456] = 4.2; archiveScores[42] = 3;
for (let i = 0; i < 22; i++) archiveScores[100 + i] = -1 - i * .05;
const archiveBytes = (values: Float32Array) => { const bytes = new Uint8Array(values.length * 4); const view = new DataView(bytes.buffer); values.forEach((v, i) => view.setFloat32(i * 4, v, true)); return bytes; };
const scoreBytes = archiveBytes(archiveScores), sourceBytes = archiveBytes(new Float32Array([1, 2, 3]));
const beforeBytes = archiveBytes(new Float32Array([1, 2, 2]));
const arrayDescriptor = (id: string, bytes: Uint8Array, offset: number) => ({ dtype: "f32le", length: bytes.length / 4, byte_length: bytes.length, offset: offset === 0 ? 0 : offset === 100 ? sourceBytes.length : sourceBytes.length + scoreBytes.length, sha256: createHash("sha256").update(bytes).digest("hex"), url: `/v1/lens/jobs/${id}/arrays/${offset}` });
function requestPreview(job: TestJob) {
  const messages = JSON.parse(job.body).input.messages;
  const index = messages.findLastIndex((message: any) => message.role === "user");
  if (index < 0 || typeof messages[index].content !== "string") return null;
  const chars = messages[index].content[Symbol.iterator]();
  let text = "";
  for (let i = 0; i < 240; i++) { const char = chars.next(); if (char.done) break; text += char.value; }
  return { message_index: index, message_count: messages.length, text: corruptPreview ? text.replace(/^./u, "!") : text, truncated: !chars.next().done };
}
const currentAssets = () => ({ ...assets, assets: assets.assets.map((asset: any) => ({ ...asset, identity: assetRevision && asset.alias === "plain" ? `${asset.identity}-revision-${assetRevision}` : asset.identity })) });
function status(job: TestJob) {
  const completed = job.ordinal === 0 && job.emptyPages > 0 && navigationStage === 3;
  return { ...statusFixture, id: job.id, revision: 18 + job.emptyPages + (job.cancelled ? 20 : 0),
    state: job.cancelled ? "cancelled" : completed ? "completed" : "running", cancel_requested: job.cancelled,
    generation: { ...statusFixture.generation, state: job.cancelled ? "cancelled" : completed ? "completed" : "running", stop_reason: job.cancelled ? "cancelled" : completed ? "token_limit" : null },
    observations: { ...statusFixture.observations, state: job.cancelled ? "partial" : completed ? "complete" : "writing" },
    result: { available: true, complete: completed || job.cancelled, url: `/v1/lens/jobs/${job.id}/result`, error: null },
  };
}
const server = liveOrigin ? null : Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(request) {
  const url = new URL(request.url);
  if (url.pathname === "/v1/lens/capabilities") return Response.json(caps);
  if (url.pathname === "/v1/lens/assets") return Response.json(currentAssets());
  if (url.pathname === "/v1/lens/jobs" && request.method === "POST") {
    const body = await request.text(); posts.push(body);
    const input = JSON.parse(body);
    let job = jobs.get(input.idempotency_key);
    const existing = Boolean(job);
    if (job && job.body !== body) return Response.json(errorFixture, { status: 409 });
    if (!job && input.preconditions) {
      const expected = input.preconditions;
      const referenced = [...new Set([...input.diagnostics.directions, ...input.diagnostics.readouts].map((row: any) => row.lens))].sort();
      const declared = Object.keys(expected.asset_identities).sort();
      if (JSON.stringify(referenced) !== JSON.stringify(declared)) throw new Error("Fixture received incomplete identity assertions");
      if (expected.model_identity !== caps.model.identity || declared.some(alias => !currentAssets().assets.some((asset: any) => asset.alias === alias && asset.identity === expected.asset_identities[alias]))) {
        return Response.json({ error: { type: "invalid_request_error", code: "binding_mismatch", message: "Fixture deployment changed; no acceptance." },
          admission: { schema_version: 1, idempotency_key: input.idempotency_key, state: "not_accepted" } }, { status: 412 });
      }
    }
    if (!job) { job = { id: jobs.size === 0 ? "job_example" : `job_variant_${jobs.size}`, key: input.idempotency_key, body, ordinal: jobs.size, cancelled: false, emptyPages: 0 }; jobs.set(job.key, job); }
    if (firstFailure) { firstFailure = false; return Response.json({ ...errorFixture, error: { ...errorFixture.error, code: "simulated_response_loss", message: "Browser fixture: accepted job, lost response; retry the same key." } }, { status: 500 }); }
    if (overloadRecovery) { overloadRecovery = false; return Response.json({ error: { type: "server_busy", code: "simulated_overload", message: "Generic worker saturation, no acceptance decision" } }, { status: 503 }); }
    return Response.json(status(job), { status: existing ? 200 : 202 });
  }
  if (url.pathname === "/v1/lens/jobs") {
    const all = [...jobs.values()].sort((a, b) => b.ordinal - a.ordinal);
    const cursor = url.searchParams.get("cursor");
    const offset = cursor === null ? 0 : all.findIndex(job => job.id === cursor) + 1;
    const selected = all.slice(offset, offset + historyPageSize);
    return Response.json({ schema_version: 1, jobs: selected.map(status), next_cursor: offset + selected.length < all.length ? selected.at(-1)!.id : null,
      ...(historyPreviews ? { request_previews: Object.fromEntries(selected.map(job => [job.id, requestPreview(job)])) } : {}) });
  }
  const binary = /^\/v1\/lens\/jobs\/([^/]+)\/arrays\/(0|100|200)$/.exec(url.pathname);
  if (binary) {
    if (![...jobs.values()].some(job => job.id === binary[1])) return new Response(null, { status: 404 });
    arrayReads++;
    return new Response(binary[2] === "0" ? sourceBytes : binary[2] === "100" ? scoreBytes : beforeBytes, { headers: { "content-type": "application/octet-stream" } });
  }
  const match = /^\/v1\/lens\/jobs\/([^/]+)(\/result|\/cancel|\/request)?$/.exec(url.pathname);
  if (match) {
    const job = [...jobs.values()].find(job => job.id === match[1]);
    if (!job) return Response.json(errorFixture, { status: 404 });
    if (match[2] === "/request") { requestReads++; if (heldRequest) await heldRequest; return Response.json({ schema_version: 1, job_id: job.id, request: JSON.parse(job.body) }); }
    if (match[2] === "/cancel") { cancelCount++; job.cancelled = true; return Response.json(status(job)); }
    if (match[2] === "/result") {
      const records = structuredClone(resultFixture.records);
      const prepared = records.find((record: { kind: string }) => record.kind === "prepared_input");
      if (job.ordinal === 0) records.find((record: any) => record.kind === "readout").scores.push({ token_id: 42, row_id: 42, label: " alternative", score: 3 }, ...Array.from({ length: 22 }, (_, i) => ({ token_id: 100 + i, row_id: 100 + i, label: ` candidate${i}`, score: archiveScores[100 + i] })));
      prepared.prompt_text = `Fixture prepared prompt, never the generated output. Identity kind: ${preparedIdentityKind ?? "missing"}`;
      prepared.prompt_bytes = [...new TextEncoder().encode(prepared.prompt_text)];
      prepared.model_identity_kind = preparedIdentityKind;
      if (JSON.parse(job.body).diagnostics.readouts.some((row: any) => row.retain === "scores_and_residual")) {
        prepared.retention_admission = { hidden_size: 3, vocabulary_size: archiveScores.length, raw_bytes_upper: sourceBytes.length + scoreBytes.length, source_arrays_upper: 1, score_arrays_upper: 1 };
        const row = records.find((record: any) => record.kind === "readout");
        row.asset_identity = null;
        row.retained = { source_key: "source-2-12", logits_key: "logits-2-12-plain" };
        const site = { position: row.position, source_layer: row.source_layer, phase: row.phase, index: row.index, input_token_id: row.input_token_id, capture_stage: row.capture_stage, provenance: row.provenance, applied_operation_ids: row.applied_operation_ids };
        records.push({ seq: 6, kind: "retained_array", key: "source-2-12", quantity: "source_residual", ...site, array: arrayDescriptor(job.id, sourceBytes, 0) });
        records.push({ seq: 7, kind: "retained_array", key: "logits-2-12-plain", quantity: "readout_logits", source_key: "source-2-12", lens: "plain", target_layer: null, asset_identity: null, score_kind: "logit", candidate_universe: "full_vocabulary", ...site, array: arrayDescriptor(job.id, scoreBytes, 100) });
        const pairs = JSON.parse(job.body).diagnostics.residual_pairs ?? [];
        if (pairs.length) {
          prepared.retention_admission.raw_bytes_upper += beforeBytes.length;
          prepared.retention_admission.before_arrays_upper = 1;
          prepared.retention_admission.pair_rows_upper = pairs.length;
          records.push({ seq: 8, kind: "retained_array", key: "before-2-12", quantity: "source_residual_before", ...site, capture_stage: "post_block_before_operations", applied_operation_ids: [], site_operation_ids: site.applied_operation_ids, array: arrayDescriptor(job.id, beforeBytes, 200) });
          for (const pair of pairs) records.push({ seq: records.length, kind: "residual_pair", id: pair.id, ...site, before_key: "before-2-12", after_key: "source-2-12", capture_stage: "whole_post_block_program", metrics: residualMetrics(new Float32Array([1, 2, 2]), new Float32Array([1, 2, 3])) });
        }
      }
      if (job.cancelled) { const terminal = records.find((record: { kind: string }) => record.kind === "generation_terminal"); terminal.state = "cancelled"; terminal.stop_reason = "cancelled"; }
      if (job.ordinal === 0) {
        const original = records.find((r: any) => r.kind === "readout");
        const extra = { ...original, retained: null, readout_id: "r-later" };
        records.push({ ...extra, seq: records.length, phase: "prefill", index: 0, position: 0, input_token_id: 1, predicts_position: 1, applied_operation_ids: [] });
        records.push({ ...original, seq: records.length, retained: null, source_layer: 13, applied_operation_ids: [], scores: [{ token_id: 42, row_id: 42, label: " alternative", score: 3 }] });
        prepared.resolved_scopes = [{ kind: "readout", id: "r-later", scope: { layers: { kind: "values", values: [12] }, prefill: { kind: "values", values: [0] }, decode: { kind: "values", values: [0] } } }];
        if (navigationStage >= 1) records.push({ seq: records.length, kind: "fixture_progress" });
        if (navigationStage >= 2) records.push({ ...extra, seq: records.length });
      }
      const cursor = url.searchParams.get("cursor");
      if (job.ordinal === 0 && cursor?.startsWith("nav-")) {
        const offset = Number(cursor.slice(4));
        return Response.json({ ...resultFixture, job_id: job.id, records: records.slice(offset), next_cursor: navigationStage < 3 ? `nav-${records.length}` : null, complete: navigationStage === 3 });
      }
      if (cursor === null) return Response.json({ ...resultFixture, job_id: job.id, records: records.slice(0, 4), next_cursor: "after-3", complete: false });
      if (cursor !== "after-3") return Response.json(errorFixture, { status: 400 });
      if (job.emptyPages++ === 0 || (job.ordinal > 0 && !job.cancelled)) return Response.json({ ...resultFixture, job_id: job.id, records: [], next_cursor: "after-3", complete: false });
      return Response.json({ ...resultFixture, job_id: job.id, records: records.slice(4), next_cursor: job.ordinal === 0 && navigationStage < 3 ? `nav-${records.length}` : null, complete: job.ordinal !== 0 || navigationStage === 3 });
    }
    return Response.json(status(job));
  }
  const path = url.pathname === "/" ? "/index.html" : url.pathname;
  if (assetPaths.has(path)) return new Response(Bun.file(`${import.meta.dir}/dist${path}`));
  return new Response("Not found", { status: 404 });
} });

const browser = Bun.spawn([Bun.env.BROWSER_BIN ?? "/Applications/Chromium.app/Contents/MacOS/Chromium", "--headless=new", "--disable-gpu", "--enable-features=CDPScreenshotNewSurface", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--no-first-run", "--no-default-browser-check", "--disable-background-networking", "--disable-sync", "--disable-extensions", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"], { stdout: "ignore", stderr: "pipe" });
const browserLog = new Response(browser.stderr).text();
let socket: WebSocket | undefined;
let stopping = false;
let interrupted = false;
let forced = false;
let forceStop: ReturnType<typeof setTimeout> | undefined;
const abort = new AbortController();
const pending = new Map<number, { resolve: (value: any) => void; reject: (reason: unknown) => void; timer: ReturnType<typeof setTimeout> }>();
function stopBrowser() {
  if (stopping) return;
  stopping = true;
  abort.abort(new Error("Browser check stopped"));
  for (const request of pending.values()) { clearTimeout(request.timer); request.reject(new Error("Browser check stopped")); }
  pending.clear();
  socket?.close();
  if (browser.exitCode === null) {
    browser.kill("SIGTERM");
    forceStop = setTimeout(() => { if (browser.exitCode === null) { forced = true; browser.kill("SIGKILL"); } }, 5000);
  }
}
function interrupt() { interrupted = true; stopBrowser(); }
process.on("SIGTERM", interrupt);
process.on("SIGINT", interrupt);
try {
  let port = "";
  for (let i = 0; i < 100; i++) {
    assert(!stopping && browser.exitCode === null, "Owned browser stopped during startup");
    const file = Bun.file(`${profile}/DevToolsActivePort`);
    if (await file.exists()) { port = (await file.text()).split("\n")[0]!; break; }
    await Bun.sleep(100);
  }
  if (!port) throw new Error("Chromium did not expose its isolated debugging endpoint");
  let target: { type: string; webSocketDebuggerUrl: string } | undefined;
  for (let attempt = 0; attempt < 100 && !target; attempt++) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(5000)]) })).json() as { type: string; webSocketDebuggerUrl: string }[];
    target = targets.find(target => target.type === "page");
    if (!target) await Bun.sleep(100);
  }
  if (!target) throw new Error("No Chromium page target");
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise<void>((resolve, reject) => {
    const failed = () => { clearTimeout(timer); reject(new Error("CDP socket failed or interrupted")); };
    const timer = setTimeout(failed, 5000);
    abort.signal.addEventListener("abort", failed, { once: true });
    socket!.onopen = () => { clearTimeout(timer); abort.signal.removeEventListener("abort", failed); resolve(); };
    socket!.onerror = failed;
    socket!.onclose = failed;
    if (abort.signal.aborted) failed();
  });
  let nextId = 0;
  const runtimeErrors: unknown[] = [];
  let screencastFrame: ((data: string) => void) | null = null;
  function send(method: string, params: Record<string, unknown> = {}): Promise<any> {
    if (stopping) return Promise.reject(new Error("Browser check interrupted"));
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, method === "Page.captureScreenshot" ? 60000 : 15000);
      pending.set(id, { resolve, reject, timer }); socket!.send(JSON.stringify({ id, method, params }));
    });
  }
  socket.onmessage = event => {
    const message = JSON.parse(String(event.data));
    if (message.id) {
      const request = pending.get(message.id); if (!request) return;
      clearTimeout(request.timer); pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error))); else request.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") runtimeErrors.push(message.params);
    else if (message.method === "Page.javascriptDialogOpening") void send("Page.handleJavaScriptDialog", { accept: true });
    else if (message.method === "Page.screencastFrame") { screencastFrame?.(message.params.data); void send("Page.screencastFrameAck", { sessionId: message.params.sessionId }); }
  };
  async function evaluate<T = unknown>(expression: string): Promise<T> {
    const response = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
  }
  async function wait(expression: string, label: string) {
    for (let i = 0; i < 150; i++) { if (await evaluate(`!!document.body && (${expression})`)) return; await Bun.sleep(100); }
    throw new Error(`Browser wait failed: ${label}`);
  }
  async function screenshot(path: string) {
    if (Bun.env.CAPTURE_SCREENSHOTS !== "1") return;
    await send("Emulation.setFocusEmulationEnabled", { enabled: true });
    await send("Page.bringToFront");
    await evaluate("document.body.getBoundingClientRect().height");
    const frame = new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => { screencastFrame = null; reject(new Error("Chromium produced no screencast frame")); }, 15000);
      screencastFrame = data => { clearTimeout(timer); screencastFrame = null; resolve(data); };
    });
    await send("Page.startScreencast", { format: "png", everyNthFrame: 1 });
    try { await Bun.write(path, Buffer.from(await frame, "base64")); }
    finally { await send("Page.stopScreencast"); }
  }
  const click = (text: string) => evaluate(`(() => { const button = [...document.querySelectorAll('button')].find(b => b.textContent === ${JSON.stringify(text)} && b.getClientRects().length); if (!button || button.disabled) throw new Error('Button unavailable: ' + ${JSON.stringify(text)}); button.click(); })()`);
  const setInput = (selector: string, value: string) => evaluate(`(() => { const input = document.querySelector(${JSON.stringify(selector)}); const proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : input.tagName === 'SELECT' ? HTMLSelectElement.prototype : HTMLInputElement.prototype; Object.getOwnPropertyDescriptor(proto, 'value').set.call(input, ${JSON.stringify(value)}); input.dispatchEvent(new Event(input.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true })); })()`);
  await send("Page.enable"); await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  if (liveOrigin && Bun.env.LENS_TEST_BASELINE_ONLY === "1") {
    await send("Page.navigate", { url: liveOrigin });
    const fitted = Bun.env.LENS_TEST_FITTED_ONLY === "1";
    const plain = Bun.env.LENS_TEST_PLAIN_ONLY === "1" || fitted;
    await wait(`document.body.textContent.includes(${JSON.stringify(plain ? "Readouts available" : "Baseline available")})`, "discovery alongside history polling");
    await setInput("#message-0", "Name an animal.");
    await setInput("#generation-mode", "no_thinking");
    await evaluate(`document.querySelector('label.check input').click()`);
    await wait(`!!document.querySelector('#prefix-text')`, "mobile typed prefill");
    await setInput("#prefix-text", "  Answer:\n");
    for (const [label, value] of [["Maximum new tokens", "8"], ["Temperature", "0.8"]]) {
      const id = await evaluate<string>(`[...document.querySelectorAll('label')].find(label => label.textContent === ${JSON.stringify(label)}).htmlFor`);
      await setInput(`[id=${JSON.stringify(id)}]`, value!);
    }
    assert(await evaluate<boolean>(`[...document.querySelectorAll('button')].filter(b => ${JSON.stringify(plain ? ["Add direction", "Add operation"] : ["Add readout", "Add direction", "Add operation"])}.includes(b.textContent)).every(b => b.disabled)`));
    if (plain) {
      await click("Add readout");
      if (fitted) {
        await evaluate(`(() => { const input = [...document.querySelectorAll('label')].find(label => label.textContent.startsWith('Readout alias')).querySelector('select'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(input, 'fit'); input.dispatchEvent(new Event('change', { bubbles: true })); })()`);
        await wait(`document.body.textContent.includes('source_deployment_equivalence_unverified')`, "fitted binding disclosure");
      }
      await wait(`document.querySelectorAll('.scope-editor select').length === 3`, "plain numeric scopes");
      await evaluate(`(() => { const input = document.querySelectorAll('.scope-editor select')[1]; Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(input, 'none'); input.dispatchEvent(new Event('change', { bubbles: true })); })()`);
      await evaluate(`(() => { const input = document.querySelectorAll('.scope-editor select')[2]; Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(input, 'values'); input.dispatchEvent(new Event('change', { bubbles: true })); })()`);
    }
    await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`, "baseline run enabled");
    await click("Run experiment");
    await wait(`!!document.querySelector('.job-id')?.textContent`, "durable baseline acceptance");
    const jobId = await evaluate<string>(`document.querySelector('.job-id').textContent`);
    await send("Page.reload");
    await wait(`document.querySelector('.job-id')?.textContent === ${JSON.stringify(jobId)}`, "selection survives reload");
    await click("02 Execution");
    await wait(`document.querySelector('.job-status')?.textContent.includes('completed')`, "baseline completed independently of viewer");
    const recorded = decodeResult(await (await fetch(`${liveOrigin}/v1/lens/jobs/${jobId}/result`)).json());
    assert(recorded.complete);
    const output = new TextDecoder().decode(new Uint8Array(recorded.records.filter(isSample).flatMap(sample => sample.piece_bytes)));
    assert(output.length > 0);
    const matches = `document.querySelector('.sampled-output')?.checkVisibility() && document.querySelector('.sampled-output').textContent === ${JSON.stringify(output)}`;
    await wait(matches, "displayed output matches durable sampled bytes");
    if (plain) {
      const rows = recorded.records.filter(isReadout);
      assert.equal(rows.length, 1);
      assert.equal(rows[0]!.lens, fitted ? "fit" : "plain");
      assert.equal(rows[0]!.phase, "decode"); assert.equal(rows[0]!.index, 0); assert.equal(rows[0]!.source_layer, 0);
      await wait(`!!document.querySelector('button[aria-label="Inspect saved layer 0"]')`, "saved plain layer");
      await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 0"]').click()`);
      if (fitted) await wait(`document.querySelector('.score-panel')?.textContent.includes('source_deployment_equivalence_unverified') && document.querySelector('.score-panel')?.textContent.includes('not the final generation distribution')`, "saved fitted provenance");
      const expected = rows[0]!.scores.map(score => [String(score.score), `Token ${score.token_id} / row ${score.row_id}`]);
      await wait(`JSON.stringify([...document.querySelectorAll('.score-panel .scores > li')].map(node => [node.querySelector('.score-value').textContent, node.querySelector('p.muted').textContent])) === ${JSON.stringify(JSON.stringify(expected))}`, "plain displayed scores equal actual saved records");
    }
    const before = await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json();
    await click("Run again: prepare same seed");
    await wait(`document.body.textContent.includes(${JSON.stringify(`Copied ${jobId} with the same seed`)})`, "copy saved request without inference");
    const copied = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1'))`);
    assert.equal(copied.prefix.text, "  Answer:\n");
    assert.equal(copied.generation.sampling.temperature, 0.8);
    assert.equal(copied.readouts.length, plain ? 1 : 0);
    await click("03 History");
    await click("Refresh history now");
    await wait(`document.querySelector('.history-list')?.textContent.includes(${JSON.stringify(jobId)})`, "mobile server history");
    assert(!await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`), "mobile baseline history overflow");
    await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
    await click("02 Execution");
    await wait(matches, "desktop reopens the same output");
    assert(!await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`), "desktop baseline overflow");
    assert.deepEqual(await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json(), before);
    assert.equal(await evaluate<number>(`document.querySelectorAll('.error-ledger details').length`), 0, "real control-pool startup must not strand discovery or report transient read saturation");
    console.log(`${fitted ? "Fitted readout" : plain ? "Plain readout" : "Baseline"} browser passed: actual same-port assets/store/producer, phone prefill/sampling/submit/reload/copy/history, exact records, desktop parity, no extra jobs. Job ${jobId}`);
  } else if (liveOrigin) {
    const fittedAlias = Bun.env.LENS_TEST_FITTED_ALIAS;
    const sourceLayer = fittedAlias ? 46 : 0;
    await send("Page.navigate", { url: liveOrigin });
    await wait(`document.body.textContent.includes('Readouts available') || document.body.textContent.includes('Lens available')`, "live readout capability discovery");
    await setInput("#message-0", "Name three primary colors.");
    await setInput("#generation-mode", "no_thinking");
    await evaluate(`document.querySelector('label.check input').click()`);
    await wait(`!!document.querySelector('#prefix-text')`, "phone assistant prefill control");
    await setInput("#prefix-text", "  Answer:\n");
    const setNumber = async (label: string, value: string) => {
      const id = await evaluate<string>(`[...document.querySelectorAll('label')].find(label => label.textContent === ${JSON.stringify(label)}).htmlFor`);
      await setInput(`[id=${JSON.stringify(id)}]`, value);
    };
    await setNumber("Maximum new tokens", "8");
    await setNumber("Temperature", "0.8");
    await click("Add readout");
    await evaluate(`(() => { const label = [...document.querySelectorAll('label')].find(label => label.textContent === 'Retain full scores and source residuals'); label.querySelector('input').click(); })()`);
    await wait(`document.querySelectorAll('.scope-editor select').length === 3`, "phone readout scopes");
    if (fittedAlias) {
      await evaluate(`(() => { const input = [...document.querySelectorAll('label')].find(label => label.textContent.startsWith('Readout alias')).querySelector('select'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(input, ${JSON.stringify(fittedAlias)}); input.dispatchEvent(new Event('change', { bubbles: true })); })()`);
      await setInput(".scope-editor .selector-editor input:not([type=checkbox])", String(sourceLayer));
      await evaluate(`document.querySelector('.scope-editor .selector-editor input:not([type=checkbox])').dispatchEvent(new FocusEvent('focusout', { bubbles: true }))`);
      await wait(`document.body.textContent.includes('source_deployment_equivalence_unverified')`, "phone fitted binding warning");
    }
    await evaluate(`(() => { const controls = document.querySelectorAll('.scope-editor select'); const set = (element, value) => { Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(element, value); element.dispatchEvent(new Event('change', { bubbles: true })); }; set(controls[1], 'none'); })()`);
    await evaluate(`(() => { const element = document.querySelectorAll('.scope-editor select')[2]; Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(element, 'values'); element.dispatchEvent(new Event('change', { bubbles: true })); })()`);
    const authoredReadouts = await evaluate<any[]>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).readouts`);
    await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`, "phone submission enabled");
    await click("Run experiment");
    await wait(`!!document.querySelector('.job-id')?.textContent`, "live durable acceptance");
    const jobId = await evaluate<string>(`document.querySelector('.job-id').textContent`);
    await send("Page.reload");
    await wait(`document.querySelector('.job-id')?.textContent === ${JSON.stringify(jobId)}`, "accepted selection survives reload");
    await click("02 Execution");
    await wait(`document.querySelector('.job-status')?.textContent.includes('completed')`, "live baseline completion after reload");
    const recorded = decodeResult(await (await fetch(`${liveOrigin}/v1/lens/jobs/${jobId}/result`)).json());
    if (!recorded.complete) throw new Error("Expected complete bounded live result");
    const retainedOutput = new TextDecoder().decode(new Uint8Array(recorded.records.filter(isSample).flatMap(sample => sample.piece_bytes)));
    if (!retainedOutput) throw new Error("Expected actual sampled bytes in live browser job");
    const outputMatches = `document.querySelector('.sampled-output')?.checkVisibility() && document.querySelector('.sampled-output').textContent === ${JSON.stringify(retainedOutput)}`;
    await wait(outputMatches, "visible sampled output equals retained token bytes, not the prepared prompt");
    const readouts = recorded.records.filter(isReadout);
    if (readouts.length !== 1 || readouts[0]!.phase !== "decode" || readouts[0]!.index !== 0 || readouts[0]!.source_layer !== sourceLayer || readouts[0]!.lens !== (fittedAlias ?? "plain")) throw new Error("Phone readout scopes or alias did not reach original consumed position");
    if (readouts[0]!.readout_id !== authoredReadouts[0].id || readouts[0]!.scores.length !== authoredReadouts[0].top_k || readouts[0]!.scores.length === 0) throw new Error("Phone readout identity or requested top-k count mismatch");
    await wait(`!!document.querySelector('button[aria-label="Inspect saved layer ${sourceLayer}"]')`, "live saved layer selection");
    await evaluate(`document.querySelector('button[aria-label="Inspect saved layer ${sourceLayer}"]').click()`);
    if (fittedAlias) await wait(`document.querySelector('.score-panel')?.textContent.includes('source_deployment_equivalence_unverified') && document.querySelector('.score-panel')?.textContent.includes('not the final generation distribution')`, "persisted fitted binding and target semantics");
    const expectedScores = readouts[0]!.scores.map(score => [String(score.score), `Token ${score.token_id} / row ${score.row_id}`]);
    const scoresMatch = `document.querySelector('.score-panel')?.checkVisibility() && JSON.stringify([...document.querySelectorAll('.score-panel .scores > li')].map(node => [node.querySelector('.score-value').textContent, node.querySelector('p.muted').textContent])) === ${JSON.stringify(JSON.stringify(expectedScores))}`;
    await wait(scoresMatch, "visible token identities and scores equal recorded native logits");
    const retainedLogits = recorded.records.filter(isRetainedArray).find(record => record.quantity === "readout_logits")!;
    assert(retainedLogits);
    const binaryScores = await (await fetch(`${liveOrigin}${retainedLogits.array.url}`)).arrayBuffer();
    const fullView = new DataView(binaryScores);
    const excluded = new Set(readouts[0]!.scores.map(score => score.token_id));
    let queryToken = 0; while (excluded.has(queryToken)) queryToken++;
    const expectedScore = fullView.getFloat32(queryToken * 4, true);
    let expectedRank = 1;
    for (let i = 0; i < retainedLogits.array.length; i++) {
      const score = fullView.getFloat32(i * 4, true);
      if (score > expectedScore || (score === expectedScore && i < queryToken)) expectedRank++;
    }
    await evaluate(`document.querySelector('.saved-measurements > summary').click()`);
    await click("Load retained scores / no inference");
    await wait(`!!document.querySelector('.retained-token')`, "live retained vocabulary load");
    await setNumber("Retained vocabulary token ID", String(queryToken));
    await wait(`Number(document.querySelector('[data-field=rank]')?.textContent) === ${expectedRank} && Number(document.querySelector('[data-field=score]')?.textContent) === ${expectedScore}`, "live query outside retained top-k");
    console.log(`Live retained query passed for token ${queryToken}, rank ${expectedRank}, outside original top-k.`);
    const jobsBefore = await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json() as { jobs: unknown[] };
    await evaluate(`Object.keys(localStorage).filter(key => key.startsWith('qwen-lens.submission.v1')).forEach(key => localStorage.removeItem(key))`);
    await click("Run again: prepare same seed");
    await wait(`document.body.textContent.includes(${JSON.stringify(`Copied ${jobId} with the same seed`)})`, "copy from durable server request");
    const copied = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1'))`);
    if (copied.prefix.text !== "  Answer:\n" || copied.generation.sampling.temperature !== 0.8 || copied.readouts.length !== 1) throw new Error("Phone copy lost prefix, sampling or readout scope");
    assert.deepEqual(copied.readouts, authoredReadouts, "Phone copy changed readout configuration or scope");
    await click("03 History");
    await click("Refresh history now");
    await wait(`document.querySelector('.history-list')?.textContent.includes(${JSON.stringify(jobId)})`, "live phone history");
    if (await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`)) throw new Error("Live phone history overflows");
    await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
    await click("02 Execution");
    await wait(outputMatches, "visible reopened output equals server sampled bytes");
    await wait(scoresMatch, "visible reopened scores and token identities equal server records");
    if (await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`)) throw new Error("Live desktop execution overflows");
    const jobsAfter = await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json() as { jobs: unknown[] };
    if (JSON.stringify(jobsBefore.jobs) !== JSON.stringify(jobsAfter.jobs)) throw new Error("History navigation or draft copy changed server jobs");
    if (fittedAlias) {
      await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
      await evaluate(`document.querySelector('.candidate-choice').click()`);
      await click("Pin + operation");
      const pinned = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).operations.at(-1).document`);
      assert.deepEqual(pinned.scope, { layers: { kind: "values", values: [46] }, decode: { kind: "values", values: [0] } });
      await click("Variant / not selected");
      await click("01 Input");
      await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`, "pinned intervention variant admitted");
      await click("Run experiment");
      await wait(`document.querySelector('.job-id')?.textContent && document.querySelector('.job-id').textContent !== ${JSON.stringify(jobId)}`, "independent intervention variant accepted");
      const variantId = await evaluate<string>(`document.querySelector('.job-id').textContent`);
      await send("Page.reload");
      await wait(`document.querySelector('.job-id')?.textContent === ${JSON.stringify(variantId)}`, "variant selection survives reload");
      await click("02 Execution");
      await wait(`document.querySelector('.job-id')?.textContent === ${JSON.stringify(variantId)} && document.querySelector('.job-status')?.textContent.includes('completed')`, "variant survives reload and completes");
      const variant = decodeResult(await (await fetch(`${liveOrigin}/v1/lens/jobs/${variantId}/result`)).json());
      assert(variant.complete);
      const applied = variant.records.filter(record => record.kind === "operation_application");
      assert.equal(applied.length, 1);
      assert.equal(applied[0]!.id, pinned.id);
      assert.equal(applied[0]!.phase, "decode");
      assert.equal(applied[0]!.index, 0);
      assert.equal(applied[0]!.layer, 46);
      assert.equal(variant.records.filter(record => record.kind === "direction_prepared").length, 1);
      const variantRows = variant.records.filter(isReadout);
      assert.equal(variantRows.length, 1);
      assert.deepEqual(variantRows[0]!.applied_operation_ids, [pinned.id]);
      const variantScores = variantRows[0]!.scores.map(score => [String(score.score), `Token ${score.token_id} / row ${score.row_id}`]);
      const variantMatches = `JSON.stringify([...document.querySelectorAll('.score-panel .scores > li')].map(node => [node.querySelector('.score-value').textContent, node.querySelector('p.muted').textContent])) === ${JSON.stringify(JSON.stringify(variantScores))}`;
      await wait(variantMatches, "variant visible scores match intervened original-forward records");
      const beforeNavigation = await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json();
      await click(`Baseline / ${jobId}`);
      await wait(scoresMatch, "baseline scores retained after variant");
      await click("03 History");
      await click("Refresh history now");
      await wait(`document.querySelector('.history-list')?.textContent.includes(${JSON.stringify(variantId)})`, "intervention variant durable history");
      await click("02 Execution");
      await click(`Variant / ${variantId}`);
      await wait(variantMatches, "intervention scores reopened without inference");
      assert.deepEqual(await (await fetch(`${liveOrigin}/v1/lens/jobs`)).json(), beforeNavigation);
      assert(!await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`), "phone intervention view overflows");
      console.log(`Live phone pin/ordered-operation variant/reload/baseline-history passed. Variant ${variantId}`);
    }
    console.log(`Live browser passed: phone editing/prefill/sampling/readout scopes, submit/reload, byte-checked tokens, tap scores matched to retained logits, server-history copy, unchanged jobs, phone/desktop bounds. Job ${jobId}`);
  } else {
  const draft = draftFromBody(JSON.stringify(requestFixture));
  await send("Page.addScriptToEvaluateOnNewDocument", { source: `if (!localStorage.getItem('qwen-lens.draft.v1')) localStorage.setItem('qwen-lens.draft.v1', ${JSON.stringify(JSON.stringify(draft))});` });
  await send("Page.navigate", { url: server!.url.href });
  await wait(`document.body.textContent.includes('Lens available')`, "actual capability discovery");
  await setInput("#message-0", "Browser fixture prompt, edited on a phone.");
  await setInput("#generation-mode", "no_thinking");
  await wait(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).messages[0].content.includes('edited on a phone')`, "durable draft edit");
  await evaluate(`(() => { const label = [...document.querySelectorAll('label')].find(label => label.textContent === 'Retain full scores and source residuals'); label.querySelector('input').click(); })()`);
  await wait(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).readouts[0].retain === 'scores_and_residual'`, "opt-in retention persisted");
  await click("Add residual pair capture");
  await wait(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).residualPairs.length === 1`, "independent pair scope persisted on phone");
  await click("Run experiment");
  await wait(`document.body.textContent.includes('simulated_response_loss')`, "lost response exposed");
  if (Number(posts.length) !== 1 || Number(jobs.size) !== 1) throw new Error("First acceptance was not singular");
  const persisted = await evaluate<string>(`JSON.parse(localStorage.getItem('qwen-lens.submission.v1')).body`);
  if (persisted !== posts[0]) throw new Error("Saved request bytes differ from submitted bytes");
  await send("Page.reload");
  await wait(`document.body.textContent.includes('Submission recovery')`, "reload recovers unresolved request");
  await click("Recover / retry exact saved request");
  await wait(`document.body.textContent.includes('simulated_overload')`, "overloaded retry remains uncertain");
  if (await evaluate<boolean>(`JSON.parse(localStorage.getItem('qwen-lens.submission.v1')).rejected === true`)) throw new Error("Generic overload erased uncertainty about accepted work");
  await send("Page.reload");
  await wait(`document.body.textContent.includes('Submission recovery')`, "overload uncertainty survives reload");
  await click("Recover / retry exact saved request");
  await wait(`document.querySelector('.sampled-output')?.checkVisibility() && document.querySelector('.sampled-output').textContent === ' light scatters'`, "visible paged sampled output is not the prepared prompt");
  if (Number(posts.length) !== 3 || posts[0] !== posts[1] || posts[0] !== posts[2] || Number(jobs.size) !== 1) throw new Error("Recovery created a rerun or changed bytes");
  await wait(`document.querySelector('section[aria-labelledby="trace-heading"] select')?.options.length === 2`, "multiple saved readouts available before token manipulation");
  await setInput('section[aria-labelledby="trace-heading"] select', "r-later");
  await wait(`document.querySelector('.selected-alias-coverage')?.textContent.includes('decode 0')`, "changing initial readout retains the implicitly inspected source position");
  await setInput('section[aria-labelledby="trace-heading"] select', "r1");
  await wait(`!!document.querySelector('button[aria-label="Inspect saved layer 12"]')`, "returning readout recovers layers without another token click");
  await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 12"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('post_block_after_operations')`, "tap score metadata");
  await wait(`!![...document.querySelectorAll('button')].find(button => button.textContent === 'Load retained scores / no inference')`, "retained cell descriptors joined");
  assert.equal(arrayReads, 0, "history and observation must not load binary payloads implicitly");
  const beforeArrayPosts = posts.length;
  await evaluate(`document.querySelector('.saved-measurements > summary').click()`);
  await click("Load retained scores / no inference");
  await wait(`!!document.querySelector('.retained-token')`, "retained full vocabulary loaded");
  const queryId = await evaluate<string>(`[...document.querySelectorAll('label')].find(label => label.textContent === 'Retained vocabulary token ID').htmlFor`);
  await setInput(`[id=${JSON.stringify(queryId)}]`, "0");
  await wait(`document.querySelector('[data-field=rank]')?.textContent === '25' && document.querySelector('[data-field=score]')?.textContent === '-4'`, "query outside saved top-k uses archived scores");
  const probability = await evaluate<number>(`Number(document.querySelector('[data-field=probability]').textContent)`);
  const expectedProbability = Math.exp(-4) / archiveScores.reduce((sum, score) => sum + Math.exp(score), 0);
  assert(Math.abs(probability - expectedProbability) < 1e-12);
  assert.equal(arrayReads, 1); assert.equal(posts.length, beforeArrayPosts);
  await click("Release loaded scores");
  await evaluate(`(() => { const select = [...document.querySelectorAll('label')].find(label => label.textContent.startsWith('Measured site')).querySelector('select'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(select, select.options[1].value); select.dispatchEvent(new Event('change', { bubbles: true })); })()`);
  await wait(`!!document.querySelector('.pair-inspector')`, "select saved whole-site pair");
  await click("Verify measured change / no inference");
  await wait(`document.body.textContent.includes('Verified from retained before/after arrays.')`, "saved pair values and metrics verified on phone");
  assert.equal(arrayReads, 3); assert.equal(posts.length, beforeArrayPosts);
  const draftBeforeNavigation = await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`);
  await setInput(".token-navigation select", "prefill");
  await evaluate(`document.querySelector('button[aria-label="Inspect prefill token 1, ID 2"]').click()`);
  await wait(`document.querySelector('[data-field=token-coverage]')?.textContent.includes('Outside saved') && !document.querySelector('.score-panel') && !document.querySelector('.pair-inspector')`, "uncaptured prompt token never inherits another site's results");
  await setInput(".token-navigation select", "decode");
  await evaluate(`document.querySelector('button[aria-label="Inspect decode token 1, ID 456"]').click()`);
  await wait(`document.querySelector('[data-field=token-coverage]')?.textContent.includes('Unconsumed sample') && !document.querySelector('.score-panel') && !document.querySelector('.pair-inspector')`, "terminal token has no invented forward");
  await evaluate(`document.querySelector('button[aria-label="Inspect decode token 0, ID 123"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('Layer 12 / decode index 0') && !!document.querySelector('.pair-inspector')`, "consumed token selects saved readout and pair");
  assert.equal(await evaluate<number>(`document.querySelectorAll('.compact-scores > li').length`), 24);
  assert(await evaluate<boolean>(`[...document.querySelectorAll('.compact-scores > li')].every(row => getComputedStyle(row).paddingTop === '0px' && row.getBoundingClientRect().height < 90)`), "computed candidate rows stay compact on phone");
  await evaluate(`(() => { const candidate = document.querySelector('button[aria-label="Select candidate 121"]'); candidate.scrollIntoView(); candidate.click(); })()`);
  await click("Adjust intervention");
  await wait(`document.activeElement.classList.contains('intervention-dock') && document.querySelector('.intervention-dock').getBoundingClientRect().top < innerHeight`, "long top-k list has direct keyboard/touch access to selected intervention controls");
  assert.equal(posts.length, beforeArrayPosts); assert.equal(arrayReads, 3);
  await evaluate(`document.querySelector('button[aria-label="Select candidate 456"]').click()`);
  assert.equal(await evaluate<number>(`[...document.querySelectorAll('.score-panel button')].filter(b => b.textContent === 'Pin + operation').length`), 1, "one contextual intervention action, not one per score row");
  assert.equal(await evaluate<string>(`document.querySelector('.token-site h4').textContent`), "Input decode 0", "scored candidate selection must not change the consumed input token");
  await wait(`document.querySelector('button[aria-label="Inspect saved layer 13"]')?.textContent.includes('Not in saved rows')`, "candidate tracking distinguishes omitted top-k from zero score");
  await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 13"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('Layer 13 / decode index 0') && document.querySelector('.intervention-dock')?.textContent.includes('score is unavailable')`, "direct layer inspection preserves tracked candidate without invented score");
  await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 12"]').click()`);
  await click("Next token");
  await wait(`document.querySelector('[data-field=token-coverage]')?.textContent.includes('Unconsumed sample')`, "next token includes terminal samples rather than skipping missing measurements");
  await click("Previous token");
  await wait(`document.querySelector('.score-panel')?.textContent.includes('Layer 12 / decode index 0')`, "previous token returns to saved measurement");
  assert.equal(arrayReads, 3); assert.equal(posts.length, beforeArrayPosts);
  assert.equal(await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`), draftBeforeNavigation);
  const traceSelect = `section[aria-labelledby="trace-heading"] select`;
  await evaluate(`document.querySelector('.advanced-grid > summary').click()`);
  await setInput(traceSelect, "r-later");
  await wait(`document.querySelector('.selected-alias-coverage')?.textContent.includes('r-later') && !document.querySelector('.score-panel')`, "missing chosen alias remains selected");
  const layerInput = await evaluate<string>(`[...document.querySelectorAll('label')].find(label => label.textContent === 'Display first layer').htmlFor`);
  const indexInput = await evaluate<string>(`[...document.querySelectorAll('label')].find(label => label.textContent === 'Display first source index').htmlFor`);
  await setInput(`[id=${JSON.stringify(layerInput)}]`, "13");
  await setInput(`[id=${JSON.stringify(indexInput)}]`, "1");
  navigationStage = 1;
  await wait(`[...document.querySelectorAll('summary')].some(s => s.textContent.startsWith('Inspect loaded records (13)'))`, "additional immutable page arrives");
  assert.equal(await evaluate<string>(`document.querySelector(${JSON.stringify(traceSelect)}).value`), "r-later");
  assert.equal(await evaluate<string>(`document.getElementById(${JSON.stringify(layerInput)}).value`), "13");
  assert.equal(await evaluate<string>(`document.getElementById(${JSON.stringify(indexInput)}).value`), "1");
  await setInput(`[id=${JSON.stringify(layerInput)}]`, "12");
  navigationStage = 2;
  await wait(`document.querySelector('.score-panel')?.textContent.includes('r-later / plain')`, "late chosen alias row resolves without retargeting");
  assert.equal(await evaluate<string>(`document.getElementById(${JSON.stringify(indexInput)}).value`), "1");
  navigationStage = 3;
  await wait(`[...document.querySelectorAll('summary')].some(s => s.textContent.startsWith('Inspect loaded records') && s.textContent.includes('publication complete'))`, "fixture publication completes");
  await setInput(traceSelect, "r1");
  await evaluate(`document.querySelector('button[aria-label="Inspect decode token 0, ID 123"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('r1 / plain')`, "explicit token selection locates chosen readout again");
  assert.equal(arrayReads, 3); assert.equal(posts.length, beforeArrayPosts);
  const beforeInvalidPin = await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`);
  for (const kind of ["unknown", undefined]) {
    preparedIdentityKind = kind;
    await click("Reconnect / reread stored records");
    await wait(`document.body.textContent.includes(${JSON.stringify(`Identity kind: ${kind ?? "missing"}`)})`, "reloaded saved provenance fixture");
    await wait(`document.querySelector('.score-panel')?.textContent.includes('Pinning unavailable')`, "unrecognized saved model provenance blocks pinning");
    await evaluate(`document.querySelectorAll('.score-panel button').forEach(button => { if (!['Pin direction', 'Pin + operation'].includes(button.textContent)) return; if (!button.disabled) throw new Error('Unvalidated pin enabled'); button.click(); })`);
    assert.equal(await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`), beforeInvalidPin);
  }
  preparedIdentityKind = "runtime_gguf_metadata_not_content_hash";
  await click("Reconnect / reread stored records");
  await wait(`!!document.querySelector('.score-panel') && [...document.querySelectorAll('.score-panel button')].some(button => button.textContent === 'Pin + operation' && !button.disabled)`, "recognized saved provenance enables matching pin");
  await click("Pin + operation");
  const pinned = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).operations.at(-1).document`);
  if (JSON.stringify(pinned.scope) !== JSON.stringify({ layers: { kind: "values", values: [12] }, decode: { kind: "values", values: [0] } })) throw new Error("Pinned scope used wrong token coordinate");
  await evaluate(`document.querySelector('.score-panel').scrollIntoView()`);
  console.log("Phone workflow passed through durable recovery and score pinning.");
  await screenshot(`${output}/phone-score.png`);
  if (await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`)) throw new Error("Phone page overflows horizontally outside its trace scroller");
  assert(await evaluate<boolean>(`document.querySelector('.exploration-measurements').getBoundingClientRect().top >= document.querySelector('.exploration-source').getBoundingClientRect().bottom`), "phone preserves document-order exploration");
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  assert(await evaluate<boolean>(`document.querySelector('.exploration-measurements').getBoundingClientRect().left >= document.querySelector('.exploration-source').getBoundingClientRect().right`), "desktop uses simultaneous source/measurement columns");
  await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 13"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('Layer 13 / decode index 0')`, "desktop preserves direct layer manipulation");
  await evaluate(`document.querySelector('button[aria-label="Inspect saved layer 12"]').click()`);
  await wait(`document.querySelector('.score-panel')?.textContent.includes('Layer 12 / decode index 0')`, "desktop returns to staged source scope");
  assert.equal(posts.length, beforeArrayPosts); assert.equal(arrayReads, 3);
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await click("Review current draft");
  await wait(`document.querySelector('#section-Input').hidden === false`, "input tab");
  await wait(`document.activeElement.id === 'section-Input'`, "review action transfers keyboard focus to current draft");
  assert.equal(posts.length, beforeArrayPosts);
  await evaluate(`document.querySelector('button[aria-label="Move operation 2 up"]').click()`);
  const reordered = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).operations[0].document`);
  if (reordered.id !== pinned.id) throw new Error("Operation reorder was not persisted");
  await click("02 Execution");
  await evaluate(`Object.keys(localStorage).filter(key => key.startsWith('qwen-lens.submission.v1')).forEach(key => localStorage.removeItem(key))`);
  await click("Run again: prepare new seed");
  await wait(`document.body.textContent.includes('Copied job_example with a new seed')`, "request restored from server history without local submission records");
  if (Number(requestReads) !== 1 || Number(posts.length) !== 3) throw new Error("Copying server history submitted inference or failed to retrieve the recorded request");
  const copied = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1'))`);
  if (copied.messages[0].content !== "Browser fixture prompt, edited on a phone." || copied.generation.sampling.seed === requestFixture.generation.sampling.seed) throw new Error("Run again did not copy the request with a new seed");
  await click("02 Execution");
  await click("Variant / not selected");
  await click("01 Input");
  await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`, "new run enabled");
  await click("Run experiment");
  await wait(`document.querySelector('.job-id')?.textContent === 'job_variant_1'`, "variant acceptance");
  if (Number(jobs.size) !== 2 || Number(posts.length) !== 4 || JSON.parse(posts[3]!).idempotency_key === JSON.parse(posts[0]!).idempotency_key) throw new Error("New run did not receive an independent key/job");
  await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Request job cancellation').disabled`, "cancellation enabled");
  await click("Request job cancellation");
  await wait(`document.querySelector('.job-status')?.textContent.includes('cancelled')`, "explicit cancellation status");
  if (cancelCount !== 1) throw new Error("Cancellation did not remain explicit and singular");
  await click("03 History");
  await click("Refresh history now");
  await wait(`document.querySelectorAll('.history-list > li').length === 2`, "real fixture job history");
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await evaluate(`window.scrollTo(0, 0)`);
  await screenshot(`${output}/desktop-history.png`);
  if (await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`)) throw new Error("Desktop layout overflows horizontally");
  await send("Page.reload");
  await wait(`document.querySelector('.job-id')?.textContent === 'job_variant_1'`, "selected job survives reload");
  if (Number(posts.length) !== 4 || cancelCount !== 1) throw new Error("Reload implicitly submitted or cancelled work");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await click("02 Execution");
  await click("Run again: prepare same seed");
  await wait(`document.body.textContent.includes('Copied job_variant_1 with the same seed')`, "bound historical request copied");
  const oldBinding = await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).binding`);
  assetRevision = 1;
  await click("Run experiment");
  await wait(`document.body.textContent.includes('binding_mismatch') && JSON.parse(localStorage.getItem('qwen-lens.submission.v1')).rejected === true`, "stale discovery rejected before acceptance");
  assert.equal(jobs.size, 2);
  const rejectedBody = await evaluate<string>(`JSON.parse(localStorage.getItem('qwen-lens.submission.v1')).body`);
  assert.deepEqual(JSON.parse(rejectedBody).preconditions, oldBinding.preconditions);
  const recovered = await fetch(`${server!.url}v1/lens/jobs`, { method: "POST", body: posts[0]!, headers: { "content-type": "application/json" } });
  assert.equal(recovered.status, 200);
  assert.equal((await recovered.json()).id, "job_example");
  assert.equal(jobs.size, 2);
  await evaluate(`document.querySelector('.deployment-details > summary').click()`);
  await click("Refresh capabilities & assets");
  await wait(`document.body.textContent.includes('Draft asset identity differs')`, "refresh explains mismatch without rebinding");
  assert.deepEqual(await evaluate<any>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).binding`), oldBinding);
  assert(await evaluate<boolean>(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`));
  await evaluate(`document.querySelector('.binding-details > summary').click()`);
  await click("Use current model & assets");
  await wait(`![...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`, "explicit target change prepares new work only");
  assert.equal(await evaluate<string>(`JSON.parse(localStorage.getItem('qwen-lens.submission.v1')).body`), rejectedBody);
  assert.equal(jobs.size, 2);
  let releaseHistory!: () => void;
  heldRequest = new Promise<void>(resolve => { releaseHistory = resolve; });
  const reads = requestReads;
  await click("02 Execution");
  await click("Run again: prepare same seed");
  for (let attempt = 0; requestReads === reads && attempt < 100; attempt++) await Bun.sleep(10);
  assert.equal(requestReads, reads + 1);
  await click("01 Input");
  await setInput("#message-0", "Newer edits must survive delayed history loading.");
  releaseHistory(); heldRequest = null;
  await wait(`document.body.textContent.includes('newer edits were preserved')`, "asynchronous copy preserves newer edits");
  assert.equal(await evaluate<string>(`JSON.parse(localStorage.getItem('qwen-lens.draft.v1')).messages[0].content`), "Newer edits must survive delayed history loading.");
  const savedDraft = await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`);
  const attempts = posts.length;
  await evaluate(`(() => { const original = Storage.prototype.setItem; Storage.prototype.setItem = function(key, value) { if (key === 'qwen-lens.draft.v1') throw new Error('Injected binding persistence failure'); return original.call(this, key, value); }; })()`);
  await click("Use current model & assets");
  await wait(`document.body.textContent.includes('Injected binding persistence failure')`, "failed identity persistence remains explicit");
  assert.equal(await evaluate<string>(`localStorage.getItem('qwen-lens.draft.v1')`), savedDraft);
  assert.equal(posts.length, attempts);
  assert(await evaluate<boolean>(`[...document.querySelectorAll('button')].find(b => b.textContent === 'Run experiment').disabled`));
  console.log("Identity safeguards passed: stale-discovery 412 with no job, accepted-key recovery after asset change, explicit rebinding, unchanged intent bytes, asynchronous edit preservation and persistence failure.");
  historyPageSize = 1;
  const postsBeforeHistory = posts.length;
  const requestReadsBeforeHistory = requestReads;
  await send("Page.reload");
  await wait(`document.querySelector('.job-id')?.textContent === 'job_variant_1'`, "read-only history test reload");
  await click("03 History");
  await wait(`document.querySelectorAll('.history-list > li').length === 1 && !!document.querySelector('.history-list .prompt-excerpt')`, "paged prompt history head");
  await click("Load older jobs");
  await wait(`document.querySelectorAll('.history-list > li').length === 2`, "older prompt history loaded");
  const previewsBefore = await evaluate<string[]>(`[...document.querySelectorAll('.history-list .prompt-excerpt')].map(node => node.textContent)`);
  assert.deepEqual(previewsBefore, ["Browser fixture prompt, edited on a phone.", "Browser fixture prompt, edited on a phone."]);
  const externalPrompt = '<img src=x onerror="throw new Error(1)">\n  ' + "\u{1f680}".repeat(245);
  const externalBody = JSON.stringify({ ...requestFixture, idempotency_key: "fixture-external-client", input: { ...requestFixture.input, messages: [{ role: "user", content: externalPrompt }] } });
  const external: TestJob = { id: "job_external_client", key: "fixture-external-client", body: externalBody, ordinal: 2, cancelled: false, emptyPages: 0 };
  jobs.set(external.key, external);
  await click("Refresh history now");
  await wait(`document.querySelectorAll('.history-list > li').length === 3 && document.querySelector('.history-list')?.textContent.includes('Excerpt: first 240')`, "another client's prompt appears without replacing older pages");
  assert(await evaluate<boolean>(`[...document.querySelectorAll('.history-list .prompt-excerpt')].some(node => node.textContent === ${JSON.stringify([...externalPrompt].slice(0, 240).join(""))})`));
  assert.equal(await evaluate<number>(`document.querySelectorAll('.history-list img').length`), 0);
  assert(!await evaluate<boolean>(`document.documentElement.scrollWidth > innerWidth`), "phone prompt previews overflow");
  const allPreviews = await evaluate<string[]>(`[...document.querySelectorAll('.history-list .prompt-excerpt')].map(node => node.textContent)`);
  historyPreviews = false;
  await click("Refresh history now");
  await wait(`![...document.querySelectorAll('button')].find(button => button.textContent === 'Refresh history now').disabled`, "legacy head refresh finished");
  assert.deepEqual(await evaluate<string[]>(`[...document.querySelectorAll('.history-list .prompt-excerpt')].map(node => node.textContent)`), allPreviews);
  historyPreviews = true; corruptPreview = true;
  await click("Refresh history now");
  await wait(`document.body.textContent.includes('Stored request preview changed')`, "immutable preview mutation reported");
  assert.deepEqual(await evaluate<string[]>(`[...document.querySelectorAll('.history-list .prompt-excerpt')].map(node => node.textContent)`), allPreviews);
  assert.equal(posts.length, postsBeforeHistory);
  assert.equal(requestReads, requestReadsBeforeHistory);
  console.log("Prompt history passed: phone previews, exact Unicode truncation, escaped markup, older-page retention, another client's job, old-server refresh and immutable-preview conflict; no POST or per-row request reads.");
  if (runtimeErrors.length) throw new Error(`Browser runtime errors: ${JSON.stringify(runtimeErrors)}`);
  console.log("Browser checks passed: phone editing, durable recovery, paged tokens, tap scores, source-scope pinning, reorder, independent variant/new seed, explicit cancellation, history, reload, desktop/phone bounds. GPU disabled; fixture-only backend.");
  console.log(Bun.env.CAPTURE_SCREENSHOTS !== "1" ? "Screenshot capture not requested; browser interactions and layout bounds checked." : `Screenshots: ${output}/phone-score.png and ${output}/desktop-history.png`);
  }
  if (runtimeErrors.length) throw new Error(`Browser runtime errors: ${JSON.stringify(runtimeErrors)}`);
} finally {
  stopBrowser();
  try {
    await browser.exited;
    await Bun.write(`${output}/chromium.log`, await browserLog);
    await server?.stop(true);
    assert(!interrupted && !forced, "Interrupted or forced browser cleanup is not a passing qualification");
  } finally {
    clearTimeout(forceStop);
    process.off("SIGTERM", interrupt);
    process.off("SIGINT", interrupt);
  }
}
