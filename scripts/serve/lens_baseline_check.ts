import { strict as assert } from "node:assert";
import { mkdir } from "node:fs/promises";
import { createConnection } from "node:net";
import { resolve } from "node:path";

// Opt-in Metal check. Owns its server handles; never targets an existing service.
const model = Bun.env.QWEN_LENS_TEST_MODEL;
if (!model) throw new Error("Set QWEN_LENS_TEST_MODEL to a qualified House Qwen3.6/3.8 GGUF.");
const root = resolve(import.meta.dir, "../..");
const output = `${root}/target/lens-baseline-${crypto.randomUUID()}`;
await mkdir(output, { recursive: true, mode: 0o700 });
const env = Object.fromEntries(Object.entries(Bun.env).filter(([key]) =>
  !key.startsWith("QWEN_") && !key.startsWith("GGML_")));
const probe = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } });
const port = probe.port;
probe.stop(true);
const origin = `http://127.0.0.1:${port}`;
const deadline = Date.now() + 140_000;
let server: ReturnType<typeof Bun.spawn> | undefined;
let launch = 0;
let forced = false;
let timedOut = false;
let stopping: Promise<{ code: number; signal: string | null }> | undefined;
const evidence: unknown[] = [];
const prefill = { channel: "reasoning", text: "Let me" };

async function stop() {
  if (stopping) return stopping;
  const child = server;
  if (!child) return;
  stopping = (async () => {
    if (child.exitCode === null) child.kill("SIGTERM");
    const timeout = setTimeout(() => {
      if (child.exitCode === null) { forced = true; child.kill("SIGKILL"); }
    }, 35_000);
    try { return { code: await child.exited, signal: child.signalCode }; }
    finally { clearTimeout(timeout); if (server === child) server = undefined; }
  })();
  return stopping;
}
const watchdog = setTimeout(() => { timedOut = true; void stop(); }, 140_000);
function remaining(max = 10_000) {
  assert(!timedOut && Date.now() < deadline, "Live protocol deadline expired");
  return Math.max(1, Math.min(max, deadline - Date.now()));
}
async function json(path: string, body?: string, expected = 200): Promise<any> {
  const response = await fetch(origin + path, {
    ...(body === undefined ? {} : { method: "POST", body, headers: { "content-type": "application/json" } }),
    signal: AbortSignal.timeout(remaining()),
  });
  const value = await response.json();
  remaining();
  assert.equal(response.status, expected, JSON.stringify(value));
  return value;
}
async function wait<T>(read: () => Promise<T>, done: (value: T) => boolean, label: string): Promise<T> {
  const until = Math.min(deadline, Date.now() + 45_000);
  while (Date.now() < until) {
    remaining();
    assert(server && server.exitCode === null, `Server exited during ${label}; see ${output}`);
    const value = await read();
    remaining();
    assert(Date.now() < until, `Deadline waiting for ${label}`);
    if (done(value)) return value;
    await Bun.sleep(25);
  }
  throw new Error(`Deadline waiting for ${label}`);
}
async function start() {
  assert(!server);
  remaining();
  stopping = undefined;
  launch++;
  server = Bun.spawn([`${root}/target/release/qwen`, "serve", "-m", model!,
    "--addr", `127.0.0.1:${port}`, "--max-context-tokens", "256", "--max-tokens", "4",
    "--snapshot-cache-mib", "0", "--durable-snapshot-dir", "off", "--template-style", "house",
    "--lens-data-dir", `${output}/jobs`], {
    cwd: root,
    env: { ...env, QWEN_METAL_LEASE_WAIT: "1", RUST_LOG: "warn,qwen_diag=info" },
    stdout: Bun.file(`${output}/server-${launch}.stdout.log`),
    stderr: Bun.file(`${output}/server-${launch}.stderr.log`),
  });
  const child = server;
  const log = Bun.file(`${output}/server-${launch}.stderr.log`);
  await wait(async () => (await log.text()).includes(`serve: listening on ${origin} `), Boolean, "owned listener");
  await wait(async () => {
    try { return (await fetch(`${origin}/v1/models`, { signal: AbortSignal.timeout(remaining(1000)) })).ok; }
    catch { return false; }
  }, Boolean, "startup");
  assert(child.exitCode === null && server === child, "Owned server exited during readiness");
}
const terminal = (job: any) => ["completed", "cancelled", "failed", "interrupted"].includes(job.state);
const status = (id: string) => json(`/v1/lens/jobs/${id}`);
const result = (id: string) => json(`/v1/lens/jobs/${id}/result`);
function request(identity: string, max = 3) {
  return JSON.stringify({ schema_version: 1, idempotency_key: crypto.randomUUID(),
    input: { kind: "messages", messages: [{ role: "user", content: "Name an animal." }],
      generation_mode: "thinking", assistant_prefill: prefill },
    generation: { max_new_tokens: max, sampling: { temperature: 0, top_k: 0, top_p: 1, min_p: 0, seed: 7 } },
    preconditions: { model_identity: identity, asset_identities: {} } });
}
async function completed(id: string) {
  const job = await wait(() => status(id), terminal, "baseline completion");
  assert.equal(job.state, "completed", JSON.stringify(job));
  assert.equal(job.result.error, null);
  const page = await result(id);
  assert(page.complete && page.next_cursor === null);
  const prepared = page.records[0];
  assert.equal(prepared.kind, "prepared_input");
  assert(prepared.prompt_text.endsWith(prefill.text));
  assert.deepEqual(prepared.prompt_bytes, [...new TextEncoder().encode(prepared.prompt_text)]);
  assert.deepEqual(prepared.assistant_prefill, prefill);
  const samples = page.records.filter((record: any) => record.kind === "sampled_token");
  assert.equal(samples.length, job.generation.sampled_tokens);
  assert.equal(samples.filter((record: any) => record.consumed).length, job.generation.consumed_generated_tokens);
  assert(job.generation.consumed_generated_tokens > 0, "Require a real decode transition, not only a first-sample stop");
  assert.equal(samples.at(-1)?.consumed, false);
  assert.equal(page.records.at(-1)?.kind, "generation_terminal");
  evidence.push({ job, page });
  return page;
}
try {
  await start();
  const caps = await json("/v1/lens/capabilities");
  assert(caps.available && caps.input_kinds.includes("messages"));
  const body = request(caps.model.identity);
  const accepted = await json("/v1/lens/jobs", body, 202);
  const baseline = await completed(accepted.id);
  const lost = request(caps.model.identity);
  await new Promise<void>((resolve, reject) => {
    const socket = createConnection({ host: "127.0.0.1", port }, () => {
      socket.write(`POST /v1/lens/jobs HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Length: ${Buffer.byteLength(lost)}\r\n\r\n${lost}`, error => {
        socket.destroy(); if (error) reject(error); else resolve();
      });
    });
    socket.on("error", reject);
    socket.setTimeout(5000, () => socket.destroy(new Error("lost-ack socket deadline")));
  });
  await wait(() => json("/v1/lens/jobs"), page => page.jobs.length === 2, "disconnected acceptance");
  const recovered = await json("/v1/lens/jobs", lost);
  const second = await completed(recovered.id);
  const samples = (page: any) => page.records.filter((record: any) => record.kind === "sampled_token");
  assert.deepEqual(samples(second), samples(baseline));
  assert.deepEqual(await result(accepted.id), baseline);
  assert.equal((await json("/v1/lens/jobs", body)).id, accepted.id);
  const ordinary = await json("/v1/responses", JSON.stringify({ model: caps.model.id, input: "Say hello.", max_output_tokens: 1, temperature: 0 }));
  assert(["completed", "incomplete"].includes(ordinary.status));
  evidence.push({ ordinary });

  const interruptedBody = request(caps.model.identity, 128);
  const active = await json("/v1/lens/jobs", interruptedBody, 202);
  const running = await wait(() => status(active.id), job => terminal(job) || job.generation.consumed_prompt_tokens > 0, "active native forward");
  assert(!terminal(running), "Fixture completed before observing active work; interruption is not qualified");
  const exit = await stop();
  assert(!forced, "Graceful stop required forced cleanup");
  assert.deepEqual(exit, { code: 143, signal: null }, "Require handled SIGTERM, not abrupt signal death");
  remaining();
  const snapshot = await Bun.file(`${output}/jobs/${active.id}/status.json`).json();
  assert.equal(snapshot.status.state, "interrupted");
  assert(snapshot.status.result.complete);
  assert.equal(snapshot.status.result.error, null);
  const bytes = await Bun.file(`${output}/jobs/${active.id}/records.jsonl`).bytes();
  const records = new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(0, snapshot.committed_bytes))
    .trimEnd().split("\n").map(line => JSON.parse(line));
  assert.equal(records.at(-1)?.kind, "generation_terminal");
  assert.equal(records.at(-1)?.state, "interrupted");
  evidence.push({ settled_before_restart: snapshot.status, terminal_record: records.at(-1) });
  evidence.push({ observed_active: running, exit });
  await start();
  assert.deepEqual(await result(accepted.id), baseline);
  assert.equal((await json("/v1/lens/jobs", body)).id, accepted.id);
  const resumed = await json("/v1/lens/jobs", interruptedBody);
  assert.equal(resumed.id, active.id);
  assert.equal(resumed.state, "interrupted");
  assert.equal(resumed.cancel_requested, false);
  assert(resumed.result.complete);
  assert.deepEqual(resumed, snapshot.status, "Restart must preserve preexisting durable settlement");
  assert.deepEqual(await status(active.id), resumed);
  assert.equal((await json("/v1/lens/jobs")).jobs.length, 3);
  evidence.push({ recovered_interruption: resumed });
  assert.deepEqual(await stop(), { code: 143, signal: null });
  assert(!forced);
  remaining();
  await Bun.write(`${output}/evidence.json`, JSON.stringify({ model, launches: launch, passed: true, evidence }, null, 2));
  remaining();
  console.log(`PASS: baseline, disconnected exact-key recovery, ordinary serving, active interruption/restart; ${output}`);
} finally {
  clearTimeout(watchdog);
  await stop();
}
