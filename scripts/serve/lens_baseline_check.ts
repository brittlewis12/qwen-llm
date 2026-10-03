import { strict as assert } from "node:assert";
import { mkdir } from "node:fs/promises";
import { createConnection } from "node:net";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

// Opt-in Metal check. Owns its server handles; never targets an existing service.
const model = Bun.env.QWEN_LENS_TEST_MODEL;
const retention = Bun.env.QWEN_LENS_TEST_RETENTION === "1";
const wide = Bun.env.QWEN_LENS_TEST_WIDE === "1";
const readouts = Bun.env.QWEN_LENS_TEST_READOUTS === "1" || retention || wide;
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
let observed: { id: string; page: any } | undefined;
let wideObserved: { id: string; page: any } | undefined;
const archived = new Map<string, Uint8Array>();

async function array(record: any) {
  const response = await fetch(origin + record.array.url, { signal: AbortSignal.timeout(remaining()) });
  assert.equal(response.status, 200);
  const bytes = new Uint8Array(await response.arrayBuffer());
  remaining();
  assert.equal(bytes.length, record.array.byte_length);
  assert.equal(bytes.length, record.array.length * 4);
  assert.equal(record.array.dtype, "f32le");
  assert.equal(createHash("sha256").update(bytes).digest("hex"), record.array.sha256);
  return bytes;
}

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
const result = (id: string) => json(`/v1/lens/jobs/${id}/result?limit=256`);
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
  if (readouts) {
    assert(caps.readout_modes.includes("full_vocabulary") && !caps.execution.baseline_only);
    const catalog = await json("/v1/lens/assets");
    const plain = catalog.assets.find((asset: any) => asset.alias === "plain");
    assert(plain?.available && plain.identity === caps.model.identity);
    const prompt = baseline.records[0].token_ids.length;
    assert(prompt > 1 && caps.model.layers > 1);
    const last = caps.model.layers - 1;
    const middle = Math.floor(last / 2);
    const authored = JSON.parse(body);
    authored.idempotency_key = crypto.randomUUID();
    authored.preconditions.asset_identities = { plain: plain.identity };
    const phases = { prefill: { kind: "values", values: [0, prompt - 1] }, decode: { kind: "values", values: [0, 1] } };
    authored.diagnostics = { directions: [], operations: [], readouts: [
      { id: "multi", lens: "plain", mode: "full_vocabulary", top_k: 5, scope: { layers: { kind: "values", values: [middle, last] }, ...phases } },
      { id: "shared", lens: "plain", mode: "full_vocabulary", top_k: 2, scope: { layers: { kind: "values", values: [last] }, ...phases } },
    ] };
    if (retention) {
      assert(caps.readout_retention_modes.includes("scores_and_residual"));
      authored.diagnostics.readouts[0].retain = "scores_and_residual";
    }
    const job = await json("/v1/lens/jobs", JSON.stringify(authored), 202);
    const page = await completed(job.id);
    const stripSequence = (page: any) => samples(page).map(({ seq, ...sample }: any) => sample);
    assert.deepEqual(stripSequence(page), stripSequence(baseline), "Readouts must not change samples or consumption");
    assert.deepEqual(page.records[0].token_ids, baseline.records[0].token_ids);
    const rows = page.records.filter((record: any) => record.kind === "readout");
    assert.equal(rows.length, 12);
    const sites = [0, prompt - 1, prompt, prompt + 1];
    const expected = new Set(sites.flatMap(position => [`multi:${middle}:${position}`, `multi:${last}:${position}`, `shared:${last}:${position}`]));
    let witnesses = 0;
    for (const row of rows) {
      assert(expected.delete(`${row.readout_id}:${row.source_layer}:${row.position}`), "Unexpected or duplicate original-forward row");
      assert.equal(row.provenance, "original_forward");
      assert.equal(row.capture_stage, "post_block_after_operations");
      assert.equal(row.predicts_position, row.position + 1);
      assert.equal(row.phase, row.position < prompt ? "prefill" : "decode");
      assert.equal(row.index, row.position < prompt ? row.position : row.position - prompt);
      assert.equal(row.scores.length, row.readout_id === "multi" ? 5 : 2);
      assert(row.scores.every((score: any) => Number.isFinite(score.score)));
      if (row.source_layer === last && row.position !== 0) {
        const witness = row.generation_logit_witness;
        assert(witness && witness.within_tolerance === true && Number.isFinite(witness.max_abs_error));
        assert.equal(witness.basis, "same_original_forward_generation_logits");
        assert.equal(witness.vocabulary_size, caps.model.vocabulary_size);
        witnesses++;
      } else assert.equal(row.generation_logit_witness, null, "No fabricated witness for a no-tail or middle-layer forward");
      if (row.readout_id === "shared") {
        assert.equal(row.cost.readout_ms, null);
        const source = rows.find((other: any) => other.readout_id === "multi" && other.source_layer === last && other.position === row.position);
        assert.deepEqual(row.scores, source.scores.slice(0, 2));
      } else assert(Number.isFinite(row.cost.readout_ms));
    }
    assert.equal(expected.size, 0);
    assert.equal(witnesses, 6, "Every expected final-layer witness must exist and pass");
    assert.equal(page.records[0].readout_admission.head_evaluations_upper, 8);
    if (retention) {
      const arrays = page.records.filter((r: any) => r.kind === "retained_array");
      assert.equal(arrays.length, 16);
      assert.equal(new Set(arrays.map((r: any) => r.key)).size, 16);
      let total = 0;
      for (const record of arrays) {
        const bytes = await array(record);
        archived.set(record.array.url, bytes);
        total += bytes.length;
        assert.equal(record.array.length, record.quantity === "source_residual" ? caps.model.hidden_size : caps.model.vocabulary_size);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let i = 0; i < record.array.length; i++) assert(Number.isFinite(view.getFloat32(i * 4, true)));
      }
      assert.equal(total, page.records[0].retention_admission.raw_bytes_upper);
      for (const row of rows) {
        if (row.readout_id === "shared") { assert.equal(row.retained, null, "Nonretaining ID must not acquire a retained reference"); continue; }
        const source = arrays.find((a: any) => a.key === row.retained.source_key);
        const logits = arrays.find((a: any) => a.key === row.retained.logits_key);
        assert(source && logits && source.seq < logits.seq && logits.seq < row.seq);
        assert.equal(source.quantity, "source_residual"); assert.equal(logits.quantity, "readout_logits");
        for (const key of ["position", "source_layer", "phase", "index", "input_token_id"]) {
          assert.equal(source[key], row[key]); assert.equal(logits[key], row[key]);
        }
        assert.equal(logits.source_key, source.key); assert.equal(logits.lens, row.lens);
        const bytes = archived.get(logits.array.url)!;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const ranks = row.scores.map(() => 1);
        for (let token = 0; token < logits.array.length; token++) {
          const value = view.getFloat32(token * 4, true);
          row.scores.forEach((score: any, index: number) => {
            const expected = view.getFloat32(score.token_id * 4, true);
            if (value > expected || (value === expected && token < score.token_id)) ranks[index]++;
          });
        }
        row.scores.forEach((score: any, index: number) => {
          assert.equal(view.getFloat32(score.token_id * 4, true), Math.fround(score.score));
          assert.equal(ranks[index], index + 1);
        });
      }
      evidence.push({ retention_gate: { arrays: arrays.length, raw_bytes: total, verified_sha256: true, full_vocabulary_rank_checked: true } });
    }
    assert.deepEqual(await result(job.id), page);
    observed = { id: job.id, page };
    evidence.push({ readout_gate: { expected_rows: 12, shared_heads: 8, passing_witnesses: witnesses, unchanged_sampling: true } });
    if (wide) {
      const layers = [...Array.from({ length: Math.min(39, last) }, (_, i) => i), last];
      assert(layers.length * sites.length > 128, "Wide fixture must exceed the event queue capacity");
      const wider = JSON.parse(body);
      wider.idempotency_key = crypto.randomUUID();
      wider.preconditions.asset_identities = { plain: plain.identity };
      wider.diagnostics = { directions: [], operations: [], readouts: [
        { id: "wide", lens: "plain", mode: "full_vocabulary", top_k: 2, scope: { layers: { kind: "values", values: layers }, ...phases } },
      ] };
      const accepted = await json("/v1/lens/jobs", JSON.stringify(wider), 202);
      const page = await completed(accepted.id);
      assert.deepEqual(stripSequence(page), stripSequence(baseline));
      const expected = new Set(sites.flatMap(position => layers.map(layer => `${position}:${layer}`)));
      let witnesses = 0;
      for (const row of page.records.filter((r: any) => r.kind === "readout")) {
        assert.equal(row.readout_id, "wide");
        assert(expected.delete(`${row.position}:${row.source_layer}`), "Duplicate or unexpected wide site");
        assert.equal(row.phase, row.position < prompt ? "prefill" : "decode");
        assert.equal(row.index, row.position < prompt ? row.position : row.position - prompt);
        assert.equal(row.scores.length, 2);
        assert(row.scores.every((s: any) => Number.isFinite(s.score)));
        if (row.source_layer === last && row.position !== 0) {
          assert.equal(row.generation_logit_witness?.within_tolerance, true); witnesses++;
        }
      }
      assert.equal(expected.size, 0); assert.equal(witnesses, 3);
      const writer = page.records.at(-1).artifact_writer;
      assert(writer.peak_record_bytes <= 8 * 1024 * 1024);
      wideObserved = { id: accepted.id, page };
      evidence.push({ wide_gate: { rows: layers.length * sites.length, unchanged_sampling: true, passing_witnesses: witnesses, writer } });
    }
  }
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
  if (observed) assert.deepEqual(await result(observed.id), observed.page);
  if (wideObserved) assert.deepEqual(await result(wideObserved.id), wideObserved.page);
  if (retention) {
    for (const record of observed!.page.records.filter((r: any) => r.kind === "retained_array")) {
      assert.deepEqual(await array(record), archived.get(record.array.url), "Retained bytes changed across restart");
    }
  }
  assert.equal((await json("/v1/lens/jobs", body)).id, accepted.id);
  const resumed = await json("/v1/lens/jobs", interruptedBody);
  assert.equal(resumed.id, active.id);
  assert.equal(resumed.state, "interrupted");
  assert.equal(resumed.cancel_requested, false);
  assert(resumed.result.complete);
  assert.deepEqual(resumed, snapshot.status, "Restart must preserve preexisting durable settlement");
  assert.deepEqual(await status(active.id), resumed);
  assert.equal((await json("/v1/lens/jobs")).jobs.length, (readouts ? 4 : 3) + Number(wide));
  evidence.push({ recovered_interruption: resumed });
  assert.deepEqual(await stop(), { code: 143, signal: null });
  assert(!forced);
  remaining();
  await Bun.write(`${output}/evidence.json`, JSON.stringify({ model, readouts, retention, wide, launches: launch, passed: true, evidence }, null, 2));
  remaining();
  console.log(`PASS: baseline${readouts ? ", shared original-forward readouts with unchanged samples and six passing witnesses" : ""}${retention ? ", retained bytes/ranks verified across restart" : ""}${wide ? ", wide-scope records and restart verified" : ""}, disconnected exact-key recovery, ordinary serving, active interruption/restart; ${output}`);
} finally {
  clearTimeout(watchdog);
  await stop();
}
