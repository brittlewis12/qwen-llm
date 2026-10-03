import { strict as assert } from "node:assert";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

// Owns one opt-in Metal test process; never targets an existing server.
const model = Bun.env.QWEN_LENS_TEST_MODEL;
const mode = Bun.env.QWEN_LENS_ORACLE_MODE ?? "readouts";
assert(["readouts", "interventions"].includes(mode), "Unknown oracle mode");
if (!model) throw new Error("Set QWEN_LENS_TEST_MODEL to a qualified House Qwen3.6/3.8 GGUF.");
const root = resolve(import.meta.dir, "../..");
const output = `${root}/target/lens-fitted-${crypto.randomUUID()}`;
await mkdir(output, { recursive: true, mode: 0o700 });
const env = Object.fromEntries(Object.entries(Bun.env).filter(([key]) => !key.startsWith("QWEN_") && !key.startsWith("GGML_")));
let child: ReturnType<typeof Bun.spawn> | undefined;
let timedOut = false;
let interrupted = false;
let forced = false;
let settlement: Promise<void> | undefined;
let deadline = Number.POSITIVE_INFINITY;
let watchdog: ReturnType<typeof setTimeout> | undefined;
function stop() {
  if (settlement) return settlement;
  const owned = child;
  if (!owned) return Promise.resolve();
  if (owned.exitCode === null) owned.kill("SIGTERM");
  const escalation = setTimeout(() => { if (owned.exitCode === null) { forced = true; owned.kill("SIGKILL"); } }, 35_000);
  settlement = (async () => { try { await owned.exited; } finally { clearTimeout(escalation); } })();
  return settlement;
}
function interrupt() { interrupted = true; void stop(); }
function check() {
  assert(!timedOut && !interrupted && !forced && Date.now() < deadline, "Interrupted/expired oracle cannot pass");
}
process.on("SIGINT", interrupt); process.on("SIGTERM", interrupt);
try {
  const build = child = Bun.spawn(["cargo", "test", "--release", "-p", "qwen-cli", "--bin", "qwen", "--no-run", "--message-format=json"], {
    cwd: root, env, stdout: "pipe", stderr: "inherit", timeout: 180_000,
  });
  const artifacts = await new Response(build.stdout).text();
  assert.equal(await build.exited, 0, "Build oracle without acquiring Metal");
  check();
  const binary = artifacts.trim().split("\n").map(line => JSON.parse(line))
    .find(item => item.reason === "compiler-artifact" && item.target.name === "qwen" && item.profile.test && item.executable)?.executable;
  assert.equal(typeof binary, "string");
  settlement = undefined;
  deadline = Date.now() + 120_000;
  const testName = mode === "readouts" ? "serve::native::fitted_live::fitted_original_forward_cpu_oracle" : "serve::native::intervention_live::ordered_interventions_cpu_oracle";
  const test = child = Bun.spawn([binary, "--exact", testName, "--ignored", "--nocapture"], {
    cwd: root, env: { ...env, QWEN_LENS_TEST_MODEL: model, QWEN_LENS_TEST_OUTPUT: `${output}/oracle`, QWEN_METAL_LEASE_WAIT: "1" },
    stdout: Bun.file(`${output}/stdout.log`), stderr: Bun.file(`${output}/stderr.log`),
  });
  watchdog = setTimeout(() => { timedOut = true; void stop(); }, 120_000);
  assert.equal(await test.exited, 0, `Oracle failed; inspect ${output}`);
  check();
  const witnesses = await Bun.file(`${output}/oracle/witnesses.json`).json();
  check();
  if (mode === "readouts") {
    assert.equal(witnesses.witness_count, 4);
    assert.equal(witnesses.identity_controls, 2);
    assert.equal(witnesses.unchanged_sampling, true);
    assert.equal(witnesses.passing_generation_witnesses, 1);
  } else {
    assert.equal(witnesses.zero_control_equal, true);
    assert.equal(witnesses.operation_only_samples_equal, true);
    assert.equal(witnesses.projection_witness_records, 6);
    assert.equal(witnesses.transformation_sites_checked, 2);
    assert.equal(witnesses.noncommuting_order_distinguished, true);
  }
} finally {
  try { await stop(); }
  finally {
    if (watchdog) clearTimeout(watchdog);
    process.off("SIGINT", interrupt); process.off("SIGTERM", interrupt);
  }
}
check();
console.log(`PASS: ${mode} original-forward numerical oracle; ${output}`);
