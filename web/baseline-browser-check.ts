import { strict as assert } from "node:assert";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";

// Production HTTP/store/writer with synthetic CPU forwards, never Metal.
const root = resolve(import.meta.dir, "..");
const compile = Bun.spawn(["cargo", "test", "-p", "qwen-cli", "--bin", "qwen", "--no-run", "--message-format=json"],
  { cwd: root, stdout: "pipe", stderr: "inherit", timeout: 120_000 });
const artifacts = await new Response(compile.stdout).text();
assert.equal(await compile.exited, 0);
const binary = artifacts.trim().split("\n").map(line => JSON.parse(line))
  .find(item => item.reason === "compiler-artifact" && item.target.name === "qwen" && item.executable)?.executable;
assert.equal(typeof binary, "string", "Cargo must identify the exact CPU test executable");
const output = `${import.meta.dir}/.browser-test/baseline-${crypto.randomUUID()}`;
await mkdir(output, { recursive: true, mode: 0o700 });
const env = Object.fromEntries(Object.entries(Bun.env).filter(([key]) => !key.startsWith("QWEN_") && !key.startsWith("GGML_")));
const stdout = Bun.file(`${output}/server.stdout.log`);
const server = Bun.spawn([binary, "--exact", "serve::control::tests::browser_baseline_child", "--ignored", "--nocapture"], {
  cwd: root, env: { ...env, QWEN_LENS_BROWSER_CHILD: "1", QWEN_LENS_BROWSER_WEB_ROOT: `${import.meta.dir}/dist`, QWEN_LENS_BROWSER_READOUTS: Bun.env.LENS_TEST_PLAIN_ONLY === "1" || Bun.env.LENS_TEST_RETENTION === "1" || Bun.env.LENS_TEST_PAIRS === "1" ? "1" : "0", QWEN_LENS_BROWSER_FITTED: Bun.env.LENS_TEST_FITTED_ONLY === "1" || Bun.env.LENS_TEST_OPERATIONS === "1" ? "1" : "0" },
  stdout, stderr: Bun.file(`${output}/server.stderr.log`),
});
let browser: ReturnType<typeof Bun.spawn> | undefined;
let timedOut = false;
let interrupted = false;
let forced = false;
let settlement: Promise<void> | undefined;
function stop() {
  if (settlement) return settlement;
  if (browser?.exitCode === null) browser.kill("SIGTERM");
  if (server.exitCode === null) server.kill("SIGTERM");
  // Chromium's owner has its own five-second escalation and must reap it first.
  const browserTimer = setTimeout(() => { if (browser?.exitCode === null) { forced = true; browser.kill("SIGKILL"); } }, 8000);
  const serverTimer = setTimeout(() => { if (server.exitCode === null) { forced = true; server.kill("SIGKILL"); } }, 5000);
  settlement = (async () => {
    try { await Promise.all([browser?.exited, server.exited]); }
    finally { clearTimeout(browserTimer); clearTimeout(serverTimer); }
  })();
  return settlement;
}
function interrupt() { interrupted = true; void stop(); }
process.on("SIGTERM", interrupt);
process.on("SIGINT", interrupt);
const watchdog = setTimeout(() => {
  timedOut = true;
  void stop();
}, 90_000);
try {
  let origin: string | undefined;
  for (let attempt = 0; attempt < 100 && !origin; attempt++) {
    assert(!interrupted && !timedOut, "CPU browser check interrupted");
    assert.equal(server.exitCode, null, `CPU server stopped; see ${output}`);
    origin = (await stdout.text()).match(/qwen-lens-browser-ready:(http:\/\/127\.0\.0\.1:\d+)/)?.[1];
    if (!origin) await Bun.sleep(50);
  }
  assert(origin, "Owned CPU server did not announce its listener");
  assert(!interrupted && !timedOut);
  browser = Bun.spawn(["bun", "run", "browser-check.ts"], { cwd: import.meta.dir,
    env: { ...env, LENS_BASELINE_ORIGIN: origin, LENS_TEST_BASELINE_ONLY: "1" }, stdout: "inherit", stderr: "inherit" });
  assert.equal(await browser.exited, 0);
  assert(!timedOut && !interrupted, "CPU browser protocol interrupted or deadline expired");
} finally {
  clearTimeout(watchdog);
  try {
    await stop();
    assert.equal(server.exitCode, 0, "CPU child must settle through the Rust test harness");
    assert(!forced && !interrupted && !timedOut, "Cleanup escalation or interruption is not a passing qualification");
    assert((await stdout.text()).includes("qwen-lens-browser-stopped"));
  } finally { process.off("SIGTERM", interrupt); process.off("SIGINT", interrupt); }
}
console.log(`PASS: CPU-owned production control loop and baseline browser; ${output}`);
