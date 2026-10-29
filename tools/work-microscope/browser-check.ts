import { strict as assert } from "node:assert";
import { constants } from "node:fs";
import { access, mkdir, readdir } from "node:fs/promises";
import { resolve } from "node:path";
import { neighborhood, tensorPath } from "./model";
import type { EvidenceBundle, FlowVariant, Investigation } from "./model";
import { MAX_PACKET_BYTES, validateBundle } from "./validate";

const chromium = "/Applications/Chromium.app/Contents/MacOS/Chromium";
const runDir = resolve(import.meta.dir, ".browser-test", crypto.randomUUID());
const profile = resolve(runDir, "profile");
const downloads = resolve(runDir, "downloads");
const abort = new AbortController();
const children: Bun.Subprocess[] = [];
const pending = new Map<number, {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}>();
let socket: WebSocket | undefined;
let nextId = 0;
let serverOutput = "";
let serverErrors = "";
let browserErrors = "";
let captureFrame: ((data: string) => void) | undefined;
const findings: string[] = [];
const warnings: string[] = [];

function stop(reason: Error) {
  if (!abort.signal.aborted) abort.abort(reason);
  for (const request of pending.values()) {
    clearTimeout(request.timer);
    request.reject(reason);
  }
  pending.clear();
}
const interrupt = (signal: string) => stop(new Error(`Browser check interrupted: ${signal}`));
process.on("SIGINT", interrupt);
process.on("SIGTERM", interrupt);
const deadline = setTimeout(() => stop(new Error("Browser check exceeded its 90-second deadline")), 90_000);

function checkRunning() {
  abort.signal.throwIfAborted();
  for (const child of children) assert(child.exitCode === null, `Owned child ${child.pid} exited unexpectedly (${child.exitCode})`);
}

async function poll<T>(read: () => Promise<T>, label: string, timeout = 6000): Promise<NonNullable<T>> {
  const until = Date.now() + timeout;
  do {
    checkRunning();
    const value = await read();
    if (value) return value as NonNullable<T>;
    await Bun.sleep(60);
  } while (Date.now() < until);
  throw new Error(`Timed out: ${label}`);
}

async function consume(stream: ReadableStream<Uint8Array>, append: (text: string) => void) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      append(decoder.decode(value, { stream: true }));
    }
    append(decoder.decode());
  } finally {
    reader.releaseLock();
  }
}

function send(method: string, params: Record<string, unknown> = {}, timeout = 5000): Promise<any> {
  checkRunning();
  if (socket?.readyState !== WebSocket.OPEN) return Promise.reject(new Error("CDP socket is not open"));
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`CDP timeout: ${method}`));
    }, timeout);
    pending.set(id, { resolve, reject, timer });
    try { socket!.send(JSON.stringify({ id, method, params })); }
    catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
  });
}

async function evaluate<T = unknown>(expression: string): Promise<T> {
  const response = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (response.exceptionDetails) throw new Error(`Browser evaluation failed: ${JSON.stringify(response.exceptionDetails)}`);
  return response.result.value as T;
}

const wait = (expression: string, label: string) => poll(() => evaluate<boolean>(`!!document.body && (${expression})`), label);
const click = (selector: string) => evaluate(`(() => {
  const button = [...document.querySelectorAll(${JSON.stringify(selector)})].find(b => b.getClientRects().length && !b.disabled);
  if (!button) throw new Error('Visible control missing: ' + ${JSON.stringify(selector)});
  button.click();
})()`);
const active = (selector: string) => `(() => {
  const b = document.querySelector(${JSON.stringify(selector)});
  return !!b && (b.getAttribute('aria-pressed') === 'true' || ['true', 'page'].includes(b.getAttribute('aria-current')) || b.classList.contains('active') || b.classList.contains('selected'));
})()`;

async function scrub(flow: FlowVariant, label: string) {
  const segments = flow.allocations.flatMap(allocation => allocation.segments);
  const states = new Set<string>();
  for (const [step, description] of flow.steps.entries()) {
    await evaluate(`(() => {
      const r = document.querySelector('#step-range');
      if (!r || Number(r.max) !== ${flow.steps.length - 1}) throw new Error('Source-order range mismatch');
      r.value = ${JSON.stringify(String(step))}; r.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await wait(`document.querySelector('#step-description')?.textContent === ${JSON.stringify(`${description.label} / ${description.detail}`)}`, `${label}: source-order description at ${step}`);
    assert.equal(await evaluate<string>(`document.querySelector('#step-counter')?.textContent`), `${step + 1} / ${flow.steps.length} boundaries; not clock time`, `${label}: displayed boundary counter at ${step}`);
    assert.equal(await evaluate<string>(`document.querySelector('#step-range').getAttribute('aria-valuetext')`), `${step + 1}: ${description.label}; source order, not time`, `${label}: accessible boundary value at ${step}`);
    const actual = await evaluate<{ rows: number; labels: string[]; active: boolean[]; unavailable: string }>(`(() => {
      const root = document.querySelector('[data-lifetimes].lifetime-panel');
      if (!root) throw new Error('Allocation/lifetime panel missing');
      return { rows: root.querySelectorAll('.allocation').length,
        labels: [...root.querySelectorAll('.allocation-segment')].map(s => s.textContent),
        active: [...root.querySelectorAll('.allocation-segment')].map(s => s.classList.contains('at-cursor')),
        unavailable: [...root.querySelectorAll('.unavailable')].filter(n => n.getClientRects().length).map(n => n.textContent).join(' ') };
    })()`);
    assert.equal(actual.rows, flow.allocations.length, `${label}: allocation row count`);
    assert.deepEqual(actual.labels, segments.map(s => s.label), `${label}: allocation windows must match packet`);
    assert.deepEqual(actual.active, segments.map(s => step >= s.start && step < s.end), `${label}: incorrect allocation highlighting at ${step}`);
    if (segments.length === 0) assert(/no backing-allocation windows are supplied/i.test(actual.unavailable), `${label}: missing windows must be explicitly unavailable, not invented`);
    else assert.equal(actual.unavailable, "", `${label}: known windows must not be reported unavailable`);
    states.add(JSON.stringify(actual.active));
  }
  if (segments.some(s => s.start > 0 || s.end < flow.steps.length)) assert(states.size > 1, `${label}: allocation highlights never change across boundaries`);
}

async function nativeRangeDrag(flow: FlowVariant, label: string) {
  await send("Emulation.setFocusEmulationEnabled", { enabled: true });
  await send("Page.bringToFront");
  const range = await evaluate<{ left: number; width: number; y: number; value: number }>(`(() => {
    const r = document.querySelector('#step-range');
    r.scrollIntoView({ block: 'center' });
    window.__microscopeRange = r;
    window.__microscopeRangeInputs = [];
    r.addEventListener('input', e => window.__microscopeRangeInputs.push({ value: Number(r.value), trusted: e.isTrusted }));
    const rect = r.getBoundingClientRect();
    return { left: rect.left, width: rect.width, y: rect.top + rect.height / 2, value: Number(r.value) };
  })()`);
  const max = flow.steps.length - 1;
  assert(max >= 2, `${label}: native drag needs multiple boundaries`);
  const xAt = (step: number) => range.left + 8 + (range.width - 16) * step / max;
  let x = xAt(range.value);
  const moves: Promise<unknown>[] = [];
  const mouse = async (type: string, params: Record<string, unknown> = {}) => {
    try { await send("Input.dispatchMouseEvent", { type, x, y: range.y, ...params }); }
    catch (error) {
      const state = await evaluate(`({ value: document.querySelector('#step-range')?.value, inputs: window.__microscopeRangeInputs })`);
      throw new Error(`${label}: native ${type} at (${x}, ${range.y}) failed; state ${JSON.stringify(state)}`, { cause: error });
    }
  };
  await mouse("mousePressed", { button: "left", buttons: 1, clickCount: 1 });
  try {
    for (const step of [...flow.steps.keys(), 0]) {
      x = xAt(step);
      // Some CPU headless builds withhold move ACKs until a compositor frame.
      // Verify each trusted input effect before advancing, without waiting on that ACK.
      moves.push(send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y: range.y, button: "left", buttons: 1 }, 2000)
        .then(() => null, error => error));
      await wait(`Number(document.querySelector('#step-range')?.value) === ${step}`, `${label}: native drag reaches boundary ${step}`);
      assert(await evaluate<boolean>(`window.__microscopeRange.isConnected && window.__microscopeRange === document.querySelector('#step-range')`), `${label}: native drag remounted the range element`);
      const description = flow.steps[step]!;
      assert.equal(await evaluate<string>(`document.querySelector('#step-description').textContent`), `${description.label} / ${description.detail}`, `${label}: native drag must update the source-order description`);
      assert.equal(await evaluate<string>(`document.querySelector('#step-counter')?.textContent`), `${step + 1} / ${flow.steps.length} boundaries; not clock time`, `${label}: native drag must update the displayed boundary counter`);
      assert.equal(await evaluate<string>(`document.querySelector('#step-range').getAttribute('aria-valuetext')`), `${step + 1}: ${description.label}; source order, not time`, `${label}: native drag accessible value`);
    }
  } finally {
    await mouse("mouseReleased", { button: "left", buttons: 0, clickCount: 1 });
  }
  const inputs = await evaluate<{ value: number; trusted: boolean }[]>(`window.__microscopeRangeInputs`);
  assert(inputs.length >= max, `${label}: drag must produce multiple native input events`);
  assert(inputs.every(input => input.trusted), `${label}: range drag must use trusted browser input, not dispatched DOM events`);
  assert(new Set(inputs.map(input => input.value)).size >= 3, `${label}: drag must traverse at least three distinct range values`);
  const replies = await Promise.all(moves);
  for (const error of replies) if (error) {
    assert(error instanceof Error && error.message === "CDP timeout: Input.dispatchMouseEvent", `${label}: native move failed: ${String(error)}`);
    if (!warnings.some(warning => warning.startsWith("Native range moves"))) warnings.push("Native range moves produced all required trusted inputs and values, but this CPU Chromium withheld some CDP move acknowledgements.");
  }
  await evaluate(`delete window.__microscopeRange; delete window.__microscopeRangeInputs`);
}

async function allocationKeyboard(flow: FlowVariant, label: string) {
  for (const allocation of flow.allocations) for (const segment of allocation.segments) {
    const focusKey = `allocation-${flow.id}-${allocation.id}-${segment.start}`;
    await evaluate(`(() => {
      const b = document.querySelector('[data-focus-key="' + CSS.escape(${JSON.stringify(focusKey)}) + '"]');
      if (!b || !b.getClientRects().length) throw new Error('Keyboard allocation control missing');
      b.focus();
    })()`);
    assert.equal(await evaluate<string>(`document.activeElement?.getAttribute('data-focus-key')`), focusKey, `${label}: allocation receives focus`);
    await key("Enter", "Enter", 13);
    const selected = flow.nodes.find(node => node.tensor === segment.tensor)!;
    await wait(`document.querySelector('[data-inspector] h2')?.textContent === ${JSON.stringify(selected.label)}`, `${label}: keyboard allocation selects its tensor`);
    assert.equal(await evaluate<string>(`document.activeElement?.getAttribute('data-focus-key')`), focusKey, `${label}: allocation activation must preserve its focus key after rendering`);
    assert(await evaluate<boolean>(`document.activeElement?.isConnected && document.activeElement.classList.contains('allocation-segment')`), `${label}: keyboard focus must remain on a mounted allocation button`);
  }
}

async function accessEvidence(investigation: Investigation, flow: FlowVariant, label: string) {
  const distinguishing = flow.nodes.find(node => flow.edges.some(edge => (edge.from === node.id || edge.to === node.id) && edge.evidenceIds.some(id => !node.evidenceIds.includes(id))));
  const node = distinguishing ?? flow.nodes.at(-1)!;
  await click(`[data-node-id="${node.id}"]`);
  const edges = flow.edges.filter(edge => edge.from === node.id || edge.to === node.id);
  assert(edges.length > 0, `${label}: edge evidence check needs an access neighborhood`);
  for (const edge of edges) {
    const selector = `[data-inspector] [data-edge-evidence="${edge.id}"]`;
    await click(`${selector} > summary`);
    await wait(`document.querySelector(${JSON.stringify(selector)})?.open`, `${label}: access basis opens for ${edge.id}`);
    const rows = await evaluate<{ label: string; kind: string; detail: string; sources: string[]; visible: boolean }[]>(`(() => {
      const details = document.querySelector(${JSON.stringify(selector)});
      return [...details.querySelectorAll('.evidence-row')].map(row => ({
        label: row.querySelector('.evidence-heading strong').textContent,
        kind: row.querySelector('.badge').textContent, detail: row.querySelector('p').textContent,
        sources: [...row.querySelectorAll('[data-source-id]')].map(b => b.dataset.sourceId),
        visible: !!row.getClientRects().length,
      }));
    })()`);
    assert.deepEqual(rows, edge.evidenceIds.map(id => {
      const evidence = investigation.evidence.find(evidence => evidence.id === id)!;
      return { label: evidence.label, kind: evidence.kind, detail: evidence.detail, sources: evidence.sourceIds, visible: true };
    }), `${label}: access basis must expose the edge's own evidence and source references, not substitute node evidence`);
  }
  if (investigation.id === "query-half") assert(distinguishing, `${label}: Q access test must cover evidence absent from the node's own basis`);
}

async function checkFollow(flow: FlowVariant, selectedId: string, label: string) {
  const node = flow.nodes.find(n => n.id === selectedId)!;
  const focused = new Set([...tensorPath(flow, node.tensor), ...neighborhood(flow, node.id)]);
  for (const enabled of [false, true]) {
    await evaluate(`(() => { const c = document.querySelector('[data-follow]'); if (!c) throw new Error('Follow control missing'); if (c.checked !== ${enabled}) c.click(); })()`);
    assert.equal(await evaluate<boolean>(`document.querySelector('[data-follow]').checked`), enabled, `${label}: follow control state`);
    const actual = await evaluate<string[]>(`[...document.querySelector('.active-flow').querySelectorAll('[data-node-id].dimmed')].map(n => n.dataset.nodeId).sort()`);
    assert.deepEqual(actual, enabled ? flow.nodes.filter(n => !focused.has(n.id)).map(n => n.id).sort() : [], `${label}: follow must highlight tensor identity and immediate access neighborhood`);
  }
}

async function key(key: string, code: string, virtualKey: number) {
  await send("Emulation.setFocusEmulationEnabled", { enabled: true });
  await send("Page.bringToFront");
  await send("Input.dispatchKeyEvent", { type: "keyDown", key, code, windowsVirtualKeyCode: virtualKey,
    ...(key === "Enter" ? { text: "\r", unmodifiedText: "\r" } : {}) });
  await send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode: virtualKey });
}

async function layout(label: string) {
  const result = await evaluate<{ overflow: boolean; small: string[]; graphScroll: boolean }>(`(() => {
    const visible = b => b.getClientRects().length && getComputedStyle(b).visibility !== 'hidden';
    const small = [...document.querySelectorAll('button')].filter(b => visible(b) && !b.closest('svg')).filter(b => {
      const r = b.getBoundingClientRect(); return r.width < 39.5 || r.height < 39.5;
    }).map(b => b.textContent.trim().slice(0, 90));
    let graphScroll = false;
    for (let node = document.querySelector('[data-node-id]'); node; node = node.parentElement) {
      if (['auto', 'scroll'].includes(getComputedStyle(node).overflowX)) graphScroll = true;
    }
    return { overflow: document.documentElement.scrollWidth > innerWidth, small, graphScroll };
  })()`);
  assert(!result.overflow, `${label}: document overflows horizontally`);
  assert.deepEqual(result.small, [], `${label}: HTML button targets must be at least 40 x 40 CSS pixels`);
  assert(result.graphScroll, `${label}: graph needs its own horizontal scroll container`);
}

async function sourceDialog(bundle: EvidenceBundle, label: string) {
  const selector = '[data-inspector] [data-source-id]';
  const id = await evaluate<string>(`[...document.querySelectorAll(${JSON.stringify(selector)})].find(b => b.getClientRects().length)?.dataset.sourceId`);
  assert(id, `${label}: inspector must expose a visible source button`);
  const snapshot = bundle.snapshots.find(source => source.id === id);
  assert(snapshot, `${label}: source button must have a bundled snapshot`);
  await click(`${selector}[data-source-id="${id}"]`);
  await wait(`document.querySelector('#source-dialog')?.open`, `${label}: source dialog opens`);
  const dialog = await evaluate<{ text: string; excerpt: string; fits: boolean; focus: boolean }>(`(() => {
    const d = document.querySelector('#source-dialog'), r = d.getBoundingClientRect();
    return { text: d.textContent, excerpt: [...d.querySelectorAll('.source-line code')].map(n => n.textContent).join('\\n'),
      fits: r.left >= 0 && r.top >= 0 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1 && d.scrollWidth <= d.clientWidth + 1,
      focus: d.contains(document.activeElement) };
  })()`);
  assert(dialog.text.includes(snapshot.sha256), `${label}: full SHA-256 missing`);
  assert(dialog.text.includes(snapshot.revision), `${label}: full source revision missing`);
  assert(dialog.text.includes(`${snapshot.path}:${snapshot.startLine}`), `${label}: pinned source path/range missing`);
  assert.equal(dialog.excerpt, snapshot.excerpt, `${label}: source excerpt differs from packet`);
  assert(dialog.fits, `${label}: source dialog clips the viewport`);
  assert(dialog.focus, `${label}: source dialog must receive keyboard focus`);
  await key("Escape", "Escape", 27);
  await wait(`!document.querySelector('#source-dialog')?.open`, `${label}: Escape closes source dialog`);
  await click(`${selector}[data-source-id="${id}"]`);
  await click('[data-close-source]');
  await wait(`!document.querySelector('#source-dialog')?.open`, `${label}: close button closes source dialog`);
}

async function importFile(path: string) {
  const { root } = await send("DOM.getDocument");
  const { nodeId } = await send("DOM.querySelector", { nodeId: root.nodeId, selector: '[data-import]' });
  assert(nodeId, "Import file input missing");
  await send("DOM.setFileInputFiles", { nodeId, files: [path] });
}

async function screenshot(name: string) {
  if (Bun.env.CAPTURE_SCREENSHOTS !== "1") return;
  const until = Date.now() + 5000;
  const remaining = () => Math.max(1, until - Date.now());
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await send("Emulation.setFocusEmulationEnabled", { enabled: true }, remaining());
    await send("Page.bringToFront", {}, remaining());
    let data: string;
    try {
      data = (await send("Page.captureScreenshot", { format: "png", fromSurface: false, captureBeyondViewport: false }, Math.min(2200, remaining()))).data;
    } catch {
      checkRunning();
      const frame = new Promise<string>((resolve, reject) => {
        captureFrame = resolve;
        timer = setTimeout(() => reject(new Error("CPU screenshot and screencast both failed to produce a frame")), Math.max(1, remaining() - 250));
      });
      [, data] = await Promise.all([
        send("Page.startScreencast", { format: "png", everyNthFrame: 1 }, Math.max(1, remaining() - 250)), frame,
      ]);
    }
    await Bun.write(resolve(runDir, `${name}.png`), Buffer.from(data, "base64"));
  } catch (error) {
    checkRunning();
    warnings.push(`${name}: capture stalled or failed; DOM checks remain valid (${String(error)})`);
  } finally {
    clearTimeout(timer);
    captureFrame = undefined;
    await send("Page.stopScreencast", {}, Math.min(250, remaining())).catch(error => {
      if (!abort.signal.aborted) warnings.push(`${name}: stop screencast: ${String(error)}`);
    });
  }
}

async function run() {
  try { await access(chromium, constants.X_OK); }
  catch { throw new Error(`BLOCKER: existing Chromium is unavailable at ${chromium}; no browser will be installed.`); }
  assert(await Bun.file(resolve(import.meta.dir, "dist/index.html")).exists(), "BLOCKER: external build is not ready; run the build separately before this check.");
  await mkdir(downloads, { recursive: true });
  const server = Bun.spawn(["bun", "run", "serve.ts"], {
    cwd: import.meta.dir, env: { ...process.env, PORT: "0" }, stdout: "pipe", stderr: "pipe",
  });
  children.push(server);
  void consume(server.stdout, text => { serverOutput = (serverOutput + text).slice(-32_768); }).catch(error => stop(new Error(`Preview stdout failed: ${error}`)));
  void consume(server.stderr, text => { serverErrors = (serverErrors + text).slice(-32_768); }).catch(error => stop(new Error(`Preview stderr failed: ${error}`)));
  const origin = await poll(async () => {
    const match = /Work Microscope:\s+(http:\/\/[^\s]+)/.exec(serverOutput);
    if (!match) return null;
    const url = new URL(match[1]!);
    assert(url.hostname === "127.0.0.1" && url.port && url.pathname === "/" && !url.username && !url.password, "Preview must announce an owned loopback URL");
    return url.origin;
  }, "owned preview URL", 10_000);
  const response = await fetch(`${origin}/evidence.json`, { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(5000)]) });
  assert(response.ok, "Bundled evidence request failed");
  const bundle = validateBundle(await response.json());
  const ids = ["query-half", "scratch-reuse", "bridge-copy"];
  assert.deepEqual(bundle.investigations.map(c => c.id).sort(), [...ids].sort());
  assert(bundle.snapshots.length > 0, "Built bundle must contain source snapshots");
  for (const source of bundle.investigations.flatMap(c => c.sources)) assert(bundle.snapshots.some(s => s.id === source.id), `Snapshot missing: ${source.id}`);

  const browser = Bun.spawn([chromium, "--headless=new", "--disable-gpu",
    "--window-size=1440,1000", "--allow-pre-commit-input", "--disable-background-timer-throttling",
    "--disable-features=PaintHolding,CalculateNativeWinOcclusion", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--no-first-run", "--no-default-browser-check",
    "--disable-background-networking", "--disable-sync", "--disable-extensions", "--disable-component-update",
    "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
  { stdout: "ignore", stderr: "pipe" });
  children.push(browser);
  void consume(browser.stderr, text => { browserErrors = (browserErrors + text).slice(-32_768); }).catch(error => stop(new Error(`Chromium stderr failed: ${error}`)));
  const port = await poll(async () => {
    const file = Bun.file(resolve(profile, "DevToolsActivePort"));
    if (!await file.exists()) return null;
    const contents = await file.text();
    if (!contents.includes("\n")) return null;
    const value = contents.split("\n")[0]!;
    assert(/^\d+$/.test(value) && Number(value) > 0 && Number(value) <= 65535, "Invalid owned Chromium debug port");
    return value;
  }, "isolated Chromium debugging endpoint", 12_000);
  const targetsResponse = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(5000)]) });
  assert(targetsResponse.ok, "Chromium target discovery failed");
  const targets = await targetsResponse.json() as { type: string; webSocketDebuggerUrl: string }[];
  const target = targets.find(t => t.type === "page");
  assert(target, "No Chromium page target");
  const endpoint = new URL(target.webSocketDebuggerUrl);
  assert(endpoint.protocol === "ws:" && endpoint.hostname === "127.0.0.1" && endpoint.port === port, "Unexpected CDP endpoint");
  socket = new WebSocket(endpoint.href);
  await new Promise<void>((resolve, reject) => {
    const failed = () => { clearTimeout(timer); abort.signal.removeEventListener("abort", failed); reject(new Error("CDP connection failed or interrupted")); };
    const timer = setTimeout(failed, 5000);
    abort.signal.addEventListener("abort", failed, { once: true });
    socket!.onopen = () => { clearTimeout(timer); abort.signal.removeEventListener("abort", failed); resolve(); };
    socket!.onerror = failed;
    socket!.onclose = failed;
    if (abort.signal.aborted) failed();
  });
  const errors: unknown[] = [];
  const forbiddenRequests: string[] = [];
  const failedResponses: string[] = [];
  let evidenceLoaded = false;
  socket.onclose = () => stop(new Error("CDP connection closed unexpectedly"));
  socket.onerror = () => stop(new Error("CDP connection failed"));
  socket.onmessage = event => {
    const message = JSON.parse(String(event.data));
    if (message.id) {
      const request = pending.get(message.id);
      if (!request) return;
      clearTimeout(request.timer); pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error))); else request.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") errors.push(message.params);
    else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") errors.push(message.params);
    else if (message.method === "Log.entryAdded" && message.params.entry.level === "error") errors.push(message.params.entry);
    else if (message.method === "Network.requestWillBeSent") {
      const url = message.params.request.url as string;
      if (!url.startsWith("blob:") && !url.startsWith("data:") && new URL(url).origin !== origin) forbiddenRequests.push(url);
    } else if (message.method === "Network.responseReceived") {
      const r = message.params.response;
      if (r.url === `${origin}/evidence.json` && r.status === 200) evidenceLoaded = true;
      if (r.status >= 400) failedResponses.push(`${r.status} ${r.url}`);
    } else if (message.method === "Page.javascriptDialogOpening") {
      errors.push(message.params);
      void send("Page.handleJavaScriptDialog", { accept: false }).catch(error => stop(error));
    } else if (message.method === "Page.screencastFrame") {
      captureFrame?.(message.params.data);
      void send("Page.screencastFrameAck", { sessionId: message.params.sessionId }).catch(error => {
        if (!abort.signal.aborted) warnings.push(`Screencast frame acknowledgement failed: ${String(error)}`);
      });
    }
  };
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Log.enable");
  await send("Network.enable");
  // Intercept before navigation: even a broken import cannot contact another origin.
  await send("Fetch.enable", { patterns: [{ urlPattern: "http://*" }, { urlPattern: "https://*" }] });
  const receive = socket.onmessage!;
  socket.onmessage = event => {
    const message = JSON.parse(String(event.data));
    if (message.method === "Fetch.requestPaused") {
      const url = message.params.request.url as string;
      const own = new URL(url).origin === origin;
      if (!own) forbiddenRequests.push(url);
      void send(own ? "Fetch.continueRequest" : "Fetch.failRequest", own
        ? { requestId: message.params.requestId } : { requestId: message.params.requestId, errorReason: "BlockedByClient" }).catch(error => stop(error));
    } else receive.call(socket!, event);
  };
  await send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: downloads });
  await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: origin });
  await wait(`document.querySelectorAll('[data-case-id]').length === 3 && !!document.querySelector('[data-inspector]')`, "bundled case navigation");
  assert(evidenceLoaded, "Browser did not successfully load bundled evidence.json");
  assert.deepEqual(await evaluate<string[]>(`[...document.querySelectorAll('[data-case-id]')].map(b => b.dataset.caseId).sort()`), [...ids].sort());
  assert(await evaluate<boolean>(active('[data-case-id="query-half"]')), "Initial case must be query-half");
  assert(await evaluate<boolean>(`!!document.querySelector('#workspace h1')?.textContent.includes('Follow the work.')`), "Workspace heading missing");
  findings.push("Bundled evidence loaded; all three built-in cases and source snapshots validated.");

  for (const id of ids) {
    const investigation = bundle.investigations.find(c => c.id === id)!;
    await click(`[data-case-id="${id}"]`);
    await wait(active(`[data-case-id="${id}"]`), `${id}: selected case`);
    await evaluate(`document.querySelector('.scope').open = true`);
    const text = await evaluate<string>("document.body.textContent");
    assert(/source[- ]reconstruction/i.test(text), `${id}: source reconstruction disclosure missing`);
    assert(/source[- ]order/i.test(text) && /not[^.\n]{0,80}(clock|time)/i.test(text), `${id}: source-order, not clock/time disclosure missing`);
    assert(/logical/i.test(text) && /DRAM/.test(text), `${id}: logical/not-DRAM disclosure missing`);
    for (const field of [investigation.scope.family, investigation.scope.phase, investigation.scope.device, investigation.scope.workload]) assert(text.includes(field), `${id}: scope field missing: ${field}`);
    for (const metric of investigation.metrics) assert(text.includes(metric.label) && text.includes(metric.value), `${id}: metric missing: ${metric.label}`);
    for (const variant of investigation.variants) {
      await click(`[data-variant-id="${variant.id}"]`);
      await wait(active(`[data-variant-id="${variant.id}"]`), `${id}/${variant.id}: variant selected`);
      const node = variant.nodes.at(-1)!;
      await click(`[data-node-id="${node.id}"]`);
      await wait(`document.querySelector('[data-inspector]')?.textContent.includes(${JSON.stringify(node.label)})`, `${id}/${variant.id}: selected node inspector`);
      await checkFollow(variant, node.id, `${id}/${variant.id}`);
      await scrub(variant, `${id}/${variant.id}`);
      if (id === "query-half" && variant.id === "after") await nativeRangeDrag(variant, `desktop/${id}/${variant.id}`);
      await allocationKeyboard(variant, `${id}/${variant.id}`);
      await accessEvidence(investigation, variant, `${id}/${variant.id}`);
      if (id === "query-half" && Bun.env.CAPTURE_SCREENSHOTS === "1") {
        await evaluate(`document.querySelector('.metrics').scrollIntoView({ block: 'start' })`);
        await screenshot(`query-half-${variant.id}`);
      }
    }
    await sourceDialog(bundle, id);
    await layout(`desktop/${id}`);
    findings.push(`Desktop ${id}: both variants, source-order windows/counter, allocation keyboard focus, edge-specific evidence and source dialog passed.`);
  }
  await evaluate(`(() => { const c = document.querySelector('[data-compare]'); if (!c) throw new Error('Comparison control missing'); if (!c.checked) c.click(); })()`);
  assert(await evaluate<boolean>(`document.querySelector('[data-compare]').checked`), "Comparison enabled");
  const current = bundle.investigations.find(c => c.id === "bridge-copy")!;
  for (const variant of current.variants) assert(await evaluate<boolean>(`!!document.querySelector('[data-node-id="${variant.nodes[0]!.id}"]')`), `Comparison missing ${variant.id} graph`);
  await layout("desktop/comparison");
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  for (const id of ids) {
    await click(`[data-case-id="${id}"]`);
    await wait(active(`[data-case-id="${id}"]`), `mobile/${id}: case selected`);
    await layout(`mobile/${id}`);
  }
  await sourceDialog(bundle, "mobile");
  await evaluate(`document.querySelector('[data-case-id="query-half"]').focus()`);
  assert(await evaluate<boolean>(`document.activeElement?.getAttribute('data-case-id') === 'query-half'`), "Keyboard navigation must focus the case button");
  await key("Enter", "Enter", 13);
  await wait(active('[data-case-id="query-half"]'), "keyboard case navigation");
  await click('[data-variant-id="after"]');
  const query = bundle.investigations.find(c => c.id === "query-half")!;
  await nativeRangeDrag(query.variants.find(v => v.id === "after")!, "mobile/query-half/after");
  await evaluate(`document.querySelector('.metrics').scrollIntoView({ block: 'start' })`);
  await screenshot("mobile");
  findings.push("Desktop 1440x1000 and phone 390x844: case/variant/node navigation, comparison, follow, scrub, scope/metrics, target sizes, internal graph scroll, source dialog and Escape passed.");
  findings.push("Native multi-position range drags preserve the mounted range and update trusted input/accessible values; keyboard allocation activation preserves focus; each access basis matches edge-scoped evidence and citations.");

  const bundledNote = `Bundled browser hypothesis ${crypto.randomUUID()}`;
  const importedNote = `Imported browser hypothesis ${crypto.randomUUID()}`;
  await evaluate(`(() => { const input = document.querySelector('#hypothesis'); input.value = ${JSON.stringify(bundledNote)}; input.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await click('[data-export]');
  const exportedPath = await poll(async () => {
    const names = await readdir(downloads);
    const complete = names.filter(name => name.endsWith(".json") && !name.endsWith(".crdownload"));
    if (complete.length !== 1 || names.some(name => name.endsWith(".crdownload"))) return null;
    const path = resolve(downloads, complete[0]!);
    return Bun.file(path).size > 0 ? path : null;
  }, "export packet download");
  const exported = validateBundle(await Bun.file(exportedPath).json());
  assert(Bun.file(exportedPath).size <= MAX_PACKET_BYTES, "Downloaded packet must fit the actual 2 MiB import byte limit");
  assert(exported.investigations.some(c => c.id === "query-half"), "Export must contain the selected case");
  assert(exported.snapshots.length > 0, "Export must retain pinned sources");
  for (const source of exported.investigations.flatMap(c => c.sources)) assert(exported.snapshots.some(s => s.id === source.id), `Export missing snapshot ${source.id}`);
  for (const snapshot of exported.snapshots) assert.deepEqual(snapshot, bundle.snapshots.find(s => s.id === snapshot.id), `Export changed pinned source ${snapshot.id}`);
  await importFile(exportedPath);
  await wait(`document.body.textContent.includes('Imported')`, "good exported packet imports");
  assert.equal(await evaluate<string>(`document.querySelector('#hypothesis').value`), "", "An imported investigation with the bundled ID must not inherit its note");
  await evaluate(`(() => { const input = document.querySelector('#hypothesis'); input.value = ${JSON.stringify(importedNote)}; input.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await click('[data-variant-id="before"]');
  assert.equal(await evaluate<string>(`document.querySelector('#hypothesis').value`), importedNote, "Imported investigation note must survive its own rerender");
  await evaluate(`(() => { const reset = [...document.querySelectorAll('button')].find(b => b.textContent === 'Bundled examples'); if (!reset) throw new Error('Bundled restore control missing'); reset.click(); })()`);
  await wait(`document.querySelector('[role="status"]')?.textContent === 'Bundled examples restored.' && document.querySelectorAll('[data-case-id]').length === 3`, "restore original bundled investigations");
  await click('[data-case-id="query-half"]');
  assert.equal(await evaluate<string>(`document.querySelector('#hypothesis').value`), bundledNote, "Restoring bundled investigations must restore the original bundled note, not the colliding imported note");
  await importFile(exportedPath);
  await wait(`document.querySelector('[role="status"]')?.textContent.includes('Imported')`, "fresh colliding-ID import");
  assert.equal(await evaluate<string>(`document.querySelector('#hypothesis').value`), "", "A fresh imported investigation object must not inherit either previous object's note");
  findings.push("Colliding-ID imports start with independent notes, preserve their own edits on rerender, and restoring bundled examples retains the original bundled hypothesis.");
  for (const investigation of exported.investigations) {
    await click(`[data-case-id="${investigation.id}"]`);
    await wait(active(`[data-case-id="${investigation.id}"]`), "import remains navigable");
  }
  const selectedCase = exported.investigations.at(-1)!.id;
  for (const [name, contents] of [["invalid.json", "{ invalid json"], ["oversized.json", " ".repeat(MAX_PACKET_BYTES + 1)]] as const) {
    // A previous alert must not satisfy the next asynchronous rejection check.
    await evaluate(`document.querySelector('[data-import]').value = ''`);
    await importFile(exportedPath);
    await wait(`document.body.textContent.includes('Imported') && ![...document.querySelectorAll('[role="alert"]')].some(a => a.textContent.trim())`, `${name}: reset to a valid import`);
    await click(`[data-case-id="${selectedCase}"]`);
    const path = resolve(runDir, name);
    await Bun.write(path, contents);
    await evaluate(`document.querySelector('[data-import]').value = ''`);
    await importFile(path);
    await wait(`!![...document.querySelectorAll('[role="alert"]')].find(a => a.textContent.trim())`, `${name}: rejection alert`);
    assert(await evaluate<boolean>(active(`[data-case-id="${selectedCase}"]`)), `${name}: rejected import must preserve the current case`);
  }
  const injected = structuredClone(exported);
  const first = injected.investigations[0]!;
  first.id = "browser-injection";
  const payload = '<img data-browser-injection src="/browser-injection-probe" onerror="window.__browserInjection=1">';
  first.title = payload;
  first.variants[0]!.nodes[0]!.label = '<svg data-browser-injection onload="window.__browserInjection=1">';
  validateBundle(injected);
  const injectionPath = resolve(runDir, "injection.json");
  await Bun.write(injectionPath, JSON.stringify(injected));
  await importFile(injectionPath);
  await wait(`!!document.querySelector('[data-case-id="browser-injection"]')`, "safe text packet imports");
  await click('[data-case-id="browser-injection"]');
  await wait(`document.body.textContent.includes(${JSON.stringify(payload)})`, "imported HTML label remains literal text");
  await click(`[data-variant-id="${first.variants[0]!.id}"]`);
  await wait(`document.querySelector('[data-node-id="${first.variants[0]!.nodes[0]!.id}"]')?.textContent.includes(${JSON.stringify(first.variants[0]!.nodes[0]!.label)})`, "imported node label remains literal text");
  assert(!await evaluate<boolean>(`!!document.querySelector('[data-browser-injection]') || !!window.__browserInjection`), "Imported labels created markup or executed code");
  await Bun.sleep(150);
  assert.deepEqual(forbiddenRequests, [], "Page attempted requests outside its owned preview");
  assert.deepEqual(failedResponses, [], "Page requests failed (including injection probes)");
  assert.deepEqual(errors, [], "Page console/runtime errors");
  findings.push("Downloaded export validates with pinned sources; good import navigates; malformed/oversized imports preserve selection; imported HTML stays text with no injected markup or external network.");
}

let failure: unknown;
try { await run(); }
catch (error) { failure = error; }
finally {
  clearTimeout(deadline);
  stop(new Error("Browser check finished"));
  if (socket) { socket.onmessage = null; socket.onclose = null; socket.onerror = null; socket.close(); }
  // Signal only direct children created above; never inspect or kill other sessions.
  for (const child of children) if (child.exitCode === null) child.kill("SIGTERM");
  await Promise.all(children.map(async child => {
    if (child.exitCode !== null) { await child.exited; return; }
    let timer: ReturnType<typeof setTimeout> | undefined;
    const exited = await Promise.race([
      child.exited.then(() => true),
      new Promise<false>(resolve => { timer = setTimeout(() => resolve(false), 3000); }),
    ]);
    clearTimeout(timer);
    if (!exited && child.exitCode === null) {
      warnings.push(`Owned child ${child.pid} required SIGKILL after its shutdown grace period.`);
      child.kill("SIGKILL");
    }
    await child.exited;
  }));
  process.removeListener("SIGINT", interrupt);
  process.removeListener("SIGTERM", interrupt);
}
for (const finding of findings) console.log(`PASS: ${finding}`);
for (const warning of warnings) console.warn(`NOTE: ${warning}`);
console.log(`Browser artifacts: ${runDir}`);
if (failure) {
  console.error(failure);
  if (serverOutput) console.error(`Preview stdout:\n${serverOutput}`);
  if (serverErrors) console.error(`Preview stderr:\n${serverErrors}`);
  if (browserErrors) console.error(`Chromium stderr:\n${browserErrors}`);
  process.exitCode = 1;
} else console.log("CPU-only Work Microscope browser check passed; owned preview and Chromium exited.");
