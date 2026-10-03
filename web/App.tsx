import { useEffect, useRef, useState } from "react";
import { ApiHttpError, createLensApi } from "./api";
import { decodeJob, decodeResult, isPrepared, isBaselineOnly, type Asset, type Capabilities, type Job } from "./contract";
import { decodeDiscovery, reconcileBinding, rebindToCurrent, recoverLegacyBinding, requireUnchangedDraft } from "./bindings";
import { DRAFT_KEY, SELECTION_KEY, draftFromSavedRequest, emptySelection, newDraft, parseDraft, parseSelection, submissionConfig, validateCapabilities } from "./draft";
import { DurableSubmission, readIntent, pruneCompletedArchives, type Intent } from "./durable";
import { InputEditor } from "./editor";
import { useHistory } from "./jobs";
import { useStored } from "./storage";
import { ExecutionViewer } from "./viewer";
import { HistoryEntry, RecoveryNotice } from "./history";

const sections = ["Input", "Execution", "History"] as const;
const api = createLensApi();
type Section = typeof sections[number];
type Failure = { at: string; context: string; message: string; headers?: [string, string][] };

export function App() {
  const [section, setSection] = useState<Section>("Input");
  const [viewVersion, setViewVersion] = useState(0);
  const reviewFocus = useRef(false);
  useEffect(() => {
    if (section === "Input" && reviewFocus.current) {
      reviewFocus.current = false;
      document.getElementById("section-Input")?.focus();
    }
  }, [section]);
  const draft = useStored(DRAFT_KEY, newDraft, parseDraft);
  const selected = useStored(SELECTION_KEY, () => ({ ...emptySelection }), parseSelection);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [assetsLoaded, setAssetsLoaded] = useState(false);
  const [failures, setFailures] = useState<Failure[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [intent, setIntent] = useState<Intent | null>(null);
  const [recoveryBlocked, setRecoveryBlocked] = useState(false);
  const discoveryBusy = useRef(false);
  const submissionBusy = useRef(false);
  const copyBusy = useRef(false);
  const alive = useRef(true);
  const [selectionText, setSelectionText] = useState(selected.value[selected.value.view]);
  const [notice, setNotice] = useState("");
  const [pruning, setPruning] = useState(false);
  const [archiveNotice, setArchiveNotice] = useState("");
  async function pruneArchives() {
    if (pruning) return;
    setPruning(true);
    try {
      if (!navigator.locks) throw new Error("Web Locks are required for cross-tab-safe cleanup.");
      const reads = createLensApi((path, init) => fetch(path, { ...init, signal: AbortSignal.timeout(10_000) }));
      const result = await navigator.locks.request("qwen-lens-submission-v1", () => pruneCompletedArchives(localStorage, reads.job));
      setArchiveNotice(`Removed ${result.removed} completed browser archives; kept ${result.kept} active, unresolved, changed or unverifiable archives. Current submission and server history are unchanged.`);
    } catch (error) { setArchiveNotice(String(error)); }
    finally { setPruning(false); }
  }
  const latestDraft = useRef(draft.value);
  latestDraft.current = draft.value;
  function updateDraft(next: typeof draft.value) {
    try {
      const bound = reconcileBinding(next, capabilities, assets);
      if (!draft.update(bound)) return false;
      latestDraft.current = bound;
      return true;
    } catch (error) { report("Bind draft edit", error); return false; }
  }
  function report(context: string, error: unknown) {
    if (!alive.current) return;
    setFailures(previous => [...previous, { at: new Date().toISOString(), context,
      message: error instanceof Error ? `${error.name}: ${error.message}${error.cause ? `\nCause: ${String(error.cause)}` : ""}${error instanceof AggregateError ? `\n${error.errors.map(String).join("\n")}` : ""}` : JSON.stringify(error),
      ...(error instanceof ApiHttpError ? { headers: [...error.response.headers] } : {}),
    }]);
  }
  const history = useHistory(report);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => setSelectionText(selected.value[selected.value.view]), [selected.value]);
  useEffect(() => {
    function recover() { try { setIntent(readIntent(localStorage)); setRecoveryBlocked(false); } catch (error) { setRecoveryBlocked(true); report("Recover durable submission", error); } }
    recover();
    window.addEventListener("storage", recover);
    void discover();
    return () => window.removeEventListener("storage", recover);
  }, []);
  async function discover() {
    if (discoveryBusy.current) return;
    discoveryBusy.current = true; setLoading(true);
    try {
      const caps = await api.capabilities();
      const catalog = await api.assets();
      const decoded = decodeDiscovery(caps, catalog);
      if (alive.current) { setCapabilities(decoded.caps); setAssets(decoded.assets); setAssetsLoaded(true); }
    } catch (error) {
      if (alive.current) { setCapabilities(null); setAssetsLoaded(false); }
      report("Discover matching model and assets", error);
    }
    discoveryBusy.current = false; if (alive.current) setLoading(false);
  }
  const validation = validateCapabilities(draft.value, capabilities, assets);
  if (loading) validation.push("Wait for capability/asset refresh to finish.");
  if (!assetsLoaded) validation.push("Registered assets have not been loaded.");
  if (draft.errors.length || selected.errors.length || recoveryBlocked) validation.push("Resolve local persistence errors before submitting.");
  if (intent && !intent.jobId && !intent.rejected) validation.push("An unresolved durable submission exists. Recover its exact request first.");

  function openJob(job: Job, slot = selected.value.view) {
    if (selected.update({ ...selected.value, [slot]: job.id, view: slot })) { setSection("Execution"); setViewVersion(value => value + 1); }
    history.merge([job]);
  }
  async function submit(retry: boolean) {
    if (submissionBusy.current || (!retry && validation.length)) return;
    const submittingDraft = latestDraft.current;
    submissionBusy.current = true; setSubmitting(true);
    try {
      if (!navigator.locks) throw new Error("Web Locks are required for cross-tab-safe submission. Use a current browser on localhost or HTTPS.");
      await navigator.locks.request("qwen-lens-submission-v1", async () => {
        const service = new DurableSubmission(localStorage, api.submit);
        if (!retry) {
          requireUnchangedDraft(submittingDraft, latestDraft.current);
          const bound = reconcileBinding(latestDraft.current, capabilities, assets);
          if (!draft.update(bound)) throw new Error("Draft identities could not be saved; no request was submitted.");
          latestDraft.current = bound;
          const config = submissionConfig(bound);
          const key = crypto.randomUUID();
          const bytes = new TextEncoder().encode(JSON.stringify({ ...config, idempotency_key: key })).byteLength;
          if (!capabilities || bytes > capabilities.limits.max_body_bytes) throw new Error(`Request is ${bytes} bytes; server body limit is ${capabilities?.limits.max_body_bytes ?? "unknown"}.`);
          service.prepare(config, key);
        }
        setIntent(readIntent(localStorage));
        const response = await service.retry();
        const job = decodeJob(response);
        const current = readIntent(localStorage);
        if (!current) throw new Error("Durable submission disappeared before confirmation.");
        service.confirmJob(current.key, job.id);
        if (alive.current) { setIntent(readIntent(localStorage)); openJob(job); setNotice(`Durably accepted ${job.id}. Viewer lifetime is independent of execution.`); }
      });
    } catch (error) { report(retry ? "Recover exact submission" : "Submit experiment", error); }
    finally {
      submissionBusy.current = false;
      if (alive.current) {
        setSubmitting(false);
        try { setIntent(readIntent(localStorage)); } catch (error) { setRecoveryBlocked(true); report("Read submission recovery state", error); }
      }
    }
  }
  async function copyRun(id: string, newSeed: boolean) {
    if (copyBusy.current) return;
    copyBusy.current = true;
    const original = latestDraft.current;
    try {
      setNotice(`Reading the server's recorded request for ${id}. No inference is being submitted.`);
      let next = draftFromSavedRequest(await api.request(id), id);
      if (next.binding.state === "legacy") {
        try {
          const page = decodeResult(await api.result(id, undefined, 1));
          if (page.job_id !== id) throw new Error("Identity result belongs to another job.");
          const prepared = page.records.find(record => record.seq === 0 && isPrepared(record));
          next = recoverLegacyBinding(next, prepared && isPrepared(prepared) ? prepared : undefined);
        } catch (error) { report("Read historical model/asset identities", error); }
      }
      if (!alive.current) return;
      requireUnchangedDraft(original, latestDraft.current);
      if (newSeed) next.generation.sampling.seed = crypto.getRandomValues(new Uint32Array(1))[0]!;
      if (updateDraft(next)) { setSection("Input"); setNotice(`Copied ${id} with ${newSeed ? "a new" : "the same"} seed. Review identities and press Run experiment to create a new idempotency key; nothing was rerun automatically.`); }
    } catch (error) { report("Prepare run again", error); }
    finally { copyBusy.current = false; }
  }

  return <div className="shell"><a className="skip-link" href="#workspace">Skip to workspace</a>
    <header><div className="masthead"><span className="eyebrow">Local inference / Instrument workspace</span><span className="status">{capabilities ? capabilities.available ? isBaselineOnly(capabilities) ? "Baseline available" : capabilities.operations.length === 0 ? "Readouts available" : "Lens available" : "Lens unavailable" : "Connecting to Lens"}</span></div><h1>Qwen <em>Lens</em><span className="crosshair" aria-hidden="true">+</span></h1><p className="intro">A closer look at inference.</p></header>
    <details className="connection deployment-details" aria-label="Server capabilities"><summary>Model / {capabilities?.model.id ?? "connecting"} / capabilities & assets</summary><div className="row-heading"><div><p className="eyebrow">Same origin / v1 / isolated diagnostic cache</p><p>{capabilities ? `${capabilities.model.id} / ${capabilities.model.layers === null ? "diagnostic model metadata unavailable" : `${capabilities.model.layers} layers`}` : "Reading actual server capabilities."}</p></div><button type="button" disabled={loading} onClick={() => void discover()}>{loading ? "Reading server..." : "Refresh capabilities & assets"}</button></div>
      {capabilities && !capabilities.available && <p className="notice">{capabilities.unavailable_reason ?? "Server did not supply an unavailable reason."}</p>}
      {capabilities && isBaselineOnly(capabilities) && <p className="notice">Baseline execution only: messages, assistant prefills, sampling and durable token history are connected. Readouts and interventions are not available yet; requests containing them are rejected, never silently ignored.</p>}
      {capabilities?.available && !isBaselineOnly(capabilities) && capabilities.operations.length === 0 && <p className="notice">Readouts use the available lens aliases at selected layers and consumed positions. Scores come from original-forward captures and the deployed output head, not inference replay. Fitted target-layer scores are not the final generation distribution. This server has no registered intervention-capable alias.</p>}
      <details><summary>Capabilities, limits and registered assets</summary><pre>{JSON.stringify({ capabilities, assets }, null, 2)}</pre></details>
    </details>
    {capabilities && !capabilities.available && <p className="notice">Lens unavailable: {capabilities.unavailable_reason ?? "Reason not supplied"}</p>}
    {notice && <p className="notice" role="status">{notice}</p>}
    <details className="connection binding-details" aria-label="Draft identity assertions"><summary>Draft binding / {draft.value.binding.state} / review or change deployment</summary>
      <p>These assertions preserve the advertised model metadata identity and registered asset fingerprints, not authenticated model weights or identical software behavior. Refreshing discovery never retargets saved work.</p>
      <details><summary>Saved draft identity assertions</summary><pre>{JSON.stringify(draft.value.binding, null, 2)}</pre></details>
      <button type="button" disabled={loading || !assetsLoaded || capabilities?.request_preconditions !== true} onClick={() => {
        if (!window.confirm("Use the current model and assets for this draft? This changes its target and may change token-ID meaning. Existing jobs and pending request bytes stay unchanged.")) return;
        try { updateDraft(rebindToCurrent(latestDraft.current, capabilities, assets)); } catch (error) { report("Explicitly rebind draft", error); }
      }}>Use current model & assets</button></details>
    {(draft.errors.length > 0 || selected.errors.length > 0) && <section role="alert" className="error-ledger"><h2>Local storage errors</h2>{[...draft.errors, ...selected.errors].map((error, index) => <pre key={index}>{error}</pre>)}<p>Original storage is retained. No submission is allowed on uncertain persistence.</p></section>}
    {intent && !intent.jobId && <section className="recovery"><h2>{intent.rejected ? "Submission rejected" : "Submission recovery"}</h2><p>Key: <code>{intent.key}</code></p><p>{intent.rejected ? "The server rejected this request before acceptance. You may correct the draft and submit with a new key, or retry this exact saved request." : "An outcome has not been associated with a validated job yet. Do not create a different request to recover it."}</p>
      <button type="button" disabled={submitting || recoveryBlocked} onClick={() => void submit(true)}>Recover / retry exact saved request</button><details><summary>Durable request and every attempt</summary><pre>{intent.body}</pre><pre>{JSON.stringify(intent.attempts, null, 2)}</pre></details></section>}
    {failures.length > 0 && <section className="error-ledger" aria-labelledby="errors-title"><h2 id="errors-title">Errors ({failures.length})</h2><p role="status">Latest: {failures.at(-1)?.context}. All errors remain below for this session.</p>{failures.map((failure, index) => <details key={index} open={index === failures.length - 1}><summary>{failure.at} / {failure.context}</summary><pre>{failure.message}</pre>{failure.headers && <pre>{failure.headers.map(([key, value]) => `${key}: ${value}`).join("\n")}</pre>}</details>)}</section>}
    <main id="workspace" tabIndex={-1}><nav aria-label="Workspace sections">{sections.map((name, index) => <button key={name} type="button" aria-current={section === name ? "page" : undefined} aria-controls={`section-${name}`} onClick={() => setSection(name)}><span className="number">0{index + 1}</span> {name}</button>)}</nav>
      <div id="section-Input" tabIndex={-1} hidden={section !== "Input"}><InputEditor draft={draft.value} update={updateDraft} caps={capabilities} assets={assets} errors={validation} busy={submitting} onRun={() => void submit(false)} /></div>
      <div id="section-Execution" hidden={section !== "Execution"}><section className="panel" aria-labelledby="execution-heading"><p className="eyebrow">02 / Explore</p><h2 id="execution-heading">Follow a token. Test a direction.</h2>
        <div className="actions" aria-label="Comparison selection">{(["baseline", "variant"] as const).map(view => <button key={view} type="button" aria-pressed={selected.value.view === view} onClick={() => selected.update({ ...selected.value, view })}>{view === "baseline" ? "Baseline" : "Variant"}{selected.value[view] ? ` / ${selected.value[view]}` : " / not selected"}</button>)}</div>
        <details className="open-job-details"><summary>Open a job by ID</summary><form onSubmit={event => { event.preventDefault(); selected.update({ ...selected.value, [selected.value.view]: selectionText.trim() }); }}><label htmlFor="selected-job">{selected.value.view} job ID</label><div className="inline-form"><input id="selected-job" value={selectionText} onChange={event => setSelectionText(event.target.value)} /><button type="submit">Open job</button></div></form></details>
        <ExecutionViewer key={`${selected.value[selected.value.view]}:${viewVersion}:${history.wasDeleted(selected.value[selected.value.view])}`} id={selected.value[selected.value.view]} caps={capabilities} assets={assets} draft={draft.value} update={updateDraft} report={report} copyRun={copyRun} onDeleted={history.observeDeleted} reviewDraft={() => { reviewFocus.current = true; setSection("Input"); }} />
      </section></div>
      <div id="section-History" hidden={section !== "History"}><section className="panel" aria-labelledby="history-heading"><p className="eyebrow">03 / Record</p><h2 id="history-heading">The server's record.</h2><button type="button" disabled={history.busy} onClick={() => void history.reload()}>Refresh history now</button>
        <p className="muted">Actual jobs refresh every 3 seconds, including while work runs. Older pages are retained and merged by ID/revision.</p>
        {history.recovery && <RecoveryNotice recovery={history.recovery} />}
        {history.storage && <p>Server storage: {history.storage.retained_jobs} / {history.storage.max_retained_jobs} retained jobs; {(history.storage.reserved_bytes / 1048576).toFixed(1)} / {(history.storage.max_store_bytes / 1048576).toFixed(0)} MiB charged; {history.storage.retry_identities} / {history.storage.max_retry_identities} permanent retry identities.</p>}
        <button type="button" disabled={pruning} onClick={() => void pruneArchives()}>{pruning ? "Checking browser archives..." : "Clean completed browser archives"}</button>
        {archiveNotice && <p role="status">{archiveNotice}</p>}
         {!history.loaded ? <p>History has not loaded; this is not an empty-server claim.</p> : history.jobs.length === 0 ? <p>{history.recovery ? "No healthy jobs are available on this page." : "The server returned no jobs."}</p> : <ol className="history-list">{history.jobs.map(job => <HistoryEntry key={job.id} job={job} preview={history.previews.get(job.id)} open={openJob} remove={history.recovery ? undefined : history.remove} />)}</ol>}
         {history.cursor !== null && <button type="button" disabled={history.busy} onClick={() => void history.more()}>Load older jobs</button>}
      </section></div>
    </main><footer><span>QWEN LENS / HTTP CONTRACT V1</span><span>Durable jobs. Independent viewers. Native scores.</span></footer>
  </div>;
}
