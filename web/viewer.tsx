import { useEffect, useMemo, useRef, useState } from "react";
import { isPrepared, isReadout, isSample, isTerminal, selectorContains, type Asset, type Capabilities, type Job, type Readout, type Score } from "./contract";
import { NumberField } from "./editor";
import { type Draft, sourceScope } from "./draft";
import { type Report, useJob } from "./jobs";
import { createLensApi } from "./api";
import { decodeJob } from "./contract";
import type { Sequenced } from "./records";
import { mergePinnedBinding, savedModelIdentity } from "./bindings";
import { RetentionViewer } from "./retention-viewer";
import { TokenNavigator } from "./token-navigator";
import { sameSite, type TokenSite } from "./token-navigation";

export function JobStatus({ job }: { job: Job }) {
  return <div className="job-status"><div className="row-heading"><strong>{job.state}</strong><span>{job.generation.sampled_tokens} sampled / {job.observations.committed_records} observations</span></div><details><summary>Generation, observation & cancellation status</summary>
    <dl className="metadata"><dt>Generation</dt><dd>{job.generation.state} / {job.generation.phase ?? "no active phase"}</dd><dt>Observations</dt><dd>{job.observations.state}</dd>
      <dt>Prompt consumed</dt><dd>{job.generation.consumed_prompt_tokens} / {job.generation.prompt_tokens}</dd><dt>Generated</dt><dd>{job.generation.sampled_tokens} sampled / {job.generation.consumed_generated_tokens} consumed</dd>
       <dt>Stop reason</dt><dd>{job.generation.stop_reason ?? "Not reported"}</dd><dt>Cancel requested</dt><dd>{job.cancel_requested ? "Yes / not a GPU completion acknowledgment" : "No"}</dd></dl></details>
    {job.generation.error !== null && <div className="error-ledger"><h4>Generation error</h4><pre>{JSON.stringify(job.generation.error, null, 2)}</pre></div>}
    {job.observations.error !== null && <div className="error-ledger"><h4>Observation error</h4><pre>{JSON.stringify(job.observations.error, null, 2)}</pre></div>}
    {job.result.error !== null && <div className="error-ledger"><h4>Result publication error</h4><pre>{JSON.stringify(job.result.error, null, 2)}</pre></div>}
  </div>;
}

export function cellAvailability(records: Sequenced[], readoutId: string, phase: "prefill" | "decode", index: number, layer: number, complete: boolean) {
  const prepared = records.find(isPrepared);
  const scope = prepared?.resolved_scopes.find(scope => scope.kind === "readout" && scope.id === readoutId)?.scope;
  if (scope && (!selectorContains(scope.layers, layer) || !selectorContains(scope[phase], index))) return "Not captured / outside scope";
  if (phase === "decode" && records.filter(isSample).some(sample => sample.index === index && !sample.consumed)) return "Not captured / unconsumed sample";
  if (complete) return "Not captured / no published record";
  return scope ? "Pending / not loaded yet" : "Unknown / scope not published";
}

export function matchingDirectionAsset(record: Readout, assets: Asset[], modelIdentity: string | undefined) {
  return modelIdentity ? assets.find(asset => asset.alias === record.lens && asset.available
    && asset.identity === (record.target_layer === null ? modelIdentity : record.asset_identity)) : undefined;
}

type TrackedToken = { tokenId: number; label: string | null };

function ScorePanel({ record, modelIdentity, caps, assets, draft, update, report, tracked, track, reviewDraft }: { record: Readout; modelIdentity: string | undefined; caps: Capabilities | null; assets: Asset[]; draft: Draft; update: (draft: Draft) => boolean; report: Report; tracked: TrackedToken | null; track: (token: TrackedToken | null) => void; reviewDraft?: () => void }) {
  const [coefficient, setCoefficient] = useState(.25);
  const [action, setAction] = useState("fixed_add");
  const [pinNotice, setPinNotice] = useState("");
  const [localRow, setLocalRow] = useState<{ seq: number; row: number } | null>(null);
  const interventionDock = useRef<HTMLDivElement>(null);
  const chosen = tracked ? record.scores.find(score => score.token_id === tracked.tokenId) : record.scores.find(score => localRow?.seq === record.seq && score.row_id === localRow.row);
  const asset = matchingDirectionAsset(record, assets, modelIdentity);
  const rowFor = (score: Score) => {
    if (score.token_id !== null && asset?.direction_rows.includes("token_id")) return { kind: "token_id", token_id: score.token_id };
    if (asset?.direction_rows.includes("template_row_id")) return { kind: "template_row_id", template_row_id: score.row_id };
    if (score.label !== null && asset?.direction_rows.includes("label")) return { kind: "label", label: score.label };
    return null;
  };
  const canPin = !!caps && !!asset && !!modelIdentity && draft.directions.length < caps.limits.max_directions;
  function pin(score: Score, operation: boolean) {
    try {
      const row = rowFor(score);
      if (!canPin || !row) throw new Error("This score has no supported direction-row selector on this server.");
      if (operation && (!caps?.operations.includes(action) || draft.operations.filter(row => row.enabled).length >= caps.limits.max_operations)) throw new Error("Operation is unavailable or the server operation count limit was reached.");
      const id = crypto.randomUUID();
      const direction = { id: `d-${id}`, lens: record.lens, row, normalization: "unit_l2" };
      const bound = mergePinnedBinding(draft, modelIdentity!, record.lens, asset!.identity);
      const saved = update({ ...bound, directions: [...bound.directions, direction], operations: operation ? [...bound.operations, { key: id, enabled: true, document: { id: `op-${id}`, scope: sourceScope(record), action: { kind: action, direction: direction.id, coefficient } } }] : bound.operations });
      if (saved) setPinNotice(`Last staged: token ${score.token_id ?? score.row_id} / ${record.lens} / layer ${record.source_layer} / ${record.phase} ${record.index}${operation ? ` / ${action} ${coefficient}` : " / direction only"} in the current draft. No work submitted.`);
    } catch (error) { report("Pin score", error); }
  }
  return <section className="score-panel" aria-labelledby="score-heading"><p className="eyebrow">Scores for the next token / position {record.predicts_position}</p><h3 id="score-heading">Layer {record.source_layer} / {record.phase} index {record.index}</h3>
    <p className="measurement-identity">{record.readout_id} / {record.lens} / raw {record.score_kind} / {record.candidate_universe}</p>
    <p className="muted">{record.applied_operation_ids.length} applied operations at this site / {record.capture_stage}</p>
    {record.target_layer !== null && <p className="muted">Fitted readout to layer {record.target_layer}, not the final generation distribution. Binding: {typeof record.binding_status === "string" ? record.binding_status : "not recorded"}.</p>}
    <p className="muted">Choose a candidate to follow it across saved layers or stage an intervention. This does not change the inspected input token.</p>
    <div className="intervention-dock" tabIndex={-1} ref={interventionDock}><h4>{chosen ? `Stage candidate ${chosen.token_id ?? chosen.row_id}` : "Select a saved candidate to intervene"}</h4>
      {tracked && !chosen && <p className="muted">Tracked token {tracked.tokenId} is not in this readout's saved rows. Its score is unavailable here, not zero.</p>}
      <p className="muted">Adds to the current draft, not automatically a copy of this historical run. {draft.operations.filter(row => row.enabled).length} enabled draft operations.</p>
      {chosen && <><div className="field-grid"><label>Pin + operation action<select value={action} onChange={event => setAction(event.target.value)}><option value="" disabled>Choose action</option>{["fixed_add", "residual_l2_fraction", "projection_ablate"].filter(kind => caps?.operations.includes(kind)).map(kind => <option key={kind}>{kind}</option>)}</select></label><NumberField label="Pinned operation coefficient (native units)" value={coefficient} onChange={setCoefficient} /></div>
        <p>Scope: <strong>{record.phase} {record.index} / layer {record.source_layer}</strong>. Lens: {record.lens}. Coefficient is in native operation units.</p>
        <div className="actions"><button type="button" disabled={!canPin || !rowFor(chosen)} onClick={() => pin(chosen, false)}>Pin direction</button><button className="primary" type="button" disabled={!canPin || !rowFor(chosen) || !caps?.operations.includes(action)} onClick={() => pin(chosen, true)}>Pin + operation</button></div></>}
      {pinNotice && <p role="status" className="notice">{pinNotice}</p>}
      {!canPin && <p className="muted">Pinning unavailable: check saved identity provenance, matching asset availability and the advertised direction limit ({caps?.limits.max_directions ?? "not loaded"}).</p>}
      {reviewDraft && <button type="button" onClick={reviewDraft}>Review current draft</button>}
    </div>
    <ol className="scores compact-scores">{record.scores.map((score, index) => <li key={`${score.row_id}-${index}`}><button type="button" className="candidate-choice" aria-pressed={chosen === score} aria-label={`Select candidate ${score.token_id ?? `row ${score.row_id}`}`} onClick={() => { setLocalRow({ seq: record.seq, row: score.row_id }); track(score.token_id === null ? null : { tokenId: score.token_id, label: score.label }); }}><span className="candidate-rank">{index + 1}</span><span><bdi className="token-label">{score.label === null ? "Label unavailable" : JSON.stringify(score.label)}</bdi><p className="muted">Token {score.token_id ?? "unavailable"} / row {score.row_id}</p></span><span className="score-value">{score.score}</span></button>{chosen === score && <button type="button" className="candidate-adjust" onClick={() => interventionDock.current?.focus()}>Adjust intervention</button>}</li>)}</ol>
    <details><summary>Readout provenance & interpretation</summary><dl className="metadata"><dt>Readout / lens</dt><dd>{record.readout_id} / {record.lens}</dd><dt>Score units</dt><dd>{record.score_kind} / not probability or confidence</dd><dt>Candidate universe</dt><dd>{record.candidate_universe}</dd>
      <dt>Source consumed position</dt><dd>{record.position} / token ID {record.input_token_id}</dd><dt>Predicts position</dt><dd>{record.predicts_position}</dd><dt>Target layer</dt><dd>{record.target_layer ?? "None"}</dd>
      <dt>Capture stage</dt><dd>{record.capture_stage}</dd><dt>Provenance</dt><dd>{record.provenance}</dd><dt>Applied operations</dt><dd>{record.applied_operation_ids.length ? record.applied_operation_ids.join(", ") : "None"}</dd>
      <dt>Measured readout cost</dt><dd>{record.cost.readout_ms === null ? "Unavailable" : `${record.cost.readout_ms} ms`}</dd></dl>
    {record.target_layer !== null && <p className="notice">Fitted transport to layer {record.target_layer}, then the deployed output head; not the final generation distribution. Binding: {typeof record.binding_status === "string" ? record.binding_status : "Not retained in this record"}. Method: {typeof record.method === "string" ? record.method : "Not retained"}.</p>}
    {record.target_layer !== null && <p className="notice">Pinning uses the transport-transposed, gamma-folded LM-head row: a deployed-logit-numerator covector, not a gradient of the normalized logit or a calibrated causal direction. The registered asset identity must match this saved readout.</p>}
    </details>
  </section>;
}

export function TraceViewer({ records, complete, caps, assets, draft, update, report, selectArray, site, selectSite, reviewDraft }: { records: Sequenced[]; complete: boolean; caps: Capabilities | null; assets: Asset[]; draft: Draft; update: (draft: Draft) => boolean; report: Report; selectArray?: (key: string) => void; site?: TokenSite | null; selectSite?: (site: TokenSite | null) => void; reviewDraft?: () => void }) {
  const readouts = useMemo(() => records.filter(isReadout), [records]);
  const ids = [...new Set(readouts.map(record => record.readout_id))];
  const [readoutId, setReadoutId] = useState("");
  const [phase, setPhase] = useState<"prefill" | "decode">("decode");
  const [layerStart, setLayerStart] = useState(0);
  const [indexStart, setIndexStart] = useState(0);
  const [layerCount, setLayerCount] = useState(6);
  const [indexCount, setIndexCount] = useState(4);
  const [selected, setSelected] = useState<number | null>(null);
  const [tracked, track] = useState<TrackedToken | null>(null);
  const previousSite = useRef<TokenSite | null | undefined>(undefined);
  useEffect(() => {
    const navigated = site !== previousSite.current;
    previousSite.current = site;
    if (site) {
      const candidates = readouts.filter(row => sameSite(row, site) && (!readoutId || row.readout_id === readoutId));
      const row = candidates.find(row => row.seq === selected) ?? candidates.find(row => row.source_layer === layerStart) ?? (navigated || !readoutId ? candidates[0] : undefined);
      if (navigated) { setPhase(site.phase); setIndexStart(site.index); if (row) setLayerStart(row.source_layer); }
      setSelected(row?.seq ?? null);
      if (!readoutId && row) setReadoutId(row.readout_id);
      return;
    }
    const first = readouts[0];
    if (!readoutId && first) { setReadoutId(first.readout_id); setPhase(first.phase); setLayerStart(first.source_layer); setIndexStart(first.index); setSelected(first.seq); }
  }, [readouts, site, readoutId]);
  const selection = readouts.find(record => record.seq === selected);
  useEffect(() => {
    if (selection?.retained && typeof selection.retained === "object" && "logits_key" in selection.retained) selectArray?.(String(selection.retained.logits_key));
    else selectArray?.("");
  }, [selected, selectArray]);
  const savedModel = savedModelIdentity(records.find(isPrepared));
  const maxLayer = Math.max(0, (caps?.model.layers ?? 0) - 1, readouts.reduce((max, record) => Math.max(max, record.source_layer), 0));
  const visibleLayers = Array.from({ length: Math.min(layerCount, Math.max(0, maxLayer - layerStart + 1)) }, (_, index) => layerStart + index);
  const visibleIndices = Array.from({ length: indexCount }, (_, index) => indexStart + index);
  const cells = new Map(readouts.filter(record => record.readout_id === readoutId && record.phase === phase).map(record => [`${record.source_layer}:${record.index}`, record]));
  const inspected = site ?? selection;
  const layers = readouts.filter(row => row.readout_id === readoutId && sameSite(row, inspected)).sort((a, b) => a.source_layer - b.source_layer);
  const chooseLayer = (record: Readout) => { setSelected(record.seq); setLayerStart(record.source_layer); if (!sameSite(site, record)) selectSite?.({ phase: record.phase, index: record.index }); };
  return <section className="readout-inspector" aria-labelledby="trace-heading"><h3 id="trace-heading" tabIndex={-1}>Layers & candidates</h3>
    {site && readoutId && !selection && <p className="selected-alias-coverage">{readoutId} / {site.phase} {site.index} / layer {layerStart}: {cellAvailability(records, readoutId, site.phase, site.index, layerStart, complete)}</p>}
    {readouts.length === 0 ? <p>{complete ? "No readouts published in the loaded result." : "No readout records loaded yet. Capture availability is not inferred."}</p> : <>
      <label>Readout<select value={readoutId} onChange={event => { const inherited = site ?? selection; setReadoutId(event.target.value); setSelected(null); if (!site && inherited) selectSite?.({ phase: inherited.phase, index: inherited.index }); }}>{ids.map(id => <option key={id}>{id}</option>)}</select></label>
      <div className="layer-context"><p>{inspected ? `Inspecting input ${inspected.phase} ${inspected.index}` : "Select an input token"} / {layers.length} saved layers</p>{tracked && <div className="actions"><span>Following candidate <bdi>{tracked.label === null ? tracked.tokenId : JSON.stringify(tracked.label)}</bdi> / ID {tracked.tokenId}</span><button type="button" onClick={() => track(null)}>Show top candidates</button></div>}</div>
      <div className="layer-strip" aria-label="Saved layers at inspected token">{layers.map(row => { const candidate = tracked ? row.scores.find(score => score.token_id === tracked.tokenId) : row.scores[0]; return <button key={row.seq} type="button" aria-pressed={selected === row.seq} aria-label={`Inspect saved layer ${row.source_layer}`} onClick={() => chooseLayer(row)}><strong>L{row.source_layer}</strong><bdi>{candidate ? candidate.label === null ? `Token ${candidate.token_id ?? candidate.row_id}` : JSON.stringify(candidate.label) : "Not in saved rows"}</bdi><span className="score-value">{candidate ? candidate.score : "Score unavailable"}</span></button>; })}</div>
      <p className="muted">Raw saved scores, not confidence. Only loaded layers in this readout are shown; a missing candidate is not a zero score.</p>
      <details className="advanced-grid"><summary>Advanced position/layer grid ({readouts.length} loaded readouts)</summary><label>Source phase<select value={phase} onChange={event => { setPhase(event.target.value as "prefill" | "decode"); setSelected(null); selectSite?.(null); }}><option value="prefill">prefill</option><option value="decode">decode</option></select></label>
      <div className="field-grid"><NumberField label="Display first layer" value={layerStart} integer min={0} max={maxLayer} onChange={setLayerStart} /><NumberField label="Display first source index" value={indexStart} integer min={0} max={Number.MAX_SAFE_INTEGER - indexCount} onChange={setIndexStart} />
        <NumberField label="Display layer count" value={layerCount} integer min={1} max={64} onChange={setLayerCount} /><NumberField label="Display index count" value={indexCount} integer min={1} max={64} onChange={setIndexCount} /></div>
      <div className="actions"><button type="button" disabled={layerStart === 0} onClick={() => setLayerStart(value => Math.max(0, value - 1))}>Layer -1</button><button type="button" disabled={layerStart >= maxLayer} onClick={() => setLayerStart(value => value + 1)}>Layer +1</button><button type="button" disabled={indexStart === 0} onClick={() => setIndexStart(value => value - 1)}>Index -1</button><button type="button" disabled={indexStart >= Number.MAX_SAFE_INTEGER - indexCount} onClick={() => setIndexStart(value => value + 1)}>Index +1</button></div>
      <div className="trace-scroll" tabIndex={0} role="region" aria-label="Readout grid, scroll horizontally"><table className="trace-grid"><caption>Rows: source layer. Columns: consumed {phase} index. At most 64 x 64 display cells.</caption><thead><tr><th scope="col">Layer</th>{visibleIndices.map(index => <th key={index} scope="col">{phase} {index}</th>)}</tr></thead><tbody>{visibleLayers.map(layer => <tr key={layer}><th scope="row">{layer}</th>{visibleIndices.map(index => {
        const record = cells.get(`${layer}:${index}`);
        return <td key={index}>{record ? <button type="button" aria-pressed={selected === record.seq} aria-label={`Layer ${layer}, ${phase} index ${index}, ${record.score_kind}`} onClick={() => { setSelected(record.seq); if (!sameSite(site, record)) selectSite?.({ phase: record.phase, index: record.index }); }}><span>{record.scores[0]?.label ?? "Inspect scores"}</span><span className="score-value">{record.scores[0]?.score ?? "No scores"}</span></button> : <span className="unavailable-cell">{cellAvailability(records, readoutId, phase, index, layer, complete)}</span>}</td>;
      })}</tr>)}</tbody></table></div></details>
    </>}
    {selection && <ScorePanel record={selection} modelIdentity={savedModel} caps={caps} assets={assets} draft={draft} update={update} report={report} tracked={tracked} track={track} reviewDraft={reviewDraft} />}
  </section>;
}

export function ExecutionViewer({ id, caps, assets, draft, update, report, copyRun, reviewDraft }: { id: string; caps: Capabilities | null; assets: Asset[]; draft: Draft; update: (draft: Draft) => boolean; report: Report; copyRun: (id: string, newSeed: boolean) => void; reviewDraft?: () => void }) {
  const view = useJob(id, report, Math.max(1, Math.min(64, caps?.limits.max_result_page_records ?? 64)));
  const [cancelling, setCancelling] = useState(false);
  const [cancelStatus, setCancelStatus] = useState<Job | null>(null);
  const [inspectSeq, setInspectSeq] = useState(0);
  const [arrayKey, setArrayKey] = useState("");
  const [site, selectSite] = useState<TokenSite | null>(null);
  useEffect(() => { setCancelStatus(null); }, [id]);
  const job = cancelStatus?.id === id && (!view.job || cancelStatus.revision > view.job.revision) ? cancelStatus : view.job;
  const prepared = view.records.find(isPrepared);
  const samples = view.records.filter(isSample).sort((a, b) => a.index - b.index);
  const generated = new TextDecoder().decode(new Uint8Array(samples.flatMap(sample => sample.piece_bytes)));
  async function cancel() {
    if (!job || cancelling || !window.confirm(`Request cancellation of ${id}? Closing the viewer alone never cancels work.`)) return;
    setCancelling(true);
    try { const result = decodeJob(await createLensApi().cancel(id)); if (result.id !== id) throw new Error("Cancellation returned the wrong job ID"); setCancelStatus(result); }
    catch (error) { report(`Cancel ${id}`, error); }
    finally { setCancelling(false); }
  }
  if (!id) return <p>Select a server job from History or submit an experiment. No job is currently selected.</p>;
  return <div><p className="job-id">{id}</p><div className="actions"><button type="button" onClick={view.reconnect}>Reconnect / reread stored records</button><label className="check"><input type="checkbox" checked={view.paused} onChange={event => view.setPaused(event.target.checked)} />Pause this viewer</label>
    <button type="button" disabled={!job || isTerminal(job) || job.cancel_requested || cancelling} onClick={() => void cancel()}>{cancelling ? "Requesting cancellation..." : "Request job cancellation"}</button></div>
    <p className="muted">Read-only polling and pause never cancel work. Reconnect rereads immutable records from zero with sequence deduplication, without inference.</p>
    {job ? <JobStatus job={job} /> : <p role="status">Waiting for an actual job status. See request errors if retrieval fails.</p>}
    <div className="actions"><button type="button" onClick={() => copyRun(id, false)}>Run again: prepare same seed</button><button type="button" onClick={() => copyRun(id, true)}>Run again: prepare new seed</button></div>
    <div className="explore-shortcuts" aria-label="Explore this run"><a href="#saved-tokens-heading">Tokens & context</a><a href="#trace-heading">Layers & scores</a></div>
    <div className="exploration-layout"><div className="exploration-source">
    <section><h3>Prepared input / exact server record</h3>{prepared ? <>{prepared.prompt_text === undefined ? <p className="notice">This record does not retain prompt text/bytes. Exact token IDs remain available; no local template rendering is substituted.</p> : <details><summary>Exact rendered prompt, including assistant prefill</summary><pre className="generated-output">{prepared.prompt_text}</pre></details>}<details><summary>Exact prepared token IDs and server metadata</summary><pre>{JSON.stringify(prepared, null, 2)}</pre></details></> : <p>No prepared_input record loaded yet.</p>}</section>
    <section><h3>Sampled output</h3><p className="muted">{samples.length} immutable token records loaded. Text is decoded from piece_bytes, not inferred reasoning or confidence. Final unconsumed samples do not acquire invented readouts.</p>
      {samples.length > 0 ? <pre className="generated-output sampled-output">{generated}</pre> : <p>No sampled-token records loaded.</p>}
      <details><summary>Sample IDs and consumption</summary><div className="sample-list">{samples.map(sample => <span key={sample.seq}>#{sample.index} / {sample.token_id} / {sample.consumed ? "consumed" : "unconsumed"}</span>)}</div></details>
    </section>
    <TokenNavigator records={view.records} site={site} select={selectSite} complete={view.complete} job={job} />
    </div><div className="exploration-measurements">
    <TraceViewer key={id} records={view.records} complete={view.complete} caps={caps} assets={assets} draft={draft} update={update} report={report} selectArray={setArrayKey} site={site} selectSite={selectSite} reviewDraft={reviewDraft} />
    <details className="saved-measurements"><summary>Before/after measurements & retained arrays{site ? ` / ${site.phase} ${site.index}` : ""}</summary><RetentionViewer records={view.records} complete={view.complete} selectedKey={arrayKey} select={setArrayKey} report={report} site={site} selectSite={selectSite} /></details>
    </div></div>
    <details><summary>Inspect loaded records ({view.records.length}) / {view.complete ? "publication complete" : "publication not complete"}</summary><p className="muted">Operation applications, terminal state, unknown record kinds and all native metadata are preserved. One record is rendered at a time.</p><NumberField label="Exact record sequence" value={inspectSeq} integer min={0} onChange={setInspectSeq} /><pre>{JSON.stringify(view.records.find(record => record.seq === inspectSeq) ?? "Sequence not loaded", null, 2)}</pre></details>
  </div>;
}
