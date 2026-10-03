import { useEffect, useMemo, useRef, useState } from "react";
import { isPrepared, isResidualPair, isRetainedArray, type RetainedArray, type ResidualPair } from "./contract";
import type { Sequenced } from "./records";
import type { Report } from "./jobs";
import { NumberField } from "./editor";
import { decodeFloats, distributionSummary, fetchArray, residualMetrics, tokenMeasurement } from "./retention";
import { sameSite, type TokenSite } from "./token-navigation";

function PairInspector({ pair, before, after, report }: { pair: ResidualPair; before: RetainedArray; after: RetainedArray; report: Report }) {
  const [verified, setVerified] = useState(false), [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  async function verify() {
    if (abort.current) return;
    const controller = new AbortController(); abort.current = controller; setBusy(true);
    try {
      const b = decodeFloats(await fetchArray(before, controller.signal), before.array.length);
      const a = decodeFloats(await fetchArray(after, controller.signal), after.array.length);
      const measured = residualMetrics(b, a);
      for (const field of ["norm_before", "norm_after", "delta_norm", "relative_delta"] as const) {
        const x = measured[field], y = pair.metrics[field];
        if (x === null || y === null ? x !== y : Math.abs(x - y) > 1e-10 * (1 + Math.abs(y))) throw new Error(`Saved pair metric disagrees with arrays: ${field}`);
      }
      if (!controller.signal.aborted) setVerified(true);
    } catch (error) { if (!controller.signal.aborted) report("Verify residual pair", error); }
    finally { if (!controller.signal.aborted) { abort.current = null; setBusy(false); } }
  }
  return <div className="pair-inspector"><p>Layer {pair.source_layer} / {pair.phase} index {pair.index} / consumed token {pair.input_token_id}</p>
    <p>Whole site program: {pair.applied_operation_ids.join(" then ") || "no effective operations"}</p>
    <dl className="metadata"><dt>Before L2 norm</dt><dd>{pair.metrics.norm_before}</dd><dt>After L2 norm</dt><dd>{pair.metrics.norm_after}</dd><dt>Measured change L2 norm</dt><dd data-field="delta-norm">{pair.metrics.delta_norm}</dd><dt>Change / before norm</dt><dd>{pair.metrics.relative_delta ?? "Undefined: before norm is zero"}</dd></dl>
    <p className="muted">Before already includes effects from earlier layers and positions. This measures the combined local program, not separate operation doses or an unmodified baseline.</p>
    <button type="button" disabled={busy || verified} onClick={() => void verify()}>Verify measured change / no inference</button>
    {busy && <p role="status">Reading the two saved arrays...</p>}{verified && <p role="status">Verified from retained before/after arrays.</p>}
  </div>;
}

function ArrayInspector({ record, report }: { record: RetainedArray; report: Report }) {
  const [scores, setScores] = useState<Float32Array | null>(null);
  const [busy, setBusy] = useState(false);
  const [token, setToken] = useState(0);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const summary = useMemo(() => scores ? distributionSummary(scores) : null, [scores]);
  const measurement = scores && summary ? tokenMeasurement(scores, token, summary) : null;
  async function load(download: boolean) {
    if (abort.current) return;
    const controller = new AbortController(); abort.current = controller; setBusy(true);
    try {
      const bytes = await fetchArray(record, controller.signal);
      if (controller.signal.aborted) return;
      if (download) {
        const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
        const link = document.createElement("a"); link.href = url; link.download = `${record.key}.f32le`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } else { setScores(decodeFloats(bytes, record.array.length)); }
    } catch (error) { if (!controller.signal.aborted) report("Read retained measurement", error); }
    finally { if (!controller.signal.aborted) { abort.current = null; setBusy(false); } }
  }
  return <div className="retained-inspector"><p>{record.quantity} / layer {record.source_layer} / {record.phase} index {record.index} / consumed token ID {record.input_token_id}{record.lens ? ` / ${record.lens}` : ""}</p>
    <p className="muted">{record.array.length.toLocaleString()} F32LE values / {record.array.byte_length.toLocaleString()} bytes. {record.quantity === "source_residual_before" ? `Captured before site operations: ${record.site_operation_ids?.join(", ") || "none"}` : `Captured after site operations: ${record.applied_operation_ids.join(", ") || "none"}`}.</p>
    <div className="actions">{record.quantity === "readout_logits" && <button type="button" disabled={busy || !!scores} onClick={() => void load(false)}>Load retained scores / no inference</button>}
      <button type="button" disabled={busy} onClick={() => void load(true)}>Download verified F32LE array</button>
      {scores && <button type="button" onClick={() => setScores(null)}>Release loaded scores</button>}</div>
    {busy && <p role="status">Reading and verifying the stored array...</p>}
    {scores && summary && measurement && <><NumberField label="Retained vocabulary token ID" value={token} integer min={0} max={scores.length - 1} onChange={setToken} />
      <dl className="metadata retained-token"><dt>Logit</dt><dd data-field="score">{measurement.score}</dd><dt>Full-vocabulary rank</dt><dd data-field="rank">{measurement.rank}</dd><dt>Readout probability / temperature 1</dt><dd data-field="probability">{measurement.probability}</dd><dt>Entropy / nats</dt><dd>{summary.entropy_nats}</dd><dt>Log partition</dt><dd>{summary.log_partition}</dd></dl>
      <p className="notice">All vocabulary entries participate, not only retained top-k. Rank is one-based, ties ordered by token ID. This is the readout softmax at temperature 1, not confidence or the configured sampler distribution. Token labels are not reconstructed using a different model.</p></>}
  </div>;
}

export function RetentionViewer({ records, complete, selectedKey, select, report, site, selectSite }: { records: Sequenced[]; complete: boolean; selectedKey: string; select: (key: string) => void; report: Report; site?: TokenSite | null; selectSite?: (site: TokenSite | null) => void }) {
  const [pairSeq, setPairSeq] = useState("");
  const arrays = records.filter(isRetainedArray);
  const prepared = records.find(isPrepared);
  const counts = prepared?.retention_admission as { source_arrays_upper?: number; score_arrays_upper?: number; raw_bytes_upper?: number; before_arrays_upper?: number; pair_rows_upper?: number } | undefined;
  const pairs = records.filter(isResidualPair);
  const visiblePairs = site ? pairs.filter(p => sameSite(p, site)) : pairs;
  const pair = visiblePairs.find(p => String(p.seq) === pairSeq) ?? (site ? visiblePairs[0] : undefined);
  const before = pair && arrays.find(a => a.key === pair.before_key), after = pair && arrays.find(a => a.key === pair.after_key);
  const selected = arrays.find(record => record.key === selectedKey);
  return <section><h3>Retained measurements</h3><p className="muted">Loaded descriptors: {arrays.filter(r => r.quantity === "source_residual").length} source arrays / {counts?.source_arrays_upper ?? "unknown"} planned upper bound; {arrays.filter(r => r.quantity === "readout_logits").length} score arrays / {counts?.score_arrays_upper ?? "unknown"} planned upper bound. {complete ? "Publication ended; early stop, cancellation or errors may leave planned sites uncaptured." : "Publication or pagination is not complete."}</p>
    {!arrays.length ? <p>No retained arrays loaded. Top-k-only records cannot recover omitted scores.</p> : <><label>Stored array<select value={selected?.key ?? ""} onChange={event => select(event.target.value)}><option value="">Select a measurement</option>{arrays.map(record => <option key={record.seq} value={record.key}>{record.quantity} / L{record.source_layer} / {record.phase} {record.index}{record.lens ? ` / ${record.lens}` : ""}</option>)}</select></label>
      {selected && <ArrayInspector key={selected.seq} record={selected} report={report} />}</>}
    <h4>Before / after pairs</h4><p className="muted">{pairs.length} loaded pair records / {counts?.pair_rows_upper ?? "unknown"} planned upper bound; {arrays.filter(a => a.quantity === "source_residual_before").length} before arrays / {counts?.before_arrays_upper ?? "unknown"} planned upper bound. Standalone arrays remain downloadable if pair publication failed.</p>
    {site && <p className="muted">Showing pairs at selected {site.phase} index {site.index}. Clear token selection to browse all sites.</p>}
    {visiblePairs.length > 0 && <label>Measured site<select value={pair ? String(pair.seq) : ""} onChange={event => { setPairSeq(event.target.value); const picked = pairs.find(p => String(p.seq) === event.target.value); if (picked) selectSite?.({ phase: picked.phase, index: picked.index }); }}><option value="">Select a pair</option>{visiblePairs.map(p => <option key={p.seq} value={p.seq}>{p.id} / L{p.source_layer} / {p.phase} {p.index}</option>)}</select></label>}
    {pair && before && after && <PairInspector key={pair.seq} pair={pair} before={before} after={after} report={report} />}
    <p className="muted">Array reads and measurement queries do no model work. Arrays record their before/after stage explicitly; none is a resumable generation checkpoint.</p>
  </section>;
}
