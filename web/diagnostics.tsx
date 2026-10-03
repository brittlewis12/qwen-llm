import { useState } from "react";
import { decodeScope, type Asset, type Capabilities, type Scope, type Selector } from "./contract";
import { isObject, layerValues, moveRow, type Draft } from "./draft";
import { JsonDocument, NumberField } from "./editor";

export const actionNames: Record<string, string> = { fixed_add: "Fixed add", residual_l2_fraction: "Residual L2 add", projection_ablate: "Projection ablate", source_to_target: "Source to target", coordinate_swap: "Coordinate swap" };

function SelectorEditor({ label, value, change, available, optional = false }: { label: string; value?: Selector; change: (value: Selector | undefined) => void; available?: number[]; optional?: boolean }) {
  const [error, setError] = useState("");
  return <div className="selector-editor"><label>{label}<select value={value?.kind ?? "none"} onChange={event => {
    setError("");
    change(event.target.value === "none" ? undefined : event.target.value === "all" ? { kind: "all" } : event.target.value === "range" ? { kind: "range", start: available?.[0] ?? 0, end: available?.[0] ?? 0 } : { kind: "values", values: available?.slice(0, 1) ?? [0] });
  }}>{optional && <option value="none">None / omit this phase</option>}<option value="all">All</option><option value="values">Exact set</option><option value="range">Inclusive range</option></select></label>
    {value?.kind === "values" && <>
      <label>Exact indices (comma-separated)<input key={JSON.stringify(value.values)} defaultValue={value.values.join(", ")} aria-invalid={!!error} onBlur={event => {
        try { const values = layerValues(event.target.value); change({ kind: "values", values }); setError(""); } catch (cause) { setError(String(cause)); }
      }} /></label>
      {available && <div className="layer-checks" aria-label={`${label} choices`}>{available.map(layer => <label className="check" key={layer}><input type="checkbox" checked={value.values.includes(layer)} onChange={event => change({ kind: "values", values: event.target.checked ? [...value.values, layer].sort((a, b) => a - b) : value.values.filter(item => item !== layer) })} />{layer}</label>)}</div>}
    </>}
    {value?.kind === "range" && <div className="field-grid"><NumberField label={`${label} start`} value={value.start} min={0} integer onChange={start => change({ ...value, start })} /><NumberField label={`${label} end (inclusive)`} value={value.end} min={value.start} integer onChange={end => change({ ...value, end })} /></div>}
    {error && <p role="alert" className="field-error">{error} Saved selector unchanged.</p>}
  </div>;
}

export function ScopeEditor({ scope, change, layers }: { scope: Scope; change: (scope: Scope) => void; layers: number[] }) {
  return <div className="scope-editor"><SelectorEditor label="Source layers" value={scope.layers} available={layers} change={value => { if (value) change({ ...scope, layers: value }); }} />
    <div className="field-grid">{(["prefill", "decode"] as const).map(phase => <SelectorEditor key={phase} label={`${phase} indices (zero-based)`} optional value={scope[phase] ?? undefined} change={value => {
      const next = { ...scope }; if (value) next[phase] = value; else delete next[phase]; change(next);
    }} />)}</div>
  </div>;
}

function editableScope(value: unknown): Scope | null { try { return decodeScope(value); } catch { return null; } }

export function DiagnosticsEditor({ draft, update, caps, assets }: { draft: Draft; update: (draft: Draft) => void; caps: Capabilities; assets: Asset[] }) {
  const [alias, setAlias] = useState("");
  const [tokenId, setTokenId] = useState(0);
  const [normalization, setNormalization] = useState("unit_l2");
  const asset = assets.find(item => item.alias === alias && item.available);
  const knownDirections = draft.directions.filter(isObject).filter(item => typeof item.id === "string");
  const layers = Array.from({ length: caps.model.layers ?? 0 }, (_, index) => index);
  const baseScope = (sourceLayers: number[]): Scope => ({ layers: { kind: "values", values: sourceLayers.slice(0, 1) }, prefill: { kind: "all" }, decode: { kind: "all" } });
  return <fieldset><legend>Diagnostics / authored order</legend>
    <p className="muted">Server limits: {caps.limits.max_directions} directions, {caps.limits.max_operations} operations, {caps.limits.max_readouts} readouts, top k {caps.limits.max_top_k}. Array order is operation order.</p>
    {caps.limits.max_head_evaluations !== undefined && <p className="muted">Aggregate per-job budget: {caps.limits.max_head_evaluations} unique position/layer/alias heads, {caps.limits.max_readout_rows} published rows and {caps.limits.max_readout_scores} scores. Overlapping readouts of the same alias share head work. Lower top k reduces stored scores, not full-vocabulary head computation.</p>}
    {caps.limits.max_operation_applications !== undefined && <p className="muted">Intervention budget: {caps.limits.max_operation_applications} event/layer applications, {caps.limits.max_direction_rows} prepared direction rows and {caps.limits.max_projection_products} projection products. Zero controls are validated, then omitted from preparation and effective applications.</p>}
    <details><summary>Pinned directions ({draft.directions.length})</summary>
      {draft.directions.map((direction, index) => <div className="operation-row" key={index}><p>{isObject(direction) ? `${String(direction.id)} / ${String(direction.lens)}` : `Opaque direction ${index + 1}`}</p>
        <JsonDocument label={`Direction ${index + 1} JSON`} value={direction} onChange={value => update({ ...draft, directions: draft.directions.map((item, i) => i === index ? value : item) })} />
        <button type="button" onClick={() => update({ ...draft, directions: draft.directions.filter((_, i) => i !== index) })}>Remove direction {index + 1}</button></div>)}
      <div className="field-grid"><label>Registered alias<select value={alias} onChange={event => setAlias(event.target.value)}><option value="">Select alias</option>{assets.map(item => <option key={item.alias} value={item.alias} disabled={!item.available || !item.direction_rows.includes("token_id")}>{item.alias}{!item.available ? ` / ${item.unavailable_reason}` : ""}</option>)}</select></label>
        <NumberField label="Token ID to pin" value={tokenId} integer min={0} max={caps.model.vocabulary_size === null ? undefined : caps.model.vocabulary_size - 1} onChange={setTokenId} />
        <label>Direction normalization<select value={normalization} onChange={event => setNormalization(event.target.value)}><option value="unit_l2">unit_l2</option><option value="as_stored">as_stored</option></select></label></div>
      <button type="button" disabled={!asset || draft.directions.length >= caps.limits.max_directions} onClick={() => update({ ...draft, directions: [...draft.directions, { id: `d-${crypto.randomUUID()}`, lens: alias, row: { kind: "token_id", token_id: tokenId }, normalization }] })}>Pin direction</button>
    </details>
    <h3>Operations</h3>
    {draft.operations.map((row, index) => {
      const document = isObject(row.document) ? row.document : null;
      const action = document && isObject(document.action) ? document.action : null;
      const scope = document ? editableScope(document.scope) : null;
      const kind = String(action?.kind);
      const supported = action && actionNames[kind] && caps.operations.includes(kind);
      const updateDocument = (patch: Record<string, unknown>) => update({ ...draft, operations: draft.operations.map(item => item.key === row.key ? { ...row, document: { ...document, ...patch } } : item) });
      return <div className="operation-row" key={row.key}><div className="row-heading"><label className="check"><input type="checkbox" checked={row.enabled} onChange={event => update({ ...draft, operations: draft.operations.map(item => item.key === row.key ? { ...row, enabled: event.target.checked } : item) })} />{index + 1}. {actionNames[kind] ?? `Opaque: ${kind}`}</label>
        <div className="actions"><button type="button" disabled={index === 0} aria-label={`Move operation ${index + 1} up`} onClick={() => update({ ...draft, operations: moveRow(draft.operations, index, -1) })}>Up</button><button type="button" disabled={index === draft.operations.length - 1} aria-label={`Move operation ${index + 1} down`} onClick={() => update({ ...draft, operations: moveRow(draft.operations, index, 1) })}>Down</button>
          <button type="button" onClick={() => update({ ...draft, operations: draft.operations.filter(item => item.key !== row.key) })}>Remove</button></div></div>
        {!supported && <p className="notice">Unsupported or unknown operation retained verbatim. Disable it to run without it; no fields are silently discarded.</p>}
        {supported && <div className="field-grid">{(["source_to_target", "coordinate_swap"].includes(kind) ? ["source", "target"] : ["direction"]).map(field => <label key={field}>{field}<select value={String(action[field] ?? "")} onChange={event => updateDocument({ action: { ...action, [field]: event.target.value } })}><option value="">Select direction</option>{knownDirections.map(direction => <option key={String(direction.id)} value={String(direction.id)}>{String(direction.id)} / {String(direction.lens)}</option>)}</select></label>)}
          <NumberField label={`${actionNames[kind]} coefficient (native units)`} value={Number(action.coefficient)} onChange={coefficient => updateDocument({ action: { ...action, coefficient } })} /></div>}
        {kind === "residual_l2_fraction" && <p className="muted">Coefficient is a fraction of residual L2 norm, not an absolute vector magnitude.</p>}
        {scope && <details><summary>Operation capture scope / not display bounds</summary><ScopeEditor scope={scope} layers={layers} change={scope => updateDocument({ scope })} /></details>}
        <details><summary>Exact operation JSON</summary><JsonDocument label={`Operation ${index + 1} document`} value={row.document} onChange={value => update({ ...draft, operations: draft.operations.map(item => item.key === row.key ? { ...row, document: value } : item) })} /></details>
      </div>;
    })}
    <div className="actions">{caps.operations.filter(kind => !!actionNames[kind]).map(kind => <button key={kind} type="button" disabled={draft.operations.filter(row => row.enabled).length >= caps.limits.max_operations} onClick={() => {
      const first = knownDirections[0];
      const source = assets.find(item => item.alias === first?.lens);
      const id = crypto.randomUUID();
      update({ ...draft, operations: [...draft.operations, { key: id, enabled: true, document: { id: `op-${id}`, scope: baseScope(source?.source_layers ?? layers), action: { kind, ...(["source_to_target", "coordinate_swap"].includes(kind) ? { source: first?.id ?? "", target: knownDirections[1]?.id ?? "" } : { direction: first?.id ?? "" }), coefficient: kind === "projection_ablate" ? 1 : .25 } } }] });
    }}>Add {actionNames[kind]}</button>)}</div>
    <h3>Readouts</h3>
    {draft.readouts.map((raw, index) => {
      const value = isObject(raw) ? raw : null;
      const selected = assets.find(asset => asset.alias === value?.lens);
      const scope = value ? editableScope(value.scope) : null;
      const change = (patch: Record<string, unknown>) => update({ ...draft, readouts: draft.readouts.map((item, i) => i === index ? { ...value, ...patch } : item) });
      return <div className="operation-row" key={index}><h4>Readout {index + 1} / {String(value?.id ?? "opaque")}</h4>
        {value && <><div className="field-grid"><label>Readout alias<select value={String(value.lens)} onChange={event => change({ lens: event.target.value })}>{assets.map(asset => <option key={asset.alias} value={asset.alias} disabled={!asset.available}>{asset.alias}</option>)}</select></label>
          <label>Candidate mode<select value={String(value.mode)} onChange={event => change({ mode: event.target.value })}><option value="">Select mode</option>{selected?.readout_modes.filter(mode => caps.readout_modes.includes(mode)).map(mode => <option key={mode}>{mode}</option>)}</select></label>
          <NumberField label="Readout top k" value={Number(value.top_k)} min={1} max={caps.limits.max_top_k} integer onChange={top_k => change({ top_k })} /></div>
          <p className="muted">{value.mode === "selected" ? "Selected uses the registered candidate bank, not the full vocabulary." : "Full-vocabulary scores retain the backend's native score units; they are not confidence."}</p>
          {caps.readout_retention_modes?.includes("scores_and_residual") && <><label className="check"><input type="checkbox" checked={value.retain === "scores_and_residual"} onChange={event => change({ retain: event.target.checked ? "scores_and_residual" : null })} />Retain full scores and source residuals</label>
            {value.retain === "scores_and_residual" && <p className="notice">Each selected layer/position/head retains {(4 * (caps.model.vocabulary_size ?? 0)).toLocaleString()} score bytes, plus {(4 * (caps.model.hidden_size ?? 0)).toLocaleString()} residual bytes shared across heads. Raw archive limit: {caps.limits.max_archive_bytes?.toLocaleString()} bytes/job. All-prefill coverage depends on server tokenization. Exact deduplicated bytes are checked before acceptance; excess is rejected, never clipped. JSON metadata uses the separate result budget.</p>}</>}
          {selected?.target_layer !== null && selected?.target_layer !== undefined && <p className="notice">Fitted target layer {selected.target_layer}; not the final generation distribution. Binding: {selected.transfer}.</p>}</>}
        {scope && <ScopeEditor scope={scope} change={scope => change({ scope })} layers={selected?.source_layers ?? layers} />}
        <details><summary>Exact readout JSON</summary><JsonDocument label={`Readout ${index + 1} document`} value={raw} onChange={value => update({ ...draft, readouts: draft.readouts.map((item, i) => i === index ? value : item) })} /></details>
        <button type="button" onClick={() => update({ ...draft, readouts: draft.readouts.filter((_, i) => i !== index) })}>Remove readout {index + 1}</button>
      </div>;
    })}
    <button type="button" disabled={!assets.some(asset => asset.available && asset.readout_modes.some(mode => caps.readout_modes.includes(mode))) || draft.readouts.length >= caps.limits.max_readouts} onClick={() => {
      const asset = assets.find(asset => asset.available && asset.readout_modes.some(mode => caps.readout_modes.includes(mode)))!;
      update({ ...draft, readouts: [...draft.readouts, { id: `r-${crypto.randomUUID()}`, lens: asset.alias, mode: asset.readout_modes.find(mode => caps.readout_modes.includes(mode)), scope: baseScope(asset.source_layers), top_k: Math.min(8, caps.limits.max_top_k) }] });
    }}>Add readout</button>
    <h3>Before / after residual measurements</h3>
    <p className="muted">Capture the combined effect of all ordered operations at selected sites. These scopes are independent of readouts and operation scopes; zero/no-operation controls are allowed. Earlier interventions may already have affected the before state.</p>
    {draft.residualPairs.map((raw, index) => {
      const value = isObject(raw) ? raw : null;
      const scope = value ? editableScope(value.scope) : null;
      const replace = (value: unknown) => update({ ...draft, residualPairs: draft.residualPairs.map((old, i) => i === index ? value : old) });
      return <div className="operation-row" key={index}><h4>Residual pair {index + 1} / {String(value?.id ?? "opaque")}</h4>
        {scope && <ScopeEditor scope={scope} layers={layers} change={scope => replace({ ...value, scope })} />}
        <JsonDocument label={`Residual pair ${index + 1} JSON`} value={raw} onChange={replace} />
        <button type="button" onClick={() => update({ ...draft, residualPairs: draft.residualPairs.filter((_, i) => i !== index) })}>Remove residual pair {index + 1}</button></div>;
    })}
    <button type="button" disabled={!caps.residual_pair_capture || draft.residualPairs.length >= (caps.limits.max_residual_pairs ?? 0)} onClick={() => {
      const first = draft.operations.find(row => row.enabled && isObject(row.document) && editableScope(row.document.scope));
      const scope = first && isObject(first.document) ? structuredClone(first.document.scope) : { layers: { kind: "values", values: [0] }, prefill: { kind: "values", values: [0] } };
      update({ ...draft, residualPairs: [...draft.residualPairs, { id: `pair-${crypto.randomUUID()}`, scope }] });
    }}>Add residual pair capture</button>
    {caps.residual_pair_capture && <p className="muted">Up to {(8 * (caps.model.hidden_size ?? 0)).toLocaleString()} raw bytes per unique pair site, less when its after residual is already retained by a readout. Shares the {caps.limits.max_archive_bytes?.toLocaleString()}-byte archive budget; {caps.limits.max_residual_pair_rows} maximum pair records. Adding a scope copies the first enabled operation's current scope when available; later edits remain independent.</p>}
  </fieldset>;
}
