import { useEffect, useId, useState } from "react";
import { moveRow, type Draft, type Message } from "./draft";
import type { Asset, Capabilities } from "./contract";
import { DiagnosticsEditor } from "./diagnostics";

export function NumberField({ label, value, onChange, integer = false, min, max }: {
  label: string; value: number; onChange: (value: number) => void; integer?: boolean; min?: number; max?: number;
}) {
  const id = useId();
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const parsed = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(parsed) && (!integer || Number.isSafeInteger(parsed))
    && (min === undefined || parsed >= min) && (max === undefined || parsed <= max);
  return <div className="field"><label htmlFor={id}>{label}</label>
    <input id={id} type="number" inputMode={integer ? "numeric" : "decimal"} value={text} step={integer ? 1 : "any"} min={min} max={max} required
      aria-invalid={!valid} onChange={event => {
        setText(event.target.value);
        const number = event.target.valueAsNumber;
        if (event.target.validity.valid && Number.isFinite(number) && (!integer || Number.isSafeInteger(number))) onChange(number);
      }} />
    {!valid && <p className="field-error">Invalid value. Saved value remains {value}.</p>}
  </div>;
}

export function JsonDocument({ value, onChange, label }: { value: unknown; onChange: (value: unknown) => void; label: string }) {
  const [text, setText] = useState(JSON.stringify(value, null, 2));
  const [error, setError] = useState("");
  const id = useId();
  useEffect(() => { setText(JSON.stringify(value, null, 2)); setError(""); }, [value]);
  const dirty = text !== JSON.stringify(value, null, 2);
  return <div><label htmlFor={id}>{label}</label><textarea id={id} className="code" rows={6} value={text} aria-invalid={!!error || dirty} onChange={event => {
    setText(event.target.value);
    try { JSON.parse(event.target.value); setError(""); } catch (cause) { setError(String(cause)); }
  }} />
    <button type="button" onClick={() => { try { onChange(JSON.parse(text)); setError(""); } catch (cause) { setError(String(cause)); } }}>Apply JSON to draft</button>
    {dirty && <p className="notice">Unapplied JSON changes block submission. Apply them or restore the saved document.</p>}
    <button type="button" disabled={!dirty} onClick={() => { setText(JSON.stringify(value, null, 2)); setError(""); }}>Restore saved document</button>
    {error && <pre className="field-error" role="alert">{error}</pre>}
  </div>;
}

export function InputEditor({ draft, update, caps, assets, errors, busy, onRun }: {
  draft: Draft; update: (draft: Draft) => void; caps: Capabilities | null; assets: Asset[];
  errors: string[]; busy: boolean; onRun: () => void;
}) {
  const [formError, setFormError] = useState("");
  function setMessage(index: number, patch: Partial<Message>) {
    update({ ...draft, messages: draft.messages.map((message, position) => position === index ? { ...message, ...patch } : message) });
  }
  return <section aria-labelledby="input-heading" className="panel">
    <p className="eyebrow">01 / Preparation</p><h2 id="input-heading">Compose an experiment.</h2>
    <p className="muted">Draft edits persist locally. A run snapshots its exact body and idempotency key before any network request.</p>
    <form onSubmit={event => {
      event.preventDefault();
      if (event.currentTarget.querySelector('[aria-invalid="true"]')) { setFormError("Resolve invalid or unapplied fields before running."); return; }
      setFormError(""); onRun();
    }}>
      <fieldset><legend>Messages</legend><p className="muted">Optional system first; user/assistant history in order. The server validates model-specific conversation and channel rules.</p>
        {draft.messages.map((message, index) => <div className="message-row" key={index}>
          <div className="row-heading"><label htmlFor={`role-${index}`}>Message {index + 1}</label><div className="actions">
            <button type="button" aria-label={`Move message ${index + 1} up`} disabled={index === 0} onClick={() => update({ ...draft, messages: moveRow(draft.messages, index, -1) })}>Up</button>
            <button type="button" aria-label={`Move message ${index + 1} down`} disabled={index === draft.messages.length - 1} onClick={() => update({ ...draft, messages: moveRow(draft.messages, index, 1) })}>Down</button>
            <button type="button" aria-label={`Remove message ${index + 1}`} onClick={() => update({ ...draft, messages: draft.messages.filter((_, position) => position !== index) })}>Remove</button>
          </div></div>
          <select id={`role-${index}`} value={message.role} onChange={event => {
            const role = event.target.value as Message["role"];
            if (message.reasoning !== undefined && role !== "assistant") { setFormError("Remove assistant reasoning explicitly before changing this role."); return; }
            setMessage(index, { role });
          }}><option value="system">System</option><option value="user">User</option><option value="assistant">Assistant</option></select>
          <label htmlFor={`message-${index}`} className="sr-only">Message {index + 1} content</label><textarea id={`message-${index}`} value={message.content} required rows={4} placeholder="Write a message..." onChange={event => setMessage(index, { content: event.target.value })} />
          {message.role === "assistant" && <details><summary>Assistant history reasoning (optional)</summary><label>Reasoning text<textarea value={message.reasoning ?? ""} rows={3} onChange={event => setMessage(index, { reasoning: event.target.value })} /></label>
            <button type="button" onClick={() => { const next = { ...message }; delete next.reasoning; update({ ...draft, messages: draft.messages.map((item, i) => i === index ? next : item) }); }}>Omit reasoning field</button></details>}
        </div>)}
        <button type="button" onClick={() => update({ ...draft, messages: [...draft.messages, { role: "user", content: "" }] })}>Add message</button>
      </fieldset>
      <fieldset><legend>Generation</legend>
        <label htmlFor="generation-mode">Native generation mode</label><select id="generation-mode" disabled={!caps} value={draft.generationMode} required onChange={event => update({ ...draft, generationMode: event.target.value })}>
          <option value="">Choose a supported mode</option>{draft.generationMode && !caps?.generation_modes.includes(draft.generationMode) && <option value={draft.generationMode} disabled>{draft.generationMode} / unsupported</option>}{caps?.generation_modes.map(mode => <option key={mode} value={mode}>{mode}</option>)}</select>
        <label className="check"><input type="checkbox" checked={draft.prefix.enabled} onChange={event => update({ ...draft, prefix: { ...draft.prefix, enabled: event.target.checked } })} />Assistant prefill</label>
        {draft.prefix.enabled && <div className="inset"><label htmlFor="prefix-channel">Prefill channel</label><select id="prefix-channel" value={draft.prefix.channel} onChange={event => update({ ...draft, prefix: { ...draft.prefix, channel: event.target.value as "reasoning" | "final" } })}>
          {!caps?.assistant_prefill_channels.includes(draft.prefix.channel) && <option value={draft.prefix.channel} disabled>{draft.prefix.channel} / unsupported</option>}{caps?.assistant_prefill_channels.filter(channel => channel === "reasoning" || channel === "final").map(channel => <option key={channel}>{channel}</option>)}</select>
          <label htmlFor="prefix-text">Exact prefill text</label><textarea id="prefix-text" rows={3} value={draft.prefix.text} onChange={event => update({ ...draft, prefix: { ...draft.prefix, text: event.target.value } })} /><p className="muted">Prompt input, not new output. Mode/channel compatibility is validated by the server.</p></div>}
        <div className="field-grid"><NumberField label="Maximum new tokens" value={draft.generation.max_new_tokens} integer min={1} max={caps?.limits.max_new_tokens} onChange={max_new_tokens => update({ ...draft, generation: { ...draft.generation, max_new_tokens } })} />
          {([ ["temperature", "Temperature", false, undefined], ["top_k", "Top k", true, undefined], ["top_p", "Top p", false, 1], ["min_p", "Min p", false, 1], ["seed", "Seed", true, Number.MAX_SAFE_INTEGER] ] as const).map(([key, label, integer, max]) => <NumberField key={key} label={label} value={draft.generation.sampling[key]} integer={integer} min={0} max={max} onChange={value => update({ ...draft, generation: { ...draft.generation, sampling: { ...draft.generation.sampling, [key]: value } } })} />)}
        </div><button type="button" onClick={() => update({ ...draft, generation: { ...draft.generation, sampling: { ...draft.generation.sampling, seed: crypto.getRandomValues(new Uint32Array(1))[0]! } } })}>New seed for draft</button>
        {caps && <p className="muted">Server: up to {caps.limits.max_new_tokens} new tokens; context {caps.limits.max_context_tokens}. Tokenization and aggregate work admission are server-validated.</p>}
      </fieldset>
      {caps ? <DiagnosticsEditor draft={draft} update={update} caps={caps} assets={assets} /> : <p className="notice">Connect to load supported diagnostics. Existing draft operations remain intact.</p>}
      {errors.length > 0 && <div className="notice"><p>Before running:</p><ul>{errors.map((error, index) => <li key={index}>{error}</li>)}</ul></div>}
      {formError && <p role="alert" className="field-error">{formError}</p>}
      <div className="form-footer"><p className="muted">Each explicit new run uses a new key. Recovery retries the original key and body.</p><button className="primary" type="submit" disabled={busy || errors.length > 0}>{busy ? "Submitting durable request..." : "Run experiment"}</button></div>
    </form>
  </section>;
}
