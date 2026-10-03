import { useEffect, useState } from "react";
import { isPrepared, isSample, type Job } from "./contract";
import { NumberField } from "./editor";
import type { Sequenced } from "./records";
import { sameSite, savedSpanLabels, savedSpanContext, pieceLabel, tokenAt, tokenCoverage, adjacentToken, TOKEN_WINDOW, type TokenSite } from "./token-navigation";

export function TokenNavigator({ records, site, select, complete, job }: { records: Sequenced[]; site: TokenSite | null; select: (site: TokenSite | null) => void; complete: boolean; job?: Job | null }) {
  const [phase, setPhase] = useState<TokenSite["phase"]>("prefill"), [start, setStart] = useState(0);
  const prepared = records.find(isPrepared), samples = records.filter(isSample);
  const count = phase === "prefill" ? prepared?.token_ids.length ?? 0 : samples.reduce((max, row) => Math.max(max, row.index + 1), 0);
  useEffect(() => { if (site) { setPhase(site.phase); setStart(Math.floor(site.index / TOKEN_WINDOW) * TOKEN_WINDOW); } }, [site?.phase, site?.index]);
  const token = site && tokenAt(records, site);
  const labels = site?.phase === "prefill" ? savedSpanLabels(prepared, site.index) : [];
  const context = site?.phase === "prefill" ? savedSpanContext(prepared, site.index) : [];
  const previous = site && adjacentToken(records, site, -1), next = site && adjacentToken(records, site, 1);
  return <section className="token-navigation" aria-label="Saved token navigation"><h3 id="saved-tokens-heading" tabIndex={-1}>Explore saved tokens</h3>
    <p className="muted">Select an input token, then explore its saved layers. Nothing is rerun.</p>
    <div className="field-grid"><label>Token phase<select value={phase} onChange={event => { setPhase(event.target.value as TokenSite["phase"]); setStart(0); }}><option value="prefill">Prepared prompt</option><option value="decode">Generated samples</option></select></label>
      <NumberField label="Token window start" integer min={0} max={Math.max(0, count - 1)} value={start} onChange={setStart} /></div>
    <div className="actions"><button type="button" disabled={start === 0} onClick={() => setStart(n => Math.max(0, n - TOKEN_WINDOW))}>Previous tokens</button><button type="button" disabled={start + TOKEN_WINDOW >= count} onClick={() => setStart(n => n + TOKEN_WINDOW)}>Next tokens</button><button type="button" disabled={!site} onClick={() => select(null)}>Clear token selection</button></div>
    <p className="muted">{count} saved {phase} indices / {TOKEN_WINDOW}-token display window</p>
    <div className="token-strip">{Array.from({ length: Math.min(TOKEN_WINDOW, Math.max(0, count - start)) }, (_, i) => {
      const candidate = { phase, index: start + i }, value = tokenAt(records, candidate);
      return value ? <button type="button" key={`${phase}-${candidate.index}`} aria-pressed={sameSite(site, candidate)} aria-label={`Inspect ${phase} token ${candidate.index}, ID ${value.id}`} onClick={() => select(candidate)}><span>{phase} {candidate.index}</span>{value.bytes && <bdi className="token-piece">{pieceLabel(value.bytes)}</bdi>}<strong>{value.id}</strong>{value.consumed === false && <span>Unconsumed</span>}</button> : <span key={candidate.index}>#{candidate.index}: not loaded</span>;
    })}</div>
    {site && <div className="token-site" aria-live="polite"><div className="row-heading"><h4>Input {site.phase} {site.index}</h4><div className="actions"><button type="button" disabled={!previous} onClick={() => previous && select(previous)}>Previous token</button><button type="button" disabled={!next} onClick={() => next && select(next)}>Next token</button></div></div>
      <p data-field="token-coverage">{tokenCoverage(records, site, complete, job)}</p>
      {context.length > 0 && <><p>Saved span context / bounded excerpts, not decoded text of this token:</p>{context.map((text, i) => <p className="prompt-excerpt" key={i}><bdi>{text}</bdi></p>)}</>}
      <details><summary>Token identity, bytes & template spans</summary><p>Token ID {token?.id ?? "not loaded"} / absolute source position {token?.position ?? "unknown"}. Readouts predict the next position, not this token.</p>
      {site.phase === "prefill" && <><p>Overlapping saved template spans (not exclusive token ownership):</p>{labels.length ? <ul>{labels.slice(0, 8).map((label, i) => <li key={i}><bdi>{label}</bdi></li>)}</ul> : <p>No matching saved span annotations.</p>}{labels.length > 8 && <p>{labels.length - 8} more span annotations in the exact prepared record.</p>}</>}
      {token?.bytes && <><p>Exact sampled piece bytes: {token.bytes.map(byte => byte.toString(16).padStart(2, "0")).join(" ") || "empty"}</p><p className="muted">A token piece can split a UTF-8 character. The continuous sampled output above remains the text authority; no per-token replacement characters are invented.</p></>}
    </details></div>}
  </section>;
}
