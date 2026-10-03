import { useState } from "react";
import { isTerminal, type Job, type RequestPreview, type RecoveryReport } from "./contract";

export function RecoveryNotice({ recovery }: { recovery: RecoveryReport }) {
  return <div className="recovery" role="alert"><h3>History recovery required</h3><p>{recovery.unavailable_count} stored jobs could not be recovered. Healthy jobs remain readable. Lens submissions and server deletion are disabled to protect accepted retry identities; ordinary serving is unchanged.</p>
    <p>Usage below excludes unavailable jobs. Inspect server diagnostics, preserve the files and repair them before restarting. Do not delete a damaged directory merely to bypass this refusal.</p>
    <details><summary>Unavailable job IDs (up to 64)</summary><ul>{recovery.unavailable_jobs.map(id => <li key={id}><code>{id}</code></li>)}</ul></details></div>;
}

export function mergeRequestPreviews(previous: ReadonlyMap<string, RequestPreview | null>, incoming?: Record<string, RequestPreview | null>) {
  const next = new Map(previous);
  for (const [id, preview] of Object.entries(incoming ?? {})) {
    if (previous.has(id)) {
      const old = previous.get(id)!;
      if (old === null ? preview !== null : preview === null || old.message_index !== preview.message_index
        || old.message_count !== preview.message_count || old.text !== preview.text || old.truncated !== preview.truncated) {
        throw new Error(`Stored request preview changed for ${id}; previous history was preserved.`);
      }
    }
    next.set(id, preview);
  }
  return next;
}

export function HistoryPreview({ preview }: { preview: RequestPreview | null | undefined }) {
  if (preview === undefined) return <p className="muted">Prompt preview not supplied.</p>;
  if (preview === null) return <p className="muted">Last user message preview unavailable.</p>;
  return <div className="history-preview"><p className="eyebrow">Last user message / {preview.message_index + 1} of {preview.message_count}</p>
    {preview.text === "" ? <p className="muted">Empty user message.</p> : <p className="prompt-excerpt" dir="auto">{preview.text}</p>}
    {preview.truncated && <p className="muted">Excerpt: first 240 Unicode code points. Full input remains in the stored request.</p>}</div>;
}

export function HistoryEntry({ job, preview, open, remove }: { job: Job; preview: RequestPreview | null | undefined; open: (job: Job, slot: "baseline" | "variant") => void; remove?: (id: string) => Promise<void> }) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  async function deletePayloads() {
    if (!remove || !confirm(`Delete prompt, results and arrays for ${job.id}? The server keeps its retry identity so old submissions cannot rerun.`)) return;
    setDeleting(true); setError("");
    try { await remove(job.id); } catch (error) { setError(String(error)); } finally { setDeleting(false); }
  }
  return <li><HistoryPreview preview={preview} /><div className="row-heading"><div><h3>{job.id}</h3>
    <p>{job.deleted ? "Logically deleted / cleanup can be retried" : job.runtime ? `Publication failed / ${job.runtime.execution_settled ? "settled" : "settling"} / saved state ${job.state}` : `${job.state} / generation ${job.generation.state} / observations ${job.observations.state}`}</p>
    <p className="muted">{new Date(job.created_at_ms).toLocaleString()} / revision {job.revision}</p></div>
    <div className="actions"><button type="button" onClick={() => open(job, "baseline")}>Open baseline</button>
      <button type="button" onClick={() => open(job, "variant")}>Open variant</button>
      {remove && <button type="button" disabled={!isTerminal(job) || !!job.runtime || deleting} onClick={() => void deletePayloads()}>{deleting ? "Deleting..." : job.deleted ? "Retry payload cleanup" : "Delete job payloads"}</button>}</div></div>
    {error && <p role="alert">{error} Deletion may have committed; retry cleanup for the same job, not inference.</p>}</li>;
}
