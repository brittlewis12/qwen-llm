import { useEffect, useRef, useState } from "react";
import { createLensApi } from "./api";
import { decodeHistory, decodeJob, decodeResult, isTerminal, type Job, type RequestPreview } from "./contract";
import { mergeRequestPreviews } from "./history";
import { RecordAccumulator, startPolling, type Sequenced } from "./records";

const api = createLensApi();
export type Report = (context: string, error: unknown) => void;

export function useJob(id: string, report: Report, pageLimit: number) {
  const [state, setState] = useState<{ id: string; job: Job | null; records: Sequenced[]; complete: boolean }>({ id, job: null, records: [], complete: false });
  const [paused, setPaused] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const reportRef = useRef(report);
  reportRef.current = report;
  useEffect(() => {
    if (!id || paused) return;
    let active = true;
    let done = false;
    let latest: Job | null = null;
    let complete = false;
    const records = new RecordAccumulator<Sequenced>();
    const observedErrors = new Set<string>();
    setState({ id, job: null, records: [], complete: false });
    const stop = startPolling(async () => {
      if (done) return;
      const job = decodeJob(await api.job(id));
      if (job.id !== id) throw new Error(`Requested job ${id}, received ${job.id}`);
      if (!active) return;
      if (!latest || job.revision >= latest.revision) latest = job;
      for (const [source, error] of [["generation", job.generation.error], ["observations", job.observations.error], ["result publication", job.result.error], ["runtime publication", job.runtime?.publication_error ?? null]] as const) {
        if (error !== null) {
          const key = `${source}:${JSON.stringify(error)}`;
          if (!observedErrors.has(key)) { observedErrors.add(key); reportRef.current(`${id} ${source}`, error); }
        }
      }
      setState({ id, job: latest, records: records.values(), complete });
      let drained = complete || !latest.result.available;
      if (latest.result.available && !complete) {
        // A small per-tick work budget is a display scheduling choice, never a capture limit.
        for (let pageIndex = 0; pageIndex < 4; pageIndex++) {
          const cursor = records.cursor;
          try {
            const page = decodeResult(await api.result(id, cursor, pageLimit));
            if (page.job_id !== id) throw new Error(`Result page belongs to ${page.job_id}, not ${id}`);
            if (!active) return;
            records.append(page.records, cursor, page.next_cursor ?? undefined);
            complete = page.complete;
            drained = complete || page.records.length === 0;
            setState({ id, job: latest, records: records.values(), complete });
            if (drained) break;
          } catch (error) { if (latest.runtime?.execution_settled) done = true; throw error; }
        }
      }
      done = latest.runtime ? latest.runtime.execution_settled && drained : isTerminal(latest) && (complete || !latest.result.available);
    }, error => reportRef.current(`Observe ${id}`, error));
    return () => { active = false; stop(); };
  }, [id, paused, refresh, pageLimit]);
  return { ...(state.id === id ? state : { id, job: null, records: [], complete: false }), paused, setPaused, reconnect: () => setRefresh(value => value + 1) };
}

export function useHistory(report: Report) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [previews, setPreviews] = useState(new Map<string, RequestPreview | null>());
  const previewRef = useRef(previews);
  const [loaded, setLoaded] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const alive = useRef(true);
  const seen = useRef(new Set<string>());
  const reportRef = useRef(report);
  reportRef.current = report;
  const merge = (incoming: Job[]) => setJobs(previous => {
    const map = new Map(previous.map(job => [job.id, job]));
    for (const job of incoming) if (!map.has(job.id) || map.get(job.id)!.revision <= job.revision) map.set(job.id, job);
    return [...map.values()].sort((a, b) => b.created_at_ms - a.created_at_ms || a.id.localeCompare(b.id));
  });
  async function load(next?: string) {
    if (inFlight.current) return;
    inFlight.current = true;
    if (alive.current) setBusy(true);
    try {
      const page = decodeHistory(await api.jobs(next));
      if (next !== undefined && page.next_cursor !== null && (page.next_cursor === next || seen.current.has(page.next_cursor))) throw new Error("History cursor did not advance.");
      if (!alive.current) return;
      const nextPreviews = mergeRequestPreviews(previewRef.current, page.request_previews);
      previewRef.current = nextPreviews; setPreviews(nextPreviews);
      merge(page.jobs); setLoaded(true);
      if (next !== undefined) seen.current.add(next);
      // Refresh the head without invalidating an in-progress traversal of older pages.
      if (next !== undefined || seen.current.size === 0) setCursor(page.next_cursor);
    } catch (error) { if (alive.current) reportRef.current("Read server history", error); }
    finally { inFlight.current = false; if (alive.current) setBusy(false); }
  }
  useEffect(() => {
    alive.current = true;
    const stop = startPolling(() => load(), error => reportRef.current("History polling", error), 3000);
    return () => { alive.current = false; stop(); };
  }, []);
  return { jobs, previews, loaded, busy, cursor, reload: () => load(), more: () => cursor === null ? Promise.resolve() : load(cursor), merge };
}
