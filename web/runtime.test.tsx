import { expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { decodeJob, isTerminal } from "./contract";
import { JobStatus } from "./viewer";

const saved = await Bun.file(`${import.meta.dir}/../crates/qwen-cli/tests/fixtures/lens_http_v1/status.json`).json();
test("runtime publication failure does not replace durable state or claim completion", () => {
  const runtime = { publication_error: { code: "artifact_writer_stopped", message: "Writer stopped" }, execution_settled: true,
    generation: { stop_reason: "token_limit", counters: { prompt_tokens: 16, consumed_prompt_tokens: 16, sampled_tokens: 3, consumed_generated_tokens: 2 }, error: null } };
  const job = decodeJob({ ...saved, state: "running", runtime });
  expect(isTerminal(job)).toBe(false);
  const html = renderToStaticMarkup(<JobStatus job={job} />);
  expect(html).toContain("Publication failed");
  expect(html).toContain("Last confirmed durable snapshot");
  expect(html).toContain("not a durability acknowledgment");
  expect(() => decodeJob({ ...saved, runtime: { ...runtime, execution_settled: "yes" } })).toThrow("runtime");
  expect(() => decodeJob({ ...saved, runtime: { ...runtime, generation: { ...runtime.generation, stop_reason: "unknown" } } })).toThrow("runtime.generation");
  expect(() => decodeJob({ ...saved, runtime: { ...runtime, publication_error: null } })).toThrow("runtime.publication_error");
  expect(decodeJob(saved).runtime).toBeUndefined();
});
