// Leverage map #4: does the stock @ai-sdk/open-responses provider carry an
// EMPTY reasoning item (what serve emits when a thinking turn closes its
// block immediately) through to the next request?
//
// Provider level only (LanguageModel doGenerate/doStream), no `ai` package:
// the provider decides (1) whether an empty reasoning item becomes a
// reasoning part, non-streamed and streamed with serve's exact event
// sequence, and (2) what a replayed reasoning part looks like on the wire,
// with and without the provider metadata the client is expected to keep.
// Whether a given client keeps the part is outside this script.
//
// Run (no install needed when a client already ships the provider):
//   OPEN_RESPONSES_MODULE=~/.cache/opencode/node_modules/@ai-sdk/open-responses/dist/index.js \
//     bun run empty_reasoning.ts > empty_reasoning_capture.json

const modulePath = process.env.OPEN_RESPONSES_MODULE ?? "@ai-sdk/open-responses";
const { createOpenResponses, VERSION } = await import(modulePath);

const NAME = "qwen-serve";
const MODEL = "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004";
const bodies: { phase: string; body: unknown }[] = [];

function envelope(output: unknown[], id: string) {
  return {
    id,
    object: "response",
    created_at: 1759700000,
    completed_at: 1759700001,
    model: MODEL,
    status: "completed",
    output,
    usage: { input_tokens: 10, output_tokens: 3, total_tokens: 13 },
    incomplete_details: null,
    error: null,
  };
}

// Serve's empty reasoning item (events.rs reasoning_item_json with no text).
const emptyReasoning = {
  id: "rs_e1",
  type: "reasoning",
  status: "completed",
  summary: [],
  content: [{ type: "reasoning_text", text: "" }],
};
const message = {
  id: "msg_e1",
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: "Hi.", annotations: [] }],
};

// Serve's SSE for an immediately closed think block (open_reasoning then
// close_reasoning: no delta events), then the visible message.
function sseBody(): string {
  const events: [string, Record<string, unknown>][] = [
    ["response.created", { response: { ...envelope([], "resp_s1"), status: "in_progress" } }],
    ["response.output_item.added", { output_index: 0, item: { id: "rs_e1", type: "reasoning", status: "in_progress", summary: [], content: [] } }],
    ["response.content_part.added", { item_id: "rs_e1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "" } }],
    ["response.reasoning.done", { item_id: "rs_e1", output_index: 0, content_index: 0, text: "" }],
    ["response.content_part.done", { item_id: "rs_e1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "" } }],
    ["response.output_item.done", { output_index: 0, item: emptyReasoning }],
    ["response.output_item.added", { output_index: 1, item: { id: "msg_e1", type: "message", role: "assistant", status: "in_progress", content: [] } }],
    ["response.content_part.added", { item_id: "msg_e1", output_index: 1, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }],
    ["response.output_text.delta", { item_id: "msg_e1", output_index: 1, content_index: 0, delta: "Hi." }],
    ["response.output_text.done", { item_id: "msg_e1", output_index: 1, content_index: 0, text: "Hi." }],
    ["response.content_part.done", { item_id: "msg_e1", output_index: 1, content_index: 0, part: { type: "output_text", text: "Hi.", annotations: [] } }],
    ["response.output_item.done", { output_index: 1, item: message }],
    ["response.completed", { response: envelope([emptyReasoning, message], "resp_s1") }],
  ];
  return (
    events
      .map(([type, payload], i) => `event: ${type}\ndata: ${JSON.stringify({ type, sequence_number: i + 1, ...payload })}`)
      .join("\n\n") + "\n\ndata: [DONE]\n\n"
  );
}

function model(phase: string, respond: () => Response) {
  const provider = createOpenResponses({
    name: NAME,
    url: "http://mock/v1/responses",
    fetch: (async (_input: unknown, init?: RequestInit) => {
      bodies.push({ phase, body: init?.body ? JSON.parse(String(init.body)) : null });
      return respond();
    }) as typeof fetch,
  });
  return provider(MODEL);
}

const json = (payload: unknown) =>
  new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } });
const sse = () =>
  new Response(sseBody(), { status: 200, headers: { "content-type": "text/event-stream" } });

const user = (text: string) => ({ role: "user", content: [{ type: "text", text }] });
const options = (prompt: unknown[]) => ({ prompt });

// (1a) Non-streamed: the reasoning part the provider returns.
const generated = await model("generate", () => json(envelope([emptyReasoning, message], "resp_g1"))).doGenerate(
  options([user("Say hi.")]),
);
const generatedReasoning = generated.content.filter((p: { type: string }) => p.type === "reasoning");

// (1b) Streamed with serve's exact events: the stream parts.
const streamed = await model("stream", sse).doStream(options([user("Say hi.")]));
const streamParts: { type: string; id?: string; delta?: string; providerMetadata?: unknown }[] = [];
const reader = streamed.stream.getReader();
for (;;) {
  const { done, value } = await reader.read();
  if (done) break;
  if (String(value.type).startsWith("reasoning")) streamParts.push(value);
}
const reasoningEnd = streamParts.find((p) => p.type === "reasoning-end");

// (2) Replays: what the next request carries for that turn.
async function replay(phase: string, reasoningPart: Record<string, unknown>) {
  await model(phase, () => json(envelope([message], "resp_r"))).doGenerate(
    options([
      user("Say hi."),
      { role: "assistant", content: [reasoningPart, { type: "text", text: "Hi." }] },
      user("Again."),
    ]),
  );
  const body = bodies.filter((b) => b.phase === phase).at(-1)?.body as { input: unknown[] };
  return body.input;
}
const withMetadata = await replay("replay-with-metadata", {
  type: "reasoning",
  text: "",
  providerOptions: reasoningEnd?.providerMetadata,
});
const withoutMetadata = await replay("replay-without-metadata", { type: "reasoning", text: "" });

console.log(
  JSON.stringify(
    {
      fixture: "empty_reasoning_capture_v1",
      provider: `@ai-sdk/open-responses@${VERSION}`,
      module: modulePath.replace(/^\/Users\/[^/]+/, "~"),
      captured_at: new Date().toISOString(),
      generate_reasoning_parts: generatedReasoning,
      stream_reasoning_parts: streamParts,
      replay_with_metadata_input: withMetadata,
      replay_without_metadata_input: withoutMetadata,
    },
    null,
    2,
  ),
);
