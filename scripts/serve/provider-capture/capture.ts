// S2-0: hermetic capture of the stock @ai-sdk/open-responses provider's
// wire behavior (k3 R3 step 0: capture provider traffic BEFORE finalizing
// item validation). No server: a mock fetch logs every request body and
// returns scripted serve-shaped responses, so we observe exactly how the
// provider (a) sends chat + tools requests, (b) replays reasoning items,
// assistant messages, and function_call/function_call_output across turns.
//
// Output: capture artifacts as JSON on stdout > provider_capture_v1.json.

import { createOpenResponses } from "@ai-sdk/open-responses";
import { generateText, streamText, tool, stepCountIs } from "ai";
import { z } from "zod";

type Capture = {
  phase: string;
  call: number;
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
};
const captures: Capture[] = [];

// Serve-shaped envelope (mirrors the conformance-passing S1 emitter).
function envelope(output: unknown[], id: string) {
  return {
    id,
    object: "response",
    created_at: 1755640000,
    completed_at: 1755640001,
    model: "Qwen3.6-35B-A3B-UD-Q4_K_S",
    status: "completed",
    output,
    store: false,
    usage: {
      input_tokens: 100,
      output_tokens: 20,
      total_tokens: 120,
      input_tokens_details: { cached_tokens: 80 },
      output_tokens_details: { reasoning_tokens: 0 },
    },
    incomplete_details: null,
    error: null,
    previous_response_id: null,
    instructions: null,
    tools: [],
    tool_choice: "auto",
    truncation: "disabled",
    parallel_tool_calls: false,
    text: { format: { type: "text" } },
    temperature: 0,
    top_p: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    top_logprobs: 0,
    reasoning: null,
    max_output_tokens: null,
    max_tool_calls: null,
    background: false,
    service_tier: "default",
    metadata: {},
    safety_identifier: null,
    prompt_cache_key: null,
  };
}

const reasoningItem = (id: string, text: string) => ({
  id,
  type: "reasoning",
  status: "completed",
  summary: [],
  content: [{ type: "reasoning_text", text }],
});
const messageItem = (id: string, text: string) => ({
  id,
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text, annotations: [] }],
});
const functionCallItem = (id: string, callId: string) => ({
  id,
  type: "function_call",
  status: "completed",
  call_id: callId,
  name: "fs_list",
  arguments: JSON.stringify({ path: "/tmp" }),
});

function makeFetch(phase: string, scripted: (() => Response)[]) {
  let call = 0;
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((value, key) => {
      headers[key] = key === "authorization" ? "<redacted>" : value;
    });
    captures.push({
      phase,
      call,
      url: String(input),
      method: init?.method ?? "GET",
      headers,
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    const respond = scripted[call];
    call += 1;
    if (!respond) throw new Error(`${phase}: unscripted call ${call - 1}`);
    return respond();
  };
}

const json = (payload: unknown) =>
  new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "content-type": "application/json" },
  });

function sse(events: [string, unknown][]) {
  const body =
    events
      .map(([type, payload]) => `event: ${type}\ndata: ${JSON.stringify(payload)}`)
      .join("\n\n") + "\n\ndata: [DONE]\n\n";
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

async function phaseChatReplay() {
  // Turn 1: reasoning + message. Turn 2: replay via result.response.messages.
  const provider = createOpenResponses({
    name: "qwen-serve",
    url: "http://mock/v1/responses",
    fetch: makeFetch("chat-replay", [
      () =>
        json(
          envelope(
            [reasoningItem("rs_1", "\nplan the answer\n"), messageItem("msg_1", "It is 5.")],
            "resp_1",
          ),
        ),
      () => json(envelope([messageItem("msg_2", "Then 15.")], "resp_2")),
    ]) as typeof fetch,
  });
  const model = provider("Qwen3.6-35B-A3B-UD-Q4_K_S");
  const first = await generateText({
    model,
    system: "You are terse.",
    messages: [{ role: "user", content: "Add 2 and 3." }],
  });
  await generateText({
    model,
    system: "You are terse.",
    messages: [
      { role: "user", content: "Add 2 and 3." },
      ...first.response.messages,
      { role: "user", content: "Now add 10." },
    ],
  });
}

async function phaseToolLoop() {
  const provider = createOpenResponses({
    name: "qwen-serve",
    url: "http://mock/v1/responses",
    fetch: makeFetch("tool-loop", [
      () =>
        json(
          envelope(
            [
              reasoningItem("rs_t1", "\nneed the listing\n"),
              functionCallItem("fc_1", "call_abc123"),
            ],
            "resp_t1",
          ),
        ),
      () => json(envelope([messageItem("msg_t2", "One file: a.txt")], "resp_t2")),
    ]) as typeof fetch,
  });
  await generateText({
    model: provider("Qwen3.6-35B-A3B-UD-Q4_K_S"),
    stopWhen: stepCountIs(3),
    tools: {
      fs_list: tool({
        description: "List directory entries",
        inputSchema: z.object({ path: z.string() }),
        execute: async ({ path }) => ({ entries: ["a.txt"], path }),
      }),
    },
    messages: [{ role: "user", content: "List files in /tmp." }],
  });
}

async function phaseStreaming() {
  const streamedReasoning = reasoningItem("rs_s1", "alias-path");
  const streamEvents: [string, unknown][] = [
    ["response.created", { type: "response.created", sequence_number: 1, response: envelope([], "resp_s1") }],
    ["response.in_progress", { type: "response.in_progress", sequence_number: 2, response: envelope([], "resp_s1") }],
    ["response.output_item.added", { type: "response.output_item.added", sequence_number: 3, output_index: 0, item: { id: "rs_s1", type: "reasoning", status: "in_progress", summary: [], content: [] } }],
    ["response.content_part.added", { type: "response.content_part.added", sequence_number: 4, item_id: "rs_s1", output_index: 0, content_index: 0, part: { type: "reasoning_text", text: "" } }],
    // Distinct payloads prove which compatibility event the provider consumes.
    ["response.reasoning.delta", { type: "response.reasoning.delta", sequence_number: 5, item_id: "rs_s1", output_index: 0, content_index: 0, delta: "canonical-path" }],
    ["response.reasoning_text.delta", { type: "response.reasoning_text.delta", sequence_number: 6, item_id: "rs_s1", output_index: 0, content_index: 0, delta: "alias-path" }],
    ["response.reasoning.done", { type: "response.reasoning.done", sequence_number: 7, item_id: "rs_s1", output_index: 0, content_index: 0, text: "alias-path" }],
    ["response.content_part.done", { type: "response.content_part.done", sequence_number: 8, item_id: "rs_s1", output_index: 0, content_index: 0, part: streamedReasoning.content[0] }],
    ["response.output_item.done", { type: "response.output_item.done", sequence_number: 9, output_index: 0, item: streamedReasoning }],
    ["response.output_item.added", { type: "response.output_item.added", sequence_number: 10, output_index: 1, item: { id: "msg_s1", type: "message", role: "assistant", status: "in_progress", content: [] } }],
    ["response.content_part.added", { type: "response.content_part.added", sequence_number: 11, item_id: "msg_s1", output_index: 1, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }],
    ["response.output_text.delta", { type: "response.output_text.delta", sequence_number: 12, item_id: "msg_s1", output_index: 1, content_index: 0, delta: "OK" }],
    ["response.output_text.done", { type: "response.output_text.done", sequence_number: 13, item_id: "msg_s1", output_index: 1, content_index: 0, text: "OK" }],
    ["response.content_part.done", { type: "response.content_part.done", sequence_number: 14, item_id: "msg_s1", output_index: 1, content_index: 0, part: { type: "output_text", text: "OK", annotations: [] } }],
    ["response.output_item.done", { type: "response.output_item.done", sequence_number: 15, output_index: 1, item: messageItem("msg_s1", "OK") }],
    ["response.completed", { type: "response.completed", sequence_number: 16, response: envelope([streamedReasoning, messageItem("msg_s1", "OK")], "resp_s1") }],
  ];
  const provider = createOpenResponses({
    name: "qwen-serve",
    url: "http://mock/v1/responses",
    fetch: makeFetch("streaming", [() => sse(streamEvents)]) as typeof fetch,
  });
  const result = streamText({
    model: provider("Qwen3.6-35B-A3B-UD-Q4_K_S"),
    prompt: "Say OK.",
  });
  const reasoningDeltas: string[] = [];
  for await (const part of result.fullStream) {
    if (part.type === "reasoning-delta") reasoningDeltas.push(part.text);
  }
  if (reasoningDeltas.length !== 1 || reasoningDeltas[0] !== "alias-path") {
    throw new Error(`streaming: expected only alias-path, got ${JSON.stringify(reasoningDeltas)}`);
  }
}

const failures: string[] = [];
for (const [name, phase] of [
  ["chat-replay", phaseChatReplay],
  ["tool-loop", phaseToolLoop],
  ["streaming", phaseStreaming],
] as const) {
  try {
    await phase();
  } catch (error) {
    failures.push(`${name}: ${error}`);
  }
}

console.log(
  JSON.stringify(
    {
      fixture: "provider_capture_v1",
      provider: "@ai-sdk/open-responses@2.0.29",
      ai_sdk: "ai@7.0.70",
      captured_at: new Date().toISOString(),
      failures,
      captures,
    },
    null,
    2,
  ),
);
