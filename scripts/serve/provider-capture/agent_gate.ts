// S2 live gate: drive a real multi-turn tool session through the stock
// @ai-sdk/open-responses provider against `qwen serve`, then report
// per-turn checkpoint hits (the gate metric is the server's
// matched_tokens, never "the session completed").

import { createOpenResponses } from "@ai-sdk/open-responses";
import { generateText, tool, stepCountIs } from "ai";
import { z } from "zod";

const BASE = process.env.QWEN_SERVE_URL ?? "http://127.0.0.1:8737/v1/responses";
const MODEL = process.env.QWEN_SERVE_MODEL ?? "Qwen3.6-35B-A3B-UD-Q4_K_S";

const files: Record<string, string[]> = {
  "/tmp": ["report.txt", "notes.md"],
  "/var": ["log"],
};
const contents: Record<string, string> = {
  "/tmp/report.txt": "Q3 revenue: 42 units.",
  "/tmp/notes.md": "Remember to check /var/log.",
};

const logFetch: typeof fetch = async (input, init) => {
  if (process.env.QWEN_TRACE) {
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    console.error("REQ >>", JSON.stringify({ tools: body?.tools?.map((t: any) => t.name), tool_choice: body?.tool_choice, input_types: body?.input?.map((i: any) => i.type + ":" + (i.role ?? "")) }));
  }
  const response = await fetch(input, init);
  if (process.env.QWEN_TRACE) {
    const clone = response.clone();
    const text = await clone.text();
    console.error("RES <<", text.slice(0, 600));
  }
  return response;
};

const provider = createOpenResponses({ name: "qwen-serve", url: BASE, fetch: logFetch });
const model = provider(MODEL);

const tools = {
  fs_list: tool({
    description: "List files in a directory",
    inputSchema: z.object({ path: z.string() }),
    execute: async ({ path }) => ({ entries: files[path] ?? [] }),
  }),
  fs_read: tool({
    description: "Read a text file",
    inputSchema: z.object({ path: z.string() }),
    execute: async ({ path }) => ({ content: contents[path] ?? "" }),
  }),
};

const prompts = [
  "List the files in /tmp using the fs_list tool.",
  "Read /tmp/report.txt with fs_read and tell me the revenue number.",
  "List /var with fs_list.",
  "Read /tmp/notes.md with fs_read and summarize it in one sentence.",
  "List /tmp again with fs_list and count the files.",
];

type TurnRow = {
  turn: number;
  steps: number;
  toolCalls: number;
  text: string;
  finishReason: string;
};

const rows: TurnRow[] = [];
const messages: Parameters<typeof generateText>[0]["messages"] = [];

for (const [index, prompt] of prompts.entries()) {
  messages.push({ role: "user", content: prompt });
  const result = await generateText({
    model,
    tools,
    stopWhen: stepCountIs(4),
    temperature: 0,
    messages,
    providerOptions: { "qwen-serve": {} },
  });
  // response.messages holds only the final step's messages; accumulate
  // across steps so tool calls/results stay in the replayed history.
  for (const step of result.steps) {
    messages.push(...step.response.messages);
  }
  rows.push({
    turn: index + 1,
    steps: result.steps.length,
    toolCalls: result.steps.reduce((n, step) => n + step.toolCalls.length, 0),
    text: result.text.slice(0, 90).replace(/\n/g, " "),
    finishReason: result.finishReason,
  });
  console.log(
    `turn ${index + 1}: steps=${result.steps.length} tool_calls=${rows[index].toolCalls} ` +
      `finish=${result.finishReason} | ${rows[index].text}`,
  );
}

console.log("\n--- summary ---");
console.log(JSON.stringify({ turns: rows.length, rows }, null, 2));
