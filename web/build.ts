import { rm } from "node:fs/promises";
import { relative } from "node:path";

const outdir = `${import.meta.dir}/dist`;
await rm(outdir, { recursive: true, force: true });
const result = await Bun.build({
  entrypoints: [`${import.meta.dir}/index.html`],
  outdir,
  target: "browser",
  minify: true,
  publicPath: "/",
  naming: { entry: "[name].[ext]", chunk: "assets/[name]-[hash].[ext]", asset: "assets/[name]-[hash].[ext]" },
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
});
if (!result.success) throw new AggregateError(result.logs, "Frontend build failed");
const files = result.outputs.map(output => ({
  path: relative(outdir, output.path),
  contentType: output.type,
  bytes: output.size,
})).sort((a, b) => a.path.localeCompare(b.path));
await Bun.write(`${outdir}/asset-manifest.json`, JSON.stringify({ version: 1, entry: "index.html", files }, null, 2) + "\n");
console.log(files.map(file => `${file.path} (${file.bytes} bytes)`).join("\n"));
