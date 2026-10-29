import { mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { buildEvidenceBundle } from "./evidence";
import { encodeBundle } from "./validate";

const outdir = resolve(import.meta.dir, "dist");
const bundle = await buildEvidenceBundle(resolve(import.meta.dir, "../.."));
await rm(outdir, { recursive: true, force: true });
await mkdir(outdir, { recursive: true });
const result = await Bun.build({
  entrypoints: [resolve(import.meta.dir, "index.html")],
  outdir,
  target: "browser",
  minify: true,
  naming: { entry: "[name].[ext]", chunk: "assets/[name]-[hash].[ext]", asset: "assets/[name]-[hash].[ext]" },
});
if (!result.success) throw new AggregateError(result.logs, "Microscope build failed");
await Bun.write(resolve(outdir, "evidence.json"), encodeBundle(bundle));
console.log(`Built ${bundle.investigations.length} investigations and ${bundle.snapshots.length} pinned source excerpts.`);
