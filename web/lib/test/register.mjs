// Lets `node --test` run lib/ TypeScript without extra deps (Node >= 22.18 strips types natively).
// Adds what Next's bundler normally provides:
//   - extensionless relative imports ("./types" -> "./types.ts")
//   - JSON imports without `with { type: "json" }` (loaded as an ES module default export)
// Usage: node --import ./lib/test/register.mjs --test "lib/**/*.test.ts"   (see `pnpm test`)
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

const HAS_EXT = /\.(?:[cm]?[jt]sx?|json)$/;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if ((specifier.startsWith("./") || specifier.startsWith("../")) && !HAS_EXT.test(specifier)) {
      for (const suffix of [".ts", ".tsx", "/index.ts"]) {
        try {
          return nextResolve(specifier + suffix, context);
        } catch {
          // try the next candidate
        }
      }
    }
    return nextResolve(specifier, context);
  },
  load(url, context, nextLoad) {
    if (url.startsWith("file:") && url.endsWith(".json")) {
      const source = `export default ${readFileSync(fileURLToPath(url), "utf8")};`;
      return { format: "module", source, shortCircuit: true };
    }
    return nextLoad(url, context);
  },
});
