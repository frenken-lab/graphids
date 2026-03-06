/**
 * Declarative Mosaic spec renderer.
 * Loads a JSON spec file and renders it via @uwdata/mosaic-spec.
 *
 * Usage in OJS cells:
 *   import { renderSpec } from "../_ojs/mosaic-renderer.js"
 *   renderSpec(await FileAttachment("figures/fig-cka-heatmap.json").url())
 */

import { parseSpec, astToDOM } from "https://cdn.jsdelivr.net/npm/@uwdata/mosaic-spec@0.21.1/+esm";
import { coordinator, wasmConnector } from "https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm";

let initialized = false;

function ensureInit() {
  if (!initialized) {
    coordinator().databaseConnector(wasmConnector());
    initialized = true;
  }
}

/**
 * Load a JSON spec file and render it into a DOM element.
 * Data file paths in the spec are resolved as absolute URLs relative to the
 * site root (e.g., "data/foo.parquet" -> "http://host/data/foo.parquet").
 * @param {string} specUrl - URL to the JSON spec file (from FileAttachment.url())
 * @returns {Promise<HTMLElement>}
 */
export async function renderSpec(specUrl) {
  ensureInit();
  const spec = await fetch(specUrl).then(r => r.json());

  // Resolve data file paths to absolute URLs from the site root.
  // Spec files use paths like "data/foo.parquet" which should resolve from site root.
  const siteRoot = new URL("/", window.location.href).href;
  if (spec.data) {
    for (const [key, def] of Object.entries(spec.data)) {
      if (def.file) {
        def.file = new URL(def.file, siteRoot).href;
      }
    }
  }

  const ast = parseSpec(spec);
  const el = await astToDOM(ast);
  return el.element;
}
