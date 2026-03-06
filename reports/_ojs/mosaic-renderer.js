/**
 * Declarative Mosaic spec renderer.
 * Loads a JSON or YAML spec file and renders it via @uwdata/mosaic-spec.
 *
 * Features:
 *   - YAML and JSON spec support
 *   - skipDataLoad: delete spec.data when tables are pre-loaded
 *   - _palette: resolve named palettes from _theme.yaml
 *   - Error boundary: renders red box on failure instead of crashing
 *
 * Usage in OJS cells:
 *   import { renderSpec } from "./_ojs/mosaic-renderer.js"
 *   renderSpec("figures/fig-cka-heatmap.yaml", { skipDataLoad: true })
 */

import { parseSpec, astToDOM } from "https://cdn.jsdelivr.net/npm/@uwdata/mosaic-spec@0.21.1/+esm";
import { coordinator, wasmConnector } from "https://cdn.jsdelivr.net/npm/@uwdata/vgplot@0.21.1/+esm";
import jsyaml from "https://cdn.jsdelivr.net/npm/js-yaml@4/+esm";

let initialized = false;
let themeCache = null;

function ensureInit() {
  if (!initialized) {
    if (!globalThis.__mosaicConnectorReady) {
      coordinator().databaseConnector(wasmConnector());
    }
    initialized = true;
  }
}

/**
 * Load and cache the shared theme file (_theme.yaml).
 * @param {string} specUrl - Any spec URL, used to derive the theme URL
 * @returns {Promise<Object|null>} Parsed theme object or null
 */
async function loadTheme(specUrl) {
  if (themeCache !== undefined && themeCache !== null) return themeCache;
  try {
    const themeUrl = specUrl.replace(/[^/]+$/, "_theme.yaml");
    const raw = await fetch(themeUrl).then(r => {
      if (!r.ok) return null;
      return r.text();
    });
    if (raw) {
      themeCache = jsyaml.load(raw);
    } else {
      themeCache = null;
    }
  } catch {
    themeCache = null;
  }
  return themeCache;
}

/**
 * Resolve _palette reference in a spec using the theme file.
 * Merges palette colorDomain/colorRange into the spec's plot or vconcat plot.
 */
function resolvePalette(spec, theme) {
  if (!spec._palette || !theme?.palettes) return;
  const palette = theme.palettes[spec._palette];
  if (!palette) return;
  delete spec._palette;

  // Apply to top-level plot attributes
  if (spec.plot && !spec.vconcat) {
    if (palette.colorDomain) spec.colorDomain = palette.colorDomain;
    if (palette.colorRange) spec.colorRange = palette.colorRange;
    return;
  }

  // Apply to vconcat — find the plot entry
  if (spec.vconcat) {
    for (const entry of spec.vconcat) {
      if (entry.plot) {
        if (palette.colorDomain) entry.colorDomain = palette.colorDomain;
        if (palette.colorRange) entry.colorRange = palette.colorRange;
        break;
      }
    }
    return;
  }

  // Fallback: apply at top level
  if (palette.colorDomain) spec.colorDomain = palette.colorDomain;
  if (palette.colorRange) spec.colorRange = palette.colorRange;
}

/**
 * Load a spec file (JSON or YAML) and render it into a DOM element.
 *
 * @param {string} specUrl - URL or path to the spec file
 * @param {Object} [opts]
 * @param {boolean} [opts.skipDataLoad=false] - If true, delete spec.data
 *   (tables already loaded by dashboard init block)
 * @returns {Promise<HTMLElement>}
 */
export async function renderSpec(specUrl, { skipDataLoad = false } = {}) {
  ensureInit();
  try {
    const raw = await fetch(specUrl).then(r => r.text());
    const spec = specUrl.endsWith(".yaml") || specUrl.endsWith(".yml")
      ? jsyaml.load(raw)
      : JSON.parse(raw);

    // Resolve _palette from shared theme
    const theme = await loadTheme(specUrl);
    if (spec._palette && theme) {
      resolvePalette(spec, theme);
    }

    if (!skipDataLoad && spec.data) {
      const siteRoot = new URL("/", window.location.href).href;
      for (const [, def] of Object.entries(spec.data)) {
        if (def.file) {
          def.file = new URL(def.file, siteRoot).href;
        }
      }
    } else if (skipDataLoad) {
      delete spec.data;
    }

    const ast = parseSpec(spec);
    const el = await astToDOM(ast);
    return el.element;
  } catch (error) {
    const el = document.createElement("div");
    el.style.cssText = "color:#f85149;padding:1rem;border:1px solid #f85149;border-radius:4px;";
    el.textContent = `[renderSpec] Failed to render ${specUrl}: ${error.message}`;
    console.error("[mosaic-renderer]", specUrl, error);
    return el;
  }
}
