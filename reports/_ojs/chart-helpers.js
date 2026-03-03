/**
 * Chart helper functions for KD-GAT dashboard.
 * Eliminates repeated Selection.single() + vconcat(menu, plot) boilerplate.
 *
 * Usage in OJS cells:
 *   import { filteredChart } from "./_ojs/chart-helpers.js"
 *   import { vg } from "./_ojs/mosaic-setup.js"
 */

import { vg } from "./mosaic-setup.js";

/**
 * Create a filtered chart with a dropdown menu bound to a Mosaic Selection.
 * Wraps the pattern: Selection.single() → menu → vconcat(menu, plot).
 *
 * @param {string} tableName - DuckDB table to populate the menu from
 * @param {function} plotFn - (selection) => vg.plot(...) — receives the shared selection
 * @param {Object} [opts]
 * @param {string} [opts.column="run_id"] - Column for the filter menu
 * @param {string} [opts.label="Run"] - Menu label
 * @param {import("@uwdata/vgplot").Selection} [opts.selection] - Shared selection (for cross-chart linking)
 * @returns Mosaic vconcat element
 */
export function filteredChart(tableName, plotFn, { column = "run_id", label = "Run", selection } = {}) {
  const sel = selection ?? vg.Selection.single();
  return vg.vconcat(
    vg.menu({ from: tableName, column, label, as: sel }),
    plotFn(sel)
  );
}

/**
 * Build Mosaic color directives for a domain/range pair.
 * Returns an array to spread into vg.plot() calls.
 *
 * @param {string[]} domain - color domain values
 * @param {string[]} range - color range hex values
 * @param {Object} [opts]
 * @param {boolean} [opts.legend=true] - show color legend
 * @param {string} [opts.label] - color legend label
 * @returns {Array} array of vg directives: [colorLegend, colorDomain, colorRange, ...]
 */
export function colorDirectives(domain, range, { legend = true, label } = {}) {
  const directives = [
    vg.colorDomain(domain),
    vg.colorRange(range),
  ];
  if (legend) directives.unshift(vg.colorLegend(true));
  if (label) directives.push(vg.colorLabel(label));
  return directives;
}
