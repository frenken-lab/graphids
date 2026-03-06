/**
 * Thin wrapper for rendering SQL query results as Observable Inputs.table().
 * Centralizes the pattern: query DuckDB → Array.from → Inputs.table.
 *
 * Usage in OJS cells:
 *   import { renderTable } from "./_ojs/table-renderer.js"
 *   renderTable(vg.sql`SELECT * FROM metrics ORDER BY f1 DESC`, { ... })
 */

import { vg } from "./mosaic-setup.js";

/**
 * Execute a SQL query and render results as an Observable table.
 *
 * @param {string} sql - SQL query string or vg.sql tagged template
 * @param {Object} [opts] - Inputs.table options (columns, header, format, sort, reverse, width)
 * @returns {Promise<HTMLElement>} Observable Inputs.table element
 */
export async function renderTable(sql, opts = {}) {
  const result = await vg.coordinator().query(sql);
  const data = Array.from(result);

  if (data.length === 0) {
    const el = document.createElement("p");
    el.style.cssText = "color:#8b949e;text-align:center;padding:2rem;";
    el.textContent = "No data available.";
    return el;
  }

  return Inputs.table(data, opts);
}
