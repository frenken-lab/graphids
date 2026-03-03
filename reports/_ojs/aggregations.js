/**
 * ML aggregation helpers for KD-GAT dashboard.
 * Pure functions — no dependencies on Mosaic, D3, or DuckDB.
 *
 * Usage in OJS cells:
 *   import { paretoFront, bestPerGroup, kdGap, durationMinutes } from "./_ojs/aggregations.js"
 */

/**
 * Compute the Pareto front (minimize minKey, maximize maxKey).
 * Returns a subset of `data` that are Pareto-optimal.
 *
 * @param {Array<Object>} data - array of data points
 * @param {string} minKey - field to minimize (e.g., "param_count_M")
 * @param {string} maxKey - field to maximize (e.g., "f1")
 * @returns {Array<Object>} Pareto-optimal points, sorted by minKey ascending
 */
export function paretoFront(data, minKey, maxKey) {
  const sorted = [...data].sort((a, b) => a[minKey] - b[minKey]);
  const pareto = [];
  let bestMax = -Infinity;
  for (let i = sorted.length - 1; i >= 0; i--) {
    if (sorted[i][maxKey] >= bestMax) {
      bestMax = sorted[i][maxKey];
      pareto.unshift(sorted[i]);
    }
  }
  return pareto;
}

/**
 * Select the best row per group (highest metricKey value).
 *
 * @param {Array<Object>} data - array of data points
 * @param {string} groupKey - field to group by (e.g., "model")
 * @param {string} metricKey - field to maximize (e.g., "f1")
 * @returns {Array<Object>} one row per unique groupKey value
 */
export function bestPerGroup(data, groupKey, metricKey) {
  const grouped = new Map();
  for (const d of data) {
    const key = d[groupKey];
    const prev = grouped.get(key);
    if (!prev || d[metricKey] > prev[metricKey]) {
      grouped.set(key, d);
    }
  }
  return Array.from(grouped.values());
}

/**
 * Compute the average KD gap (teacher - student) for a given metric.
 *
 * @param {Array<Object>} transfers - kd_transfer_json.data array
 * @param {string} metric - metric name to filter on (e.g., "f1")
 * @returns {number|null} average gap, or null if no data
 */
export function kdGap(transfers, metric) {
  const filtered = transfers.filter(d => d.metric_name === metric);
  if (filtered.length === 0) return null;
  const gaps = filtered.map(d => d.teacher_value - d.student_value);
  return gaps.reduce((a, b) => a + b, 0) / gaps.length;
}

/**
 * Compute duration in minutes from a run object.
 * Tries duration_seconds first, then falls back to started_at/completed_at timestamps.
 *
 * @param {Object} run - run object with duration_seconds or started_at/completed_at
 * @returns {number|null} duration in minutes, or null if not computable
 */
export function durationMinutes(run) {
  if (run.duration_seconds != null && run.duration_seconds > 0) {
    return run.duration_seconds / 60.0;
  }
  if (run.started_at != null && run.completed_at != null) {
    const start = new Date(run.started_at);
    const end = new Date(run.completed_at);
    const seconds = (end - start) / 1000;
    return seconds > 0 ? seconds / 60.0 : null;
  }
  return null;
}
