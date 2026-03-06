/**
 * SQL query utilities for safe parameterization.
 */

/**
 * Escape a value for safe use in SQL string literals (prevents SQL injection).
 * @param {string|number} value - Value to escape
 * @returns {string} Escaped string safe for SQL single-quoted literals
 */
export function safeEq(value) {
  return String(value).replace(/'/g, "''");
}
