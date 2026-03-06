---
name: verify-site
description: Run Playwright headless browser verification on KD-GAT Quarto site (dashboard + paper)
user_invocable: true
arguments:
  - name: page
    description: "Page to verify: dashboard, paper, slides, playground, or 'all' (default: all)"
    required: false
---

# /verify-site — Runtime Verification for KD-GAT Quarto Site

Verify KD-GAT reports site for runtime JS/OJS errors, 404s, and rendering bugs using Playwright MCP.

## Prerequisites

- On OSC (Playwright MCP installs Chromium to `~/.cache/ms-playwright/`).
- `.env` file sourced (Quarto needs env vars).

## Step 1: Build and Serve

```bash
source .env && quarto render reports/
```

If the render fails, stop and report the error.

**CRITICAL**: Always pick a fresh port. Kill any existing server and start new:

```bash
pkill -f "python3 -m http.server" 2>/dev/null
sleep 1
PORT=8770
cd reports/_site && python3 -m http.server $PORT &
sleep 2
curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/dashboard.html
```

Verify the response is 200 before proceeding. If the port is busy, increment and retry.

## Step 2: Page Verification

### Page Map

| Argument | URL Path | Wait Time | Special Handling |
|----------|----------|-----------|------------------|
| `dashboard` | `/dashboard.html` | 5s initial + 5s per tab | Must click ALL 9 tabs |
| `paper` | `/paper/paper.html` | 15s | Heavy OJS, many figures |
| `slides` | `/slides.html` | 5s | RevealJS |
| `playground` | `/playground.html` | 10s | Interactive cells |

If argument is `all` or omitted, verify dashboard and paper (primary). Include slides and playground if time permits.

### For Each Page

#### 2a. Navigate and Wait

```
browser_navigate(url)
```

Then wait for async content. Use `browser_run_code`:
```javascript
async (page) => {
  await page.waitForTimeout(WAIT_TIME_MS);
  return 'loaded';
}
```

#### 2b. Check Console Errors

```
browser_console_messages(level="error", filename="/tmp/kdgat_verify_errors.log")
```

Read the file. Filter out:
- `favicon.ico` 404 (harmless, ignore)

ALL other errors are real. Common error patterns:
- `vg.XXX is not a function` → mark/function not available in bundled vgplot version
- `Unrecognized mark type: XXX` → mark not in mosaic-spec version
- `TypeError: Cannot read properties of undefined` → rendering crash, often mark API mismatch
- `Error evaluating OJS cell` → OJS cell failed, read the cell code in the error
- `Failed to load resource: 404` → missing data file (Parquet, JSON)

#### 2c. Check Network for 404s

```
browser_network_requests
```

Scan for failed requests on `.parquet`, `.json`, `.js`, `.css` files. Report any 404s.

#### 2d. Check Warnings

```
browser_console_messages(level="warning", filename="/tmp/kdgat_verify_warnings.log")
```

Look for:
- DuckDB-WASM init warnings
- Mosaic coordinator warnings
- Missing table references

#### 2e. Screenshot

```
browser_take_screenshot(filename="/tmp/kdgat_verify_PAGE.png")
```

### Dashboard-Specific: Tab Verification

**This is critical. Dashboard tabs lazy-load OJS cells. Errors only appear after clicking each tab.**

Use `browser_run_code` to click through all tabs (browser_snapshot times out on heavy dashboard):

```javascript
async (page) => {
  const tabNames = ['Performance', 'Training', 'GAT & DQN', 'Knowledge Distillation',
                     'Loss Landscape', 'Graph Structure', 'Datasets', 'Staging'];
  for (const name of tabNames) {
    const tabs = await page.$$('[role="tab"]');
    for (const tab of tabs) {
      const text = (await tab.textContent()).trim();
      if (text === name) {
        await tab.click();
        await page.waitForTimeout(4000);
        break;
      }
    }
  }
  await page.waitForTimeout(5000);
  return 'visited all tabs';
}
```

Note: This will timeout on `page._snapshotForAI` — that's expected. Ignore the timeout error and proceed to check console messages.

After visiting all tabs, collect errors:
```
browser_console_messages(level="error", filename="/tmp/kdgat_verify_dashboard_all.log")
```

### Paper-Specific

The paper has many OJS cells including:
- Mosaic JSON spec figures (rendered by `mosaic-renderer.js`)
- Observable Plot figures
- D3 force graph
- Mosaic imperative API plots

Wait 15s after navigation for all cells to resolve. Then check errors.

Also verify data file accessibility:
```javascript
async (page) => {
  const files = ['data/graph_samples.json', 'data/graph_statistics.parquet',
                  'data/datasets.json', 'data/embeddings.parquet'];
  const results = {};
  for (const f of files) {
    const resp = await fetch('/' + f, { method: 'HEAD' });
    results[f] = resp.status;
  }
  return results;
}
```

## Step 3: Cleanup

```bash
pkill -f "python3 -m http.server" 2>/dev/null
browser_close
```

## Step 4: Report

Summarize as a table:

| Page | Console Errors | 404s | Warnings | Status |
|------|---------------|------|----------|--------|

For dashboard, also report per-tab status if errors were found on specific tabs.

Mark each page as PASS (0 real errors) or FAIL. List specific errors for failures.

## Known Limitations

- `browser_snapshot` times out on dashboard (heavy DOM with many OJS cells) — use `browser_run_code`/`browser_evaluate`
- Python `http.server` caches files — always restart server after re-rendering
- DuckDB-WASM needs 3-5s to initialize; Mosaic plots need 5-10s to render
- Quarto dashboard tabs lazy-load — errors only surface after clicking each tab
- Paper OJS cells resolve asynchronously — wait 15s minimum
- `tickX` mark compiles but crashes at render in mosaic-plot@0.21.1 — use `dotX` instead
- `boxX`/`boxY` don't exist in vgplot@0.21.1

## Quick Mark Compatibility Reference (vgplot@0.21.1)

Available: `barX`, `barY`, `dot`, `dotX`, `dotY`, `line`, `lineX`, `lineY`, `areaX`, `areaY`,
`rectX`, `rectY`, `ruleX`, `ruleY`, `tickX`, `tickY`, `text`, `textX`, `textY`, `cell`,
`frame`, `hexbin`, `contour`, `density`, `densityX`, `densityY`, `arrow`, `link`, `image`, `spike`, `vector`

NOT available: `boxX`, `boxY` (added in later versions)

Caution: `tickX`/`tickY` parse OK but crash during render. Use `dotX`/`dotY` for strip plots.
