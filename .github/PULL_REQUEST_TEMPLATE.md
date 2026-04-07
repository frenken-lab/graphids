## What changed

<!-- 1-3 sentences: what and why -->

## How to test

<!-- How to verify this works. For model/training changes: which dataset + stage to run. For config changes: which jsonnet to render. -->

## Checklist

- [ ] `ruff check graphids/ tests/` passes
- [ ] Jsonnet stages render without errors (`jsonnet configs/stages/*.jsonnet`)
- [ ] No hardcoded paths, secrets, or login-node-unsafe imports
- [ ] Updated `CLAUDE.md` or `docs/` if architecture changed
