# KD-GAT Session Plan

> Last updated: 2026-03-04

## Priority: Wait for Cache Rebuild → Export → Verify

Cache rebuild jobs submitted (SLURM 44485258-63) with `include_attack_type=True`.
Once complete, run export and verify visualizations.

### Post-Cache Steps

```bash
# 1. Check job status
squeue -u $USER

# 2. Verify caches have attack_type metadata
.venv/bin/python -c "
import torch
g = torch.load('data/cache/hcrl_ch/processed_graphs.pt', map_location='cpu', weights_only=False)
g0 = g[0] if isinstance(g, list) else g.data_list[0]
print(f'x: {g0.x.shape}, edge_attr: {g0.edge_attr.shape}')
print(f'attack_type: {g0.attack_type}, node_y: {g0.node_y.shape}')
print(f'id_entropy: {g0.id_entropy}')
"

# 3. Run export (generates graph_samples.json v2)
.venv/bin/python -m graphids.pipeline.export

# 4. Render reports
quarto render reports/

# 5. Verify in browser (WSL only)
quarto preview reports/
```

### Verification Checklist

- [ ] Cache rebuild completes for all 6 datasets (check SLURM logs)
- [ ] Cached graphs have `attack_type`, `node_attack_type`, `node_y`, `id_entropy`, 26-D `x`, 11-D `edge_attr`
- [ ] `python -m graphids.pipeline.export` produces `graph_samples.json` v2 schema
- [ ] `quarto render reports/` succeeds (16/16 pages)
- [ ] Browser: force graph renders with all 7 color modes
- [ ] Browser: edge tooltips show 11-D features
- [ ] Browser: color legend correct per mode
- [ ] Browser: 0 JS console errors
- [ ] Paper figure renders with attack_type coloring

## In Progress

- Cache rebuild (SLURM jobs 44485258-63, CPU partition, ~5-20 min each)

## Blocked

(none)

## Open Questions

### Is RL justified for fusion? (dqn.py, fusion.py)

The DQN fusion agent is a **contextual bandit**, not a sequential MDP:
- `next_state == state` (no transitions), `done == False` always
- Bellman target `r + gamma * max Q(s, a')` is self-referential (s' = s)
- 15-D states are pre-cached, i.i.d. — no environment dynamics

**Experiment needed**: Compare F1/accuracy across all three fusion methods on held-out test sets:
1. DQN (vectorized, current) — RL with gamma=0.99, replay buffer, target network
2. DQN with gamma=0 — proper bandit, target = immediate reward, no target network
3. MLPFusionAgent — supervised BCE, already vectorized
4. WeightedAvgFusionAgent — single learned alpha

If MLP matches DQN, drop the RL framing entirely. This saves code complexity and
training time (~100x faster even after vectorization).

**Additional fix needed**: Training uses `(alpha > 0.5)` as prediction (alpha is a
fusion weight, not a score). Validation uses the proper fused score
`(1-alpha)*anomaly + alpha*gat_prob > 0.5`. The training reward signal is based on
a semantically wrong prediction. If DQN is kept, this should be fixed.

## Next Up

- Run full pipeline retrain on rebuilt caches
- Run tests: `bash scripts/slurm/run_tests_slurm.sh`
- Fusion method comparison experiment (see Open Questions above)
- Loss landscape analysis on retrained models
- Evaluate research questions R1–R3

## Completed

- Force graph & visualization rework: 7 color modes, edge tooltips, attack_type support (2026-03-04)
- `export_graph_samples()` + v2 JSON schema (2026-03-04)
- `include_attack_type=True` default in preprocessing pipeline (2026-03-04)
- evaluation.py: attack_type capture in embeddings.npz (2026-03-04)
- `rebuild_all_caches.sh` + `build_test_cache.sh` fixes (2026-03-04)
- Feature engineering v2.0.0: 15 new node features (11→26-D), GATv2 switch, paper updated (2026-03-03)
- Loss landscape stage fixes + dashboard tab (2026-03-03)
- SLURM account migration PAS3209→PAS1266 (2026-03-03)
- (older items in git log)
