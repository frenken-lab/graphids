# KD-GAT Session Plan

> Last updated: 2026-03-23

## Active Plan

### Ablation pipeline — READY TO RUN (pending model variants)

The manifest orchestrator is built (`758d66f`). `ablation.yaml` has 16 configs covering 6 paper claims. Dry-run produces 62 dedup'd SLURM jobs (vs 160 naive). But several configs reference model/stage variants that don't exist yet.

**What's done:**
- `ManifestBuilder` — agnostic YAML plan builder (add/factorial/sweep_axis/write)
- `submit_manifest()` — reads any YAML manifest, builds SLURM DAG with stage deduplication via `identity_keys` in `pipeline.yaml`
- `ablation.yaml` — 16 configs using Hydra dotlist keys, `sweep` + `defaults` + `configs` format
- CLI: `python -m graphids.pipeline.orchestration.manifest ablation.yaml --dry-run`

**What's missing to run the ablation:**

1. **GCN/SAGE/GPS conv variants in `_make_conv`** — `conv_gatv1` and `conv_gps` configs override `vgae.conv_type` and `gat.conv_type`. The autoencoder and GAT modules need to handle `conv_type` values beyond `gatv2`. Check what `_make_conv()` currently supports in the VGAE and GAT architectures.

2. **GAE unsupervised method** — `unsup_gae` config needs a GAE (non-variational) autoencoder. Currently only VGAE exists. GAE is simpler (no KL divergence, no reparameterization). Decide: separate model_type `gae` in pipeline.yaml, or a flag on the existing VGAE module (`variational: false`)?

3. **DGI unsupervised method** — `unsup_dgi` config needs Deep Graph Infomax. This is a different architecture (contrastive, not reconstructive). Likely a separate model_type + stage function. Lower priority — can defer to a future ablation round.

4. **`normal` stage evaluation path** — `gat_only` config runs `stages: [normal, evaluation]`. The evaluation stage currently expects a fusion checkpoint (depends_on fusion). When running without fusion, evaluation needs to handle missing upstream — either skip fusion metrics or evaluate the GAT directly.

5. **`vgae_only` evaluation path** — same issue: `stages: [autoencoder, evaluation]` skips GAT and fusion. Evaluation needs to handle VGAE-only scoring.

6. **`small_kd` scale** — `kd_student` config uses `scale: small_kd`. This needs:
   - A `small_kd` preset in `models.yaml` (or reuse `small` with KD auxiliaries)
   - Teacher dependency logic in the orchestrator (when scale contains `_kd`, add dependency on `large`-scale teacher job)
   - KD loss wiring in the training loop

**Suggested approach for next session:**
1. Start with what's simplest: verify `normal` and `vgae_only` eval paths work
2. Add `gat` (v1) conv_type support to `_make_conv()` — likely a 1-line change
3. Run the subset of configs that work now: `--filter loss_x_curriculum_ce_curriculum loss_x_curriculum_focal_curriculum fusion_bandit fusion_weighted_avg`
4. Add remaining model variants incrementally

## Recently Completed

- Generic manifest orchestrator (`758d66f`)
- Deduplicate graph assembly + ablation builder (`c7b5231`)
- Fix preprocessing parallelism (`83fd3f5`)
- GPU training efficiency: deterministic batch sizing (`d88e2ed`)

## In Progress

- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) — running on HF Spaces

## Blocked

(none)

## 3-Pillar Architecture (target)

| Pillar | Owner | Current state |
|--------|-------|---------------|
| **Config** | Hydra Compose + Pydantic | **Done** — 5-file config layer, Hydra config groups, lake_root-only |
| **Orchestration** | submitit + graphlib | **Done** — manifest-driven SLURM DAG with stage deduplication |
| **ML Training** | Lightning modules + stages | **Done** — All models use `trainer.test()` for eval, `trainer.fit()` for training |
| **I/O** | Lightning CSVLogger + ModelCheckpoint + callbacks | **Done** — No custom storage layer |

## Open Questions

- **GAE vs DGI**: Should these be separate model_types in pipeline.yaml or flags on the VGAE module?
- **Eval without fusion**: Should evaluation stage auto-detect missing upstream, or should manifest configs specify a different eval stage?

## Key Reference Documents

- `ablation.yaml` — 16-config experiment manifest
- `graphids/pipeline/orchestration/manifest.py` — orchestrator
- `graphids/config/pipeline.yaml` — DAG topology + identity_keys
