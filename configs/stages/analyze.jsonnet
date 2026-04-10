// Analysis artifacts — dispatches on model_type.
//
// Usage: python -m graphids analyze --config configs/stages/analyze.jsonnet \
//          --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa \
//          --analyzer.model_type vgae
//
// Artifacts by model_type:
//   vgae   → embeddings, loss landscape (51×51)
//   dgi    → embeddings, loss landscape (51×51) — unsupervised, same shape as VGAE
//   gat    → embeddings, attention, CKA, loss landscape
//   fusion → fusion policy visualization

local artifacts = {
  vgae: {
    embeddings: true,
    landscape: true,
    landscape_resolution: 51,
    landscape_scale: 1.0,
  },
  dgi: {
    embeddings: true,
    landscape: true,
    landscape_resolution: 51,
    landscape_scale: 1.0,
  },
  gat: {
    embeddings: true,
    attention: true,
    cka: true,
    landscape: true,
  },
  fusion: {
    embeddings: false,
    fusion_policy: true,
  },
};

function(model_type='vgae')
  { analyzer: { model_type: model_type } + artifacts[model_type] }
