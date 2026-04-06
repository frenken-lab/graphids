// VGAE analysis artifacts (embeddings, loss landscape).
// Usage: python -m graphids analyze --config configs/stages/analyze_vgae.jsonnet \
//          --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
{
  analyzer: {
    model_type: "vgae",
    embeddings: true,
    landscape: true,
    landscape_resolution: 51,
    landscape_scale: 1.0,
  },
}
