// Fusion analysis artifacts (DQN/bandit policy).
// Usage: python -m graphids analyze --config configs/stages/analyze_fusion.jsonnet \
//          --analyzer.ckpt_path path/to/fusion.ckpt --analyzer.dataset hcrl_sa \
//          --analyzer.vgae_ckpt_path path/to/vgae.ckpt --analyzer.gat_ckpt_path path/to/gat.ckpt
{
  analyzer: {
    model_type: "fusion",
    embeddings: false,
    fusion_policy: true,
  },
}
