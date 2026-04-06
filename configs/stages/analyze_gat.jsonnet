// GAT analysis artifacts (embeddings, attention, CKA, loss landscape).
// Usage: python -m graphids analyze --config configs/stages/analyze_gat.jsonnet \
//          --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa \
//          --analyzer.cka_teacher_ckpt path/to/teacher.ckpt
{
  analyzer: {
    model_type: "gat",
    embeddings: true,
    attention: true,
    cka: true,
    landscape: true,
  },
}
