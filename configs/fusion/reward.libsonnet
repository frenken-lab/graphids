// Fusion reward shaping constants — paper methodology.md §Stage 3 Adaptive Fusion.
//
// These are FIXED methodological choices, not ablation axes. DQN and bandit
// share this identical reward. The only per-run tunable is `vgae_weights`
// (the convex combination over the three VGAE reconstruction components),
// which each method libsonnet sets separately.
//
// Do NOT override in sweeps — if you find yourself wanting to, add a new
// libsonnet alongside this one and update the paper.
{
  correct: 3.0,
  incorrect: -3.0,
  confidence_weight: 0.5,
  combined_conf_weight: 0.3,
  disagreement_penalty: -1.0,
  overconf_penalty: -1.5,
  balance_weight: 0.3,
}
