// Fusion reward shaping constants — paper methodology §Stage 3 Adaptive Fusion.
//
// FIXED methodological choices, not ablation axes. Shared by bandit + dqn
// (the two RL-based fusion methods). MLP and weighted_avg don't use a
// reward signal — supervised gradient losses, not bandits.
//
// Do NOT override in sweeps. If you find yourself wanting to, add a new
// libsonnet alongside this one and update the paper.

{
  vgae_weights: [0.4, 0.3, 0.3],
  correct: 3.0,
  incorrect: -3.0,
  confidence_weight: 0.5,
  combined_conf_weight: 0.3,
  disagreement_penalty: -1.0,
  overconf_penalty: -1.5,
  balance_weight: 0.3,
}
