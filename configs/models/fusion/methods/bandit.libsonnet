// Bandit fusion (UCB over pretrained teacher outputs).
// Ported from graphids/config/fusion/methods/bandit.yaml.

local reward = import '../reward.libsonnet';

{
  model: {
    class_path: 'graphids.core.models.fusion.bandit.BanditFusionModule',
    init_args: {
      decision_threshold: 0.5,
      reward_kwargs: {
        vgae_weights: [0.4, 0.3, 0.3],
        correct: reward.correct,
        incorrect: reward.incorrect,
        confidence_weight: reward.confidence_weight,
        combined_conf_weight: reward.combined_conf_weight,
        disagreement_penalty: reward.disagreement_penalty,
        overconf_penalty: reward.overconf_penalty,
        balance_weight: reward.balance_weight,
      },
      ucb_alpha: 1.0,
      lambda_reg: 1.0,
      backbone_lr: 0.001,
      backbone_retrain_freq: 50,
      backbone_epochs: 5,
      buffer_size: 100000,
    },
  },
}
