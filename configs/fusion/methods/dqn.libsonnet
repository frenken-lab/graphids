// DQN fusion (Q-learning over pretrained teacher outputs).
// Ported from graphids/config/fusion/methods/dqn.yaml.

local reward = import '../reward.libsonnet';

{
  model: {
    class_path: 'graphids.core.models.fusion.dqn.DQNFusionModule',
    init_args: {
      lr: 0.001,
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
      epsilon: 0.2,
      epsilon_decay: 0.995,
      min_epsilon: 0.01,
      gpu_training_steps: 1,
      weight_decay: 0.00001,
      buffer_size: 50000,
    },
  },
}
