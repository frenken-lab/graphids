// DQN fusion (Q-learning over pretrained teacher outputs).
// reward_kwargs comes from the shared methodology constants — not tunable.

local reward = import '_reward.libsonnet';

function(decision_threshold=0.5, lr=0.001, epsilon=0.2, epsilon_decay=0.995,
         min_epsilon=0.01, buffer_size=50000, weight_decay=0.00001)
  {
    model: {
      class_path: 'graphids.core.models.fusion.dqn.DQNFusionModule',
      init_args: {
        lr: lr,
        decision_threshold: decision_threshold,
        epsilon: epsilon,
        epsilon_decay: epsilon_decay,
        min_epsilon: min_epsilon,
        buffer_size: buffer_size,
        weight_decay: weight_decay,
        gpu_training_steps: 1,
        reward_kwargs: reward,
      },
    },
  }
