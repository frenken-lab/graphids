// Bandit fusion (UCB over pretrained teacher outputs).
// reward_kwargs comes from the shared methodology constants — not tunable.

local reward = import '_reward.libsonnet';

function(decision_threshold=0.5, ucb_alpha=1.0, lambda_reg=1.0,
         backbone_lr=0.001, buffer_size=100000)
  {
    model: {
      class_path: 'graphids.core.models.fusion.bandit.BanditFusionModule',
      init_args: {
        decision_threshold: decision_threshold,
        ucb_alpha: ucb_alpha,
        lambda_reg: lambda_reg,
        backbone_lr: backbone_lr,
        backbone_retrain_freq: 50,
        backbone_epochs: 5,
        buffer_size: buffer_size,
        reward_kwargs: reward,
      },
    },
  }
