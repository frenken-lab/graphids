// Fusion stage base defaults — shared by every fusion method.
//
// Trainer runs fusion on CPU in precision 32 — RL fusion methods
// (bandit, dqn) disable automatic optimization and the RL loop dominates
// wall time, so GPU offers no speedup and fp16 is unsafe for Q-learning.
//
// Monitors are val_acc / max (fusion is the only stage tracking a
// classification metric at training time).

{
  callbacks: {
    checkpoint+: { init_args+: { monitor: 'val_acc', mode: 'max' } },
    early_stopping+: { init_args+: { monitor: 'val_acc', mode: 'max', patience: 200 } },
  },

  trainer: {
    accelerator: 'cpu',
    precision: '32-true',
    max_epochs: 1500,
    gradient_clip_val: null,
    log_every_n_steps: 10,
  },

  data: {
    class_path: 'graphids.core.data.datamodule.FusionDataModule',
    init_args: {},
  },
}
