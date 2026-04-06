// Fusion stage base defaults — shared by every fusion method.
//
// Ported from:
//   graphids/config/stages/fusion.yaml  (trainer + data + monitor blocks)
//   graphids/config/fusion/base.yaml    (placeholder — was `{}`)
//
// The trainer runs fusion on CPU in precision 32 — RL fusion methods
// (bandit, dqn) disable automatic optimization and the RL loop dominates
// wall time, so GPU offers no speedup and fp16 is unsafe for Q-learning.
//
// Monitors are val_acc / max (fusion stages are the only ones that track
// a classification metric at training time). _STAGE_MONITORS in
// orchestrate/resolve.py asserts this.

{
  trainer: {
    accelerator: 'cpu',
    precision: 32,
    max_epochs: 1500,
    gradient_clip_val: null,
    log_every_n_steps: 10,
  },

  checkpoint: {
    monitor: 'val_acc',
    mode: 'max',
  },

  early_stopping: {
    monitor: 'val_acc',
    mode: 'max',
    patience: 200,
  },

  data: {
    class_path: 'graphids.core.preprocessing.datamodule.FusionDataModule',
    init_args: {
      max_samples: 150000,
      max_val_samples: 30000,
      eval_batch_size: 256,
    },
  },
}
