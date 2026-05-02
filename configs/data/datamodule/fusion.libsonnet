// FusionDataModule wrapper — reads cached VGAE+GAT states from disk.
// No source primitive: the cache_dir IS the source for fusion.
// `method` flips RL vs supervised batching internally.

function(dataset, seed, method, batch_size=128, episode_sample_size=20000)
  {
    data: {
      class_path: 'graphids.core.data.datamodule.fusion.FusionDataModule',
      init_args: {
        cached_states_dir: std.native('paths.states_dir')(dataset, seed),
        method: method,
        batch_size: batch_size,
        episode_sample_size: episode_sample_size,
      },
    },
  }
