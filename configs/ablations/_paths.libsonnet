// Shared path helpers for the configs/ablations/ tree.
//
// Every ablation's run_dir and every upstream ckpt path is derived
// from (lake_root, dataset, seed) so the submit call collapses to just
// `--tla dataset=... --tla seed=...`.
{
  run_dir(lake_root, dataset, group, variant, seed)::
    '%s/%s/ablations/%s/%s/seed_%d' % [lake_root, dataset, group, variant, seed],

  vgae_ckpt(lake_root, dataset, seed)::
    '%s/%s/ablations/unsupervised/vgae/seed_%d/checkpoints/best_model.ckpt' % [lake_root, dataset, seed],

  states_dir(lake_root, dataset, seed)::
    '%s/%s/ablations/fusion_states/seed_%d' % [lake_root, dataset, seed],
}
