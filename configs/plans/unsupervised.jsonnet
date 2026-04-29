// VGAE unsupervised pretraining plan.
//
// Run before ofat.jsonnet and fusion.jsonnet — both consume the vgae
// checkpoint (curriculum_vgae sampler + extract-states respectively).
// Validate val_discrimination_ratio ≥ 1.5 on a smoke run before
// promoting to full dataset.

local lib = import '_lib.libsonnet';

function(dataset, seed)
{
  nodes: lib.fit_test('unsupervised/vgae.jsonnet', timeout_min=210),
}
