// Fusion plan: extract VGAE+GAT latent states, then train fusion methods.
//
// Prerequisites (from separate plans, both must be FINISHED):
//   - unsupervised.jsonnet: vgae
//   - ofat.jsonnet: focal  (gat_loss/focal)
// Ckpt paths are auto-derived from (dataset, seed) via paths.best_ckpt.

local lib = import '_lib.libsonnet';

function(dataset, seed)

local extract_states_command =
  std.join(' ', [
    'python -m graphids extract-fusion-states',
    '--vgae-ckpt ' + std.native('paths.best_ckpt')(dataset, 'unsupervised', 'vgae', seed),
    '--gat-ckpt ' + std.native('paths.best_ckpt')(dataset, 'gat_loss', 'focal', seed),
    '--dataset ' + dataset,
    '--seed ' + seed,
    '--output-dir ' + std.native('paths.states_dir')(dataset, seed),
  ]);

{
  nodes:
    // Extract VGAE + focal GAT latent states → tensor cache.
    [{
      name: 'extract-states',
      command: extract_states_command,
      deps: [],
      cross_plan_deps: ['vgae', 'focal'],
      mode: 'gpu',
      mem_gb: 36,
      timeout_min: 30,
    }]
    // Fusion methods fan out from extract-states.
    // mode='cpu': fusion runs on cached state tensors (0% GPU compute confirmed).
    + lib.fit_test('fusion/bandit.jsonnet', deps=['extract-states'], mode='cpu')
    + lib.fit_test('fusion/dqn.jsonnet', deps=['extract-states'], mode='cpu')
    + lib.fit_test('fusion/mlp.jsonnet', deps=['extract-states'], mode='cpu')
    + lib.fit_test('fusion/weighted_avg.jsonnet', deps=['extract-states'], mode='cpu'),
}
