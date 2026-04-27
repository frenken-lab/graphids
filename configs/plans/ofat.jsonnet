// OFAT (one-factor-at-a-time) ablation plan.
//
// Returns `{ nodes: [Node, ...] }` for `graphids run` to consume.
// Each Node is one `graphids submit` call; deps reference other nodes
// by name. Test peers are explicit (cpu/long, 32 GB, 30 min).
//
// Single source of truth for the ablation topology — Python reads it
// via `graphids.config.jsonnet.render`. To add a variant: add a
// `fit_test(...)` line; to change test resources: edit `fit_test`.

function(dataset, seed)

// `name` defaults to the variant (preset stem) — override only when the
// plan needs distinct topology nodes for the same preset. group / variant
// are derived from the preset path by `graphids.slurm.dag.Node`.
local fit_test(preset, name=null, deps=[], timeout_min=null) =
  local stem = std.split(std.split(preset, '/')[1], '.')[0];
  local nm = if name != null then name else stem;
  [
    {
      name: nm,
      preset: preset,
      action: 'fit',
      deps: deps,
      [if timeout_min != null then 'timeout_min']: timeout_min,
    },
    {
      name: nm + '-test',
      preset: preset,
      action: 'test',
      deps: [nm],
      mode: 'cpu',
      length: 'long',
      mem_gb: 32,
      timeout_min: 30,
    },
  ];

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
    // Stage 0 — VGAE baseline (upstream for curriculum_vgae + extract-states).
    fit_test('unsupervised/vgae.jsonnet', timeout_min=210)
    // Stage 1 — standalone parallel variants.
    + fit_test('gat_sampling/none.jsonnet')
    + fit_test('gat_sampling/curriculum_random.jsonnet')
    + fit_test('gat_loss/ce.jsonnet')
    + fit_test('gat_loss/weighted_ce.jsonnet')
    + fit_test('gat_loss/focal.jsonnet')
    + fit_test('gat_loss/tau_norm.jsonnet')
    + fit_test('id_encoding/lookup.jsonnet')
    + fit_test('id_encoding/learned_unk.jsonnet')
    + fit_test('id_encoding/hash.jsonnet')
    // Stage 2 — curriculum_vgae fans in to vgae.
    + fit_test('gat_sampling/curriculum_vgae.jsonnet', deps=['vgae'])
    // Stage 3 — extract fusion states (vgae + focal encoders → tensor cache).
    + [{
      name: 'extract-states',
      command: extract_states_command,
      deps: ['vgae', 'focal'],
      mode: 'gpu',
      mem_gb: 36,
      timeout_min: 30,
    }]
    // Stage 4 — fusion methods fan out from extract-states.
    + fit_test('fusion/bandit.jsonnet', deps=['extract-states'])
    + fit_test('fusion/dqn.jsonnet', deps=['extract-states'])
    + fit_test('fusion/mlp.jsonnet', deps=['extract-states'])
    + fit_test('fusion/weighted_avg.jsonnet', deps=['extract-states']),
}
