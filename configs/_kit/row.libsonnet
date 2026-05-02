// Row builders for plan jsonnets.
// Identity strings synthesized via `std.format` — no native callback.
// Inline asserts fail loudly at render if a preset forgot _meta.
//
// Resource separation: `_resources.mode` is archetype-fixed (composer attaches
// it). `length` is per-row (smoke vs production), passed by the plan.
// `accelerator` is derived from mode — single source of truth.

local identity_for(rendered) = {
  local m = rendered._meta,
  run_name: std.format('%s_%s_%s_seed%d', [m.group, m.variant, m.dataset, m.seed]),
  run_dir: std.native('paths.run_dir')(m.dataset, m.group, m.variant, m.seed),
  jobname: std.format('%s-%s-%s', [m.model_type, m.scale, m.variant]),
};

local strip_meta(o) =
  { [k]: o[k] for k in std.objectFields(o) if !std.startsWith(k, '_') };

local accelerator_for(mode) = if mode == 'cpu' then 'cpu' else 'auto';

{
  fit(name, rendered, length='long'):: {
    assert std.objectHas(rendered, '_meta') : 'preset missing _meta',
    assert std.objectHas(rendered, '_resources') : 'preset missing _resources',
    local mode = rendered._resources.mode,
    name: name,
    action: 'fit',
    identity: identity_for(rendered),
    // Structured meta survives onto the row so Python (MLflow tags,
    // experiment name, runstate queries) can read group/variant/dataset/
    // seed/model_type/scale without parsing the run_name string.
    meta: rendered._meta,
    // Inject the derived accelerator into trainer at row-emit time so the
    // rendered_config in the blueprint is self-contained.
    rendered_config: strip_meta(rendered) + {
      trainer+: { accelerator: accelerator_for(mode) },
    },
    upstreams: std.get(rendered, '_upstreams', []),
    resources: { mode: mode, length: length },
  },
  test(name, rendered, length='long'):: self.fit(name + '-test', rendered, length=length) + { action: 'test' },
  cmd(name, command, mode='cpu', length='short'):: {
    name: name,
    action: 'cmd',
    command: command,
    resources: { mode: mode, length: length },
  },
  // One-shot fusion-feature extraction. Idempotent on output_dir — repeated
  // submissions hit the cache check in `extract_fusion_states`.
  extract(name, dataset, extractor_ckpts, output_dir,
          mode='gpu', length='short',
          max_samples=150000, max_val_samples=30000, batch_size=256,
          seed=42, window_size=100, stride=100, val_fraction=0.2):: {
    name: name,
    action: 'extract',
    dataset: dataset,
    extractor_ckpts: extractor_ckpts,
    output_dir: output_dir,
    resources: { mode: mode, length: length },
    max_samples: max_samples,
    max_val_samples: max_val_samples,
    batch_size: batch_size,
    seed: seed,
    window_size: window_size,
    stride: stride,
    val_fraction: val_fraction,
  },
}
