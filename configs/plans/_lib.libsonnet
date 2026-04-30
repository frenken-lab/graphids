// Shared helpers for plan files.

{
  // Produces a fit node + a mode-matched test peer for one ablation preset.
  // Test inherits the fit node's effective mode (gpu by default, cpu when
  // the fit node explicitly sets mode='cpu', e.g. fusion stages).
  fit_test(preset, name=null, deps=[], timeout_min=null, mode=null, cross_plan_deps=[])::
    local stem = std.split(std.split(preset, '/')[1], '.')[0];
    local nm = if name != null then name else stem;
    local effective_mode = if mode != null then mode else 'gpu';
    [
      {
        name: nm,
        preset: preset,
        action: 'fit',
        deps: deps,
        [if mode != null then 'mode']: mode,
        [if timeout_min != null then 'timeout_min']: timeout_min,
        [if std.length(cross_plan_deps) > 0 then 'cross_plan_deps']: cross_plan_deps,
      },
      {
        name: nm + '-test',
        preset: preset,
        action: 'test',
        deps: [nm],
        mode: effective_mode,
        length: 'long',
        mem_gb: 32,
        timeout_min: 30,
      },
    ],
}
