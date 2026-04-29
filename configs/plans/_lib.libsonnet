// Shared helpers for plan files.

{
  // Produces a fit node + a cpu test peer for one ablation preset.
  fit_test(preset, name=null, deps=[], timeout_min=null, mode=null, cross_plan_deps=[])::
    local stem = std.split(std.split(preset, '/')[1], '.')[0];
    local nm = if name != null then name else stem;
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
        mode: 'cpu',
        length: 'long',
        mem_gb: 32,
        timeout_min: 30,
      },
    ],
}
