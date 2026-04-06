// Recipe expansion helpers (Jsonnet port of graphids.config.recipe_expand).
{
  _override_string(value)::
    if std.type(value) == 'boolean' then
      if value then 'true' else 'false'
    else if value == null then
      'None'
    else
      std.toString(value),

  _name_string(value)::
    if std.type(value) == 'boolean' then
      if value then 'True' else 'False'
    else if value == null then
      'None'
    else
      std.toString(value),

  _is_scalar(value)::
    std.type(value) == 'string' || std.type(value) == 'number' || std.type(value) == 'boolean' || value == null,

  // Flatten nested dict to dotted-key CLI strings.
  // {"trainer": {"max_epochs": 2}} -> {"trainer.max_epochs": "2"}
  flatten_dict(d, prefix='')::
    std.foldl(
      function(acc, key)
        local value = d[key];
        local full = if prefix == '' then key else prefix + '.' + key;
        if std.type(value) == 'object' then
          acc + $.flatten_dict(value, full)
        else if $._is_scalar(value) then
          acc + { [full]: $._override_string(value) }
        else
          error 'Non-scalar value for override key ' + full + ': ' + std.type(value),
      std.objectFields(d),
      {},
    ),

  _stage_chain(stage)::
    if stage == 'fusion' then
      ['autoencoder', 'supervised', 'fusion']
    else if stage == 'supervised' then
      ['autoencoder', 'supervised']
    else
      [stage],

  // Cartesian product of an array of arrays (go-jsonnet has no std.cartesianProduct).
  _cartesian(arrays)::
    if std.length(arrays) == 0 then [[]]
    else
      local rest = $._cartesian(arrays[1:]);
      [[x] + r for x in arrays[0] for r in rest],

  _axis_combo_map(keys, combo)::
    if std.length(keys) == 0 then
      {}
    else
      { [keys[i]]: combo[i] for i in std.range(0, std.length(keys) - 1) },

  _expand_sweep(sweep)::
    local scales = if std.type(sweep.scale) == 'array' then sweep.scale else [sweep.scale];
    local methods =
      if std.objectHas(sweep, 'fusion_method') && sweep.fusion_method != null then
        if std.type(sweep.fusion_method) == 'array' then sweep.fusion_method else [sweep.fusion_method]
      else
        [null];
    local init_args =
      if std.objectHas(sweep, 'model_overrides') && std.objectHas(sweep.model_overrides, 'init_args')
      then sweep.model_overrides.init_args
      else {};
    local axis_keys = std.objectFields(init_args);
    local axis_values = [
      if std.type(init_args[k]) == 'array' then init_args[k] else [init_args[k]]
      for k in axis_keys
    ];
    local combos =
      if std.length(axis_values) == 0 then [[]] else $._cartesian(axis_values);
    std.foldl(
      function(acc, entry) acc + entry,
      [
        local axis_map = $._axis_combo_map(axis_keys, combo);
        local core = {
          scale: scale,
          stages: $._stage_chain(sweep.stage),
        }
        + (if method != null then { fusion_method: method } else {});
        local axis_overrides = {
          [k]: axis_map[k]
          for k in std.objectFields(axis_map)
          if std.member(['conv_type', 'loss_fn', 'variational'], k)
        };
        local over = core + axis_overrides;
        local suffix_keys = [k for k in std.sort(std.objectFields(over)) if k != 'stages'];
        local suffix = std.join('_', [k + '-' + $._name_string(over[k]) for k in suffix_keys]);
        local raw_name = sweep.model_family + '_' + sweep.stage + '_' + suffix;
        { [std.strReplace(raw_name, '/', '_')]: over }
        for scale in scales
        for method in methods
        for combo in combos
      ],
      {},
    ),

  expand(recipe, valid_scales, valid_fusion_methods)::
    local defaults = if std.objectHas(recipe, 'overrides') then recipe.overrides else {};
    local sweeps = if std.objectHas(recipe, 'sweeps') then recipe.sweeps else [];
    local selection = if std.objectHas(recipe, 'selection') then recipe.selection else null;
    local configs_from_sweeps = std.foldl(
      function(acc, sweep) acc + $._expand_sweep(sweep),
      sweeps,
      {},
    );
    local configs_from_selection =
      if selection == null then
        {}
      else
        local scales = if std.length(selection.scales) > 0 then selection.scales else valid_scales;
        local methods =
          if std.length(selection.fusion_methods) > 0
          then selection.fusion_methods
          else valid_fusion_methods;
        std.foldl(
          function(acc, entry) acc + entry,
          [
            if family == 'fusion' then
              { [family + '_' + stage + '_' + scale + '_' + method]: {
                  stages: $._stage_chain(stage),
                  scale: scale,
                  fusion_method: method,
                } }
            else
              { [family + '_' + stage + '_' + scale]: {
                  stages: $._stage_chain(stage),
                  scale: scale,
                } }
            for family in selection.model_families
            for stage in (if std.objectHas(selection.stages, family) then selection.stages[family] else [])
            for scale in scales
            for method in (if family == 'fusion' then methods else [null])
          ],
          {},
        );
    local configs = configs_from_sweeps + configs_from_selection;
    local seeds = if std.objectHas(recipe, 'seeds') && std.length(recipe.seeds) > 0 then recipe.seeds else [42];

    assert std.length(std.objectFields(configs)) > 0 :
      'Recipe contains no runnable configs after expansion. Provide at least one sweep or selection block.';

    {
      defaults: defaults,
      configs: configs,
      sweep: { seeds: seeds },
      trainer_overrides:
        if std.objectHas(recipe, 'trainer_overrides')
        then $.flatten_dict(recipe.trainer_overrides)
        else {},
      stage_overrides:
        if std.objectHas(recipe, 'stage_overrides') then
          { [stage]: $.flatten_dict(recipe.stage_overrides[stage])
            for stage in std.objectFields(recipe.stage_overrides)
          }
        else {},
      resource_overrides:
        if std.objectHas(recipe, 'resource_overrides') then recipe.resource_overrides else {},
    },
}
