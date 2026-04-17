// Jsonnet helpers shared by every stage.
//
// `apply_dotted(overrides)` reproduces the pre-migration
// `yaml_utils.apply_dotted_overrides` primitive: turn a flat dict of
// dotted-key strings into a nested object, with deep-merge semantics
// (so multiple dotted keys under the same parent compose rather than
// replace each other).

{
  // {"trainer.max_epochs": "50", "data.init_args.num_workers": "4"}
  // ->
  // { trainer+: { max_epochs: "50" },
  //   data+: { init_args+: { num_workers: "4" } } }
  apply_dotted(overrides)::
    std.foldl(
      function(acc, key) acc + $._nest(std.split(key, '.'), overrides[key]),
      std.objectFields(overrides),
      {},
    ),

  // Recursively nest a split path back into a deep-merge-friendly object.
  // Intermediate nodes use `+:` so multiple dotted keys under the same
  // parent compose correctly (without `+:`, the last key would clobber
  // the whole sibling tree).
  _nest(path, value)::
    if std.length(path) == 1 then
      { [path[0]]: value }
    else
      { [path[0]]+: $._nest(path[1:], value) },
}
