// CAN bus data source — CANBusSource only (raw byte reader).
// Registry lookup validates `dataset` and surfaces metadata.
// `overrides` is the escape hatch for source-level ablations
// (e.g. {window_size: 200, val_fraction: 0.3}).

local registry = import '../datasets.json';

function(dataset, seed, overrides={})
  if !std.objectHas(registry, dataset) then
    error 'unknown dataset: ' + dataset
          + ' (registry: ' + std.join(', ', std.objectFields(registry)) + ')'
  else
    {
      class_path: 'graphids.core.data.datasets.can_bus.CANBusSource',
      init_args: std.mergePatch({
        name: dataset,
        seed: seed,
        window_size: 100,
        stride: 100,
        val_fraction: 0.2,
      }, overrides),
    }
