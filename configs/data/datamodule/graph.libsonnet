// GraphDataModule wrapper — takes a source block, produces the data block.
// Used by unsupervised + supervised archetypes.
//
// Curriculum mode (sampler='curriculum') requires a scorer: pass a
// {class_path, init_args} block. The composer + Python `_resolve_nested`
// handle instantiation. `start_ratio`/`end_ratio`/`max_epochs`/`num_tiers`
// flow through as datamodule init_args.
//
// `overrides` is the merge knob for niche datamodule ablations not
// covered by the named knobs.

function(source,
         label_filter=null,
         conv_type='gatv2',
         heads=4,
         sampler='standard',
         scorer=null,
         curriculum_start_ratio=1.0,
         curriculum_end_ratio=10.0,
         curriculum_max_epochs=300,
         num_tiers=10,
         overrides={})
  {
    data: {
      class_path: 'graphids.core.data.datamodule.GraphDataModule',
      init_args: std.mergePatch(
        {
          dataset: source,
          conv_type: conv_type,
          heads: heads,
          sampler: sampler,
        }
        + (if label_filter != null then { label_filter: label_filter } else {})
        + (if sampler == 'curriculum' then {
             scorer: scorer,
             curriculum_start_ratio: curriculum_start_ratio,
             curriculum_end_ratio: curriculum_end_ratio,
             max_epochs: curriculum_max_epochs,
             num_tiers: num_tiers,
           } else {}),
        overrides,
      ),
    },
  }
