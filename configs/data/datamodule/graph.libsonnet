// GraphDataModule wrapper — takes a source block, produces the data block.
// Used by unsupervised + supervised archetypes.
//
// Curriculum mode (sampler='curriculum') swaps in CurriculumDataModule and
// requires a scorer: pass a {class_path, init_args} block. The composer +
// Python `_resolve_nested` handle instantiation.
// `start_ratio`/`end_ratio`/`max_epochs`/`num_tiers` flow through as
// CurriculumDataModule init_args.
//
// `overrides` is the merge knob for niche datamodule ablations not
// covered by the named knobs.

function(source,
         label_filter=null,
         sampler='standard',
         scorer=null,
         curriculum_start_ratio=1.0,
         curriculum_end_ratio=10.0,
         curriculum_max_epochs=300,
         num_tiers=10,
         overrides={})
  // conv_type/heads are model properties — read by GraphModuleBase.compute_budget
  // off the model's own hparams, not duplicated onto the DataModule.
  local is_curriculum = sampler == 'curriculum';
  {
    data: {
      class_path: if is_curriculum
                  then 'graphids.core.data.datamodule.CurriculumDataModule'
                  else 'graphids.core.data.datamodule.GraphDataModule',
      init_args: std.mergePatch(
        { dataset: source }
        + (if label_filter != null then { label_filter: label_filter } else {})
        + (if is_curriculum then {
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
