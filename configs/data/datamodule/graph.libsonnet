// GraphDataModule wrapper — takes a source block, produces the data block.
// Used by unsupervised + supervised archetypes.
//
// Curriculum learning lives at the loss end: pass ``difficulty`` (a
// ``{class_path, init_args}`` block for a function returning a per-graph
// score tensor) and optionally ``scope_label`` (default 0 — "y == 0 is
// in curriculum, others bypass"). The datamodule attaches per-graph
// ``difficulty`` + ``in_scope`` at setup; ``g.losses.curriculum(...)``
// reads them off the batch.
//
// `overrides` is the merge knob for niche datamodule ablations not
// covered by the named knobs.

function(source,
         label_filter=null,
         difficulty=null,
         scope_label=0,
         overrides={})
  // conv_type/heads are model properties — read by GraphModuleBase.compute_budget
  // off the model's own hparams, not duplicated onto the DataModule.
  {
    data: {
      class_path: 'graphids.core.data.datamodule.GraphDataModule',
      init_args: std.mergePatch(
        { dataset: source }
        + (if label_filter != null then { label_filter: label_filter } else {})
        + (if difficulty != null then {
             difficulty: difficulty,
             scope_label: scope_label,
           } else {}),
        overrides,
      ),
    },
  }
