// Curriculum-weighted loss wrapper — composes a base classification loss
// with a visibility schedule. The base loss is forced to ``reduction='none'``
// so the wrapper can apply per-example weights before reducing.
//
// Usage from a plan:
//   loss = g.losses.curriculum(g.losses.focal())
//   loss = g.losses.curriculum(g.losses.focal(), schedule={
//     class_path: 'graphids.core.data.preprocessing.curriculum.LinearRampSchedule',
//     init_args: { start_ratio: 1.0, end_ratio: 10.0, max_epochs: 300 },
//   })
//
// Difficulty + scope live on the datamodule (``g.data.datamodule.graph``
// with ``difficulty`` + ``scope_label`` kwargs) — the loss reads
// ``batch.difficulty`` / ``batch.in_scope`` produced by collation.

local default_schedule = {
  class_path: 'graphids.core.data.preprocessing.curriculum.LinearRampSchedule',
  init_args: { start_ratio: 1.0, end_ratio: 10.0, max_epochs: 300 },
};

function(base, schedule=default_schedule)
  // Force reduction='none' on the base. ``+:`` ensures we deep-merge into
  // existing init_args (e.g. focal's gamma) rather than replacing them.
  local base_per_example = base.loss_fn + {
    init_args+: { reduction: 'none' },
  };
  {
    loss_fn: {
      class_path: 'graphids.core.losses.CurriculumWeightedLoss',
      init_args: {
        base_loss: base_per_example,
        schedule: schedule,
      },
    },
  }
