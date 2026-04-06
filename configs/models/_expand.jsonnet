// Expand a model's base + scale config from a family libsonnet.
// Used by recipe expansion to preview model config without a full stage render.
function(family, model_type, scale)
  local lib = import (family + '.libsonnet');
  lib[model_type].base + lib[model_type].scales[scale]
