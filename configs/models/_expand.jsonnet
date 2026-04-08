// Expand a model's base + scale config from a family libsonnet.
// Used by budget probe to preview model config without a full stage render.
local libs = {
  unsupervised: import 'unsupervised.libsonnet',
  supervised: import 'supervised.libsonnet',
  fusion: import 'fusion.libsonnet',
};
function(family, model_type, scale)
  libs[family][model_type][scale]
