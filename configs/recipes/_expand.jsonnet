local recipes = import '../_lib/recipes.libsonnet';

function(recipe, valid_scales, valid_fusion_methods)
  recipes.expand(recipe, valid_scales, valid_fusion_methods)
