"""Python-native config composition.

A plan module under ``graphids.configs.plans`` exposes
``build(dataset, seed) -> list[dict]``. The single composer
``graphids.configs.compose.compose`` (with thin ``fusion`` wrapper)
combines primitives (``graphids.configs.primitives``) into a frozen
:class:`RowSpec` whose ``rendered`` is a typed
:class:`graphids.configs.blueprint.RenderedConfig`. Output dicts are validated
end-to-end by :class:`graphids.configs.blueprint.BlueprintArray`.
"""
