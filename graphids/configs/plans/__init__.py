"""Plan modules.

Each plan exposes ``build(*, dataset: str, seed: int) -> list[dict]``.
``graphids run <name> --dataset X --seed N`` imports
``graphids.configs.plans.<name>`` and calls ``build`` to produce a row
array validated by :class:`graphids.configs.blueprint.BlueprintArray`.
"""
