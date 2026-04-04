from graphids.config.recipe_expand import expand_recipe_configs


def test_expand_sweep_with_kd_auxiliary() -> None:
    recipe = {
        "sweeps": [
            {
                "model_family": "gat",
                "stage": "normal",
                "scale": ["small"],
                "model_overrides": {"init_args": {"loss_fn": ["ce"]}},
                "kd": {
                    "type": "kd",
                    "alpha": 0.5,
                    "teacher_config": "gat_normal_large",
                    "teacher_scale": "large",
                    "temperature": 2.0,
                },
            }
        ]
    }

    expanded = expand_recipe_configs(
        recipe,
        valid_scales=frozenset({"small", "large"}),
        valid_fusion_methods=frozenset({"bandit", "dqn"}),
    )

    assert expanded["configs"]
    config = next(iter(expanded["configs"].values()))
    assert "auxiliaries" in config
    assert config["auxiliaries"][0]["type"] == "kd"
    assert config["auxiliaries"][0]["teacher_config"] == "gat_normal_large"
    assert config["auxiliaries"][0]["teacher_scale"] == "large"
