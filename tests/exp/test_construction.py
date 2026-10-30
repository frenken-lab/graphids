"""Construction errors and metric conversion at the Lightning boundary."""

import pytest

from graphids.exp.run import _scalar_metrics, build_component, resolve_spec


def test_component_builder_errors_propagate_without_retry(monkeypatch):
    """A TypeError inside a builder must keep its original identity and traceback."""
    from graphids.data.config import CANEventDataConfig

    error = TypeError("bug inside builder")
    calls = []

    def broken_build(self):
        calls.append("build")
        raise error

    monkeypatch.setattr(CANEventDataConfig, "build", broken_build)
    with pytest.raises(TypeError) as caught:
        build_component(
            {
                "type": "can_event_dm",
                "manifest_path": "unused",
                "datasets": ["toy"],
                "cache_root": "unused",
            }
        )
    assert caught.value is error
    assert calls == ["build"]
    with pytest.raises(ValueError, match="unknown component type"):
        resolve_spec({"type": "missing_component"})


def test_nested_component_specs_build_through_builtin_and_class_paths(tmp_path):
    """Nested specs retain data contracts and resolve the moved callback path."""
    from graphids.callbacks import Sha256ModelCheckpoint
    from graphids.data.config import CANEventDataConfig
    from graphids.data.events import CANEventDataModule
    from graphids.data.representation import CANEventRepresentationConfig
    from graphids.models.config import EventGraphMambaConfig
    from graphids.models.event_graph_mamba import EventGraphMamba

    data_args = dict(
        manifest_path="unused",
        datasets=["toy"],
        cache_root="unused",
        representation_cfg={
            "class_path": "graphids.data.representation.CANEventRepresentationConfig"
        },
    )
    specs = [
        {"type": "can_event_dm", **data_args},
        {
            "class_path": "graphids.data.config.CANEventDataConfig",
            "init_args": data_args,
        },
        {"type": "event_graph_mamba", "width": 8, "backend": "reference"},
        {
            "class_path": "graphids.callbacks.Sha256ModelCheckpoint",
            "init_args": {"dirpath": str(tmp_path), "monitor": "val_loss"},
        },
    ]
    resolved = resolve_spec({"components": specs})["components"]
    assert isinstance(resolved[0], CANEventDataConfig)
    assert resolved[0] == resolved[1]
    assert resolved[0].representation_cfg == CANEventRepresentationConfig()
    assert isinstance(resolved[2], EventGraphMambaConfig)
    assert isinstance(resolved[3], Sha256ModelCheckpoint)
    assert isinstance(build_component(resolved[0]), CANEventDataModule)
    assert isinstance(build_component(resolved[2]), EventGraphMamba)
    assert build_component(resolved[3]) is resolved[3]


def test_config_imports_do_not_load_training_dependencies():
    """Config inspection remains usable before torch or Lightning is installed."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'lightning', 'mamba_ssm'}:
            raise AssertionError(f'config import loaded {fullname}')

sys.meta_path.insert(0, NoTrainingImports())
import graphids.data.config
import graphids.models.config
import graphids.exp.config
from graphids.exp.run import resolve_spec

resolve_spec({'type': 'event_graph_mamba'})
resolve_spec({
    'type': 'can_event_dm', 'manifest_path': 'unused',
    'datasets': ['toy'], 'cache_root': 'unused',
})
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_scalar_metrics_preserve_scalars_and_ignore_vectors():
    """Offline ingestion receives numeric scalars, without hiding conversion errors."""
    import numpy as np
    import torch

    metrics = {
        "loss": torch.tensor(0.25, requires_grad=True),
        "count": 3,
        "numpy": np.float32(0.5),
        "vector": torch.ones(2),
        "empty": torch.empty(0),
        "metadata": "unscored",
    }
    assert _scalar_metrics(metrics) == {"loss": 0.25, "count": 3.0, "numpy": 0.5}
