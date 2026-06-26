"""Shared infrastructure for temporal event models."""

from __future__ import annotations

from typing import Any

import torch

from graphids.core.models.base import _ModelBase


class TemporalModuleBase(_ModelBase):
    """Base class for models that consume PyG ``TemporalData`` batches."""

    automatic_optimization = True

    def prepare_from_datamodule(self, dm) -> None:
        already_built = getattr(self, "_built", False)
        if not already_built:
            for k in ("num_ids", "in_channels", "num_classes"):
                v = getattr(dm, k)
                setattr(self, k, v)
                self.hparams[k] = v
            self._build()
            self._built = True

        tests = getattr(dm, "test_data", None)
        self._test_set_names = list(tests.keys()) if tests else ["test"]
        self._attack_type_names = dict(getattr(dm, "attack_type_names", {0: "benign"}))

    def _init_post(self, locals_dict: dict[str, Any]) -> None:
        self._store_init_kwargs(locals_dict)
        self._built = False
        if int(getattr(self, "num_ids", 0)) > 0 and int(getattr(self, "in_channels", 0)) > 0:
            self._build()
            self._built = True

    def _build(self) -> None:
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=float(getattr(self.hparams, "lr", 1e-3)),
            weight_decay=float(getattr(self.hparams, "weight_decay", 0.0)),
        )

    @staticmethod
    def scored_mask(batch) -> torch.Tensor:
        mask = getattr(batch, "is_scored", None)
        if mask is None:
            return torch.ones_like(batch.y, dtype=torch.bool)
        return mask.bool()
