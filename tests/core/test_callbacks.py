"""TauNormCallback contract — classifier-key resolution.

CONTRACT: ``TauNormCallback._resolve_classifier_key`` must target the
top-level ``fc_layers.<N>.weight`` keys produced by GAT's collapsed
``_build()`` (no ``self.model = ...`` indirection — see
``graphids/core/models/base.py:516``). Lightning's ``state_dict`` for
``GAT(GraphModuleBase)`` writes ``fc_layers.<N>.weight`` directly; the
old ``model.fc_layers.<N>.weight`` prefix would silently fail every
GAT τ-norm run with a KeyError at fit-end.
"""

from __future__ import annotations

import pytest
import torch

from graphids.core.callbacks import TauNormCallback


def _synthetic_gat_state(num_classes: int = 2, fc_input_dim: int = 32) -> dict[str, torch.Tensor]:
    """Mirror GAT._build's fc_layers shape: [Linear, ReLU, Dropout, ..., Linear].

    fc_layers.0 / fc_layers.3 are intermediate Linears (out=fc_input_dim);
    fc_layers.6 is the final classifier (out=num_classes). ReLU/Dropout
    contribute no parameters. Other modules (input_encoder, convs, jk)
    are present so the resolver must filter to the fc_layers prefix.
    """
    return {
        "input_encoder.embedding.weight": torch.randn(64, 16),
        "convs.0.lin.weight": torch.randn(384, 64),
        "jk.lstm.weight_ih_l0": torch.randn(1024, 384),
        "fc_layers.0.weight": torch.randn(fc_input_dim, fc_input_dim),
        "fc_layers.0.bias": torch.randn(fc_input_dim),
        "fc_layers.3.weight": torch.randn(fc_input_dim, fc_input_dim),
        "fc_layers.3.bias": torch.randn(fc_input_dim),
        "fc_layers.6.weight": torch.randn(num_classes, fc_input_dim),
        "fc_layers.6.bias": torch.randn(num_classes),
    }


class TestTauNormResolveClassifierKey:
    def test_picks_highest_indexed_2row_linear(self):
        state = _synthetic_gat_state()
        assert TauNormCallback._resolve_classifier_key(state) == "fc_layers.6.weight"

    def test_ignores_intermediate_fc_layers(self):
        # INVARIANT: intermediate fc Linears (fc_input_dim rows) must be
        # skipped — only the num_classes-row final layer is a classifier.
        state = _synthetic_gat_state(num_classes=2, fc_input_dim=32)
        key = TauNormCallback._resolve_classifier_key(state)
        assert state[key].shape[0] == 2

    def test_raises_on_no_fc_layers(self):
        # CONTRACT: τ-norm only supports GAT-shaped models. Non-GAT
        # state_dicts (no fc_layers) must fail loudly, not silently no-op.
        bad_state = {"input_encoder.embedding.weight": torch.randn(64, 16)}
        with pytest.raises(KeyError, match="fc_layers"):
            TauNormCallback._resolve_classifier_key(bad_state)

    def test_rejects_legacy_model_prefix(self):
        # REGRESSION: pre-collapse ckpts had ``model.fc_layers.<N>.weight``;
        # ``safe_load_checkpoint`` (base.py:520) strips the ``model.``
        # prefix on load, so by the time TauNorm sees a freshly-saved
        # Lightning ckpt the prefix is gone. If a state_dict somehow
        # arrives with only the legacy prefix, the resolver should NOT
        # match it (would indicate a load-path bug worth surfacing).
        legacy_state = {
            "model.fc_layers.6.weight": torch.randn(2, 32),
            "model.fc_layers.6.bias": torch.randn(2),
        }
        with pytest.raises(KeyError, match="fc_layers"):
            TauNormCallback._resolve_classifier_key(legacy_state)
