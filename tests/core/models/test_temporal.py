"""Temporal model + Lightning module + dataset tests."""

from __future__ import annotations

import torch

from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, make_graph


class TestTemporalStage:
    """Temporal model + Lightning module + dataset."""

    @staticmethod
    def _make_sequence(seq_len=4, num_nodes=8):
        return [make_graph(num_nodes=num_nodes) for _ in range(seq_len)]

    def test_temporal_classifier_forward(self):
        from graphids.core.models.temporal_family.temporal import TemporalGraphClassifier
        from graphids.core.models.supervised.gat import GATWithJK

        spatial = GATWithJK(
            num_ids=NUM_IDS, in_channels=IN_CHANNELS, hidden_channels=16,
            out_channels=2, num_layers=2, heads=2, dropout=0.0,
            num_fc_layers=2, embedding_dim=4, conv_type="gatv2",
            edge_dim=EDGE_DIM, pool_aggrs=("mean",), proj_dim=0,
        )
        model = TemporalGraphClassifier(
            spatial_encoder=spatial, spatial_dim=32, num_classes=2,
            temporal_hidden=16, temporal_heads=2, temporal_layers=1,
            max_seq_len=4,
        )
        sequences = [self._make_sequence() for _ in range(3)]
        model.eval()
        with torch.no_grad():
            logits = model(sequences)
        assert logits.shape == (3, 2)

    def test_collate_produces_correct_shapes(self):
        from graphids.core.preprocessing._temporal import collate_temporal
        batch_data = [
            ([make_graph() for _ in range(4)], 0),
            ([make_graph() for _ in range(4)], 1),
        ]
        graph_sequences, labels = collate_temporal(batch_data)
        assert len(graph_sequences) == 2
        assert len(graph_sequences[0]) == 4
        assert labels.shape == (2,)

    def test_temporal_lightning_module_has_test_metrics(self):
        from graphids.core.models.temporal_family.temporal import TemporalLightningModule
        assert hasattr(TemporalLightningModule, "test_step")
