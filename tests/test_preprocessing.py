"""
Comprehensive test suite for CAN-Graph preprocessing functionality.

Tests cover the new modular preprocessing pipeline (Phase 3):
    - EntityVocabulary (build, encode, persistence)
    - IRSchema validation
    - CANBusAdapter (file discovery, raw→IR conversion)
    - GraphEngine (sliding window graph construction)
    - GraphDataset (wrapper, stats, consistency validation)
    - process_dataset (end-to-end pipeline)

Run with: python -m pytest tests/test_preprocessing.py -v
"""

import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphids.config import EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT
from graphids.core.preprocessing.adapters.can_bus import CANBusAdapter
from graphids.core.preprocessing.dataset import GraphDataset
from graphids.core.preprocessing.engine import GraphEngine
from graphids.core.preprocessing.parallel import process_dataset
from graphids.core.preprocessing.schema import CAN_BUS_SCHEMA, IRSchema
from graphids.core.preprocessing.vocabulary import EntityVocabulary, _safe_hex_to_int


class TestHexConversion(unittest.TestCase):
    """Test hex-to-decimal conversion robustness."""

    def test_valid_hex_strings(self):
        self.assertEqual(_safe_hex_to_int("1A"), 26)
        self.assertEqual(_safe_hex_to_int("FF"), 255)
        self.assertEqual(_safe_hex_to_int("0"), 0)

    def test_invalid_inputs(self):
        self.assertIsNone(_safe_hex_to_int("XYZ"))
        self.assertIsNone(_safe_hex_to_int(""))
        self.assertIsNone(_safe_hex_to_int(None))

    def test_numeric_inputs(self):
        self.assertEqual(_safe_hex_to_int(123), 123)
        self.assertEqual(_safe_hex_to_int(0), 0)


class TestEntityVocabulary(unittest.TestCase):
    """Test EntityVocabulary build, encode, and persistence."""

    def test_build_from_ids(self):
        vocab = EntityVocabulary.build_from_ids([100, 200, 300])
        self.assertEqual(len(vocab), 4)  # 3 IDs + OOV
        self.assertIn("OOV", vocab)
        self.assertIn(100, vocab)

    def test_encode_known(self):
        vocab = EntityVocabulary.build_from_ids([10, 20, 30])
        idx = vocab.encode(10)
        self.assertEqual(idx, 0)  # sorted: 10=0, 20=1, 30=2

    def test_encode_oov(self):
        vocab = EntityVocabulary.build_from_ids([10, 20])
        oov = vocab.oov_index
        self.assertEqual(vocab.encode(999), oov)

    def test_encode_series(self):
        vocab = EntityVocabulary.build_from_ids([10, 20, 30])
        series = pd.Series([10, 20, 999])
        encoded = vocab.encode_series(series)
        self.assertEqual(encoded.iloc[0], vocab.encode(10))
        self.assertEqual(encoded.iloc[2], vocab.oov_index)

    def test_roundtrip_persistence(self):
        import tempfile

        vocab = EntityVocabulary.build_from_ids([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            vocab.save(f.name)
            loaded = EntityVocabulary.load(f.name)
        self.assertEqual(len(loaded), len(vocab))
        self.assertEqual(loaded.encode(1), vocab.encode(1))
        self.assertEqual(loaded.oov_index, vocab.oov_index)

    def test_from_legacy_mapping(self):
        legacy = {100: 0, 200: 1, "OOV": 2}
        vocab = EntityVocabulary.from_legacy_mapping(legacy)
        self.assertEqual(vocab.encode(100), 0)
        self.assertEqual(vocab.oov_index, 2)

    def test_from_legacy_mapping_adds_oov(self):
        legacy = {100: 0, 200: 1}
        vocab = EntityVocabulary.from_legacy_mapping(legacy)
        self.assertIn("OOV", vocab)
        self.assertEqual(vocab.oov_index, 2)


class TestIRSchema(unittest.TestCase):
    """Test IRSchema validation."""

    def test_can_bus_schema_columns(self):
        schema = CAN_BUS_SCHEMA
        cols = schema.columns
        self.assertEqual(cols[0], "entity_id")
        self.assertEqual(cols[-1], "label")
        self.assertEqual(len(cols), 1 + 8 + 3)  # entity + 8 features + src/tgt/label

    def test_validate_valid_df(self):
        schema = IRSchema(num_features=2)
        df = pd.DataFrame(
            {
                "entity_id": [0, 1, 2],
                "feature_0": [0.1, 0.2, 0.3],
                "feature_1": [0.4, 0.5, 0.6],
                "source_id": [0, 1, 0],
                "target_id": [1, 2, 1],
                "label": [0, 1, 0],
            }
        )
        result = schema.validate(df)
        self.assertEqual(len(result), 3)

    def test_validate_missing_column(self):
        schema = IRSchema(num_features=2)
        df = pd.DataFrame(
            {
                "entity_id": [0],
                "feature_0": [0.1],
                # missing feature_1
                "source_id": [0],
                "target_id": [1],
                "label": [0],
            }
        )
        with self.assertRaises(ValueError):
            schema.validate(df)

    def test_col_indices(self):
        schema = IRSchema(num_features=8)
        # entity_id(0) + feature_0..7(1..8) + source_id(9) + target_id(10) + label(11)
        self.assertEqual(schema.col_source, 9)
        self.assertEqual(schema.col_target, 10)
        self.assertEqual(schema.col_label, 11)


class TestGraphEngine(unittest.TestCase):
    """Test GraphEngine graph construction."""

    def _make_ir_df(self, n_rows=200, n_features=8, n_ids=5):
        """Create a synthetic IR DataFrame for testing."""
        rng = np.random.RandomState(42)
        schema = IRSchema(num_features=n_features)
        ids = list(range(n_ids))

        data = {"entity_id": rng.choice(ids, n_rows)}
        for i in range(n_features):
            data[f"feature_{i}"] = rng.rand(n_rows)

        sources = rng.choice(ids, n_rows)
        targets = np.roll(sources, -1)  # simple temporal adjacency
        data["source_id"] = sources
        data["target_id"] = targets
        data["label"] = rng.choice([0, 1], n_rows, p=[0.9, 0.1])

        return pd.DataFrame(data), schema

    def test_create_graphs_returns_list(self):
        ir_df, schema = self._make_ir_df()
        engine = GraphEngine(schema, window_size=50, stride=50)
        graphs = engine.create_graphs(ir_df)
        self.assertIsInstance(graphs, list)
        self.assertGreater(len(graphs), 0)

    def test_graph_structure(self):
        ir_df, schema = self._make_ir_df()
        engine = GraphEngine(schema, window_size=50, stride=50)
        graphs = engine.create_graphs(ir_df)

        for g in graphs[:5]:
            self.assertIsInstance(g, Data)
            self.assertIsNotNone(g.x)
            self.assertIsNotNone(g.edge_index)
            self.assertIsNotNone(g.edge_attr)
            self.assertIsNotNone(g.y)

            # Feature dimensions (derived from schema, not hardcoded constant)
            self.assertEqual(g.x.size(1), engine.node_feature_count)
            self.assertEqual(g.edge_attr.size(1), EDGE_FEATURE_COUNT)

            # Dtypes
            self.assertEqual(g.x.dtype, torch.float)
            self.assertEqual(g.edge_index.dtype, torch.long)

            # No NaN/Inf
            self.assertFalse(torch.isnan(g.x).any())
            self.assertFalse(torch.isinf(g.x).any())
            self.assertFalse(torch.isnan(g.edge_attr).any())
            self.assertFalse(torch.isinf(g.edge_attr).any())

    def test_window_count(self):
        ir_df, schema = self._make_ir_df(n_rows=200)
        engine = GraphEngine(schema, window_size=50, stride=50)
        graphs = engine.create_graphs(ir_df)
        expected = max(1, (200 - 50) // 50 + 1)  # = 4
        self.assertEqual(len(graphs), expected)


class TestCANBusAdapter(unittest.TestCase):
    """Test CAN bus adapter file discovery and conversion."""

    @classmethod
    def setUpClass(cls):
        cls.test_root = "data/automotive/hcrl_sa"
        import os

        if not os.path.isdir(cls.test_root):
            raise unittest.SkipTest(f"Test data not available at {cls.test_root}")

    def test_discover_files(self):
        adapter = CANBusAdapter()
        files = adapter.discover_files(self.test_root, "train_")
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(f.suffix == ".csv")

    def test_build_vocabulary(self):
        adapter = CANBusAdapter()
        files = adapter.discover_files(self.test_root, "train_")
        vocab = adapter.build_vocabulary(files)
        self.assertIsInstance(vocab, EntityVocabulary)
        self.assertGreater(len(vocab), 1)  # more than just OOV
        self.assertIn("OOV", vocab)

    def test_read_and_convert(self):
        adapter = CANBusAdapter()
        files = adapter.discover_files(self.test_root, "train_")
        if not files:
            self.skipTest("No CSV files found")

        vocab = adapter.build_vocabulary(files)
        ir_df = adapter.read_and_convert(files[0], vocab)

        # Check IR schema conformance
        self.assertEqual(list(ir_df.columns), adapter.schema.columns)
        self.assertFalse(ir_df.isnull().any().any())

        # Check feature normalization (bytes should be in [0, 1])
        for col in [f"feature_{i}" for i in range(8)]:
            self.assertTrue(
                (ir_df[col] >= 0).all() and (ir_df[col] <= 1).all(),
                f"{col} not properly normalized",
            )


class TestGraphDataset(unittest.TestCase):
    """Test GraphDataset wrapper."""

    def _make_graphs(self, n=10):
        graphs = []
        for i in range(n):
            x = torch.randn(5, NODE_FEATURE_COUNT)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
            edge_attr = torch.randn(3, EDGE_FEATURE_COUNT)
            y = torch.tensor(i % 2, dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        return graphs

    def test_basic_ops(self):
        graphs = self._make_graphs()
        ds = GraphDataset(graphs)
        self.assertEqual(len(ds), 10)
        self.assertIsInstance(ds[0], Data)

    def test_stats(self):
        graphs = self._make_graphs()
        ds = GraphDataset(graphs)
        stats = ds.get_stats()
        self.assertEqual(stats["num_graphs"], 10)
        self.assertIn("avg_nodes", stats)
        self.assertEqual(stats["normal_graphs"] + stats["attack_graphs"], 10)

    def test_inconsistent_features_raises(self):
        g1 = Data(x=torch.randn(3, 5), edge_index=torch.zeros(2, 1, dtype=torch.long))
        g2 = Data(x=torch.randn(3, 7), edge_index=torch.zeros(2, 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            GraphDataset([g1, g2])


class TestEndToEnd(unittest.TestCase):
    """End-to-end test using real data (skipped if unavailable)."""

    @classmethod
    def setUpClass(cls):
        cls.test_root = "data/automotive/hcrl_sa"
        import os

        if not os.path.isdir(cls.test_root):
            raise unittest.SkipTest(f"Test data not available at {cls.test_root}")

    def test_process_dataset(self):
        graphs, vocab_dict = process_dataset(
            self.test_root,
            split="train_",
            return_vocab=True,
            verbose=True,
        )
        self.assertIsInstance(graphs, list)
        self.assertGreater(len(graphs), 0)
        self.assertIsInstance(vocab_dict, dict)
        self.assertIn("OOV", vocab_dict)

    def test_graph_quality(self):
        graphs = process_dataset(self.test_root, split="train_")
        for g in graphs[:10]:
            self.assertEqual(g.x.size(1), NODE_FEATURE_COUNT)
            self.assertEqual(g.edge_attr.size(1), EDGE_FEATURE_COUNT)
            self.assertFalse(torch.isnan(g.x).any())
            self.assertFalse(torch.isnan(g.edge_attr).any())
            # Payload features should be in [0, 1]
            payload = g.x[:, 1:9]
            self.assertTrue(torch.all(payload >= 0) and torch.all(payload <= 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
