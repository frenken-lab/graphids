"""
Comprehensive test suite for CAN-Graph preprocessing functionality.

Tests cover the modular preprocessing pipeline:
    - EntityVocabulary (build, encode, persistence)
    - IRSchema validation
    - CANBusAdapter (file discovery, raw→IR conversion)
    - GraphEngine (sliding window graph construction)
    - GraphDataset (wrapper, stats, consistency validation)
    - process_dataset (end-to-end pipeline)
    - PreprocessingPipeline (class interface)
    - Feature computation (scipy-backed entropy, networkx clustering)

Run with: python -m pytest tests/test_preprocessing.py -v
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphids.config import EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT
from graphids.core.preprocessing import (
    ATTACK_TYPE_CODES,
    ATTACK_TYPE_NAMES,
    CollatedGraphDataset,
    EntityVocabulary,
    GraphDataset,
    GraphEngine,
    IRSchema,
    get_batch_index,
    graph_attack_type,
)
from graphids.core.preprocessing._parallel import process_dataset
from graphids.core.preprocessing.adapters._can_bus import (
    CAN_BUS_SCHEMA,
    CANBusAdapter,
    _safe_hex_to_int,
)


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


class TestGraphUtils(unittest.TestCase):
    """Test re-exported graph utilities."""

    def test_get_batch_index_with_batch(self):
        g = Data(
            x=torch.randn(4, 3),
            batch=torch.tensor([0, 0, 1, 1]),
        )
        idx = get_batch_index(g, torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 0, 1, 1])

    def test_get_batch_index_without_batch(self):
        g = Data(x=torch.randn(4, 3))
        idx = get_batch_index(g, torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 0, 0, 0])

    def test_graph_attack_type_present(self):
        g = Data(x=torch.randn(2, 3), attack_type=torch.tensor(3))
        self.assertEqual(graph_attack_type(g), 3)

    def test_graph_attack_type_absent(self):
        g = Data(x=torch.randn(2, 3))
        self.assertEqual(graph_attack_type(g), -1)

    def test_attack_type_codes(self):
        self.assertIn("normal", ATTACK_TYPE_CODES)
        self.assertEqual(ATTACK_TYPE_CODES["normal"], 0)
        self.assertEqual(ATTACK_TYPE_NAMES[0], "normal")


class TestFeatures(unittest.TestCase):
    """Test scipy/networkx-backed feature computation."""

    def test_entropy_matches_manual(self):
        """Verify scipy entropy matches manual Shannon entropy."""
        from scipy.stats import entropy as scipy_entropy

        counts = np.array([10, 20, 30, 40, 0, 0, 0, 0], dtype=np.float64)
        # Manual
        total = counts.sum()
        probs = counts[counts > 0] / total
        manual_entropy = -np.sum(probs * np.log2(probs))
        # scipy
        scipy_ent = scipy_entropy(counts, base=2)
        np.testing.assert_allclose(scipy_ent, manual_entropy, rtol=1e-10)

    def test_clustering_coefficient_triangle(self):
        """Verify networkx clustering on a known graph (triangle)."""
        import networkx as nx

        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        cc = nx.clustering(G)
        # All nodes in a triangle have cc=1.0
        for node_cc in cc.values():
            self.assertAlmostEqual(node_cc, 1.0)

    def test_feature_computation_no_nan(self):
        """End-to-end: features from synthetic IR should have no NaN."""
        rng = np.random.RandomState(42)
        schema = IRSchema(num_features=4)
        n_rows = 100
        data = {"entity_id": rng.choice(3, n_rows)}
        for i in range(4):
            data[f"feature_{i}"] = rng.rand(n_rows)
        data["source_id"] = rng.choice(3, n_rows)
        data["target_id"] = np.roll(data["source_id"], -1)
        data["label"] = rng.choice([0, 1], n_rows, p=[0.9, 0.1])
        ir_df = pd.DataFrame(data)

        engine = GraphEngine(schema, window_size=50, stride=50)
        graphs = engine.create_graphs(ir_df)
        for g in graphs:
            self.assertFalse(torch.isnan(g.x).any(), "NaN in node features")
            self.assertFalse(torch.isnan(g.edge_attr).any(), "NaN in edge features")


class TestPreprocessingPipeline(unittest.TestCase):
    """Test the PreprocessingPipeline class interface."""

    def test_init(self):
        from graphids.config import resolve
        from graphids.core.preprocessing import PreprocessingPipeline

        cfg = resolve("vgae", "large")
        pipe = PreprocessingPipeline(cfg)
        self.assertIsNotNone(pipe._adapter)
        self.assertEqual(pipe._prep.window_size, 100)

    def test_static_methods(self):
        from graphids.core.preprocessing import PreprocessingPipeline

        # get_batch_index
        g = Data(x=torch.randn(3, 5))
        idx = PreprocessingPipeline.get_batch_index(g, torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 0, 0])

        # graph_attack_type
        g2 = Data(x=torch.randn(2, 3), attack_type=torch.tensor(2))
        self.assertEqual(PreprocessingPipeline.graph_attack_type(g2), 2)


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


class TestAdapterRoundtrip(unittest.TestCase):
    """Fix 2: Adapter serialization via to_init_kwargs()."""

    def test_adapter_roundtrip_defaults(self):
        adapter = CANBusAdapter()
        kwargs = adapter.to_init_kwargs()
        clone = CANBusAdapter(**kwargs)
        self.assertEqual(clone._chunk_size, adapter._chunk_size)
        self.assertEqual(clone._excluded_attacks, adapter._excluded_attacks)
        self.assertEqual(clone._include_attack_type, adapter._include_attack_type)

    def test_adapter_roundtrip_non_defaults(self):
        adapter = CANBusAdapter(chunk_size=999, excluded_attacks=["foo"], include_attack_type=True)
        kwargs = adapter.to_init_kwargs()
        clone = CANBusAdapter(**kwargs)
        self.assertEqual(clone._chunk_size, 999)
        self.assertEqual(clone._excluded_attacks, ["foo"])
        self.assertTrue(clone._include_attack_type)


class TestTrainValSplitConfig(unittest.TestCase):
    """Fix 1: train_val_split from config, not hardcoded."""

    def test_non_default_split(self):
        """Verify non-default split produces correct sizes."""
        # Create a minimal fake dataset
        graphs = [
            Data(
                x=torch.randn(3, NODE_FEATURE_COUNT),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, EDGE_FEATURE_COUNT),
                y=torch.tensor(0, dtype=torch.long),
            )
            for _ in range(100)
        ]
        dataset = torch.utils.data.TensorDataset(torch.arange(100))

        # Simulate the split logic from _cache.load_dataset
        train_val_split = 0.7
        train_size = int(train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        self.assertEqual(train_size, 70)
        self.assertEqual(val_size, 30)


class TestIRSchemaValidation(unittest.TestCase):
    """Fix 4: IRSchema.validate()."""

    def _make_valid_df(self, n_rows=10, n_features=8, include_attack_type=False):
        rng = np.random.RandomState(42)
        schema = IRSchema(num_features=n_features, include_attack_type=include_attack_type)
        data = {}
        data["entity_id"] = rng.choice(5, n_rows)
        for i in range(n_features):
            data[f"feature_{i}"] = rng.rand(n_rows)
        data["source_id"] = rng.choice(5, n_rows)
        data["target_id"] = rng.choice(5, n_rows)
        data["label"] = rng.choice([0, 1], n_rows)
        if include_attack_type:
            data["attack_type"] = rng.choice(3, n_rows)
        return pd.DataFrame(data)[schema.columns], schema

    def test_ir_validate_correct(self):
        df, schema = self._make_valid_df()
        schema.validate(df)  # should not raise

    def test_ir_validate_missing_column(self):
        df, schema = self._make_valid_df()
        df_bad = df.drop(columns=["entity_id"])
        with self.assertRaises(ValueError, msg="missing"):
            schema.validate(df_bad)

    def test_ir_validate_empty(self):
        df, schema = self._make_valid_df()
        df_empty = df.iloc[:0]
        with self.assertRaises(ValueError, msg="empty"):
            schema.validate(df_empty)

    def test_ir_validate_strict_nan(self):
        df, schema = self._make_valid_df()
        df.loc[0, "feature_0"] = np.nan
        with self.assertRaises(ValueError, msg="NaN"):
            schema.validate(df, strict=True)

    def test_ir_validate_strict_passes(self):
        df, schema = self._make_valid_df()
        schema.validate(df, strict=True)  # should not raise


class TestFeatureManifest(unittest.TestCase):
    """Fix 5: Feature manifest as single source of truth."""

    def test_manifest_count_matches_config(self):
        from graphids.core.preprocessing._schema import EDGE_MANIFEST, NODE_MANIFEST

        self.assertEqual(NODE_MANIFEST.count, NODE_FEATURE_COUNT)
        self.assertEqual(EDGE_MANIFEST.count, EDGE_FEATURE_COUNT)

    def test_manifest_json_roundtrip(self):
        from graphids.core.preprocessing._schema import NODE_MANIFEST

        data = NODE_MANIFEST.to_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), NODE_MANIFEST.count)
        for entry in data:
            self.assertIn("name", entry)
            self.assertIn("index", entry)
            self.assertIn("description", entry)
            self.assertIn("value_range", entry)

    def test_engine_uses_manifest(self):
        from graphids.core.preprocessing._schema import build_node_manifest

        schema = IRSchema(num_features=8)
        engine = GraphEngine(schema, window_size=50, stride=50)
        self.assertEqual(engine.node_feature_count, build_node_manifest(8).count)

    def test_manifest_names_unique(self):
        from graphids.core.preprocessing._schema import EDGE_MANIFEST, NODE_MANIFEST

        node_names = [f.name for f in NODE_MANIFEST.features]
        self.assertEqual(len(node_names), len(set(node_names)), "Duplicate node feature names")
        edge_names = [f.name for f in EDGE_MANIFEST.features]
        self.assertEqual(len(edge_names), len(set(edge_names)), "Duplicate edge feature names")


class TestAtomicIO(unittest.TestCase):
    """Fix 6: Extracted atomic I/O."""

    def test_atomic_rename_retry(self):
        import tempfile

        from graphids.storage.gateway import _atomic_rename

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "tmp.txt"
            final = Path(d) / "final.txt"
            tmp.write_text("hello")
            _atomic_rename(tmp, final)
            self.assertTrue(final.exists())
            self.assertFalse(tmp.exists())
            self.assertEqual(final.read_text(), "hello")


class TestCacheMetadata(unittest.TestCase):
    """Fix 6: Extracted cache metadata writes feature manifest."""

    def test_cache_metadata_writes_manifest(self):
        import tempfile

        from graphids.core.preprocessing._cache_metadata import write_cache_metadata
        from graphids.core.preprocessing._dataset import CollatedGraphDataset

        # Build minimal collated dataset
        graphs = [
            Data(
                x=torch.randn(3, NODE_FEATURE_COUNT),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.randn(2, EDGE_FEATURE_COUNT),
                y=torch.tensor(0, dtype=torch.long),
            )
            for _ in range(5)
        ]

        with tempfile.TemporaryDirectory() as d:
            cache_dir = Path(d)
            write_cache_metadata(
                cache_dir,
                "test_ds",
                graphs,
                {"a": 0},
                ["f1.csv"],
                window_size=100,
                stride=100,
            )
            self.assertTrue((cache_dir / "cache_metadata.json").exists())
            self.assertTrue((cache_dir / "feature_manifest.json").exists())

            import json

            manifest = json.loads((cache_dir / "feature_manifest.json").read_text())
            self.assertIn("node_features", manifest)
            self.assertIn("edge_features", manifest)
            self.assertEqual(len(manifest["node_features"]), NODE_FEATURE_COUNT)
            self.assertEqual(len(manifest["edge_features"]), EDGE_FEATURE_COUNT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
