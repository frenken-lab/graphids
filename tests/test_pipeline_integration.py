"""Pipeline integration tests.

Exercises the full stage-to-stage contract with synthetic data on CPU.
Catches the class of bugs that have repeatedly crashed the SLURM pipeline:
  - Config serialization round-trip failures
  - Checkpoint save/load dimension mismatches (strict=True)
  - Frozen config propagation between stages
  - Path construction / aux suffix logic
  - Dead config params (config values that don't affect the model)
  - Validation gaps (missing files not caught early)

Run with:  python -m pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from tests.conftest import IN_CHANNELS, NUM_IDS, _make_dataset, _make_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_root(tmp_path):
    """Temporary experiment root directory."""
    return tmp_path / "experimentruns"


@pytest.fixture()
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Config serialization must preserve every field exactly."""

    def test_save_load_identity(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.resolver import resolve

        cfg = resolve("vgae", "small", dataset="set_01")
        p = tmp_path / "config.json"
        cfg.save(p)
        loaded = PipelineConfig.load(p)
        assert cfg == loaded

    def test_all_model_scale_round_trip(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.resolver import list_models, resolve

        for model_type, scales in list_models().items():
            for scale in scales:
                cfg = resolve(model_type, scale, dataset="hcrl_sa")
                p = tmp_path / f"{model_type}_{scale}.json"
                cfg.save(p)
                loaded = PipelineConfig.load(p)
                assert cfg == loaded, f"Round-trip failed for ({model_type}, {scale})"

    def test_tuple_survives_json(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.schema import VGAEArchitecture

        cfg = PipelineConfig(vgae=VGAEArchitecture(hidden_dims=(100, 50, 25)))
        p = tmp_path / "cfg.json"
        cfg.save(p)
        loaded = PipelineConfig.load(p)
        assert isinstance(loaded.vgae.hidden_dims, tuple)
        assert loaded.vgae.hidden_dims == (100, 50, 25)

    def test_legacy_flat_json_loads(self, tmp_path):
        """Old flat config.json files must still load correctly."""
        import json

        from graphids.config import PipelineConfig

        flat = {
            "dataset": "hcrl_sa",
            "model_size": "student",
            "seed": 42,
            "lr": 0.001,
            "max_epochs": 300,
            "batch_size": 4096,
            "patience": 50,
            "vgae_hidden_dims": [80, 40, 16],
            "vgae_latent_dim": 16,
            "vgae_heads": 1,
            "vgae_embedding_dim": 4,
            "vgae_dropout": 0.1,
            "gat_hidden": 24,
            "gat_layers": 2,
            "gat_heads": 4,
            "gat_dropout": 0.1,
            "gat_embedding_dim": 8,
            "gat_fc_layers": 3,
            "dqn_hidden": 160,
            "dqn_layers": 2,
            "dqn_gamma": 0.99,
            "use_kd": True,
            "teacher_path": "/some/path",
            "kd_temperature": 4.0,
            "kd_alpha": 0.7,
            "experiment_root": "experimentruns",
            "device": "cuda",
        }
        p = tmp_path / "legacy.json"
        p.write_text(json.dumps(flat))
        cfg = PipelineConfig.load(p)
        assert cfg.scale == "small"
        assert cfg.vgae.hidden_dims == (80, 40, 16)
        assert cfg.has_kd is True
        assert cfg.kd.temperature == 4.0


# ---------------------------------------------------------------------------
# 2. Model construction matches config
# ---------------------------------------------------------------------------


class TestModelMatchesConfig:
    """Every config param must actually affect the model it controls."""

    def test_vgae_dims_from_config(self):
        from graphids.config.resolver import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        for scale in ("large", "small"):
            cfg = resolve("vgae", scale)
            conv_type = cfg.vgae.conv_type
            model = GraphAutoencoderNeighborhood(
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
                hidden_dims=list(cfg.vgae.hidden_dims),
                latent_dim=cfg.vgae.latent_dim,
                encoder_heads=cfg.vgae.heads,
                embedding_dim=cfg.vgae.embedding_dim,
                dropout=cfg.vgae.dropout,
                conv_type=conv_type,
                edge_dim=cfg.vgae.edge_dim if conv_type in ("gatv2", "transformer") else None,
            )
            g = _make_graph()
            batch = torch.zeros(g.x.size(0), dtype=torch.long)
            model.eval()
            with torch.no_grad():
                out = model(g.x, g.edge_index, batch, edge_attr=g.edge_attr)
            z = out[3]
            assert z.shape[1] == cfg.vgae.latent_dim, (
                f"{scale} VGAE latent dim mismatch: got {z.shape[1]}, expected {cfg.vgae.latent_dim}"
            )

    def test_gat_dims_from_config(self):
        from graphids.config.resolver import resolve
        from graphids.core.models.gat import GATWithJK

        for scale in ("large", "small"):
            cfg = resolve("gat", scale)
            conv_type = cfg.gat.conv_type
            model = GATWithJK(
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
                hidden_channels=cfg.gat.hidden,
                out_channels=2,
                num_layers=cfg.gat.layers,
                heads=cfg.gat.heads,
                dropout=cfg.gat.dropout,
                num_fc_layers=cfg.gat.fc_layers,
                embedding_dim=cfg.gat.embedding_dim,
                conv_type=conv_type,
                edge_dim=cfg.gat.edge_dim if conv_type in ("gatv2", "transformer") else None,
            )
            g = _make_graph()
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            model.eval()
            with torch.no_grad():
                out = model(g)
            assert out.shape == (1, 2), f"{scale} GAT output shape wrong: {out.shape}"

    def test_dqn_dims_from_config(self):
        """dqn.hidden and dqn.layers must actually change QNetwork architecture."""
        from graphids.config.resolver import resolve
        from graphids.core.models.dqn import QNetwork

        large_cfg = resolve("dqn", "large")
        small_cfg = resolve("dqn", "small")

        large_net = QNetwork(
            15,
            large_cfg.fusion.alpha_steps,
            hidden_dim=large_cfg.dqn.hidden,
            num_layers=large_cfg.dqn.layers,
        )
        small_net = QNetwork(
            15,
            small_cfg.fusion.alpha_steps,
            hidden_dim=small_cfg.dqn.hidden,
            num_layers=small_cfg.dqn.layers,
        )

        large_params = sum(p.numel() for p in large_net.parameters())
        small_params = sum(p.numel() for p in small_net.parameters())
        assert large_params != small_params, (
            f"Large and small DQN have identical param count ({large_params}). "
            f"dqn.hidden/dqn.layers config params are not being used."
        )

    def test_dqn_agent_uses_config_batch_size(self):
        """Agent must use the config batch_size, not override it."""
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.registry import fusion_state_dim

        agent = EnhancedDQNFusionAgent(
            batch_size=64,
            buffer_size=500,
            device="cpu",
            state_dim=fusion_state_dim(),
        )
        assert agent.batch_size == 64
        assert agent.buffer_size == 500


# ---------------------------------------------------------------------------
# 3. Checkpoint save -> load round-trip (strict=True)
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """Saving then loading a model must reproduce identical weights."""

    def test_vgae_checkpoint_roundtrip(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.resolver import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        cfg = resolve("vgae", "small")
        conv_type = cfg.vgae.conv_type
        model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(cfg.vgae.hidden_dims),
            latent_dim=cfg.vgae.latent_dim,
            encoder_heads=cfg.vgae.heads,
            embedding_dim=cfg.vgae.embedding_dim,
            dropout=cfg.vgae.dropout,
            conv_type=conv_type,
            edge_dim=cfg.vgae.edge_dim if conv_type in ("gatv2", "transformer") else None,
        )
        ckpt = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), ckpt)
        cfg.save(tmp_path / "config.json")

        loaded_cfg = PipelineConfig.load(tmp_path / "config.json")
        loaded_conv = loaded_cfg.vgae.conv_type
        model2 = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(loaded_cfg.vgae.hidden_dims),
            latent_dim=loaded_cfg.vgae.latent_dim,
            encoder_heads=loaded_cfg.vgae.heads,
            embedding_dim=loaded_cfg.vgae.embedding_dim,
            dropout=loaded_cfg.vgae.dropout,
            conv_type=loaded_conv,
            edge_dim=loaded_cfg.vgae.edge_dim if loaded_conv in ("gatv2", "transformer") else None,
        )
        model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

        for (n1, p1), (_n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_gat_checkpoint_roundtrip(self, tmp_path):
        from graphids.config.resolver import resolve
        from graphids.core.models.gat import GATWithJK

        cfg = resolve("gat", "large")
        conv_type = cfg.gat.conv_type
        model = GATWithJK(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_channels=cfg.gat.hidden,
            out_channels=2,
            num_layers=cfg.gat.layers,
            heads=cfg.gat.heads,
            dropout=cfg.gat.dropout,
            num_fc_layers=cfg.gat.fc_layers,
            embedding_dim=cfg.gat.embedding_dim,
            conv_type=conv_type,
            edge_dim=cfg.gat.edge_dim if conv_type in ("gatv2", "transformer") else None,
        )
        ckpt = tmp_path / "best_model.pt"
        torch.save(model.state_dict(), ckpt)

        model2 = GATWithJK(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_channels=cfg.gat.hidden,
            out_channels=2,
            num_layers=cfg.gat.layers,
            heads=cfg.gat.heads,
            dropout=cfg.gat.dropout,
            num_fc_layers=cfg.gat.fc_layers,
            embedding_dim=cfg.gat.embedding_dim,
            conv_type=conv_type,
            edge_dim=cfg.gat.edge_dim if conv_type in ("gatv2", "transformer") else None,
        )
        model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

        for (n1, p1), (_n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_dqn_checkpoint_roundtrip(self, tmp_path):
        from graphids.config.resolver import resolve
        from graphids.core.models.dqn import QNetwork

        cfg = resolve("dqn", "large")
        net = QNetwork(
            15, cfg.fusion.alpha_steps, hidden_dim=cfg.dqn.hidden, num_layers=cfg.dqn.layers
        )

        ckpt = tmp_path / "best_model.pt"
        torch.save({"q_network": net.state_dict()}, ckpt)

        net2 = QNetwork(
            15, cfg.fusion.alpha_steps, hidden_dim=cfg.dqn.hidden, num_layers=cfg.dqn.layers
        )
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        net2.load_state_dict(sd["q_network"])

        for (n1, p1), (_n2, p2) in zip(net.named_parameters(), net2.named_parameters()):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_wrong_dims_crash_loudly(self, tmp_path):
        """Loading a checkpoint into a model with wrong dims must raise, not silently corrupt."""
        from graphids.config.resolver import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        large_cfg = resolve("vgae", "large")
        small_cfg = resolve("vgae", "small")
        conv_type = large_cfg.vgae.conv_type

        teacher = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(large_cfg.vgae.hidden_dims),
            latent_dim=large_cfg.vgae.latent_dim,
            encoder_heads=large_cfg.vgae.heads,
            embedding_dim=large_cfg.vgae.embedding_dim,
            conv_type=conv_type,
            edge_dim=large_cfg.vgae.edge_dim if conv_type in ("gatv2", "transformer") else None,
        )
        ckpt = tmp_path / "teacher.pt"
        torch.save(teacher.state_dict(), ckpt)

        student = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(small_cfg.vgae.hidden_dims),
            latent_dim=small_cfg.vgae.latent_dim,
            encoder_heads=small_cfg.vgae.heads,
            embedding_dim=small_cfg.vgae.embedding_dim,
            conv_type=conv_type,
            edge_dim=small_cfg.vgae.edge_dim if conv_type in ("gatv2", "transformer") else None,
        )

        with pytest.raises(RuntimeError, match="size mismatch"):
            student.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))


# ---------------------------------------------------------------------------
# 4. Teacher loading contract
# ---------------------------------------------------------------------------


class TestTeacherLoading:
    """Teacher loading must use the teacher's frozen config, not the student's."""

    def _save_teacher(self, tmp_path, model_type):
        """Save a teacher model + config to a temp directory."""
        from graphids.config import checkpoint_path, config_path, stage_dir
        from graphids.config.resolver import resolve

        stage_map = {"vgae": "autoencoder", "gat": "curriculum", "dqn": "fusion"}
        stage = stage_map[model_type]
        cfg = resolve(model_type, "large", experiment_root=str(tmp_path))

        sd = stage_dir(cfg, stage)
        sd.mkdir(parents=True, exist_ok=True)

        if model_type == "vgae":
            from graphids.core.models.vgae import GraphAutoencoderNeighborhood

            model = GraphAutoencoderNeighborhood(
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
                hidden_dims=list(cfg.vgae.hidden_dims),
                latent_dim=cfg.vgae.latent_dim,
                encoder_heads=cfg.vgae.heads,
                embedding_dim=cfg.vgae.embedding_dim,
                dropout=cfg.vgae.dropout,
                conv_type=cfg.vgae.conv_type,
                edge_dim=cfg.vgae.edge_dim
                if cfg.vgae.conv_type in ("gatv2", "transformer")
                else None,
            )
            torch.save(model.state_dict(), checkpoint_path(cfg, stage))
        elif model_type == "gat":
            from graphids.core.models.gat import GATWithJK

            model = GATWithJK(
                num_ids=NUM_IDS,
                in_channels=IN_CHANNELS,
                hidden_channels=cfg.gat.hidden,
                out_channels=2,
                num_layers=cfg.gat.layers,
                heads=cfg.gat.heads,
                dropout=cfg.gat.dropout,
                num_fc_layers=cfg.gat.fc_layers,
                embedding_dim=cfg.gat.embedding_dim,
                conv_type=cfg.gat.conv_type,
                edge_dim=cfg.gat.edge_dim
                if cfg.gat.conv_type in ("gatv2", "transformer")
                else None,
            )
            torch.save(model.state_dict(), checkpoint_path(cfg, stage))
        elif model_type == "dqn":
            from graphids.core.models.dqn import QNetwork

            net = QNetwork(
                15, cfg.fusion.alpha_steps, hidden_dim=cfg.dqn.hidden, num_layers=cfg.dqn.layers
            )
            torch.save({"q_network": net.state_dict()}, checkpoint_path(cfg, stage))

        cfg.save(config_path(cfg, stage))
        return str(checkpoint_path(cfg, stage))

    def test_vgae_teacher_loads_own_dims(self, tmp_path):
        from graphids.config.resolver import resolve
        from graphids.pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "vgae")
        student_cfg = resolve("vgae", "small")
        teacher = load_teacher(
            teacher_path, "vgae", student_cfg, NUM_IDS, IN_CHANNELS, torch.device("cpu")
        )
        assert teacher is not None

    def test_gat_teacher_loads_own_dims(self, tmp_path):
        from graphids.config.resolver import resolve
        from graphids.pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "gat")
        student_cfg = resolve("gat", "small")
        teacher = load_teacher(
            teacher_path, "gat", student_cfg, NUM_IDS, IN_CHANNELS, torch.device("cpu")
        )
        assert teacher is not None

    def test_dqn_teacher_loads_own_dims(self, tmp_path):
        from graphids.config.resolver import resolve
        from graphids.pipeline.stages.utils import load_teacher

        teacher_path = self._save_teacher(tmp_path, "dqn")
        student_cfg = resolve("dqn", "small")
        teacher = load_teacher(
            teacher_path, "dqn", student_cfg, NUM_IDS, IN_CHANNELS, torch.device("cpu")
        )
        assert teacher is not None

    def test_missing_teacher_config_raises(self, tmp_path):
        """Missing teacher config.json must raise, not silently fall back."""
        from graphids.config.resolver import resolve
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood
        from graphids.pipeline.stages.utils import load_teacher

        cfg = resolve("vgae", "large")
        model = GraphAutoencoderNeighborhood(
            num_ids=NUM_IDS,
            in_channels=IN_CHANNELS,
            hidden_dims=list(cfg.vgae.hidden_dims),
            latent_dim=cfg.vgae.latent_dim,
            encoder_heads=cfg.vgae.heads,
            embedding_dim=cfg.vgae.embedding_dim,
            conv_type=cfg.vgae.conv_type,
            edge_dim=cfg.vgae.edge_dim if cfg.vgae.conv_type in ("gatv2", "transformer") else None,
        )
        ckpt = tmp_path / "orphan" / "best_model.pt"
        ckpt.parent.mkdir(parents=True)
        torch.save(model.state_dict(), ckpt)
        with pytest.raises(FileNotFoundError, match="Teacher config not found"):
            load_teacher(str(ckpt), "vgae", cfg, NUM_IDS, IN_CHANNELS, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 5. Path construction
# ---------------------------------------------------------------------------


class TestPathConstruction:
    """Path logic must be consistent between config-based and string-based variants."""

    def test_aux_suffix(self):
        from graphids.config import PipelineConfig, checkpoint_path, run_id
        from graphids.config.schema import AuxiliaryConfig

        large = PipelineConfig(model_type="vgae", scale="large", dataset="hcrl_sa")
        small_kd = PipelineConfig(
            model_type="gat",
            scale="small",
            dataset="hcrl_sa",
            auxiliaries=[AuxiliaryConfig(type="kd")],
        )
        small_no_kd = PipelineConfig(model_type="gat", scale="small", dataset="hcrl_sa")

        assert run_id(large, "autoencoder") == "hcrl_sa/vgae_large_autoencoder"
        assert run_id(small_kd, "curriculum") == "hcrl_sa/gat_small_curriculum_kd"
        assert run_id(small_no_kd, "curriculum") == "hcrl_sa/gat_small_curriculum"

        # Ensure no double suffix
        assert "_kd_kd" not in run_id(small_kd, "curriculum")

    def test_checkpoint_path_matches_str_variant(self):
        """PipelineConfig checkpoint_path must produce the same string as checkpoint_path_str."""
        from graphids.config import PipelineConfig, checkpoint_path, checkpoint_path_str
        from graphids.config.schema import AuxiliaryConfig

        cases = [
            ("vgae", "large", False),
            ("gat", "small", True),
            ("gat", "small", False),
            ("dqn", "large", False),
        ]
        for model_type, scale, kd in cases:
            aux_list = [AuxiliaryConfig(type="kd")] if kd else []
            aux_str = "kd" if kd else ""
            cfg = PipelineConfig(
                dataset="hcrl_sa",
                model_type=model_type,
                scale=scale,
                auxiliaries=aux_list,
            )
            for stage in ["autoencoder", "curriculum", "fusion"]:
                py_path = str(checkpoint_path(cfg, stage))
                str_path = checkpoint_path_str("hcrl_sa", model_type, scale, stage, aux=aux_str)
                assert py_path == str_path, (
                    f"Path mismatch for ({model_type}, {scale}, {stage}, kd={kd}): "
                    f"cfg-based={py_path} vs str-based={str_path}"
                )

    def test_metrics_path_str(self):
        from graphids.config import PipelineConfig, metrics_path, metrics_path_str
        from graphids.config.schema import AuxiliaryConfig

        for model_type, scale, kd in [("vgae", "large", False), ("gat", "small", True)]:
            aux_list = [AuxiliaryConfig(type="kd")] if kd else []
            aux_str = "kd" if kd else ""
            cfg = PipelineConfig(
                dataset="hcrl_sa",
                model_type=model_type,
                scale=scale,
                auxiliaries=aux_list,
            )
            py_path = str(metrics_path(cfg, "evaluation"))
            str_path = metrics_path_str("hcrl_sa", model_type, scale, "evaluation", aux=aux_str)
            assert py_path == str_path

    def test_benchmark_path_str(self):
        from graphids.config import benchmark_path_str

        assert (
            benchmark_path_str("hcrl_sa", "vgae", "large", "autoencoder")
            == "experimentruns/hcrl_sa/vgae_large_autoencoder/benchmark.tsv"
        )
        assert (
            benchmark_path_str("set_01", "dqn", "small", "fusion", aux="kd")
            == "experimentruns/set_01/dqn_small_fusion_kd/benchmark.tsv"
        )


# ---------------------------------------------------------------------------
# 6. Validation catches missing prerequisites
# ---------------------------------------------------------------------------


class TestValidation:
    """Validator must catch missing files before SLURM submission."""

    def test_missing_teacher_checkpoint(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.schema import AuxiliaryConfig
        from graphids.pipeline.validate import validate

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="small",
            auxiliaries=[
                AuxiliaryConfig(
                    type="kd", model_path=str(tmp_path / "nonexistent" / "best_model.pt")
                )
            ],
            experiment_root=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Teacher checkpoint not found"):
            validate(cfg, "autoencoder")

    def test_missing_teacher_config(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.schema import AuxiliaryConfig
        from graphids.pipeline.validate import validate

        teacher_dir = tmp_path / "teacher_autoencoder"
        teacher_dir.mkdir(parents=True)
        (teacher_dir / "best_model.pt").write_bytes(b"fake")

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="small",
            auxiliaries=[AuxiliaryConfig(type="kd", model_path=str(teacher_dir / "best_model.pt"))],
            experiment_root=str(tmp_path),
        )
        with pytest.raises(ValueError, match="Teacher config not found"):
            validate(cfg, "autoencoder")

    def test_missing_prerequisite_config(self, tmp_path):
        from graphids.config import PipelineConfig, stage_dir
        from graphids.pipeline.validate import validate

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="gat",
            scale="large",
            experiment_root=str(tmp_path),
        )
        sd = stage_dir(cfg, "autoencoder")
        sd.mkdir(parents=True)
        (sd / "best_model.pt").write_bytes(b"fake")

        with pytest.raises(ValueError, match="config"):
            validate(cfg, "curriculum")

    def test_valid_config_passes(self, tmp_path):
        """A fully valid configuration must not raise."""
        from graphids.config import PipelineConfig
        from graphids.pipeline.validate import validate

        data_path = Path("data/automotive/hcrl_sa")
        if not data_path.exists():
            pytest.skip("Test data not available")

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="large",
            experiment_root=str(tmp_path),
        )
        validate(cfg, "autoencoder")


# ---------------------------------------------------------------------------
# 7. Frozen config propagation between stages
# ---------------------------------------------------------------------------


class TestFrozenConfigPropagation:
    """Downstream stages must load upstream configs with correct architecture dims."""

    def test_curriculum_loads_vgae_small_dims(self, tmp_path):
        """When curriculum loads frozen VGAE config, it must get small dims, not large.

        load_frozen_cfg resolves "autoencoder" → model_type="vgae" automatically,
        so a GAT config can find the VGAE autoencoder config.
        """
        from graphids.config import config_path, stage_dir
        from graphids.config.resolver import resolve
        from graphids.pipeline.stages.utils import load_frozen_cfg

        vgae_cfg = resolve(
            "vgae",
            "small",
            auxiliaries="kd_standard",
            dataset="hcrl_sa",
            experiment_root=str(tmp_path),
        )
        sd = stage_dir(vgae_cfg, "autoencoder")
        sd.mkdir(parents=True)
        vgae_cfg.save(config_path(vgae_cfg, "autoencoder"))

        # A GAT config with same dataset/scale/aux should resolve to the VGAE path
        curr_cfg = resolve(
            "gat",
            "small",
            auxiliaries="kd_standard",
            dataset="hcrl_sa",
            experiment_root=str(tmp_path),
        )
        frozen = load_frozen_cfg(curr_cfg, "autoencoder")
        assert frozen.vgae.hidden_dims == (80, 40, 16), (
            f"Got large dims {frozen.vgae.hidden_dims} instead of small (80, 40, 16)"
        )
        assert frozen.vgae.latent_dim == 16

    def test_missing_frozen_config_raises(self, tmp_path):
        from graphids.config import PipelineConfig
        from graphids.config.schema import AuxiliaryConfig
        from graphids.pipeline.stages.utils import load_frozen_cfg

        cfg = PipelineConfig(
            dataset="hcrl_sa",
            model_type="vgae",
            scale="small",
            auxiliaries=[AuxiliaryConfig(type="kd")],
            experiment_root=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match="Frozen config not found"):
            load_frozen_cfg(cfg, "autoencoder")


# ---------------------------------------------------------------------------
# 8. MMAP limit constant is consistent
# ---------------------------------------------------------------------------


class TestMmapConstant:
    def test_single_source_of_truth(self):
        from graphids.config.constants import MMAP_TENSOR_LIMIT

        assert isinstance(MMAP_TENSOR_LIMIT, int)
        assert MMAP_TENSOR_LIMIT > 0


# ---------------------------------------------------------------------------
# 9. Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Pydantic validates field constraints."""

    def test_invalid_model_type_raises(self):
        from graphids.config import PipelineConfig

        with pytest.raises(Exception):
            PipelineConfig(model_type="invalid_model")

    def test_invalid_scale_raises(self):
        from graphids.config import PipelineConfig

        with pytest.raises(Exception):
            PipelineConfig(scale="mega")

    def test_negative_lr_raises(self):
        from graphids.config.schema import TrainingConfig

        with pytest.raises(Exception):
            TrainingConfig(lr=-0.001)

    def test_sub_configs_are_frozen(self):
        from graphids.config import PipelineConfig

        cfg = PipelineConfig()
        with pytest.raises(Exception):
            cfg.vgae.latent_dim = 999

    def test_resolver_list_models(self):
        from graphids.config.resolver import list_models

        models = list_models()
        assert "vgae" in models
        assert "gat" in models
        assert "dqn" in models
        assert "large" in models["vgae"]
        assert "small" in models["vgae"]

    def test_resolver_list_auxiliaries(self):
        from graphids.config.resolver import list_auxiliaries

        aux = list_auxiliaries()
        assert "none" in aux
        assert "kd_standard" in aux

    def test_has_kd_property(self):
        from graphids.config import PipelineConfig
        from graphids.config.schema import AuxiliaryConfig

        cfg_no_kd = PipelineConfig()
        assert cfg_no_kd.has_kd is False
        assert cfg_no_kd.kd is None

        cfg_kd = PipelineConfig(auxiliaries=[AuxiliaryConfig(type="kd", model_path="/x")])
        assert cfg_kd.has_kd is True
        assert cfg_kd.kd.model_path == "/x"

    def test_active_arch(self):
        from graphids.config import PipelineConfig

        cfg = PipelineConfig(model_type="gat")
        assert cfg.active_arch == cfg.gat
