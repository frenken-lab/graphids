"""Ownership, eager transitions, and CUDA partial-network training equivalence."""

from __future__ import annotations

import copy

import pytest
import torch

from graphids.core.data.datamodule.stream_lanes import TemporalStreamLaneBatch
from graphids.core.models.temporal import TemporalHybridModel
from graphids.core.models.temporal._hybrid.cuda_graph import (
    ScoredCudaGraph,
    eligibility_reason,
    mixed_scored_index,
)


def batch(channels=17, lanes=2):
    chunk = 32
    shape = (lanes, chunk)
    return TemporalStreamLaneBatch(
        src=torch.randint(0, 4, shape),
        dst=torch.randint(0, 4, shape),
        t=torch.arange(chunk).float().expand(lanes, -1).clone(),
        msg=torch.rand(lanes, chunk, channels),
        y=torch.randint(0, 2, shape),
        attack_type=torch.zeros(shape, dtype=torch.long),
        event_id=torch.arange(lanes * chunk).reshape(shape),
        is_scored=torch.ones(shape, dtype=torch.bool),
        stream_id=torch.zeros(shape, dtype=torch.long),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        lane_reset=torch.zeros(lanes, dtype=torch.bool),
        lane_stream_end=torch.zeros(lanes, dtype=torch.bool),
        reset_after=torch.zeros(shape, dtype=torch.bool),
        successor_index=torch.ones(shape, dtype=torch.long),
        successor_dst=torch.randint(0, 4, shape),
        successor_msg=torch.rand(lanes, chunk, channels),
        lane_section_end=torch.zeros(lanes, dtype=torch.bool),
    )


def model(channels=17, dropout=0.0, kernels="torch"):
    return TemporalHybridModel(
        num_ids=4,
        in_channels=channels,
        num_classes=2,
        objective="joint",
        scale="small",
        input={"hidden": 8, "embedding_dim": 4, "dropout": dropout},
        backbone={"type": "ssm_lite", "layers": 1, "dropout": dropout},
        anomaly={"mode": "nll"},
        rhythm={"enabled": True},
        motif={"enabled": True, "length": 3, "embedding_dim": 4, "time_dim": 4},
        scan={
            "cuda_graph": True,
            "rhythm_kernel": kernels,
            "time_kernel": kernels,
            "motif_writer": kernels,
        },
    )


def assert_tensors(actual, expected, *, gradients=False):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if value is None:
            assert expected[key] is None
        else:
            torch.testing.assert_close(
                value,
                expected[key],
                rtol=3e-4 if gradients else 1e-4,
                atol=3e-5 if gradients else 2e-5,
                msg=lambda msg, key=key: f"{key}: {msg}",
            )


@pytest.mark.parametrize(
    "field", ["lane_reset", "lane_section_end", "lane_stream_end", "reset_after"]
)
def test_boundary_fallback(field):
    b = batch()
    assert eligibility_reason(b) is None
    getattr(b, field).view(-1)[0] = True
    assert eligibility_reason(b) == field


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("valid_mask", False, "partial_lanes"),
        ("is_scored", False, "warmup_or_mixed"),
        ("successor_index", -1, "missing_successor"),
    ],
)
def test_metadata_fallback(field, value, reason):
    b = batch()
    getattr(b, field).view(-1)[0] = value
    assert eligibility_reason(b) == reason


@pytest.mark.parametrize("channels", [4, 17])
def test_regular_path_preserves_targets_losses_and_state(channels):
    torch.manual_seed(123)
    m = model(channels)
    b = batch(channels)
    _, state = m._forward_with_state(b)
    state = m._detach_state(state)
    out, next_state = m._forward_with_state(b, state)
    regular, next_regular = m._forward_lane_with_state(b, state, regular_scored=True)
    assert_tensors(regular, out)
    assert_tensors(next_regular, next_state)
    assert_tensors(
        m._loss_terms(regular, b, regular_scored=True), m._loss_terms(out, b)
    )


def test_cpu_graph_falls_back_and_keeps_checkpoint_keys():
    m, b = model(), batch()
    keys = tuple(m.state_dict())
    m.on_before_batch_transfer(b, 0)
    m.log = lambda *args, **kwargs: None
    m.training_step(b, 0).backward()
    assert tuple(m.state_dict()) == keys
    assert m.cuda_graph_report()["fallback_batches"] == {"cpu": 1}
    m.on_load_checkpoint({})
    assert m.cuda_graph_report() == {"enabled": False}


def test_mixed_index_requires_fixed_shape_and_no_boundaries():
    b = batch(lanes=256)
    b.is_scored[0].zero_()
    b.warmup_mask = ~b.is_scored
    index = mixed_scored_index(b)
    assert index is not None and index.numel() == 8160
    assert index[0] == 32 and index[-1] == 8191
    b.is_scored[1, 0] = False
    assert mixed_scored_index(b) is None
    b.is_scored[1, 0] = True
    b.lane_reset[0] = True
    assert mixed_scored_index(b) is None


def test_graph_setting_prepares_common_mixed_batch():
    m = model()
    b = batch(lanes=256)
    b.is_scored[0].zero_()
    b.warmup_mask = ~b.is_scored
    m.on_before_batch_transfer(b, 0)
    assert m._cuda_graph_reason == "warmup_or_mixed"
    assert m._cpu_scored_index is not None
    assert m._cpu_scored_index.numel() == 8160


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_graph_outputs_gradients_and_adam():
    torch.manual_seed(512)
    eager = model(dropout=0.0).cuda()
    captured = copy.deepcopy(eager)
    cpu_batch = batch(lanes=256)
    cpu_batch.is_scored[0].zero_()
    cpu_batch.warmup_mask = ~cpu_batch.is_scored
    scored_index = mixed_scored_index(cpu_batch)
    assert scored_index is not None
    b = cpu_batch.to("cuda")
    with torch.no_grad():
        _, eager_state = eager._forward_with_state(b)
    graph_state = {k: None if v is None else v.clone() for k, v in eager_state.items()}
    eager_opt, graph_opt = eager.configure_optimizers(), captured.configure_optimizers()
    graph = ScoredCudaGraph()
    regular = batch(lanes=256).to("cuda")
    captured._cpu_scored_events = regular.y.numel()
    assert graph.run(captured, regular, graph_state, None) is not None, graph.setup_error
    for _ in range(2):
        eager_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        out, next_state = eager._forward_with_state(b, eager_state)
        terms = eager._loss_terms(out, b)
        captured._cpu_scored_events = 8160
        captured._cpu_scored_index = scored_index
        result = graph.run(captured, b, graph_state, "warmup_or_mixed")
        assert result is not None, graph.mixed_setup_error
        mixed_out, mixed_state, mixed_terms = result
        assert_tensors(mixed_out, out)
        assert_tensors(mixed_state, next_state)
        assert_tensors(mixed_terms, terms)
        terms["total"].backward()
        mixed_terms["total"].backward()
        assert_tensors(
            dict(captured.named_parameters()), dict(eager.named_parameters()), gradients=True
        )
        assert_tensors(
            dict((n, p.grad) for n, p in captured.named_parameters()),
            dict((n, p.grad) for n, p in eager.named_parameters()), gradients=True,
        )
        eager_opt.step()
        graph_opt.step()
        assert_tensors(captured.state_dict(), eager.state_dict(), gradients=True)
        for actual, expected in zip(graph_opt.state.values(), eager_opt.state.values(), strict=True):
            assert_tensors(actual, expected, gradients=True)
        eager_state, graph_state = eager._detach_state(next_state), mixed_state
    assert graph.report()["mixed_replays"] == 2
    assert graph.report()["scored_replay_events"] == 24512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kernels", ["torch", "triton"])
@pytest.mark.parametrize("channels", [4, 17])
def test_cuda_consecutive_outputs_gradients_adam_and_ownership(kernels, channels):
    torch.manual_seed(456)
    eager = model(channels, kernels=kernels).cuda()
    captured = copy.deepcopy(eager)
    b = batch(channels).to("cuda")
    _, state = eager._forward_with_state(b)
    state = eager._detach_state(state)
    graph_state = {k: None if v is None else v.clone() for k, v in state.items()}
    graph = ScoredCudaGraph()
    eager_opt, graph_opt = eager.configure_optimizers(), captured.configure_optimizers()
    initial_keys = tuple(captured.state_dict())
    snapshots = []
    for step in range(3):
        eager_opt.zero_grad(set_to_none=True)
        graph_opt.zero_grad(set_to_none=True)
        out, next_state = eager._forward_with_state(b, state)
        terms = eager._loss_terms(out, b)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        weights = {k: v.clone() for k, v in captured.state_dict().items()}
        live = {k: None if v is None else v.clone() for k, v in graph_state.items()}
        captured._cpu_scored_events = b.y.numel()
        result = graph.run(captured, b, graph_state, None)
        assert result is not None, graph.setup_error
        regular, next_regular, regular_terms = result
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(
            cuda_rng, torch.cuda.get_rng_state()
        )  # no dropout in this comparison
        assert_tensors(captured.state_dict(), weights)
        assert_tensors(graph_state, live)
        assert graph_opt.state == {} if step == 0 else bool(graph_opt.state)
        assert_tensors(regular, out)
        assert_tensors(regular_terms, terms)
        assert_tensors(next_regular, next_state)
        terms["total"].backward()
        regular_terms["total"].backward()
        assert_tensors(
            dict((n, p.grad) for n, p in captured.named_parameters()),
            dict((n, p.grad) for n, p in eager.named_parameters()),
            gradients=True,
        )
        eager_opt.step()
        graph_opt.step()
        assert_tensors(captured.state_dict(), eager.state_dict(), gradients=True)
        for actual, expected in zip(
            graph_opt.state.values(), eager_opt.state.values(), strict=True
        ):
            assert_tensors(actual, expected, gradients=True)
        snapshots.append(
            (
                next_regular,
                {k: None if v is None else v.clone() for k, v in next_regular.items()},
                regular_terms,
                {k: v.detach().clone() for k, v in regular_terms.items()},
            )
        )
        state, graph_state = eager._detach_state(next_state), next_regular
        b.t += 32
    for saved_state, expected_state, saved_terms, expected_terms in snapshots:
        assert_tensors(saved_state, expected_state)
        assert_tensors(saved_terms, expected_terms)
    assert tuple(captured.state_dict()) == initial_keys
    assert graph.replays == 3
    # Graph -> eager section/terminal reset -> initialized graph resumes.
    b.lane_section_end[0] = True
    _, state = eager._forward_with_state(b, state)
    _, graph_state = captured._forward_with_state(b, graph_state)
    assert_tensors(graph_state, state)
    b.lane_section_end.zero_()
    graph_state = captured._detach_state(graph_state)
    state = eager._detach_state(state)
    torch.testing.assert_close(
        graph.run(captured, b, graph_state, None)[0]["features"],
        eager._forward_with_state(b, state)[0]["features"],
        rtol=1e-4,
        atol=2e-5,
    )
    fresh = model(channels, kernels=kernels).cuda()
    fresh.load_state_dict(captured.state_dict())
    fresh.eval()
    captured.eval()
    assert_tensors(fresh.forward_temporal(b), captured.forward_temporal(b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_dropout_rng_and_fresh_randomness():
    torch.manual_seed(789)
    m, b = model(dropout=0.1).cuda(), batch().to("cuda")
    with torch.no_grad():
        _, state = m._forward_with_state(b)
    state = m._detach_state(state)
    graph = ScoredCudaGraph()
    # Compare first replay RNG with an already captured replay after restoring the seed.
    before = torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        expected, expected_state = m._forward_with_state(b, state)
    eager_after = torch.cuda.get_rng_state().clone()
    torch.cuda.set_rng_state(before)
    first = graph.run(m, b, state, None)
    assert first is not None, graph.setup_error
    after = torch.cuda.get_rng_state().clone()
    assert_tensors(first[0], expected)
    assert_tensors(first[1], expected_state)
    assert torch.equal(after, eager_after)
    torch.cuda.set_rng_state(before)
    repeat = graph.run(m, b, state, None)
    assert_tensors(first[0], repeat[0])
    assert torch.equal(after, torch.cuda.get_rng_state())
    fresh = graph.run(m, b, state, None)
    assert not torch.equal(first[0]["features"], fresh[0]["features"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lightning_graph_transitions_warmup_and_checkpoint(tmp_path):
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

    batches = [batch() for _ in range(9)]
    batches[0].lane_reset.fill_(True)
    batches[0].is_scored.zero_()
    batches[2].valid_mask[0, -1] = False
    batches[3].lane_section_end[0] = True
    batches[4].lane_reset[0] = True
    batches[5].lane_stream_end[0] = True
    batches[5].reset_after[0, -1] = True
    batches[6].successor_index[0, -1] = -1
    m = model(dropout=0.1, kernels="triton")
    loader = DataLoader(batches, batch_size=None, num_workers=0)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(m, train_dataloaders=loader)
    assert m._optimizer_steps == 8
    report = m.cuda_graph_report()
    assert report["replays"] == 3, report
    assert report["fallback_batches"]["warmup_or_mixed"] == 1
    assert not report["disabled"], report
    path = tmp_path / "graph.ckpt"
    trainer.save_checkpoint(path)
    loaded = TemporalHybridModel.load_from_checkpoint(path).cuda().eval()
    m.cuda().eval()
    b = batches[-1].to("cuda")
    assert_tensors(loaded.forward_temporal(b), m.forward_temporal(b))
    trainer.validate(m, dataloaders=loader, ckpt_path=str(path), verbose=False)
    assert m.cuda_graph_report() == {"enabled": False}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_setup_failure_preserves_live_state_rng_and_allows_eager(monkeypatch):
    m, b = model(dropout=0.1).cuda(), batch().to("cuda")
    with torch.no_grad():
        _, state = m._forward_with_state(b)
    state = m._detach_state(state)
    weights = {key: value.clone() for key, value in m.state_dict().items()}
    saved_state = {
        key: None if value is None else value.clone() for key, value in state.items()
    }
    before = torch.cuda.get_rng_state().clone()

    def fail(*args, **kwargs):
        torch.rand(8, device="cuda")
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", fail)
    graph = ScoredCudaGraph()
    with pytest.warns(RuntimeWarning, match="injected setup failure"):
        assert graph.run(m, b, state, None) is None
    assert torch.equal(before, torch.cuda.get_rng_state())
    assert_tensors(m.state_dict(), weights)
    assert_tensors(state, saved_state)
    out, _ = m._forward_with_state(b, state)
    m._loss_terms(out, b)["total"].backward()
    assert graph.disabled
    assert graph.run(m, b, state, None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_preserves_gradient_accumulation():
    eager = model().cuda()
    captured = copy.deepcopy(eager)
    b = batch().to("cuda")
    with torch.no_grad():
        _, state = eager._forward_with_state(b)
    graph = ScoredCudaGraph()
    for step in range(2):
        b.msg = b.msg + 0.2 * step
        out, _ = eager._forward_with_state(b, state)
        eager._loss_terms(out, b)["total"].backward()
        result = graph.run(captured, b, state, None)
        assert result is not None, graph.setup_error
        result[2]["total"].backward()
    assert_tensors(
        {n: p.grad for n, p in captured.named_parameters()},
        {n: p.grad for n, p in eager.named_parameters()},
        gradients=True,
    )
