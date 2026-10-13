"""Actual optimized-kernel gates; run only in a GPU allocation."""

import pytest
import torch

from graphids.core.models.graph_mamba import VehicleGraphMambaJoint
from tests.core.models.test_graph_mamba import make_batch

pytestmark = [
    pytest.mark.slurm,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"),
]


def test_optimized_forward_backward_and_streaming(tmp_path):
    batch, _, _ = make_batch(tmp_path)
    batch = batch.to("cuda")
    torch.manual_seed(42)
    official = VehicleGraphMambaJoint(width=16, dropout=0, backend="official").cuda()
    reference = VehicleGraphMambaJoint(width=16, dropout=0, backend="reference").cuda()
    reference.load_state_dict(official.state_dict())
    expected, actual = reference(batch), official(batch)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=2e-4, atol=2e-5)
    loss, _ = official.losses(actual, batch)
    ref_loss, _ = reference.losses(expected, batch)
    loss.backward()
    ref_loss.backward()
    for (name, p), (other, q) in zip(
        official.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, rtol=5e-3, atol=5e-5)
    official.eval()
    with torch.no_grad():
        streamed, _ = official.stream(batch)
    for name in actual:
        torch.testing.assert_close(actual[name], streamed[name], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_is_finite(tmp_path, dtype):
    batch, _, _ = make_batch(tmp_path)
    batch = batch.to("cuda")
    net = VehicleGraphMambaJoint(width=32, dropout=0, backend="official").cuda()
    with torch.autocast("cuda", dtype=dtype):
        loss, _ = net.losses(net(batch), batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())
