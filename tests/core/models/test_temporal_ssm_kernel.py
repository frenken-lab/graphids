import pytest
import torch

from graphids.core.models.temporal._hybrid.kernels.ssm import ssm_recurrence


def test_ssm_primitive_build_preserves_opt_in():
    from types import SimpleNamespace

    from graphids.primitives_models import TemporalHybridCfg
    cfg = TemporalHybridCfg.model_validate({'backbone': {'type': 'ssm_lite'}, 'scan': {'ssm_kernel': 'triton'}})
    model = cfg.build()
    model.prepare_from_datamodule(SimpleNamespace(num_ids=4, in_channels=17, num_classes=2))
    assert model.backbone.cells[0].ssm_kernel == 'triton'
    with pytest.raises(ValueError):
        TemporalHybridCfg.model_validate({'scan': {'ssm_kernel': 'triton'}})


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(0, 8), (1, 8), (17, 64), (256, 64)])
def test_ssm_forward_and_all_input_gradients(device, shape):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA unavailable')
        pytest.importorskip('triton')
    generator = torch.Generator().manual_seed(42)
    values = [torch.randn(shape, generator=generator).to(device).requires_grad_() for _ in range(4)]
    reference = [value.detach().clone().requires_grad_() for value in values]
    actual = ssm_recurrence(*values, kernel='triton')
    expected = ssm_recurrence(*reference)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    if device == 'cuda':
        assert type(actual.grad_fn).__name__ == '_RecurrenceBackward'
    # Noncontiguous upstream gradient exercises the backward materialization.
    grad = torch.randn((shape[1], shape[0]), generator=generator).t().to(device)
    actual.backward(grad)
    expected.backward(grad)
    for value, ref in zip(values, reference, strict=True):
        torch.testing.assert_close(value.grad, ref.grad, rtol=1e-4, atol=2e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_ssm_strided_double_fallback_and_saturation(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    for dtype in (torch.float32, torch.float64):
        values = [torch.linspace(-80, 80, 24, device=device, dtype=dtype).reshape(3, 8)[:, ::2].requires_grad_()
                  for _ in range(4)]
        actual = ssm_recurrence(*values, kernel='triton')
        expected = ssm_recurrence(*values)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        assert all(torch.isfinite(value.grad).all() for value in values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_ssm_fused_saturated_inputs():
    values = [torch.linspace(-80, 80, 1024, device='cuda').view(16, 64).requires_grad_() for _ in range(4)]
    reference = [value.detach().clone().requires_grad_() for value in values]
    result = ssm_recurrence(*values, kernel='triton')
    expected = ssm_recurrence(*reference)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)
    result.sum().backward()
    expected.sum().backward()
    for value, ref in zip(values, reference, strict=True):
        torch.testing.assert_close(value.grad, ref.grad, rtol=1e-4, atol=2e-6)
