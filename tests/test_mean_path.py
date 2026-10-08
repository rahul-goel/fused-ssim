"""Mean-only training path (train=True, img1 requires grad) against the map path."""
import pytest
import torch
import fused_ssim as ssim
from fused_ssim import fused_ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(shape, channels_last):
    torch.manual_seed(29)
    x = torch.rand(shape, device="cuda")
    y = torch.rand_like(x)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    return x, y


def map_mean(x, y, padding):
    """The map path, reduced by torch: what fused_ssim returned before the mean path existed."""
    return ssim.FusedSSIMMap.apply(.0001, .0009, x, y, padding, True, 2).mean()


def assert_grad_close(actual, expected):
    # elementwise, scaled to the gradient magnitude: 1e-5 of the largest entry plus 1e-5 relative
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())


@pytest.mark.parametrize("padding", ["same", "valid"])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("shape", [(2, 3, 19, 27), (1, 4, 64, 80), (1, 1, 33, 40)])
@pytest.mark.parametrize("outer", ["identity", "nonlinear"])
def test_mean_path_matches_map_path(padding, channels_last, shape, outer):
    x, y = make(shape, channels_last)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    f = (lambda v: v) if outer == "identity" else (lambda v: (1 - v) ** 2)
    va = fused_ssim(a, y, padding=padding)
    vb = map_mean(b, y, padding)
    # only the order of the final summation differs
    torch.testing.assert_close(va, vb, rtol=1e-6, atol=1e-7)
    f(va).backward()
    f(vb).backward()
    assert_grad_close(a.grad, b.grad)
    if channels_last:
        assert a.grad.is_contiguous(memory_format=torch.channels_last)


def test_mean_path_is_used_and_map_path_otherwise(monkeypatch):
    x, y = make((1, 3, 32, 32), False)
    calls = []
    real = ssim.fusedssim_sum
    monkeypatch.setattr(ssim, "fusedssim_sum", lambda *a: calls.append(1) or real(*a))
    fused_ssim(x.clone().requires_grad_(), y)
    assert calls == [1]
    fused_ssim(x, y)                                  # no grad needed
    fused_ssim(x.clone().requires_grad_(), y, train=False)
    assert calls == [1]


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (1, 3, 8, 12), (1, 3, 12, 8), (1, 3, 10, 12), (1, 3, 10, 10)])
def test_empty_valid_region_value_and_gradient(shape):
    """valid padding crops 5 pixels per side; nothing is left when a side is <= 10."""
    x, y = make(shape, True)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    expected = map_mean(b, y, "valid")                # mean of an empty tensor
    actual = fused_ssim(a, y, padding="valid")
    assert torch.isnan(expected) and torch.isnan(actual)
    expected.backward()
    actual.backward()
    assert torch.count_nonzero(b.grad) == 0
    assert torch.count_nonzero(a.grad) == 0


@pytest.mark.parametrize("deterministic", [False, True])
def test_mean_path_is_deterministic(monkeypatch, deterministic):
    """The mean path is used in both modes; value and gradient are bitwise reproducible."""
    x, y = make((2, 3, 67, 93), False)
    called = []
    real = ssim.fusedssim_sum
    monkeypatch.setattr(ssim, "fusedssim_sum", lambda *a: called.append(1) or real(*a))
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        runs = []
        for _ in range(3):
            a = x.clone().requires_grad_()
            v = fused_ssim(a, y, padding="valid")
            v.backward()
            runs.append((v.detach(), a.grad))
    finally:
        torch.use_deterministic_algorithms(previous)
    assert len(called) == 3
    for v, g in runs[1:]:
        assert torch.equal(v, runs[0][0]) and torch.equal(g, runs[0][1])
    b = x.clone().requires_grad_()
    map_mean(b, y, "valid").backward()
    assert_grad_close(runs[0][1], b.grad)


@pytest.mark.parametrize("deterministic", [False, True])
def test_mean_path_on_side_stream(deterministic):
    """Partial-sum buffer, kernel, sum and backward must all be ordered on the current stream."""
    x, y = make((1, 3, 37, 45), False)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        ref = x.clone().requires_grad_()
        expected = fused_ssim(ref, y, padding="valid")
        expected.backward()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            img = x.clone().requires_grad_()
            actual = fused_ssim(img, y.clone(), padding="valid")
            actual.backward()
        stream.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert_grad_close(img.grad, ref.grad)
