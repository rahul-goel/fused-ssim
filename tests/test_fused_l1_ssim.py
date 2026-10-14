"""fused_l1_ssim against F.l1_loss + fused_ssim, and its input contract."""
import pytest
import torch
import torch.nn.functional as F
import fused_ssim as ssim
from fused_ssim import fused_l1_ssim, fused_ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(shape=(2, 3, 19, 27), channels_last=False):
    torch.manual_seed(29)
    x = torch.rand(shape, device="cuda")
    y = torch.rand_like(x)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    return x, y


def assert_grad_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item())


@pytest.mark.parametrize("padding", ["same", "valid"])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("outer", ["sum", "nonlinear", "l1_only", "ssim_only"])
def test_matches_separate_losses(padding, channels_last, outer):
    x, y = make(channels_last=channels_last)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    l1_a, s_a = fused_l1_ssim(a, y, padding=padding)
    l1_b, s_b = F.l1_loss(b, y), fused_ssim(b, y, padding=padding)
    torch.testing.assert_close(l1_a, l1_b, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(s_a, s_b, rtol=1e-6, atol=1e-7)
    loss = {"sum": lambda l, s: .8 * l + .2 * (1 - s),
            "nonlinear": lambda l, s: (.8 * l + .2 * (1 - s)) ** 2,
            "l1_only": lambda l, s: l,
            "ssim_only": lambda l, s: 1 - s}[outer]
    loss(l1_a, s_a).backward()
    loss(l1_b, s_b).backward()
    assert_grad_close(a.grad, b.grad)


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (1, 3, 8, 12), (1, 3, 12, 8), (1, 3, 10, 12)])
def test_empty_valid_region(shape):
    x, y = make(shape, True)
    a = x.clone().requires_grad_()
    l1, s = fused_l1_ssim(a, y, padding="valid")
    torch.testing.assert_close(l1, F.l1_loss(x, y), rtol=1e-6, atol=1e-7)
    assert torch.isnan(s)
    s.backward()
    assert torch.count_nonzero(a.grad) == 0


def test_rejects_invalid_inputs_before_kernel_launch(monkeypatch):
    monkeypatch.setattr(ssim, "fusedssim_sum", lambda *a: pytest.fail("invalid input reached the kernel"))
    x, y = make()
    with pytest.raises(ValueError, match="CUDA"):
        fused_l1_ssim(x.cpu(), y.cpu())
    with pytest.raises(ValueError, match="same CUDA device"):
        fused_l1_ssim(x, y.cpu())
    with pytest.raises(ValueError, match="float32"):
        fused_l1_ssim(x, y.half())
    with pytest.raises(ValueError, match="4D"):
        fused_l1_ssim(x, y[:, :, :-1])
    with pytest.raises(ValueError, match="4D"):
        fused_l1_ssim(x[0], y[0])
    with pytest.raises(ValueError, match="fixed target"):
        fused_l1_ssim(x, y.requires_grad_())


@pytest.mark.parametrize("deterministic", [False, True])
def test_bitwise_reproducible(deterministic):
    x, y = make((2, 3, 67, 93))
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        runs = []
        for _ in range(3):
            a = x.clone().requires_grad_()
            l1, s = fused_l1_ssim(a, y, padding="valid")
            (l1 + s).backward()
            runs.append((l1.detach(), s.detach(), a.grad))
    finally:
        torch.use_deterministic_algorithms(previous)
    for r in runs[1:]:
        assert all(torch.equal(u, v) for u, v in zip(r, runs[0]))


@pytest.mark.parametrize("deterministic", [False, True])
def test_on_side_stream(deterministic):
    x, y = make((1, 3, 37, 45))
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        ref = x.clone().requires_grad_()
        l1_e, s_e = fused_l1_ssim(ref, y, padding="valid")
        (l1_e + s_e).backward()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            img = x.clone().requires_grad_()
            l1, s = fused_l1_ssim(img, y.clone(), padding="valid")
            (l1 + s).backward()
        stream.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous)
    torch.testing.assert_close(l1, l1_e, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(s, s_e, rtol=1e-6, atol=1e-7)
    assert_grad_close(img.grad, ref.grad)
