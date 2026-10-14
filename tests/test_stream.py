"""The CUDA kernels must run on PyTorch's current stream, not the legacy default stream.

The work below runs on a side stream behind a long `_sleep`. A kernel launched on the default
stream does not wait for it, so it reads the inputs before they are written and returns wrong
values (typically 0) or wrong gradients.
"""
import pytest
import torch
from fused_ssim import fused_ssim, fused_ssim3d

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def run(fn, shape, side_stream):
    torch.manual_seed(0)
    base = torch.rand(shape, device="cuda")
    target = torch.rand(shape, device="cuda")
    torch.cuda.synchronize()
    if not side_stream:
        x = base.clone().requires_grad_()
        out = fn(x, target.clone())
        out.backward()
        torch.cuda.synchronize()
        return out.detach(), x.grad
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)  # about 25 ms: the launches below must queue behind it
        x = base.clone().requires_grad_()
        out = fn(x, target.clone())
        out.backward()
    stream.synchronize()
    return out.detach(), x.grad


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_2d_forward_backward_on_side_stream(padding):
    fn = lambda x, y: fused_ssim(x, y, padding=padding)
    shape = (1, 3, 37, 45)
    ref, ref_grad = run(fn, shape, False)
    out, grad = run(fn, shape, True)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(grad, ref_grad)


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_3d_forward_backward_on_side_stream(padding):
    fn = lambda x, y: fused_ssim3d(x, y, padding=padding)
    shape = (1, 1, 14, 24, 28)
    ref, ref_grad = run(fn, shape, False)
    out, grad = run(fn, shape, True)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(grad, ref_grad)
