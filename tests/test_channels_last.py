"""Channels-last inputs are read in place; results must match the contiguous NCHW path."""
import pytest
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(x, y, padding):
    """float64 SSIM with the same Gaussian window and zero padding, differentiable by autograd."""
    x, y = x.double(), y.double()
    c = x.shape[1]
    g = torch.exp(-((torch.arange(11, device=x.device, dtype=torch.float64) - 5) ** 2) / 4.5)
    w = (g / g.sum())
    w = (w[:, None] * w[None, :]).expand(c, 1, 11, 11).contiguous()
    conv = lambda t: F.conv2d(t, w, padding=5, groups=c)
    mu1, mu2 = conv(x), conv(y)
    s1, s2, s12 = conv(x * x) - mu1 * mu1, conv(y * y) - mu2 * mu2, conv(x * y) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))
    return (m[:, :, 5:-5, 5:-5] if padding == "valid" else m).mean()


def make(shape, x_cl, y_cl):
    torch.manual_seed(7)
    x = torch.rand(shape, device="cuda")
    y = torch.rand(shape, device="cuda")
    if x_cl:
        x = x.contiguous(memory_format=torch.channels_last)
    if y_cl:
        y = y.contiguous(memory_format=torch.channels_last)
    return x, y


@pytest.mark.parametrize("padding", ["same", "valid"])
@pytest.mark.parametrize("shape", [(1, 3, 19, 27), (2, 4, 33, 17), (1, 1, 20, 21), (2, 3, 40, 64)])
@pytest.mark.parametrize("x_cl,y_cl", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("train", [True, False])
def test_channels_last_matches_contiguous_and_reference(padding, shape, x_cl, y_cl, train):
    x, y = make(shape, x_cl, y_cl)
    xc, yc = x.contiguous(), y.contiguous()  # NCHW copies with identical values
    a = x.clone().requires_grad_()
    b = xc.clone().requires_grad_()
    r = xc.clone().requires_grad_()
    va = fused_ssim(a, y, padding=padding, train=train)
    vb = fused_ssim(b, yc, padding=padding, train=train)
    vr = reference(r, yc, padding)
    # Same per-pixel arithmetic; only the order in which torch's mean sums the map differs.
    torch.testing.assert_close(va, vb, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(va.double(), vr, rtol=0, atol=2e-6)
    if train:
        va.backward()
        vb.backward()
        vr.backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        # fp32 fast-math kernel vs fp64 reference, elementwise
        torch.testing.assert_close(a.grad.double(), r.grad.double(), rtol=1e-3, atol=2e-9)
        if x_cl:
            assert a.grad.is_contiguous(memory_format=torch.channels_last)


def test_channels_last_view_of_nhwc_render():
    """The gsplat use case: an NHWC tensor passed as a permute(0, 3, 1, 2) view."""
    torch.manual_seed(3)
    nhwc_x = torch.rand(1, 37, 45, 3, device="cuda", requires_grad=True)
    nhwc_y = torch.rand(1, 37, 45, 3, device="cuda")
    v = fused_ssim(nhwc_x.permute(0, 3, 1, 2), nhwc_y.permute(0, 3, 1, 2), padding="valid")
    v.backward()
    x2 = nhwc_x.detach().permute(0, 3, 1, 2).contiguous().requires_grad_()
    v2 = fused_ssim(x2, nhwc_y.permute(0, 3, 1, 2).contiguous(), padding="valid")
    v2.backward()
    torch.testing.assert_close(v, v2, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(nhwc_x.grad.permute(0, 3, 1, 2), x2.grad, rtol=0, atol=0)
