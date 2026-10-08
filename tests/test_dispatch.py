"""Host-only dispatch checks (no GPU needed): which layout reaches the native 2D kernel."""
import torch
import fused_ssim as ssim


def test_non_cuda_channels_last_is_made_contiguous(monkeypatch):
    """Metal and SYCL kernels write flat NCHW buffers, so they must never see channels-last."""
    received = []

    def forward(c1, c2, x, y, train):
        received.append(x)
        return (torch.ones_like(x),) + tuple(torch.zeros_like(x) for _ in range(3))

    monkeypatch.setattr(ssim, "fusedssim", forward, raising=False)
    x = torch.rand(1, 3, 13, 17).contiguous(memory_format=torch.channels_last)
    assert not x.is_contiguous()
    assert ssim.fused_ssim(x, x).item() == 1
    assert received[0].is_contiguous()
    torch.testing.assert_close(received[0], x)
