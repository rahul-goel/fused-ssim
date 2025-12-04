import torch
from fused_ssim import fused_l1_ssim_loss, fused_ssim
import time

ssim_weight = 0.2
image_shape = [3, 1080, 1920]


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_ssim_map_loss(image, gt_image, ssim_weight):
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - ssim_weight) * Ll1 + ssim_weight * (1.0 - ssim_value)
    loss.backward()
    return loss, image.grad.clone()


def fused_l1_ssim_loss_map(image, gt_image, ssim_weight):
    loss = fused_l1_ssim_loss(image.unsqueeze(0), gt_image.unsqueeze(0), ssim_weight)
    loss.backward()
    return loss, image.grad.clone()


def test_fused_l1_ssim_loss():
    # generate test data
    gt_image_data = torch.rand(image_shape, device="cuda")
    image_data = torch.rand(image_shape, device="cuda")

    # 1st result: before version
    image1 = image_data.clone().requires_grad_(True)
    gt_image1 = gt_image_data.clone()
    before_loss, before_grad = l1_ssim_map_loss(image1, gt_image1, ssim_weight)

    # 2nd result: after version, using the same data but different tensor
    image2 = image_data.clone().requires_grad_(True)
    gt_image2 = gt_image_data.clone()
    after_loss, after_grad = fused_l1_ssim_loss_map(image2, gt_image2, ssim_weight)

    assert torch.isclose(before_loss, after_loss)
    assert torch.isclose(before_grad, after_grad).all()


def benchmark_fused_l1_ssim_loss():
    print("benchmarking with shape", image_shape)

    gt_image = torch.rand(image_shape, device="cuda")
    image = torch.rand(image_shape, device="cuda").requires_grad_(True)

    iterations = 100
    begin = time.time()
    for _ in range(iterations):
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - ssim_weight) * Ll1 + ssim_weight * (1.0 - ssim_value)
        loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    time_forward_backward_before = (end - begin) / iterations * 1000
    print(
        "l1 + fused_ssim Time (forward + backward):", time_forward_backward_before, "ms"
    )

    begin = time.time()
    for _ in range(iterations):
        loss = fused_l1_ssim_loss(
            image.unsqueeze(0), gt_image.unsqueeze(0), ssim_weight
        )
        loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    time_forward_backward_after = (end - begin) / iterations * 1000
    print(
        f"fused_l1_ssim_loss Time (forward + backward): {time_forward_backward_after} ms ({(time_forward_backward_before - time_forward_backward_after) / time_forward_backward_before * 100:.2f}% faster)"
    )


if __name__ == "__main__":
    for _ in range(100):
        test_fused_l1_ssim_loss()

    benchmark_fused_l1_ssim_loss()
