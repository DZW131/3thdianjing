import torch
import torch.nn.functional as F


def _sliding_positions(length, window_size, stride):
    if length <= window_size:
        return [0]
    positions = list(range(0, max(1, length - window_size + 1), stride))
    if positions[-1] != length - window_size:
        positions.append(length - window_size)
    return positions


def _make_blend_weight(height, width, device):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    distance = torch.maximum(torch.abs(xx), torch.abs(yy))
    weight = 1.0 - 0.5 * distance
    return torch.clamp(weight, min=0.25).unsqueeze(0).unsqueeze(0)


def sliding_window_inference(model, image, num_classes, window_size, stride):
    batch_size, _, height, width = image.shape
    if batch_size != 1:
        raise ValueError("Sliding window inference currently expects batch_size=1, got {}".format(batch_size))

    if height <= window_size and width <= window_size:
        return model(image)

    device = image.device
    logits_sum = torch.zeros((1, num_classes, height, width), device=device)
    weight_sum = torch.zeros((1, 1, height, width), device=device)

    y_positions = _sliding_positions(height, window_size, stride)
    x_positions = _sliding_positions(width, window_size, stride)

    for top in y_positions:
        for left in x_positions:
            bottom = min(height, top + window_size)
            right = min(width, left + window_size)

            patch = image[:, :, top:bottom, left:right]
            patch_height = bottom - top
            patch_width = right - left

            pad_height = max(0, window_size - patch_height)
            pad_width = max(0, window_size - patch_width)
            if pad_height or pad_width:
                patch = F.pad(patch, (0, pad_width, 0, pad_height), mode="constant", value=0.0)

            logits = model(patch)[:, :, :patch_height, :patch_width]
            weight = _make_blend_weight(patch_height, patch_width, device=device)

            logits_sum[:, :, top:bottom, left:right] += logits * weight
            weight_sum[:, :, top:bottom, left:right] += weight

    return logits_sum / torch.clamp(weight_sum, min=1e-6)


def predict_logits(model, image, num_classes, inference_mode="direct", window_size=None, stride=None):
    if inference_mode != "sliding":
        return model(image)
    if not window_size or not stride:
        raise ValueError("Sliding inference requires both window_size and stride.")
    return sliding_window_inference(model, image, num_classes, window_size, stride)
