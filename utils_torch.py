import os
import random
import numpy as np
import torch
import imageio.v2 as imageio


def deprocess_image(img_tensor):
    """
    img_tensor: [1, C, H, W] or [C, H, W], float in [0,1]
    return: HWC uint8
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def save_image(path, img_tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = deprocess_image(img_tensor)
    imageio.imwrite(path, img)


def neuron_covered(model_layer_dict):
    covered = sum(model_layer_dict.values())
    total = len(model_layer_dict)
    ratio = covered / float(total) if total > 0 else 0.0
    return covered, total, ratio


def init_coverage_table_single(model_name, layer_names):
    """
    model_layer_dict key format:
        (model_name, layer_name, neuron_idx)
    """
    model_layer_dict = {}
    for layer_name, num_neurons in layer_names.items():
        for idx in range(num_neurons):
            model_layer_dict[(model_name, layer_name, idx)] = False
    return model_layer_dict


def get_feature_num_neurons(features):
    """
    features: dict[str, Tensor]
    Conv feature map [B,C,H,W] -> channel 수 C
    FC-ish feature [B,C] 또는 [B,C,1,1] -> C
    """
    out = {}
    for name, feat in features.items():
        if feat.dim() == 4:
            out[name] = feat.size(1)
        elif feat.dim() == 2:
            out[name] = feat.size(1)
        else:
            out[name] = feat.shape[1]
    return out


def update_coverage_from_features(features, model_name, model_layer_dict, threshold=0.5):
    """
    Channel mean activation > threshold 이면 covered 처리
    """
    for layer_name, feat in features.items():
        if feat.dim() == 4:
            act = feat.mean(dim=(0, 2, 3))
        elif feat.dim() == 2:
            act = feat.mean(dim=0)
        else:
            continue

        for idx in range(act.size(0)):
            key = (model_name, layer_name, idx)
            if key in model_layer_dict and act[idx].item() > threshold:
                model_layer_dict[key] = True


def neuron_to_cover(model_layer_dict):
    """
    아직 cover되지 않은 뉴런 우선 선택
    Returns:
        (model_name, layer_name, neuron_idx)
    """
    not_covered = [k for k, v in model_layer_dict.items() if not v]
    if len(not_covered) > 0:
        return random.choice(not_covered)
    return random.choice(list(model_layer_dict.keys()))


def normalize_gradient(grad):
    denom = torch.sqrt(torch.mean(torch.square(grad))) + 1e-8
    return grad / denom


def constraint_light(grads):
    """
    전체 밝기 변화 비슷하게 유도
    grads: [1,C,H,W]
    """
    g = grads.mean(dim=(1, 2, 3), keepdim=True)
    return g.expand_as(grads)


def constraint_occl(grads, start_point, occlusion_size):
    """
    지정 박스 내부만 gradient 유지
    start_point: (x, y)
    occlusion_size: (w, h)
    """
    x, y = start_point
    w, h = occlusion_size

    mask = torch.zeros_like(grads)
    _, _, H, W = grads.shape

    x2 = min(x + w, W)
    y2 = min(y + h, H)
    mask[:, :, y:y2, x:x2] = 1.0
    return grads * mask


def constraint_black(grads):
    """
    더 어두워지는 방향만 허용
    """
    return torch.clamp(grads, max=0.0)


def get_argmax_label(logits):
    return torch.argmax(logits, dim=1).item()


def disagreement_found(preds):
    return not all(p == preds[0] for p in preds)


def make_objective(logits_list, orig_label, target_model, weight_diff,
                   feature_dicts, target_neurons, weight_nc):
    """
    logits_list: [logits1, logits2, logits3]
    target_neurons:
        [
          (model_name, layer_name, idx),
          (model_name, layer_name, idx),
          (model_name, layer_name, idx)
        ]
    """
    losses = []
    for i, logits in enumerate(logits_list):
        val = logits[0, orig_label]
        if i == target_model:
            losses.append(-weight_diff * val)
        else:
            losses.append(val)

    neuron_losses = []
    for feature_dict, (_, layer_name, neuron_idx) in zip(feature_dicts, target_neurons):
        feat = feature_dict[layer_name]
        if feat.dim() == 4:
            neuron_losses.append(feat[:, neuron_idx, :, :].mean())
        elif feat.dim() == 2:
            neuron_losses.append(feat[:, neuron_idx].mean())
        else:
            feat2 = feat.reshape(feat.shape[0], feat.shape[1], -1)
            neuron_losses.append(feat2[:, neuron_idx].mean())

    total = sum(losses) + weight_nc * sum(neuron_losses)
    return total