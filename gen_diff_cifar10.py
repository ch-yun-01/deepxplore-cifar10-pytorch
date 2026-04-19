# gen_diff_cifar10.py
import os
import argparse
import random

import torch
import torchvision
import torchvision.transforms as transforms

from models_torch import load_model
from configs import bcolors
from utils_torch import (
    save_image,
    neuron_covered,
    init_coverage_table_single,
    get_feature_num_neurons,
    update_coverage_from_features,
    neuron_to_cover,
    normalize_gradient,
    constraint_light,
    constraint_occl,
    constraint_black,
    get_argmax_label,
    disagreement_found,
    make_objective,
)

# CIFAR-10 정규화 통계값 (학습 시 사용한 값과 동일해야 함)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def clamp_to_valid_range(img: torch.Tensor) -> torch.Tensor:
    """
    정규화된 입력값을 원본 [0,1] 픽셀 공간 기준으로 clamp한다.
    정규화 공식: x_norm = (x - mean) / std
    역산하면: x = x_norm * std + mean
    따라서 유효 범위: [(0 - mean) / std, (1 - mean) / std]
    """
    mean = torch.tensor(CIFAR10_MEAN, device=img.device).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=img.device).view(1, 3, 1, 1)
    lo = (0.0 - mean) / std   # 정규화 공간에서의 최솟값
    hi = (1.0 - mean) / std   # 정규화 공간에서의 최댓값
    return torch.clamp(img, lo, hi)


def main():
    parser = argparse.ArgumentParser(
        description="Difference-inducing input generation for CIFAR-10"
    )
    parser.add_argument("transformation", choices=["light", "occl", "blackout"])
    parser.add_argument("weight_diff",    type=float)
    parser.add_argument("weight_nc",      type=float)
    parser.add_argument("step",           type=float)
    parser.add_argument("seeds",          type=int)
    parser.add_argument("grad_iterations",type=int)
    parser.add_argument("threshold",      type=float)

    parser.add_argument("-t", "--target_model", choices=[0, 1, 2], default=0, type=int)

    parser.add_argument("--start_x",  type=int, default=0)
    parser.add_argument("--start_y",  type=int, default=0)
    parser.add_argument("--occl_w",   type=int, default=8)
    parser.add_argument("--occl_h",   type=int, default=8)

    parser.add_argument("--model1_path", default="./checkpoints/model1.pth")
    parser.add_argument("--model2_path", default="./checkpoints/model2.pth")
    parser.add_argument("--model3_path", default="./checkpoints/model3.pth")
    parser.add_argument("--output_dir",  default="./generated_inputs")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # FIX 5: 변수명을 cifar10_transform으로 명확히 구분
    # (원본 코드의 normalize 변수명이 gradient normalize 함수와 혼동될 수 있었음)
    cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=cifar10_transform,
    )

    # 모델 로드
    model1 = load_model(args.model1_path, device)
    model2 = load_model(args.model2_path, device)
    model3 = load_model(args.model3_path, device)

    # coverage 초기화 — dummy forward로 feature 구조 파악
    dummy_img, _ = testset[0]
    dummy_img = dummy_img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, feat1 = model1.forward_with_features(dummy_img)
        _, feat2 = model2.forward_with_features(dummy_img)
        _, feat3 = model3.forward_with_features(dummy_img)

    layer_info1 = get_feature_num_neurons(feat1)
    layer_info2 = get_feature_num_neurons(feat2)
    layer_info3 = get_feature_num_neurons(feat3)

    model_layer_dict1 = init_coverage_table_single("model1", layer_info1)
    model_layer_dict2 = init_coverage_table_single("model2", layer_info2)
    model_layer_dict3 = init_coverage_table_single("model3", layer_info3)

    print(bcolors.OKBLUE + "Start generating inputs..." + bcolors.ENDC)

    found_count = 0

    for seed_idx in range(args.seeds):
        img, _ = random.choice(testset)

        # FIX 4: gradient 추적이 필요한 gen_img를 루프 진입 전에 한 번만 준비
        # 매 iteration 내부에서 requires_grad_(True)를 반복하면
        # 새 텐서 할당 후 grad 추적이 끊길 수 있음
        gen_img  = img.unsqueeze(0).to(device)
        orig_img = gen_img.clone().detach()

        # 시드가 이미 불일치를 유발하는지 확인
        with torch.no_grad():
            logits1, feat1 = model1.forward_with_features(gen_img)
            logits2, feat2 = model2.forward_with_features(gen_img)
            logits3, feat3 = model3.forward_with_features(gen_img)

        label1 = get_argmax_label(logits1)
        label2 = get_argmax_label(logits2)
        label3 = get_argmax_label(logits3)

        if disagreement_found([label1, label2, label3]):
            print(
                bcolors.OKGREEN +
                f"[seed {seed_idx}] already differs: {label1},{label2},{label3}" +
                bcolors.ENDC
            )
            # coverage update
            update_coverage_from_features(feat1, "model1", model_layer_dict1, args.threshold)
            update_coverage_from_features(feat2, "model2", model_layer_dict2, args.threshold)
            update_coverage_from_features(feat3, "model3", model_layer_dict3, args.threshold)

            save_image(
                os.path.join(args.output_dir, f"already_{seed_idx}_{label1}_{label2}_{label3}.png"),
                gen_img,
            )
            found_count += 1
            continue

        orig_label = label1

        for iters in range(args.grad_iterations):

            # FIX 4: 매 step마다 새 텐서에서 grad를 추적
            gen_img = gen_img.detach().requires_grad_(True)

            # FIX 3: forward 경로를 forward_with_features로 통일
            # (아래 불일치 확인도 동일한 함수를 쓰므로 결과가 일관됨)
            logits1, feat1 = model1.forward_with_features(gen_img)
            logits2, feat2 = model2.forward_with_features(gen_img)
            logits3, feat3 = model3.forward_with_features(gen_img)

            n1 = neuron_to_cover(model_layer_dict1)
            n2 = neuron_to_cover(model_layer_dict2)
            n3 = neuron_to_cover(model_layer_dict3)

            objective = make_objective(
                logits_list=[logits1, logits2, logits3],
                orig_label=orig_label,
                target_model=args.target_model,
                weight_diff=args.weight_diff,
                feature_dicts=[feat1, feat2, feat3],
                target_neurons=[n1, n2, n3],
                weight_nc=args.weight_nc,
            )

            model1.zero_grad()
            model2.zero_grad()
            model3.zero_grad()

            objective.backward()

            grads = normalize_gradient(gen_img.grad.detach())

            if args.transformation == "light":
                grads = constraint_light(grads)
            elif args.transformation == "occl":
                grads = constraint_occl(
                    grads,
                    start_point=(args.start_x, args.start_y),
                    occlusion_size=(args.occl_w, args.occl_h),
                )
            elif args.transformation == "blackout":
                grads = constraint_black(grads)

            # FIX 1: 정규화된 공간에서의 유효 범위로 clamp
            # (원본: clamp(0,1)은 정규화 전 픽셀 범위이므로 정규화된 텐서에 부적절)
            gen_img = clamp_to_valid_range(gen_img.detach() + grads * args.step)

            # FIX 3: 불일치 확인도 forward_with_features로 통일
            with torch.no_grad():
                logits1_new, feat1_new = model1.forward_with_features(gen_img)
                logits2_new, feat2_new = model2.forward_with_features(gen_img)
                logits3_new, feat3_new = model3.forward_with_features(gen_img)

            pred1 = get_argmax_label(logits1_new)
            pred2 = get_argmax_label(logits2_new)
            pred3 = get_argmax_label(logits3_new)

            if disagreement_found([pred1, pred2, pred3]):
                print(
                    bcolors.OKGREEN +
                    f"[seed {seed_idx}, iter {iters}] diff: {pred1},{pred2},{pred3}" +
                    bcolors.ENDC
                )

                # FIX 2: coverage update 추가 (원본 코드에서 누락됐던 부분)
                update_coverage_from_features(feat1_new, model_layer_dict1, args.threshold)
                update_coverage_from_features(feat2_new, model_layer_dict2, args.threshold)
                update_coverage_from_features(feat3_new, model_layer_dict3, args.threshold)

                c1, t1, r1 = neuron_covered(model_layer_dict1)
                c2, t2, r2 = neuron_covered(model_layer_dict2)
                c3, t3, r3 = neuron_covered(model_layer_dict3)
                avg_nc = (c1 + c2 + c3) / float(t1 + t2 + t3)
                print(
                    bcolors.OKGREEN +
                    f"  NC — M1:{r1:.3f} M2:{r2:.3f} M3:{r3:.3f} avg:{avg_nc:.3f}" +
                    bcolors.ENDC
                )

                base = f"{args.transformation}_{seed_idx}_{iters}_{pred1}_{pred2}_{pred3}"
                save_image(os.path.join(args.output_dir, base + ".png"),      gen_img)
                save_image(os.path.join(args.output_dir, base + "_orig.png"), orig_img)

                found_count += 1
                break

    # 최종 coverage 출력
    c1, t1, r1 = neuron_covered(model_layer_dict1)
    c2, t2, r2 = neuron_covered(model_layer_dict2)
    c3, t3, r3 = neuron_covered(model_layer_dict3)
    avg_nc = (c1 + c2 + c3) / float(t1 + t2 + t3)
    print(bcolors.OKBLUE +
          f"Finished. Found {found_count}/{args.seeds} inputs. "
          f"Avg NC: {avg_nc:.4f}" +
          bcolors.ENDC)


if __name__ == "__main__":
    main()